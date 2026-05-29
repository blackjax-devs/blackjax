import numpy as np
import jax
import jax.numpy as jnp
import scipy.optimize
from pymc.sampling.jax import get_jaxified_logp
from blackjax.mcmc.slingshot import init_adaptation, dual_averaging_step
import blackjax

def sample_slingshot(
    pymc_model, 
    draws=1000, 
    tune=1000, 
    chains=16, 
    proposals=1000, 
    target_accept=0.65, 
    num_temperatures=1,
    random_seed=42
):
    """
    High-level user API to sample a PyMC model using the parallel Slingshot engine.
    Supports native JAX-driven Parallel Tempering via the num_temperatures parameter.
    """
    dim = len(pymc_model.value_vars)
    var_names = [v.name for v in pymc_model.value_vars]
    
    # 1. Internal Graph Compilation
    raw_logp = get_jaxified_logp(pymc_model, negative_logp=True)
    logdensity_fn = lambda theta: raw_logp([theta[i] for i in range(dim)])
    
    # 2. Internal MAP Optimization
    def neg_log_density(theta): return -logdensity_fn(theta)
    val_and_grad_fn = jax.jit(jax.value_and_grad(neg_log_density))
    
    def scipy_objective(theta_np):
        val, grad = val_and_grad_fn(jnp.array(theta_np))
        return np.array(val).astype(np.float64), np.array(grad).astype(np.float64)
        
    opt_result = scipy.optimize.minimize(
        scipy_objective, jnp.zeros(dim), method="BFGS", jac=True
    )
    map_estimate = jnp.array(opt_result.x)
    
    # 3. State Initialization Setup
    rng_key = jax.random.PRNGKey(random_seed)
    init_key, warmup_key, sample_key = jax.random.split(rng_key, 3)
    
    jitter = jax.random.normal(init_key, (chains, dim)) * 0.01
    warm_start_positions = map_estimate + jitter

    # Handle standard vs tempered layout paths
    if num_temperatures == 1:
        states = jax.vmap(lambda p: blackjax.slingshot(logdensity_fn, step_size=1.0, num_proposals=proposals).init(p))(warm_start_positions)
        da_states = jax.vmap(lambda ss: init_adaptation(ss, dim))(jnp.ones(chains) * 0.1)
        
        @jax.jit
        def warmup_step(carry, step_key):
            states, da_states = carry
            keys = jax.random.split(step_key, chains)
            def single_chain_warmup(key, state, da_state):
                step_size = jnp.exp(da_state.log_step_size)
                algo = blackjax.slingshot(logdensity_fn, step_size=step_size, num_proposals=proposals, cholesky=da_state.cholesky)
                next_state, info = algo.step(key, state)
                acc_rate = getattr(info, "acceptance_rate", target_accept)
                next_da_state = dual_averaging_step(da_state, acc_rate, next_state.position, target_rate=target_accept)
                min_log_step = jnp.log(0.05)
                return next_state, next_da_state._replace(
                    log_step_size=jnp.maximum(next_da_state.log_step_size, min_log_step),
                    log_step_size_bar=jnp.maximum(next_da_state.log_step_size_bar, min_log_step)
                )
            return jax.vmap(single_chain_warmup)(keys, states, da_states), None

        @jax.jit
        def sample_step(carry_states, step_key):
            keys = jax.random.split(step_key, chains)
            def single_chain_sample(key, state, step_size, cholesky):
                return blackjax.slingshot(logdensity_fn, step_size=step_size, num_proposals=proposals, cholesky=cholesky).step(key, state)
            next_states, info = jax.vmap(single_chain_sample)(keys, carry_states, final_step_sizes, final_choleskys)
            return next_states, next_states.position

        warmup_keys = jax.random.split(warmup_key, tune)
        (states, da_states), _ = jax.lax.scan(warmup_step, (states, da_states), warmup_keys)
        final_step_sizes = jnp.exp(da_states.log_step_size_bar)
        final_choleskys = da_states.cholesky
        
        sample_keys = jax.random.split(sample_key, draws)
        _, positions = jax.lax.scan(sample_step, states, sample_keys)

    else:
        # Parallel Tempering Multi-Grid Architecture
        betas = jnp.array([1.0 / (2.0**i) for i in range(num_temperatures)])
        
        def init_temp_level(beta):
            tempered_fn = lambda theta: beta * logdensity_fn(theta)
            states_level = jax.vmap(lambda p: blackjax.slingshot(tempered_fn, step_size=1.0, num_proposals=proposals).init(p))(warm_start_positions)
            da_states_level = jax.vmap(lambda ss: init_adaptation(ss, dim))(jnp.ones(chains) * 0.1)
            return states_level, da_states_level
            
        states, da_states = jax.vmap(init_temp_level)(betas)

        @jax.jit
        def tempered_warmup_step(carry, step_key):
            states, da_states = carry
            sample_key, swap_key = jax.random.split(step_key)
            
            def single_temp_step(beta, states_level, da_states_level, keys_level):
                tempered_fn = lambda theta: beta * logdensity_fn(theta)
                def single_chain_step(key, state, da_state):
                    step_size = jnp.exp(da_state.log_step_size)
                    algo = blackjax.slingshot(tempered_fn, step_size=step_size, num_proposals=proposals, cholesky=da_state.cholesky)
                    next_state, info = algo.step(key, state)
                    acc_rate = getattr(info, "acceptance_rate", target_accept)
                    next_da_state = dual_averaging_step(da_state, acc_rate, next_state.position, target_rate=target_accept)
                    min_log_step = jnp.log(0.05)
                    return next_state, next_da_state._replace(
                        log_step_size=jnp.maximum(next_da_state.log_step_size, min_log_step),
                        log_step_size_bar=jnp.maximum(next_da_state.log_step_size_bar, min_log_step)
                    )
                return jax.vmap(single_chain_step)(keys_level, states_level, da_states_level)

            keys = jax.random.split(sample_key, num_temperatures * chains).reshape(num_temperatures, chains, 2)
            next_states, next_da_states = jax.vmap(single_temp_step)(betas, states, da_states, keys)
            
            for i in range(num_temperatures - 1):
                j = i + 1
                state_i = jax.tree.map(lambda x: x[i], next_states)
                state_j = jax.tree.map(lambda x: x[j], next_states)
                logp_i = jax.vmap(logdensity_fn)(state_i.position)
                logp_j = jax.vmap(logdensity_fn)(state_j.position)
                log_alpha = (betas[i] - betas[j]) * (logp_j - logp_i)
                
                swap_key, subkey = jax.random.split(swap_key)
                do_swap = jnp.log(jax.random.uniform(subkey, shape=(chains,))) < log_alpha
                
                def update_full_tree(full_leaf, leaf_i, leaf_j):
                    mask = jnp.reshape(do_swap, (chains,) + (1,) * (leaf_i.ndim - 1))
                    return full_leaf.at[i].set(jnp.where(mask, leaf_j, leaf_i)).at[j].set(jnp.where(mask, leaf_i, leaf_j))
                next_states = jax.tree.map(update_full_tree, next_states, state_i, state_j)
                
            return (next_states, next_da_states), None

        @jax.jit
        def tempered_sample_step(carry_states, step_key):
            sample_key, swap_key = jax.random.split(step_key)
            
            def single_temp_sample(beta, states_level, step_sizes_level, choleskys_level, keys_level):
                tempered_fn = lambda theta: beta * logdensity_fn(theta)
                def single_chain_sample(key, state, step_size, cholesky):
                    algo = blackjax.slingshot(tempered_fn, step_size=step_size, num_proposals=proposals, cholesky=cholesky)
                    next_state, _ = algo.step(key, state)
                    return next_state, next_state.position
                return jax.vmap(single_chain_sample)(keys_level, states_level, step_sizes_level, choleskys_level)
                
            keys = jax.random.split(sample_key, num_temperatures * chains).reshape(num_temperatures, chains, 2)
            next_states, positions = jax.vmap(single_temp_sample)(betas, carry_states, final_step_sizes, final_choleskys, keys)
            
            for i in range(num_temperatures - 1):
                j = i + 1
                state_i = jax.tree.map(lambda x: x[i], next_states)
                state_j = jax.tree.map(lambda x: x[j], next_states)
                logp_i = jax.vmap(logdensity_fn)(state_i.position)
                logp_j = jax.vmap(logdensity_fn)(state_j.position)
                log_alpha = (betas[i] - betas[j]) * (logp_j - logp_i)
                
                swap_key, subkey = jax.random.split(swap_key)
                do_swap = jnp.log(jax.random.uniform(subkey, shape=(chains,))) < log_alpha
                
                def update_full_tree(full_leaf, leaf_i, leaf_j):
                    mask = jnp.reshape(do_swap, (chains,) + (1,) * (leaf_i.ndim - 1))
                    return full_leaf.at[i].set(jnp.where(mask, leaf_j, leaf_i)).at[j].set(jnp.where(mask, leaf_i, leaf_j))
                next_states = jax.tree.map(update_full_tree, next_states, state_i, state_j)
                
                pos_i, pos_j = positions[i], positions[j]
                mask_pos = jnp.reshape(do_swap, (chains,) + (1,) * (pos_i.ndim - 1))
                positions = positions.at[i].set(jnp.where(mask_pos, pos_j, pos_i)).at[j].set(jnp.where(mask_pos, pos_i, pos_j))
                
            return next_states, positions

        warmup_keys = jax.random.split(warmup_key, tune)
        (states, da_states), _ = jax.lax.scan(tempered_warmup_step, (states, da_states), warmup_keys)
        final_step_sizes = jnp.exp(da_states.log_step_size_bar)
        final_choleskys = da_states.cholesky
        
        sample_keys = jax.random.split(sample_key, draws)
        _, full_grid_positions = jax.lax.scan(tempered_sample_step, states, sample_keys)
        positions = full_grid_positions[:, 0, :, :]

    # 4. Map Named Dictionary Output
    samples_dict = {}
    for idx, name in enumerate(var_names):
        samples_dict[name] = np.array(positions[:, :, idx])
        
    return samples_dict
