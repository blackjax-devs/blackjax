import numpy as np
import jax
import jax.numpy as jnp
import scipy.optimize
import arviz as az
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
    target_swap_accept=0.30,
    num_temperatures=1,
    random_seed=42
):
    """
    Production-grade user API to sample a PyMC model using the parallel Slingshot engine.
    Features native JAX-driven Adaptive Parallel Tempering via Sigmoid-Ratio Space.
    """
    dim = len(pymc_model.value_vars)
    var_names = [v.name for v in pymc_model.value_vars]
    
    # 1. Graph Compilation & MAP Optimization
    raw_logp = get_jaxified_logp(pymc_model, negative_logp=True)
    logdensity_fn = lambda theta: raw_logp([theta[i] for i in range(dim)])
    
    def neg_log_density(theta): return -logdensity_fn(theta)
    val_and_grad_fn = jax.jit(jax.value_and_grad(neg_log_density))
    
    def scipy_objective(theta_np):
        val, grad = val_and_grad_fn(jnp.array(theta_np))
        return np.array(val).astype(np.float64), np.array(grad).astype(np.float64)
        
    opt_result = scipy.optimize.minimize(
        scipy_objective, jnp.zeros(dim), method="BFGS", jac=True
    )
    map_estimate = jnp.array(opt_result.x)
    
    # 2. State Initialization
    rng_key = jax.random.PRNGKey(random_seed)
    init_key, warmup_key, sample_key = jax.random.split(rng_key, 3)
    
    jitter = jax.random.normal(init_key, (chains, dim)) * 0.01
    warm_start_positions = map_estimate + jitter

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
        swap_rates_out = None

    else:
        # Initial logit ratios mapping to stable 0.5 geometric reduction steps
        init_logit_r = jnp.zeros(num_temperatures - 1)
        
        def init_temp_level(beta):
            tempered_fn = lambda theta: beta * logdensity_fn(theta)
            states_level = jax.vmap(lambda p: blackjax.slingshot(tempered_fn, step_size=1.0, num_proposals=proposals).init(p))(warm_start_positions)
            da_states_level = jax.vmap(lambda ss: init_adaptation(ss, dim))(jnp.ones(chains) * 0.1)
            return states_level, da_states_level
            
        init_betas = jnp.array([1.0 / (2.0**i) for i in range(num_temperatures)])
        states, da_states = jax.vmap(init_temp_level)(init_betas)

        @jax.jit
        def tempered_warmup_step(carry, input_tuple):
            states, da_states, logit_r = carry
            step_key, iteration = input_tuple
            sample_key, swap_key = jax.random.split(step_key)
            
            # Construct strictly monotonic betas from logit ratio space
            r = jax.nn.sigmoid(logit_r)
            betas_list = [1.0]
            for idx in range(num_temperatures - 1):
                betas_list.append(betas_list[-1] * r[idx])
            betas = jnp.array(betas_list)
            
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
            
            gamma = 1.0 / jnp.power(iteration + 1, 0.6)
            
            for i in range(num_temperatures - 1):
                j = i + 1
                state_i = jax.tree.map(lambda x: x[i], next_states)
                state_j = jax.tree.map(lambda x: x[j], next_states)
                logp_i = jax.vmap(logdensity_fn)(state_i.position)
                logp_j = jax.vmap(logdensity_fn)(state_j.position)
                
                log_alpha = (betas[i] - betas[j]) * (logp_j - logp_i)
                mean_p_accept = jnp.mean(jnp.minimum(1.0, jnp.exp(log_alpha)))
                
                swap_key, subkey = jax.random.split(swap_key)
                do_swap = jnp.log(jax.random.uniform(subkey, shape=(chains,))) < log_alpha
                
                def update_full_tree(full_leaf, leaf_i, leaf_j):
                    mask = jnp.reshape(do_swap, (chains,) + (1,) * (leaf_i.ndim - 1))
                    return full_leaf.at[i].set(jnp.where(mask, leaf_j, leaf_i)).at[j].set(jnp.where(mask, leaf_i, leaf_j))
                next_states = jax.tree.map(update_full_tree, next_states, state_i, state_j)
                
                # Adapt the ratio boundaries directly to target acceptance rate
                logit_r = logit_r.at[i].add(-gamma * (mean_p_accept - target_swap_accept))
                
            return (next_states, next_da_states, logit_r), None

        @jax.jit
        def tempered_sample_step(carry, step_key):
            carry_states, betas = carry
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
            
            step_swaps = jnp.zeros(num_temperatures - 1)
            for i in range(num_temperatures - 1):
                j = i + 1
                state_i = jax.tree.map(lambda x: x[i], next_states)
                state_j = jax.tree.map(lambda x: x[j], next_states)
                logp_i = jax.vmap(logdensity_fn)(state_i.position)
                logp_j = jax.vmap(logdensity_fn)(state_j.position)
                log_alpha = (betas[i] - betas[j]) * (logp_j - logp_i)
                
                swap_key, subkey = jax.random.split(swap_key)
                do_swap = jnp.log(jax.random.uniform(subkey, shape=(chains,))) < log_alpha
                step_swaps = step_swaps.at[i].set(jnp.mean(do_swap.astype(jnp.float32)))
                
                def update_full_tree(full_leaf, leaf_i, leaf_j):
                    mask = jnp.reshape(do_swap, (chains,) + (1,) * (leaf_i.ndim - 1))
                    return full_leaf.at[i].set(jnp.where(mask, leaf_j, leaf_i)).at[j].set(jnp.where(mask, leaf_i, leaf_j))
                next_states = jax.tree.map(update_full_tree, next_states, state_i, state_j)
                
                pos_i, pos_j = positions[i], positions[j]
                mask_pos = jnp.reshape(do_swap, (chains,) + (1,) * (pos_i.ndim - 1))
                positions = positions.at[i].set(jnp.where(mask_pos, pos_j, pos_i)).at[j].set(jnp.where(mask_pos, pos_i, pos_j))
                
            return (next_states, betas), (positions, step_swaps)

        # Run Warmup Scan
        warmup_keys = jax.random.split(warmup_key, tune)
        warmup_inputs = (warmup_keys, jnp.arange(tune))
        (states, da_states, adapted_logit_r), _ = jax.lax.scan(tempered_warmup_step, (states, da_states, init_logit_r), warmup_inputs)
        final_step_sizes = jnp.exp(da_states.log_step_size_bar)
        final_choleskys = da_states.cholesky
        
        # Construct final production betas from optimized warmup ratios
        final_r = jax.nn.sigmoid(adapted_logit_r)
        final_betas_list = [1.0]
        for idx in range(num_temperatures - 1):
            final_betas_list.append(final_betas_list[-1] * final_r[idx])
        final_betas = jnp.array(final_betas_list)
        
        # Run Production Scan
        sample_keys = jax.random.split(sample_key, draws)
        _, (full_grid_positions, swap_history) = jax.lax.scan(tempered_sample_step, (states, final_betas), sample_keys)
        positions = full_grid_positions[:, 0, :, :]
        
        swap_history_np = np.array(swap_history)
        swap_rates_out = np.repeat(swap_history_np[np.newaxis, :, :], chains, axis=0)

    # 3. Direct Native ArviZ InferenceData Generation
    posterior_dict = {}
    for idx, name in enumerate(var_names):
        posterior_dict[name] = np.swapaxes(positions[:, :, idx], 0, 1)
        
    sample_stats = {}
    if swap_rates_out is not None:
        sample_stats["swap_acceptance_rate"] = swap_rates_out

    idata = az.from_dict(
        posterior=posterior_dict,
        sample_stats=sample_stats if num_temperatures > 1 else None
    )
    return idata
