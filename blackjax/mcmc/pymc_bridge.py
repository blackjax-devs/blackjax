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
    random_seed=42
):
    """
    High-level user API to sample a PyMC model using the parallel Slingshot engine.
    Returns a dictionary of posterior samples mapped to variable names.
    """
    dim = pymc_model.ndim
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
    
    # 3. State Initialization
    rng_key = jax.random.PRNGKey(random_seed)
    init_key, warmup_key, sample_key = jax.random.split(rng_key, 3)
    
    jitter = jax.random.normal(init_key, (chains, dim)) * 0.01
    warm_start_positions = map_estimate + jitter
    
    states = jax.vmap(lambda p: blackjax.slingshot(logdensity_fn, step_size=1.0, num_proposals=proposals).init(p))(warm_start_positions)
    da_states = jax.vmap(lambda ss: init_adaptation(ss, dim))(jnp.ones(chains) * 0.1)
    
    # 4. Warmup Scan Execution
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
            next_da_state = next_da_state._replace(
                log_step_size=jnp.maximum(next_da_state.log_step_size, min_log_step),
                log_step_size_bar=jnp.maximum(next_da_state.log_step_size_bar, min_log_step)
            )
            return next_state, next_da_state

        next_states, next_da_states = jax.vmap(single_chain_warmup)(keys, states, da_states)
        return (next_states, next_da_states), None

    warmup_keys = jax.random.split(warmup_key, tune)
    (states, da_states), _ = jax.lax.scan(warmup_step, (states, da_states), warmup_keys)
    
    final_step_sizes = jnp.exp(da_states.log_step_size_bar)
    final_choleskys = da_states.cholesky
    
    # 5. Production Sampling Execution
    @jax.jit
    def sample_step(carry_states, step_key):
        keys = jax.random.split(step_key, chains)
        def single_chain_sample(key, state, step_size, cholesky):
            algo = blackjax.slingshot(logdensity_fn, step_size=step_size, num_proposals=proposals, cholesky=cholesky)
            next_state, _ = algo.step(key, state)
            return next_state, next_state.position
            
        next_states, positions = jax.vmap(single_chain_sample)(keys, carry_states, final_step_sizes, final_choleskys)
        return next_states, positions

    sample_keys = jax.random.split(sample_key, draws)
    _, positions = jax.lax.scan(sample_step, states, sample_keys)
    
    # 6. Map to Named Dictionary Output
    samples_dict = {}
    for idx, name in enumerate(var_names):
        samples_dict[name] = np.array(positions[:, :, idx])
        
    return samples_dict