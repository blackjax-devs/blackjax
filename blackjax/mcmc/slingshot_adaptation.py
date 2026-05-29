import jax
import jax.numpy as jnp
import blackjax
from blackjax.mcmc.slingshot import init_adaptation, dual_averaging_step

def run_warmup(
    rng_key: jax.random.PRNGKey,
    logdensity_fn,
    initial_positions: jnp.ndarray,
    num_proposals: int = 1000,
    num_warmup: int = 1000,
    target_rate: float = 0.65
):
    """
    High-level warmup wrapper for the Slingshot MP-MCMC sampler.
    Automatically handles step-size dual averaging and dense Cholesky preconditioning.
    """
    num_chains, dim = initial_positions.shape
    
    # 1. Initialize states and dual-averaging vectors
    init_chain_vmap = jax.vmap(lambda pos: blackjax.slingshot(logdensity_fn, 1.0, num_proposals).init(pos))
    states = init_chain_vmap(initial_positions)
    
    init_adapt_vmap = jax.vmap(lambda ss: init_adaptation(ss, dim))
    da_states = init_adapt_vmap(jnp.ones(num_chains) * 0.1) 
    
    # 2. The JIT-compiled warmup loop
    @jax.jit
    def warmup_step(carry, step_key):
        states, da_states = carry
        keys = jax.random.split(step_key, num_chains)
        
        def single_chain_warmup(key, state, da_state):
            step_size = jnp.exp(da_state.log_step_size) 
            algo = blackjax.slingshot(
                logdensity_fn, 
                step_size=step_size, 
                num_proposals=num_proposals, 
                cholesky=da_state.cholesky
            )
            next_state, info = algo.step(key, state)
            
            acceptance_rate = getattr(info, "acceptance_rate", target_rate)
            next_da_state = dual_averaging_step(
                da_state, 
                acceptance_rate,
                next_state.position,
                target_rate=target_rate
            )
            
            # Protect MALA momentum with a step-size floor
            min_log_step = jnp.log(0.05)
            next_da_state = next_da_state._replace(
                log_step_size=jnp.maximum(next_da_state.log_step_size, min_log_step),
                log_step_size_bar=jnp.maximum(next_da_state.log_step_size_bar, min_log_step)
            )
            
            return next_state, next_da_state
            
        next_states, next_da_states = jax.vmap(single_chain_warmup)(keys, states, da_states)
        return (next_states, next_da_states), None

    # 3. Execute the scan
    warmup_keys = jax.random.split(rng_key, num_warmup)
    (final_states, final_da_states), _ = jax.lax.scan(warmup_step, (states, da_states), warmup_keys)
    
    # Extract the optimally tuned parameters
    final_step_sizes = jnp.exp(final_da_states.log_step_size_bar)
    final_choleskys = final_da_states.cholesky
    
    return final_states, final_step_sizes, final_choleskys