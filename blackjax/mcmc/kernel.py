from typing import Callable, NamedTuple, Any
import jax
import jax.numpy as jnp

# =============================================================================
# 1. IMMUTABLE STATE AND INFO PYTREES
# =============================================================================
class ParallelTemperingState(NamedTuple):
    position: jax.Array              
    logdensity: jax.Array            
    inner_state: Any                 
    beta: jax.Array                  

class SlingshotState(NamedTuple):
    """Wrapper to match the driver's expected state.pt_state API"""
    pt_state: ParallelTemperingState

class ParallelTemperingInfo(NamedTuple):
    inner_info: Any                  
    swap_acceptance: jax.Array       

# =============================================================================
# 2. THE FACTORY INTERFACE
# =============================================================================
def build_kernel(
    logdensity_fn: Callable[[jax.Array], float],
    kernel_factory: Any,  
    inner_params: dict, 
    static_ladder: bool = False,
    coupled_adaptation: bool = True
):
    """Factory that builds the pure JAX PT-NUTS kernel."""

    def one_rung_init(pos, beta_val):
        def local_logdensity(p):
            return beta_val * logdensity_fn(p)
        kernel_instance = kernel_factory(local_logdensity, **inner_params)
        return kernel_instance.init(pos)

    def init(initial_positions: jax.Array, beta: jax.Array) -> SlingshotState:
        sharded_inner_states = jax.vmap(one_rung_init)(initial_positions, beta)
        base_logdensities = jax.vmap(logdensity_fn)(initial_positions)
        
        pt_state = ParallelTemperingState(
            position=initial_positions,
            logdensity=base_logdensities,
            inner_state=sharded_inner_states,
            beta=beta
        )
        return SlingshotState(pt_state=pt_state)

    def step(rng_key: jax.Array, state: SlingshotState) -> tuple[SlingshotState, ParallelTemperingInfo]:
        pt_state = state.pt_state
        key_mutation, key_swap = jax.random.split(rng_key)
        num_rungs = pt_state.beta.shape[0]
        
        # --- STAGE 1: LOCAL MUTATION ---
        mutation_keys = jax.random.split(key_mutation, num_rungs)
        
        def one_rung_step(key, inner_state, beta_val):
            def local_logdensity(p):
                return beta_val * logdensity_fn(p)
            kernel_instance = kernel_factory(local_logdensity, **inner_params)
            return kernel_instance.step(key, inner_state)
        
        new_inner_states, inner_info = jax.vmap(one_rung_step)(mutation_keys, pt_state.inner_state, pt_state.beta)
        new_positions = new_inner_states.position
        new_base_logdensities = jax.vmap(logdensity_fn)(new_positions)
        
        # --- STAGE 2: GLOBAL SWAP ---
        key_even, key_odd = jax.random.split(key_swap)
        
        def execute_swap_phase(rng_key, current_positions, current_logdensities, is_even):
            start_idx = 0 if is_even else 1
            idx_i = jnp.arange(start_idx, num_rungs - 1, 2)
            idx_j = idx_i + 1
            
            if idx_i.shape[0] == 0:
                empty_mask = jnp.zeros((0,), dtype=bool)
                return current_positions, current_logdensities, empty_mask, idx_i
            
            pos_i, pos_j = current_positions[idx_i], current_positions[idx_j]
            logdeb_i, logdeb_j = current_logdensities[idx_i], current_logdensities[idx_j]
            beta_i, beta_j = pt_state.beta[idx_i], pt_state.beta[idx_j]
            
            delta_beta = beta_i - beta_j
            delta_logdensity = logdeb_j - logdeb_i
            swap_log_prob = delta_beta * delta_logdensity
            
            random_draws = jax.random.uniform(rng_key, shape=(idx_i.shape[0],))
            accept_mask = jnp.log(random_draws) < swap_log_prob
            
            mask_expanded = jnp.expand_dims(accept_mask, axis=-1)
            swapped_pos_i = jnp.where(mask_expanded, pos_j, pos_i)
            swapped_pos_j = jnp.where(mask_expanded, pos_i, pos_j)
            swapped_log_i = jnp.where(accept_mask, logdeb_j, logdeb_i)
            swapped_log_j = jnp.where(accept_mask, logdeb_i, logdeb_j)
            
            updated_positions = current_positions.at[idx_i].set(swapped_pos_i).at[idx_j].set(swapped_pos_j)
            updated_logdensities = current_logdensities.at[idx_i].set(swapped_log_i).at[idx_j].set(swapped_log_j)
            
            return updated_positions, updated_logdensities, accept_mask, idx_i

        global_swap_record = jnp.zeros(num_rungs - 1, dtype=bool)

        pos_after_even, log_after_even, mask_even, idx_even = execute_swap_phase(
            key_even, new_positions, new_base_logdensities, is_even=True
        )
        global_swap_record = global_swap_record.at[idx_even].set(mask_even)
        
        final_positions, final_logdensities, mask_odd, idx_odd = execute_swap_phase(
            key_odd, pos_after_even, log_after_even, is_even=False
        )
        if idx_odd.shape[0] > 0:
            global_swap_record = global_swap_record.at[idx_odd].set(mask_odd)
        
        # --- STAGE 3: THE FIX (Rebuild Inner States) ---
        final_inner_sampler_state = jax.vmap(one_rung_init)(final_positions, pt_state.beta)

        final_pt_state = ParallelTemperingState(
            position=final_positions,
            logdensity=final_logdensities,
            inner_state=final_inner_sampler_state,
            beta=pt_state.beta
        )
        
        info = ParallelTemperingInfo(
            inner_info=inner_info,
            swap_acceptance=global_swap_record
        )

        return SlingshotState(pt_state=final_pt_state), info

    return init, step