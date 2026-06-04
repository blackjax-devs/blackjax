import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
import blackjax  
import time

from .kernel import build_kernel 

def run_sharded_pt_nuts(
    logdensity_fn, 
    initial_positions, 
    num_rungs=16, 
    num_warmup=1000, 
    num_samples=2000, 
    step_size=0.05,
    rng_key_seed=42,
    min_beta=0.01,        # <--- NEW PARAMETER
    static_ladder=False   # <--- NEW PARAMETER
):
    """Executes SPMD Parallel Tempering NUTS across multiple devices."""
    num_chains = initial_positions.shape[0]
    
    devices = jax.devices()
    print(f"Slingshot Engine: Distributing {num_rungs} rungs for {num_chains} chains across {len(devices)} devices...")

    mesh = Mesh(devices, axis_names=('rungs_axis',))
    rung_sharding = NamedSharding(mesh, P('rungs_axis'))
    pos_sharding = NamedSharding(mesh, P(None, 'rungs_axis', None))

    init_fn, step_fn = build_kernel(
        logdensity_fn=logdensity_fn,
        kernel_factory=blackjax.nuts, 
        inner_params={"step_size": step_size, "inverse_mass_matrix": jnp.ones(initial_positions.shape[-1])},
        static_ladder=static_ladder,  # <--- PASS PARAMETER TO KERNEL
        coupled_adaptation=True
    )
    
    vmap_init = jax.vmap(init_fn, in_axes=(0, None))
    vmap_step = jax.vmap(step_fn, in_axes=(0, 0))

    rng_key = jax.random.PRNGKey(rng_key_seed)
    _, sample_key = jax.random.split(rng_key)
    
    # <--- USE MIN_BETA FOR THE LADDER FLOOR
    raw_ladder = jnp.geomspace(1.0, min_beta, num_rungs) 
    sharded_ladder = jax.device_put(raw_ladder, rung_sharding)
    sharded_pos = jax.device_put(initial_positions, pos_sharding)

    state = vmap_init(sharded_pos, sharded_ladder)

    @jax.jit
    def run_all_chains(current_state, key):
        def one_step(carry, step_key):
            chain_keys = jax.random.split(step_key, num_chains)
            new_state, _ = vmap_step(chain_keys, carry)
            cold_positions = new_state.pt_state.position[:, 0, :]
            return new_state, cold_positions
            
        keys = jax.random.split(key, num_warmup + num_samples)
        final_state, positions = jax.lax.scan(one_step, current_state, keys)
        return jnp.swapaxes(positions, 0, 1)

    t0 = time.time()
    positions = run_all_chains(state, sample_key)
    positions.block_until_ready() 
    print(f"Sharded multi-chain execution completed in {time.time() - t0:.2f} seconds.")
    
    return positions