import pymc as pm
import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
from pymc.sampling.jax import get_jaxified_logp
import arviz as az
import numpy as np

from .driver import run_sharded_pt_nuts

def sample_slingshot(
    model: pm.Model = None,
    num_chains: int = 4,
    num_rungs: int = 16,
    num_warmup: int = 1000,
    num_samples: int = 2000,
    jitter_scale: float = 0.1,
    rng_seed: int = 42,
    min_beta: float = 0.01,        # <--- NEW PARAMETER
    static_ladder: bool = False    # <--- NEW PARAMETER
) -> az.InferenceData:
    """Compiles ANY PyMC model to JAX and samples it using Slingshot PT-NUTS."""
    model = pm.modelcontext(model)
    print("Compiling PyMC model to JAX...")
    
    value_vars = model.value_vars
    initial_point = model.initial_point()
    
    init_list = [initial_point[v.name] for v in value_vars]
    
    flat_init, unravel_fn = ravel_pytree(init_list)
    D = flat_init.shape[0] 
    
    jax_logp_fn = get_jaxified_logp(model)
    
    @jax.jit
    def flattened_logprob(flat_position):
        unraveled_list = unravel_fn(flat_position)
        return jax_logp_fn(unraveled_list)

    print(f"Model dimensions: {D}. Generating initial particles...")
    np.random.seed(rng_seed)
    
    jitter = np.random.normal(loc=0.0, scale=jitter_scale, size=(num_chains, num_rungs, D))
    initial_positions = flat_init + jitter
    initial_positions = jnp.array(initial_positions)

    raw_samples = run_sharded_pt_nuts(
        logdensity_fn=flattened_logprob,
        initial_positions=initial_positions,
        num_rungs=num_rungs,
        num_warmup=num_warmup,
        num_samples=num_samples,
        rng_key_seed=rng_seed,
        min_beta=min_beta,          # <--- PASS TO DRIVER
        static_ladder=static_ladder # <--- PASS TO DRIVER
    )
    
    valid_samples = np.array(raw_samples)[:, num_warmup:, :]
    
    print("Unflattening results and packaging into ArviZ InferenceData...")
    posterior_dict = {}
    current_idx = 0
    
    for v in value_vars:
        var_name = v.name
        var_shape = initial_point[var_name].shape
        var_size = np.prod(var_shape, dtype=int)
        
        flat_slice = valid_samples[:, :, current_idx : current_idx + var_size]
        posterior_dict[var_name] = flat_slice.reshape((num_chains, num_samples) + var_shape)
        
        current_idx += var_size

    return az.from_dict(posterior=posterior_dict)