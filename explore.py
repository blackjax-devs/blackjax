import jax
import jax.numpy as jnp
from benchmarks.mcmc.sampling_algorithms import samplers
import blackjax
from blackjax.mcmc.mhmclmc import mhmclmc, rescale
from blackjax.mcmc.hmc import hmc
from blackjax.mcmc.dynamic_hmc import dynamic_hmc
from blackjax.mcmc.integrators import isokinetic_mclachlan
from blackjax.util import run_inference_algorithm





init_key, tune_key, run_key = jax.random.split(jax.random.PRNGKey(0), 3)

def logdensity_fn(x):
    return -0.5 * jnp.sum(jnp.square(x))

initial_position = jnp.ones(10,)


def run_mclmc(logdensity_fn, num_steps, initial_position):
    key = jax.random.PRNGKey(0)
    init_key, tune_key, run_key = jax.random.split(key, 3)


    initial_state = blackjax.mcmc.mclmc.init(
        position=initial_position, logdensity_fn=logdensity_fn, rng_key=init_key
    )

    kernel = blackjax.mcmc.mclmc.build_kernel(
        logdensity_fn=logdensity_fn,
        integrator=blackjax.mcmc.integrators.isokinetic_mclachlan,
    )

    (
        blackjax_state_after_tuning,
        blackjax_mclmc_sampler_params,
    ) = blackjax.mclmc_find_L_and_step_size(
        mclmc_kernel=kernel,
        num_steps=num_steps,
        state=initial_state,
        rng_key=tune_key,
    )

    print(blackjax_mclmc_sampler_params)

# out = run_hmc(initial_position)
out = samplers["mhmclmc"](logdensity_fn=logdensity_fn, num_steps=5000, initial_position=initial_position, key=jax.random.PRNGKey(0))
print(out.mean(axis=0) )


