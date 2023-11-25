import math
from typing import NamedTuple

import jax
import jax.numpy as jnp
from chex import Array
from scipy.fftpack import next_fast_len  # type: ignore

import blackjax
from blackjax.adaptation.step_size import MCLMCAdaptationState, tune
from blackjax.base import SamplingAlgorithm
from blackjax.mcmc.integrators import noneuclidean_mclachlan
from blackjax.mcmc.mclmc import build_kernel

# from blackjax.diagnostics import effective_sample_size
from blackjax.types import PRNGKey


def logdensity_fn(x):
    return -0.5 * jnp.sum(jnp.square(x))


def run_sampling_algorithm(
    sampling_algorithm: SamplingAlgorithm, num_steps: int, initial_val, rng_key
):
    keys = jax.random.split(rng_key, num_steps)
    state = sampling_algorithm.init(initial_val)
    _, info = jax.lax.scan(lambda s, k: (sampling_algorithm.step(k, s)), state, keys)
    return info

def tune_and_run(logdensity_fn, key, dim, num_steps):
    main_key, tune_key = jax.random.split(key)
    identity = lambda x: x

    params, state = tune(
        params=MCLMCAdaptationState(L=math.sqrt(dim), step_size=math.sqrt(dim) * 0.4),
        kernel=build_kernel(
            logdensity_fn, integrator=noneuclidean_mclachlan, transform=identity
        ),
        num_steps=num_steps,
        rng_key=tune_key,
    )

    mclmc = blackjax.mcmc.mclmc.mclmc(
        logdensity_fn=logdensity_fn,
        transform=lambda x: x,
        L=params.L,
        step_size=params.step_size,
    )

    print(
        f"L is {params.L} and should be {1.3147894144058228} and step_size is {params.step_size} and should be {0.6470216512680054}"
    )
    return run_sampling_algorithm(
        sampling_algorithm=mclmc,
        num_steps=num_steps,
        initial_val=state.position,
        rng_key=main_key,
    )

out = tune_and_run(logdensity_fn=logdensity_fn, key=jax.random.PRNGKey(0), dim=2, num_steps=10000)

print(jnp.mean(out.transformed_x, axis=0))



# assert params.L==1.3147894144058228 and params.step_size==0.6470216512680054
# assert jnp.allclose(jnp.mean(out.transformed_x, axis=0), jnp.array([1.9507202e-03, 2.8414153e-05]))
assert jnp.allclose(jnp.mean(out.transformed_x, axis=0), jnp.array([0.00296992, 0.00087555]))



# def test_mclmc(self):
#     """Test the MCLMC kernel."""
#     init_key0, init_key1, inference_key = jax.random.split(self.key, 3)
#     x_data = jax.random.normal(init_key0, shape=(1000, 1))
#     y_data = 3 * x_data + jax.random.normal(init_key1, shape=x_data.shape)

#     logposterior_fn_ = functools.partial(
#         self.regression_logprob, x=x_data, preds=y_data
#     )
#     logposterior_fn = lambda x: logposterior_fn_(**x)

#     mala = blackjax.mcmc.mclmc.mclmc(logposterior_fn, 1e-5)
#     state = mala.init({"coefs": 1.0, "log_scale": 1.0})
#     states = inference_loop(mala.step, 10_000, inference_key, state)

#     coefs_samples = states.position["coefs"][3000:]
#     scale_samples = np.exp(states.position["log_scale"][3000:])

#     np.testing.assert_allclose(np.mean(scale_samples), 1.0, atol=1e-1)
#     np.testing.assert_allclose(np.mean(coefs_samples), 3.0, atol=1e-1)