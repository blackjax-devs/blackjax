import functools
import math
from typing import NamedTuple

import jax
import jax.numpy as jnp
from chex import Array
import numpy as np
from scipy import stats
from scipy.fftpack import next_fast_len  # type: ignore

import blackjax
from blackjax.adaptation.step_size import MCLMCAdaptationState, tune
from blackjax.base import SamplingAlgorithm
from blackjax.mcmc.integrators import noneuclidean_mclachlan
from blackjax.mcmc.mclmc import build_kernel

# from blackjax.diagnostics import effective_sample_size
from blackjax.types import PRNGKey
# from tests.mcmc.test_sampling import inference_loop


def logdensity_fn(x):
    return -0.5 * jnp.sum(jnp.square(x))


def run_sampling_algorithm(
    sampling_algorithm: SamplingAlgorithm, num_steps: int, initial_val, rng_key
):
    keys = jax.random.split(rng_key, num_steps)
    state = sampling_algorithm.init(initial_val)
    _, info = jax.lax.scan(lambda s, k: (sampling_algorithm.step(k, s)), state, keys)
    return info

def tune_and_run(position, logdensity_fn, key, dim, num_steps):
    main_key, tune_key = jax.random.split(key)

    # params, state = tune(
    #     position=position,
    #     params=MCLMCAdaptationState(L=math.sqrt(dim), step_size=math.sqrt(dim) * 0.4),
    #     logdensity_fn=logdensity_fn,
    #     num_steps=num_steps,
    #     rng_key=tune_key,
    # )
    # print(
    #     f"L is {params.L} and should be {1.3147894144058228} and step_size is {params.step_size} and should be {0.6470216512680054}"
    # )

    mclmc = blackjax.mcmc.mclmc.mclmc(
        logdensity_fn=logdensity_fn,
        transform=lambda x: x,
        # L=params.L,
        # step_size=params.step_size,
        L=math.sqrt(dim), step_size=math.sqrt(dim) * 0.4
    )


    return run_sampling_algorithm(
        sampling_algorithm=mclmc,
        num_steps=num_steps,
        # initial_val=state.position,
        initial_val=position,
        rng_key=main_key,
    )

out = tune_and_run(position=jnp.array([10.0, 10.0]), logdensity_fn=logdensity_fn, key=jax.random.PRNGKey(0), dim=2, num_steps=10000)

print(jnp.mean(out.transformed_x, axis=0))



# # assert params.L==1.3147894144058228 and params.step_size==0.6470216512680054
# # assert jnp.allclose(jnp.mean(out.transformed_x, axis=0), jnp.array([1.9507202e-03, 2.8414153e-05]))
# assert jnp.allclose(jnp.mean(out.transformed_x, axis=0), jnp.array([0.00296992, 0.00087555]))

