from functools import partial
from typing import Any, Callable, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

import blackjax.mcmc.hmc as hmc
import blackjax.mcmc.integrators as integrators
import blackjax.mcmc.metrics as metrics
import blackjax.mcmc.proposal as proposal
import blackjax.mcmc.termination as termination
import blackjax.mcmc.trajectory as trajectory
from blackjax.base import SamplingAlgorithm
from blackjax.types import ArrayLikeTree, ArrayTree, PRNGKey
import blackjax


def build_kernel(kernel_1, kernel_2, logdensity_fn):
    def kernel(state, rng_key):
        next_x_state, info = kernel_1(
            rng_key=rng_key,
            state=state['x'],
            step_size=1e-3,
            inverse_mass_matrix=jnp.ones(state['x'].position.shape[0]),
            num_integration_steps=1,
            logdensity_fn=partial(logdensity_fn, pseudofermion=state['y']),
        )
        next_y_state = kernel_2(state['y'])
        return {'x': next_x_state, 'y': next_y_state}, info
    return kernel

def init(position, logdensity_fn, pseudofermion, init_1, init_2, rng_key  ):
    key_1, key_2 = jax.random.split(rng_key, 2)
    state_b = init_1(position, partial(logdensity_fn, pseudofermion=pseudofermion) )
    state_pf = init_2(pseudofermion)

    return {'x': state_b, 'y': state_pf}


def as_top_level_api(
    kernel_1,
    kernel_2,
    init_1,
    init_2,
    logdensity_fn: Callable,
) -> SamplingAlgorithm:
    
    # kernel = build_kernel(integrator, divergence_threshold)

    return None

    return SamplingAlgorithm(init_fn, step_fn)


