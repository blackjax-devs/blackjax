# Copyright 2024- Will Handley & David Yallup
from typing import Callable, NamedTuple, Optional

import jax
import jax.numpy as jnp

from blackjax.types import Array, ArrayLikeTree, ArrayTree, PRNGKey
from blackjax.base import SamplingAlgorithm
import blackjax.ns.base as base


class NSState(NamedTuple):
    """State of the Nested Sampler.

    Live points must be a ArrayTree, each leave represents a variable from the posterior,
    being an array of size `(nlive, ...)`.

    Examples (three particles):
        - Single univariate posterior:
            [ Array([[1.], [1.2], [3.4]]) ]
        - Single bivariate  posterior:
            [ Array([[1,2], [3,4], [5,6]]) ]
        - Two variables, each univariate:
            [ Array([[1.], [1.2], [3.4]]),
            Array([[50.], [51], [55]]) ]
        - Two variables, first one bivariate, second one 4-variate:
            [ Array([[1., 2.], [1.2, 0.5], [3.4, 50]]),
            Array([[50., 51., 52., 51], [51., 52., 52. ,54.], [55., 60, 60, 70]]) ]
    """

    particles: ArrayTree
    logL: Array # The log-likelihood of the particles
    logL_birth: Array # The hard likelihood threshold of each particle at birth
    logL_star: float # The current hard likelihood threshold
    create_parameters: ArrayTree # NOTE num_repeats?
    delete_parameters: ArrayTree # NOTE num_repeats?


class NSInfo(NamedTuple):
    """Additional information on the NS step.

    """
    update_info: NamedTuple

def init(particles: ArrayLikeTree, logL_fn, init_create_params, init_delete_params):
    logL_star = -jnp.inf
    num_particles = jax.tree_util.tree_flatten(particles)[0][0].shape[0]
    logL_birth = logL_star * jnp.ones(num_particles)
    logL = logL_fn(particles)
    return NSState(particles, logL, logL_birth, logL_star, init_create_params, init_delete_params)

def build_kernel(
    logL_fn: Callable,
    create_fn: Callable,
    delete_fn: Callable,
) -> Callable:
    r"""Build a Nested Sampling by running a creation and deletion step.

    Parameters
    ----------
    logL_fn: Callable
        A function that assigns a weight to the particles.
    create_fn: Callable
        Function that takes an array of keys and particles and returns
        new particles.
    delete_fn: Callable
        Function that takes an array of keys and particles and deletes some
        particles.

    Returns
    -------
    A callable that takes a rng_key and a NSState that contains the current state
    of the chain and that returns a new state of the chain along with
    information about the transition.

    """

    def kernel(rng_key: PRNGKey, state: base.NSState) -> tuple[base.NSState, base.NSInfo]:
        # Create new particles
        create_parameters = state.create_parameters
        particles, create_info = create_fn(rng_key, state.particles, logL_fn, state.logL_star, create_parameters)
        num_particles = jax.tree_util.tree_flatten(particles)[0][0].shape[0] # Not good jax -- improve with sgpt
        logL_birth = state.logL_star * jnp.ones(num_particles)

        # Add these to the jax array
        particles = jnp.concatenate((state.particles, particles), axis=0)
        logL = jnp.concatenate((state.logL, create_info["logL"]))
        logL_birth = jnp.concatenate((state.logL_birth, logL_birth))

        # Find particles to delete
        idx = delete_fn(rng_key, logL, state.delete_parameters)
        update_info = {"particles": particles[~idx], "logL": logL[~idx], "logL_birth": logL_birth[~idx]}

        logL_star = logL[idx].min()

        return base.NSState(
            particles[idx],
            logL[idx],
            logL_birth[idx],
            logL_star,
            state.create_parameters,
            state.delete_parameters,
        ), base.NSInfo(update_info)

    return kernel

def as_top_level_api(
    logL_fn: Callable,
    create_fn: Callable,
    delete_fn: Callable,
) -> SamplingAlgorithm:
    """Implements the (basic) user interface for the Adaptive Tempered SMC kernel.

    Parameters
    ----------
    logL_fn: Callable
        A function that assigns a weight to the particles.
    create_fn: Callable
        Function that takes an array of keys and particles and returns
        new particles.
    delete_fn: Callable
        Function that takes an array of keys and particles and deletes some
        particles.
    Returns
    -------
    A ``SamplingAlgorithm``.

    """
    kernel = build_kernel(
        logL_fn,
        create_fn,
        delete_fn,
    )

    def init_fn(position: ArrayLikeTree, rng_key=None):
        del rng_key
        return init(position, logL_fn, {}, {})

    def step_fn(rng_key: PRNGKey, state):
        return kernel(
            rng_key,
            state
        )

    return SamplingAlgorithm(init_fn, step_fn)
