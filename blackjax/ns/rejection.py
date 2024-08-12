# Copyright 2024- Will Handley & David Yallup
from functools import partial
from typing import Callable

import jax
import jax.numpy as jnp

import blackjax.ns.base as base
from blackjax.base import SamplingAlgorithm
from blackjax.ns.base import build_kernel, init
from blackjax.types import ArrayLikeTree, PRNGKey

__all__ = ["init", "as_top_level_api", "build_kernel"]


def create_fn(rng_key, particles, prior, logL_fn, logL_star, create_parameters):
    # num_particles, ndims = jax.tree_util.tree_flatten(particles)[0][0].shape
    # ndims = jax.tree_util.tree_flatten(particles)[0][0].shape
    num_particles, ndims = particles.shape

    def body_fun(carry, xs):
        def cond_fun(carry):
            _, logL, _ = carry
            return logL <= logL_star

        def inner_body(carry):
            rng_key, _, _ = carry
            rng_key, subkey = jax.random.split(rng_key)
            particle = prior(seed=subkey)
            logL = logL_fn(particle)
            return rng_key, logL, particle

        rng_key = carry
        rng_key, step_rng = jax.random.split(rng_key)
        _, final_logL, particle = jax.lax.while_loop(
            cond_fun, inner_body, (step_rng, -jnp.inf, jnp.zeros(ndims))
        )
        return rng_key, (particle, final_logL)

    logLs = jnp.ones(num_particles) * -jnp.inf
    rng_key, init_key = jax.random.split(rng_key)
    _, new_particles = jax.lax.scan(body_fun, init_key, (particles, logLs))

    return new_particles[0], new_particles[1]


def delete_fn(rng_key, logL, delete_parameters, n_delete):
    val, idx = jax.lax.top_k(-logL, n_delete)
    return val, idx


def as_top_level_api(
    logPrior_fn: Callable,
    logL_fn: Callable,
    n_delete: int = 1,
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

    delete_func = partial(delete_fn, n_delete=n_delete)

    kernel = build_kernel(
        logPrior_fn,
        logL_fn,
        create_fn,
        delete_func,
    )

    def init_fn(position: ArrayLikeTree, rng_key=None):
        del rng_key
        return init(position, logL_fn, {})

    def step_fn(rng_key: PRNGKey, state):
        return kernel(rng_key, state)

    return SamplingAlgorithm(init_fn, step_fn)
