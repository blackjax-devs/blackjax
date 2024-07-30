# Copyright 2024- Will Handley & David Yallup
from typing import Callable

import jax
import jax.numpy as jnp

import blackjax.ns.base as base
from blackjax.base import SamplingAlgorithm
from blackjax.types import ArrayLikeTree, PRNGKey

from blackjax.ns.base import init, build_kernel

__all__ = ["init", "as_top_level_api", "build_kernel"]


def create_fn(rng_key, particles, logL_fn, logL_star, create_parameters):
    num_particles, ndims = jax.tree_util.tree_flatten(particles)[0][0].shape

    def body_fun(carry, xs):
        def cond_fun(carry):
            _, logL, _ = carry
            return logL <= logL_star

        def inner_body(carry):
            rng_key, _, _ = carry
            rng_key, subkey = jax.random.split(rng_key)
            particle = jax.random.uniform(subkey, (ndims,))
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


def delete_fn(rng_key, logL, delete_parameters):
    idx = logL > logL.min()
    return idx


def as_top_level_api(
    logL_fn: Callable,
) -> SamplingAlgorithm:
    """Implements a rejection sampling nested sampling algo

    Parameters
    ----------
    logL_fn: Callable
        A function that assigns a weight to the particles.
    Returns
    -------
    A ``SamplingAlgorithm``.

    """
    return base.as_top_level_api(logL_fn, create_fn, delete_fn)
