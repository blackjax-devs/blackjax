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

    def cond_fun(carry):
        
        _, logL, _, _ = carry
        return logL <= logL_star

    def body_fun(carry, xs):
        rng_key, _, _, mh_accept = carry
        rng_key, subkey = jax.random.split(rng_key)
        particle = jax.random.uniform(subkey, (ndims,))
        mh_accept = jnp.logical_or(jax.random.uniform(subkey) > 0.5, mh_accept)
        logL = logL_fn(particle)
        return (rng_key, logL, particle, mh_accept), particle


    new_particles = jax.lax.scan(body_fun, (rng_key, -jnp.inf, jnp.zeros(ndims), jnp.zeros(ndims, dtype=bool)), particles)

    init_val = (rng_key, -jnp.inf, jnp.zeros(ndims), jnp.zeros(ndims, dtype=bool))
    final_rng_key, final_logL, final_particle = jax.lax.while_loop(cond_fun, body_fun, init_val)

    return jnp.array([final_particle]), { "logL": jnp.array([final_logL]) }


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
