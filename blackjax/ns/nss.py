import functools
from functools import partial
from typing import Callable

import jax
import jax.numpy as jnp
from blackjax import SamplingAlgorithm
from blackjax.mcmc.ss import build_kernel as build_slice_kernel
from blackjax.mcmc.ss import default_generate_slice_direction_fn, default_stepper_fn
from blackjax.mcmc.ss import init as slice_init
from blackjax.ns.adaptive import build_kernel, init
from blackjax.ns.base import NSInfo, NSState, delete_fn
from blackjax.ns.utils import get_first_row
from blackjax.smc.tuning.from_particles import (
    particles_as_rows,
    particles_covariance_matrix,
)
from blackjax.types import ArrayLikeTree, PRNGKey
from jax.flatten_util import ravel_pytree

__all__ = ["init", "as_top_level_api"]


def default_predict_fn(key, **kwargs):
    cov = kwargs["cov"]
    row = get_first_row(cov)
    _, unravel_fn = ravel_pytree(row)
    cov = particles_as_rows(cov)
    n = default_generate_slice_direction_fn(key, cov)
    return unravel_fn(n)


def default_train_fn(state, info):
    cov = particles_covariance_matrix(state.particles)
    single_particle = get_first_row(state.particles)
    _, unravel_fn = ravel_pytree(single_particle)
    return {"cov": jax.vmap(unravel_fn)(cov)}


def as_top_level_api(
    logprior_fn: Callable,
    loglikelihood_fn: Callable,
    num_mcmc_steps: int,
    n_delete: int = 1,
    stepper_fn: Callable = default_stepper_fn,
    train_fn: Callable = default_train_fn,
    predict_fn: Callable = default_predict_fn,
) -> SamplingAlgorithm:
    """Implements the a baseline Nested Slice Sampling kernel.

    Parameters
    ----------
    logprior_fn: Callable
        A function that computes the log prior probability.
    loglikelihood_fn: Callable
        A function that computes the log likelihood.
    num_mcmc_steps: int
        Number of MCMC steps to perform. Recommended is 5 times the dimension of the parameter space.
    n_delete: int, optional
        Number of particles to delete in each iteration. Default is 1.
    stepper_fn: Callable, optional
        Optional function to add the proposal to the current state, if needed

    Returns
    -------
    SamplingAlgorithm
        A sampling algorithm object.
    """
    delete_func = partial(delete_fn, n_delete=n_delete)

    def mcmc_build_kernel(**kwargs):
        return build_slice_kernel(partial(predict_fn, **kwargs), stepper_fn)

    mcmc_init_fn = slice_init
    mcmc_parameter_update_fn = train_fn

    def init_fn(particles: ArrayLikeTree, rng_key=None):
        del rng_key
        return init(particles, loglikelihood_fn, logprior_fn, mcmc_parameter_update_fn)

    step_fn = build_kernel(
        logprior_fn,
        loglikelihood_fn,
        delete_func,
        mcmc_build_kernel,
        mcmc_init_fn,
        mcmc_parameter_update_fn,
        num_mcmc_steps,
    )

    return SamplingAlgorithm(init_fn, step_fn)
