import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
import functools

from functools import partial
from blackjax.ns.base import delete_fn
from typing import Callable
from blackjax.ns.base import NSInfo, NSState
from blackjax import SamplingAlgorithm
from blackjax.ns.adaptive import build_kernel, init
from blackjax.types import ArrayLikeTree, PRNGKey
from blackjax.smc.tuning.from_particles import particles_covariance_matrix, particles_as_rows
from blackjax.mcmc.ss import build_kernel as build_slice_kernel
from blackjax.mcmc.ss import init as slice_init
from blackjax.mcmc.ss import default_stepper
from blackjax.ns.utils import get_first_row
from blackjax.mcmc.ss import default_proposal_distribution

__all__ = ["init", "as_top_level_api"]

def default_predict_fn(key, **kwargs):
    cov = kwargs["cov"]
    row = get_first_row(cov)
    _, unravel_fn = ravel_pytree(row)
    cov = particles_as_rows(cov) 
    n = default_proposal_distribution(key, cov)
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
    stepper: Callable = default_stepper,
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
    stepper: Callable, optional
        Optional function to add the proposal to the current state, if needed

    Returns
    -------
    SamplingAlgorithm
        A sampling algorithm object.
    """
    delete_func = partial(delete_fn, n_delete=n_delete)

    def mcmc_build_kernel(**kwargs):
        def proposal_distribution(key):
            return predict_fn(key, **kwargs)
        return build_slice_kernel(proposal_distribution, stepper)

    mcmc_init_fn = slice_init
    mcmc_parameter_update_fn = train_fn

    kernel = build_kernel(
        logprior_fn,
        loglikelihood_fn,
        delete_func,
        mcmc_build_kernel,
        mcmc_init_fn,
        mcmc_parameter_update_fn,
        num_mcmc_steps,
    )

    def init_fn(particles: ArrayLikeTree, rng_key=None):
        del rng_key
        return init(particles, loglikelihood_fn, mcmc_parameter_update_fn)

    def step_fn(rng_key: PRNGKey, state):
        return kernel(rng_key, state)

    return SamplingAlgorithm(init_fn, step_fn)
