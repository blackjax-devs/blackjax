import jax
import jax.numpy as jnp

from functools import partial
from blackjax.ns.base import delete_fn
from typing import Callable
from blackjax.ns.base import NSInfo, NSState
from blackjax import SamplingAlgorithm
from blackjax.ns.adaptive import build_kernel, init
from blackjax.types import ArrayLikeTree, PRNGKey
from blackjax.smc.tuning.from_particles import particles_covariance_matrix
from blackjax.mcmc.ss import build_kernel as build_slice_kernel
from blackjax.mcmc.ss import init as slice_init

__all__ = ["init", "as_top_level_api", "build_kernel"]

def default_stepper(x, n, t):
    return jax.tree_map(lambda x, n: x + t * n, x, n)

def default_predict_fn(key, **kwargs):
    cov = kwargs["cov"]
    n = jax.random.multivariate_normal(
        key, mean=jnp.zeros(cov.shape[0]), cov=cov
    )
    invcov = jnp.linalg.inv(cov)
    norm = jnp.sqrt(jnp.einsum("...i,...ij,...j", n, invcov, n))
    n = n / norm[..., None]
    return n

def default_train_fn(particles):
    cov = particles_covariance_matrix(particles)
    return {"cov": jnp.atleast_2d(cov)}


def as_top_level_api(
    logprior_fn: Callable,
    loglikelihood_fn: Callable,
    num_mcmc_steps: int,
    n_delete: int = 1,
    ravel_fn: Callable = lambda x: x,
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
    ravel_fn: Callable, optional
        Optional function to ravel the proposal to the same shape as the state space, if needed
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
            return ravel_fn(predict_fn(key, **kwargs))
        return build_slice_kernel(proposal_distribution, stepper)

    mcmc_init_fn = slice_init

    def mcmc_parameter_update_fn(state, _):
        return train_fn(state.particles)

    kernel = build_kernel(
        logprior_fn,
        loglikelihood_fn,
        delete_func,
        mcmc_build_kernel,
        mcmc_init_fn,
        mcmc_parameter_update_fn,
        num_mcmc_steps,
    )

    def init_fn(position: ArrayLikeTree, rng_key=None):
        del rng_key
        return init(position, loglikelihood_fn, mcmc_parameter_update_fn)

    def step_fn(rng_key: PRNGKey, state):
        return kernel(rng_key, state)

    return SamplingAlgorithm(init_fn, step_fn)
