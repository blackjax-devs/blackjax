"""Adaptive nested sampling.

A nested sampling outer kernel that updates its inner kernel parameters at each iteration.

Inner kernel is the 'mcmc kernel' (typically slice sampling, but could be anything)
"""
from functools import partial
from typing import Callable, Dict

import jax
import jax.numpy as jnp

from blackjax import SamplingAlgorithm
from blackjax.ns.base import NSInfo, NSState
from blackjax.ns.base import build_kernel as base_build_kernel
from blackjax.ns.base import delete_fn
from blackjax.ns.base import init as init_base
from blackjax.smc.inner_kernel_tuning import StateWithParameterOverride
from blackjax.types import ArrayLikeTree, ArrayTree, PRNGKey

__all__ = ["init", "as_top_level_api", "build_kernel"]


def init(particles, loglikelihood_fn, parameter_update_fn):
    state = init_base(particles, loglikelihood_fn)
    initial_parameter_value = parameter_update_fn(
        state, NSInfo(state, state, state, None)
    )
    return StateWithParameterOverride(state, initial_parameter_value)


def build_kernel(
    logprior_fn: Callable,
    loglikelihood_fn: Callable,
    delete_fn: Callable,
    mcmc_build_kernel: Callable,
    mcmc_init_fn: Callable,
    mcmc_parameter_update_fn: Callable,
    num_mcmc_steps: int,
) -> Callable:
    r"""Build an adaptive Nested Sampling kernel. Tunes the inner kernel parameters
    at each iteration.

    Parameters
    ----------
    logprior_fn : Callable
        A function that computes the log prior probability.
    loglikelihood_fn : Callable
        A function that computes the log likelihood.
    delete_fn : Callable
        Function that takes an array of log likelihoods and marks particles for deletion and updates.
    mcmc_build_kernel:
        A function that builds an mcmc kernel
    mcmc_init_fn
        A callable that initializes the inner kernel
    mcmc_parameter_update_fn : Callable[[NSState, NSInfo], Dict[str, ArrayTree]]
        Function that updates the parameters of the inner kernel.
    num_mcmc_steps: int
        Number of MCMC steps to perform. Recommended is 5 times the dimension of the parameter space.

    Returns
    -------
    Callable
        A function that takes a rng_key and a NSState that contains the current state
        of the chain and returns a new state of the chain along with
        information about the transition.
    """

    def kernel(
        rng_key: PRNGKey,
        state: StateWithParameterOverride,
    ) -> tuple[StateWithParameterOverride, NSInfo]:
        step_fn = base_build_kernel(
            logprior_fn,
            loglikelihood_fn,
            delete_fn,
            mcmc_build_kernel,
            mcmc_init_fn,
            num_mcmc_steps,
        )
        new_state, info = step_fn(
            rng_key, state.sampler_state, state.parameter_override
        )
        new_parameter_override = mcmc_parameter_update_fn(new_state, info)
        return (
            StateWithParameterOverride(new_state, new_parameter_override),
            info,
        )

    return kernel


def as_top_level_api(
    logprior_fn: Callable,
    loglikelihood_fn: Callable,
    mcmc_build_kernel: Callable,
    mcmc_init_fn: Callable,
    mcmc_parameter_update_fn: Callable[[NSState, NSInfo], Dict[str, ArrayTree]],
    num_mcmc_steps: int,
    n_delete: int = 1,
) -> SamplingAlgorithm:
    """Implements the (basic) user interface for the Adaptive Nested Sampling kernel.

    Parameters
    ----------
    logprior_fn : Callable
        A function that computes the log prior probability.
    loglikelihood_fn : Callable
        A function that computes the log likelihood.
    mcmc_build_kernel:
        A function that builds an mcmc kernel
    mcmc_init_fn
        A callable that initializes the inner kernel
    mcmc_parameter_update_fn : Callable[[NSState, NSInfo], Dict[str, ArrayTree]]
        A function that updates the parameters given the current state and info.
    num_mcmc_steps: int
        Number of MCMC steps to perform. Recommended is 5 times the dimension of the parameter space.
    n_delete : int, optional
        Number of particles to delete in each iteration. Default is 1.

    Returns
    -------
    SamplingAlgorithm
        A sampling algorithm object.
    """
    delete_fn = partial(delete_fn, n_delete=n_delete)

    kernel = build_kernel(
        logprior_fn,
        loglikelihood_fn,
        delete_fn,
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
