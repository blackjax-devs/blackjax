# Copyright 2024- Will Handley & David Yallup
from functools import partial
from typing import Callable, Dict

import jax.numpy as jnp

from blackjax import SamplingAlgorithm
from blackjax.ns.base_ns import NSInfo, NSState
from blackjax.ns.base_ns import build_kernel as base_ns
from blackjax.ns.base_ns import delete_fn
from blackjax.ns.base_ns import init as init_base
from blackjax.ns.vectorized_slice import build_kernel as slice_kernel
from blackjax.ns.vectorized_slice import init as slice_init
from blackjax.smc.inner_kernel_tuning import StateWithParameterOverride
from blackjax.smc.tuning.from_particles import particles_covariance_matrix
from blackjax.types import ArrayLikeTree, ArrayTree, PRNGKey

__all__ = ["init", "as_top_level_api", "build_kernel", "ssns"]


def init(position, loglikelihood_fn, parameter_update_function):
    state = init_base(position, loglikelihood_fn)
    initial_parameter_value = parameter_update_function(
        state, NSInfo(state, state, state, None)
    )
    return StateWithParameterOverride(state, initial_parameter_value)


def build_kernel(
    logprior_fn: Callable,
    loglikelihood_fn: Callable,
    delete_fn: Callable,
    mcmc_step_fn: Callable,
    mcmc_init_fn: Callable,
    parameter_update_fn: Callable,
    num_mcmc_steps: int = 10,
) -> Callable:
    r"""Build a Nested Sampling by running a creation and deletion step.
    Parameters
        Parameters
    ----------
    logprior_fn : Callable
        A function that computes the log prior probability.
    loglikelihood_fn : Callable
        A function that computes the log likelihood.
    delete_fn : Callable
        Function that takes an array of keys and particles and deletes some
        particles.
    parameter_update_fn : Callable[[NSState, NSInfo], Dict[str, ArrayTree]]
        Function that updates the parameters of the inner kernel.
    num_mcmc_steps : int, optional
        Number of MCMC steps to perform, by default 10.

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
        step_fn = base_ns(
            logprior_fn,
            loglikelihood_fn,
            delete_fn,
            mcmc_step_fn,
            mcmc_init_fn,
            num_mcmc_steps,
        )
        new_state, info = step_fn(
            rng_key, state.sampler_state, state.parameter_override
        )
        new_parameter_override = parameter_update_fn(new_state, info)
        return (
            StateWithParameterOverride(new_state, new_parameter_override),
            info,
        )

    return kernel


def as_top_level_api(
    logprior_fn: Callable,
    loglikelihood_fn: Callable,
    mcmc_step_fn: Callable,
    mcmc_init_fn: Callable,
    parameter_update_fn: Callable[[NSState, NSInfo], Dict[str, ArrayTree]],
    num_mcmc_steps: int = 10,
    n_delete: int = 1,
) -> SamplingAlgorithm:
    """Implements the (basic) user interface for the Adaptive Nested Sampling kernel.

    Parameters
    ----------
    logprior_fn : Callable
        A function that computes the log prior probability.
    loglikelihood_fn : Callable
        A function that computes the log likelihood.
    mcmc_step_fn : Callable
        A function that performs a single MCMC step.
    mcmc_init_fn : Callable
        A function that initializes the MCMC sampler.
    parameter_update_fn : Callable[[NSState, NSInfo], Dict[str, ArrayTree]]
        A function that updates the parameters given the current state and info.
    num_mcmc_steps : int, optional
        Number of MCMC steps to perform. Default is 10.
    n_delete : int, optional
        Number of particles to delete in each iteration. Default is 1.

    Returns
    -------
    SamplingAlgorithm
        A sampling algorithm object.
    """
    delete_func = partial(delete_fn, n_delete=n_delete)

    kernel = build_kernel(
        logprior_fn,
        loglikelihood_fn,
        delete_func,
        mcmc_step_fn,
        mcmc_init_fn,
        parameter_update_fn,
        num_mcmc_steps,
    )

    def init_fn(position: ArrayLikeTree, rng_key=None):
        del rng_key
        return init(position, loglikelihood_fn, parameter_update_fn)

    def step_fn(rng_key: PRNGKey, state):
        return kernel(rng_key, state)

    return SamplingAlgorithm(init_fn, step_fn)


def ssns(
    logprior_fn: Callable,
    loglikelihood_fn: Callable,
    num_mcmc_steps: int,
    n_delete: int = 1,
) -> SamplingAlgorithm:
    """Implements the a baseline Adaptive Slice Sampling Nested Sampling kernel.

    Parameters
    ----------
    logprior_fn: Callable
        A function that computes the log prior probability.
    loglikelihood_fn: Callable
        A function that computes the log likelihood.
    num_mcmc_steps: int, optional
        Number of MCMC steps to perform. Recommended is 5 times the dimension of the parameter space.
    n_delete: int, optional
        Number of particles to delete in each iteration. Default is 1.

    Returns
    -------
    SamplingAlgorithm
        A sampling algorithm object.
    """
    delete_func = partial(delete_fn, n_delete=n_delete)
    mcmc_step_fn = slice_kernel
    mcmc_init_fn = slice_init

    def parameter_update_fn(state, info):
        cov = jnp.atleast_2d(particles_covariance_matrix(state.particles))
        return {"cov": cov}

    kernel = build_kernel(
        logprior_fn,
        loglikelihood_fn,
        delete_func,
        mcmc_step_fn,
        mcmc_init_fn,
        parameter_update_fn,
        num_mcmc_steps,
    )

    def init_fn(position: ArrayLikeTree, rng_key=None):
        del rng_key
        return init(position, loglikelihood_fn, parameter_update_fn)

    def step_fn(rng_key: PRNGKey, state):
        return kernel(rng_key, state)

    return SamplingAlgorithm(init_fn, step_fn)
