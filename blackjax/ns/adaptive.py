# Copyright 2020- The Blackjax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Adaptive Nested Sampling for BlackJAX.

This module provides an adaptive version of the Nested Sampling algorithm.
In this variant, the parameters of the inner MCMC kernel, which is used to
sample new live points, are updated (tuned) at each iteration of the
Nested Sampling loop. This adaptation is based on the information from the
current set of live particles or the history of the sampling process,
allowing the MCMC kernel to adjust to the changing characteristics of the
constrained prior distribution as the likelihood threshold increases.
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


def init(
    particles: ArrayLikeTree,
    loglikelihood_fn: Callable,
    logprior_fn: Callable,
    mcmc_parameter_update_fn: Callable[[NSState, NSInfo], Dict[str, ArrayTree]],
) -> StateWithParameterOverride:
    """Initializes the state for the Adaptive Nested Sampler.

    This involves initializing the base Nested Sampler state and then computing
    the initial set of parameters for the inner MCMC kernel using the
    `mcmc_parameter_update_fn`.

    Parameters
    ----------
    particles
        An initial set of particles (PyTree of arrays) drawn from the prior
        distribution.
    loglikelihood_fn
        A function that computes the log-likelihood of a single particle.
    logprior_fn
        A function that computes the log-prior of a single particle.
    mcmc_parameter_update_fn
        A function that, given the current `NSState` and `NSInfo` (though info might be None at initialization),
        computes a dictionary of parameters for the inner MCMC kernel.

    Returns
    -------
    StateWithParameterOverride
        The initial state for the adaptive Nested Sampler, including the
        initial MCMC kernel parameters.
    """
    state = init_base(particles, loglikelihood_fn, logprior_fn)
    initial_parameter_value = mcmc_parameter_update_fn(
        state, NSInfo(state, None, None, None, None)
    )
    return StateWithParameterOverride(state, initial_parameter_value)


def build_kernel(
    logprior_fn: Callable,
    loglikelihood_fn: Callable,
    delete_fn: Callable,
    mcmc_build_kernel: Callable,
    mcmc_init_fn: Callable,
    mcmc_parameter_update_fn: Callable[[NSState, NSInfo], Dict[str, ArrayTree]],
    num_mcmc_steps: int,
) -> Callable:
    """Build an adaptive Nested Sampling kernel.

    This kernel extends the base Nested Sampling kernel by re-computing/tuning
    the parameters for the inner MCMC kernel at each step. The
    `mcmc_parameter_update_fn` is called after each NS step to determine the
    MCMC parameters for the *next* NS step.

    Parameters
    ----------
    logprior_fn
        A function that computes the log-prior probability of a single particle.
    loglikelihood_fn
        A function that computes the log-likelihood of a single particle.
    delete_fn
        A function `(rng_key, current_ns_state) -> (dead_indices, live_indices_for_resampling)`
        that identifies particles to be deleted and selects live particles
        to be starting points for new particle generation.
    mcmc_build_kernel
        A function that, when called with MCMC parameters, returns an MCMC kernel function.
    mcmc_init_fn
        A function `(position, logdensity_fn) -> mcmc_state` that initializes
        the state for the MCMC kernel.
    mcmc_parameter_update_fn
        A function that takes the `NSState` and `NSInfo` from the completed NS step
        and returns a dictionary of parameters to be used for the MCMC kernel
        in the *next* NS step.
    num_mcmc_steps
        The number of MCMC steps to run for each new particle generation.

    Returns
    -------
    Callable
        A kernel function for adaptive Nested Sampling:
        `(rng_key, current_adapted_ns_state) -> (new_adapted_ns_state, ns_info)`.
        The `current_adapted_ns_state` is of type `StateWithParameterOverride`.
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
    delete_fn: Callable,
    n_delete: int = 1,
) -> SamplingAlgorithm:
    """Creates an Adaptive Nested Sampling algorithm.

    This convenience function wraps the adaptive `build_kernel` and `init`
    functions into a `SamplingAlgorithm` object. The inner MCMC kernel's
    parameters are tuned at each step using `mcmc_parameter_update_fn`.

    Parameters
    ----------
    logprior_fn
        A function that computes the log-prior probability of a single particle.
    loglikelihood_fn
        A function that computes the log-likelihood of a single particle.
    mcmc_build_kernel
        A function that, when called with MCMC parameters, returns an MCMC kernel.
    mcmc_init_fn
        A function that initializes the state for the MCMC kernel.
    mcmc_parameter_update_fn
        A function that takes the `NSState` and `NSInfo` from a completed NS step
        and returns a dictionary of parameters for the MCMC kernel for the next step.
    num_mcmc_steps
        The number of MCMC steps to run for each new particle generation.
    n_delete
        The number of particles to delete and replace at each NS step.
        Defaults to 1.

    Returns
    -------
    SamplingAlgorithm
        A `SamplingAlgorithm` tuple containing `init` and `step` functions for
        the configured Adaptive Nested Sampler. The state managed by this
        algorithm is of type `StateWithParameterOverride`.
    """
    delete_fn = partial(delete_fn, n_delete=n_delete)

    step_fn = build_kernel(
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
        return init(particles, loglikelihood_fn, logprior_fn, mcmc_parameter_update_fn)

    return SamplingAlgorithm(init_fn, step_fn)
