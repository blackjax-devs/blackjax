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
In this variant, the parameters of the inner kernel, which is used to
sample new live points, are updated (tuned) at each iteration of the
Nested Sampling loop. This adaptation is based on the information from the
current set of live particles or the history of the sampling process,
allowing the kernel to adjust to the changing characteristics of the
constrained prior distribution as the likelihood threshold increases.
"""

from typing import Callable, Dict, Tuple, Optional, Any

from blackjax.ns.base import NSInfo, NSState
from blackjax.ns.base import build_kernel as base_build_kernel
from blackjax.ns.base import init as init_base
from blackjax.smc.inner_kernel_tuning import StateWithParameterOverride
from blackjax.types import ArrayLikeTree, ArrayTree, PRNGKey, Array

__all__ = ["init", "build_kernel"]


def init(
    particles: ArrayLikeTree,
    loglikelihood_fn: Callable[[ArrayTree], float],
    logprior_fn: Callable[[ArrayTree], float],
    update_inner_kernel: Callable[[NSState, Optional[NSInfo]], Dict[str, ArrayTree]],
) -> StateWithParameterOverride[NSState, Dict[str, ArrayTree]]:
    """Initializes the state for the Adaptive Nested Sampler.

    This involves initializing the base Nested Sampler state and then computing
    the initial set of parameters for the inner kernel using `update_inner_kernel`.

    Parameters
    ----------
    particles
        An initial set of particles (PyTree of arrays) drawn from the prior
        distribution.
    loglikelihood_fn
        A function that computes the log-likelihood of a single particle.
    logprior_fn
        A function that computes the log-prior of a single particle.
    update_inner_kernel
        A function that, given the current `NSState` and `NSInfo` (which will
        be `None` at initialization), computes a dictionary of parameters
        for the inner kernel.

    Returns
    -------
    StateWithParameterOverride
        The initial state for the adaptive Nested Sampler, including the
        initial kernel parameters.
    """
    state = init_base(particles, loglikelihood_fn, logprior_fn)
    initial_parameter_value = update_inner_kernel(state, None)
    return StateWithParameterOverride(state, initial_parameter_value)


def build_kernel(
    logprior_fn: Callable[[ArrayTree], float],
    loglikelihood_fn: Callable[[ArrayTree], float],
    delete_fn: Callable[[PRNGKey, NSState], Tuple[Array, Array, Array]],
    inner_init_fn: Callable[[ArrayTree], Any], # Type of inner state can vary
    inner_kernel: Callable[..., Callable[[PRNGKey, Any, Callable[[ArrayTree], float]], Tuple[Any, Any]]], # Higher-order fn
    update_inner_kernel: Callable[[NSState, NSInfo], Dict[str, ArrayTree]],
) -> Callable[[PRNGKey, StateWithParameterOverride[NSState, Dict[str, ArrayTree]]], Tuple[StateWithParameterOverride[NSState, Dict[str, ArrayTree]], NSInfo]]:
    """Build an adaptive Nested Sampling kernel.

    This kernel extends the base Nested Sampling kernel by re-computing/tuning
    the parameters for the inner kernel at each step. The `update_inner_kernel`
    function is called after each NS step to determine the parameters for the
    inner kernel in the *next* NS step.

    Parameters
    ----------
    logprior_fn
        A function that computes the log-prior probability of a single particle.
    loglikelihood_fn
        A function that computes the log-likelihood of a single particle.
    delete_fn
        A function `(rng_key, current_ns_state) -> (dead_indices,
        target_update_indices, live_indices_for_resampling)` that identifies
        particles to be deleted and selects live particles to be starting points
        for new particle generation.
    inner_init_fn
        A function `(initial_position: ArrayTree) -> inner_state` used to
        initialize the state for the inner kernel. The `logdensity_fn`
        for this inner kernel will be partially applied before this init function
        is called within the main NS loop.
    inner_kernel
        This kernel function has the signature
        `(rng_key, inner_mcmc_state, constrained_logdensity_fn, **inner_kernel_parameters) -> (new_inner_state, inner_info)`.
    update_inner_kernel
        A function that takes the `NSState` and `NSInfo` from the completed NS
        step and returns a dictionary of parameters to be used for the inner
        kernel in the *next* NS step.

    Returns
    -------
    Callable
        A kernel function for adaptive Nested Sampling. It takes an `rng_key`
        and the current `StateWithParameterOverride` (which bundles the `NSState`
        and current inner kernel parameters) and returns a tuple containing the
        new `StateWithParameterOverride` and the `NSInfo` for the step.
    """

    base_kernel = base_build_kernel(
        logprior_fn,
        loglikelihood_fn,
        delete_fn,
        inner_init_fn,
        inner_kernel,
    )

    def kernel(
        rng_key: PRNGKey,
        state: StateWithParameterOverride[NSState, Dict[str, ArrayTree]],
    ) -> Tuple[StateWithParameterOverride[NSState, Dict[str, ArrayTree]], NSInfo]:
        """Performs one step of adaptive Nested Sampling.

        This involves running a step of the base Nested Sampling algorithm using
        the current inner kernel parameters, and then updating these parameters
        for the next step.

        Parameters
        ----------
        rng_key
            A JAX PRNG key.
        state
            The current `StateWithParameterOverride`, containing the `NSState`
            and the inner kernel parameters.

        Returns
        -------
        tuple[StateWithParameterOverride, NSInfo]
            A tuple with the new `StateWithParameterOverride` (including updated
            inner kernel parameters) and the `NSInfo` for this step.
        """
        new_state, info = basekernel(
            rng_key, state.sampler_state, state.parameter_override
        )
        new_parameter_override = update_inner_kernel(new_state, info)
        return (
            StateWithParameterOverride(new_state, new_parameter_override),
            info,
        )

    return kernel
