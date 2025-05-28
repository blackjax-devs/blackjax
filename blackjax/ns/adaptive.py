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

from typing import Callable, Dict, Optional

import jax.numpy as jnp

from blackjax.ns.base import NSInfo, NSState
from blackjax.ns.base import build_kernel as base_build_kernel
from blackjax.ns.base import init as base_init
from blackjax.types import Array, ArrayLikeTree, ArrayTree, PRNGKey

__all__ = ["init", "build_kernel"]


def init(
    particles: ArrayLikeTree,
    logprior_fn: Callable,
    loglikelihood_fn: Callable,
    loglikelihood_birth: Array = -jnp.nan,
    update_inner_kernel_params_fn: Optional[Callable] = None,
) -> NSState:
    """Initializes the Nested Sampler state.

    Parameters
    ----------
    particles
        An initial set of particles (PyTree of arrays) drawn from the prior
        distribution. The leading dimension of each leaf array must be equal to
        the number of particles.
    loglikelihood_fn
        A function that computes the log-likelihood of a single particle.
    logprior_fn
        A function that computes the log-prior of a single particle.
    loglikelihood_birth
        The initial log-likelihood birth threshold. Defaults to -NaN, which
        implies no initial likelihood constraint beyond the prior.
    update_inner_kernel_params_fn
        A function that takes the `NSState`, `NSInfo` from the completed NS
        step, and the current inner kernel parameters dictionary, and returns
        a dictionary of parameters to be used for the kernel in the *next* NS step.

    Returns
    -------
    NSState
        The initial state of the Nested Sampler.
    """
    state = base_init(particles, logprior_fn, loglikelihood_fn, loglikelihood_birth)
    if update_inner_kernel_params_fn is not None:
        inner_kernel_params = update_inner_kernel_params_fn(state, None, {})
        state = state._replace(inner_kernel_params=inner_kernel_params)
    return state


def build_kernel(
    logprior_fn: Callable,
    loglikelihood_fn: Callable,
    delete_fn: Callable,
    inner_kernel: Callable,
    update_inner_kernel_params_fn: Callable[
        [NSState, NSInfo, Dict[str, ArrayTree]], Dict[str, ArrayTree]
    ],
) -> Callable:
    """Build an adaptive Nested Sampling kernel.

    This kernel extends the base Nested Sampling kernel by re-computing/tuning
    the parameters for the inner kernel at each step. The `update_inner_kernel_params_fn`
    is called after each NS step to determine the parameters for the *next* NS
    step.

    Parameters
    ----------
    logprior_fn
        A function that computes the log-prior probability of a single particle.
    loglikelihood_fn
        A function that computes the log-likelihood of a single particle.
    delete_fn
        this particle deletion function has the signature
        `(rng_key, current_state) -> (dead_idx, target_update_idx, start_idx)`
        and identifies particles to be deleted, particles to be updated, and
        selects live particles to be starting points for the inner kernel
        for new particle generation.
    inner_kernel
        This kernel function has the signature
        `(rng_key, inner_state, logprior_fn, loglikelihood_fn, loglikelihood_0, inner_kernel_params) -> (new_inner_state, inner_info)`,
        and is used to generate new particles.
    update_inner_kernel_params_fn
        A function that takes the `NSState`, `NSInfo` from the completed NS
        step, and the current inner kernel parameters dictionary, and returns
        a dictionary of parameters to be used for the kernel in the *next* NS step.

    Returns
    -------
    Callable
        A kernel function for adaptive Nested Sampling. It takes an `rng_key` and the
        current `NSState` and returns a tuple containing the new `NSState` and
        the `NSInfo` for the step.
    """

    base_kernel = base_build_kernel(
        logprior_fn,
        loglikelihood_fn,
        delete_fn,
        inner_kernel,
    )

    def kernel(rng_key: PRNGKey, state: NSState) -> tuple[NSState, NSInfo]:
        """Performs one step of adaptive Nested Sampling.

        This involves running a step of the base Nested Sampling algorithm using
        the current inner kernel parameters, and then updating these parameters
        for the next step.

        Parameters
        ----------
        rng_key
            A JAX PRNG key.
        state
            The current `NSState`.

        Returns
        -------
        tuple[NSState, NSInfo]
            A tuple with the new `NSState` (including updated inner kernel
            parameters) and the `NSInfo` for this step.
        """
        new_state, info = base_kernel(rng_key, state)

        inner_kernel_params = update_inner_kernel_params_fn(
            new_state, info, new_state.inner_kernel_params
        )
        new_state = new_state._replace(inner_kernel_params=inner_kernel_params)
        return new_state, info

    return kernel
