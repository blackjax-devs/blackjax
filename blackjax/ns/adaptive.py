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

from typing import Callable, Dict, NamedTuple


from blackjax.ns.base import NSInfo, NSState
from blackjax.ns.base import build_kernel as base_build_kernel
from blackjax.ns.base import init as base_init
from blackjax.types import ArrayLikeTree, ArrayTree, PRNGKey, Array
import jax.numpy as jnp

__all__ = ["init", "build_kernel"]



class AdaptiveNSState(NamedTuple):
    base_state: NSState
    inner_kernel_params: Dict[str, ArrayTree]

    # Define properties to access base_state fields directly
    @property
    def particles(self) -> ArrayLikeTree:
        return self.base_state.particles

    @property
    def loglikelihood(self) -> Array:
        return self.base_state.loglikelihood

    @property
    def loglikelihood_birth(self) -> Array:
        return self.base_state.loglikelihood_birth

    @property
    def logprior(self) -> Array:
        return self.base_state.logprior

    @property
    def pid(self) -> Array:
        return self.base_state.pid

    @property
    def logX(self) -> float:
        return self.base_state.logX

    @property
    def logZ_live(self) -> float:
        return self.base_state.logZ_live

    @property
    def logZ(self) -> float:
        return self.base_state.logZ




def init(
    particles: ArrayLikeTree,
    logprior_fn: Callable,
    loglikelihood_fn: Callable,
    loglikelihood_birth: Array = -jnp.nan,
    update_inner_kernel_params_fn: Callable = lambda state: {},
) -> AdaptiveNSState:
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

    Returns
    -------
    AdaptiveNSState
        The initial state of the Nested Sampler.
    """
    base_state = base_init(particles, logprior_fn, loglikelihood_fn, loglikelihood_birth)
    inner_kernel_params = update_inner_kernel_params_fn(base_state, None)
    return AdaptiveNSState(base_state, inner_kernel_params)


def build_kernel(
    logprior_fn: Callable,
    loglikelihood_fn: Callable,
    delete_fn: Callable,
    inner_init_fn: Callable,
    inner_kernel: Callable,
    update_inner_kernel_params_fn: Callable[[NSState, NSInfo], Dict[str, ArrayTree]],
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
        A function `(rng_key, current_ns_state) -> (dead_indices,
        live_indices_for_resampling)` that identifies particles to be deleted
        and selects live particles to be starting points for new particle
        generation.
    inner_init_fn
        A function `(initial_position: ArrayTree) -> inner_state` used to
        initialize the state for the inner kernel. The `logdensity_fn`
        for this inner kernel will be partially applied before this init
        function is called within the main NS loop.
    inner_kernel
        A function that, when called with inner_kernel parameters, returns a
        kernel function `(rng_key, state, logdensity_fn) -> (new_state, info)`.
    update_inner_kernel_params_fn
        A function that takes the `NSState` and `NSInfo` from the completed NS
        step and returns a dictionary of parameters to be used for the kernel
        in the *next* NS step.

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
        inner_init_fn,
        inner_kernel,
    )

    def kernel(rng_key: PRNGKey, state: AdaptiveNSState) -> tuple[AdaptiveNSState, NSInfo]:
        """Performs one step of adaptive Nested Sampling.

        This involves running a step of the base Nested Sampling algorithm using
        the current inner kernel parameters, and then updating these parameters
        for the next step.

        Parameters
        ----------
        rng_key
            A JAX PRNG key.
        state
            The current `AdaptiveNSState`.

        Returns
        -------
        tuple[AdaptiveNSState, NSInfo]
            A tuple with the new `AdaptiveNSState` (including updated inner kernel
            parameters) and the `NSInfo` for this step.
        """
        base_state = state.base_state
        new_base_state, info = base_kernel(rng_key, base_state, state.inner_kernel_params)
        inner_kernel_params = update_inner_kernel_params_fn(new_base_state, info, state.inner_kernel_params)
        new_state = AdaptiveNSState(
            base_state=new_base_state,
            inner_kernel_params=inner_kernel_params,
        )
        return new_state, info

    return kernel
