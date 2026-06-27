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

This combines the SMC equivalent of Adaptive Tempering and inner kernel tuning in one file.
"""

from functools import partial
from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp

from blackjax.ns.base import NSInfo, NSState, StateWithLogLikelihood
from blackjax.ns.base import build_kernel as base_build_kernel
from blackjax.ns.base import init as base_init
from blackjax.ns.integrator import NSIntegrator, init_integrator, update_integrator
from blackjax.types import ArrayLikeTree, ArrayTree, PRNGKey

__all__ = ["init", "build_kernel"]


class AdaptiveNSState(NamedTuple):
    """State of the adaptive Nested Sampling chain.

    particles
        The ``StateWithLogLikelihood`` of the current live particles.
    integrator
        The ``NSIntegrator`` that tracks evidence-related statistics.
    inner_kernel_params
        Parameters for the inner kernel used to generate new particles.
    """

    particles: StateWithLogLikelihood
    integrator: NSIntegrator
    inner_kernel_params: dict[str, ArrayTree]


def init(
    positions: ArrayLikeTree,
    init_state_fn: Callable,
    loglikelihood_birth: float = jnp.nan,
    update_inner_kernel_params_fn: Callable | None = None,
    rng_key: PRNGKey | None = None,
) -> AdaptiveNSState:
    """Initialize the adaptive Nested Sampling state from live positions.

    Parameters
    ----------
    positions
        Initial positions of the live particles.
    init_state_fn
        Maps positions to a ``StateWithLogLikelihood`` (typically vmapped).
    loglikelihood_birth
        Birth log-likelihood assigned to the initial particles.
    update_inner_kernel_params_fn
        Optional ``(rng_key, state, info, params) -> params`` used to seed the
        inner kernel parameters; if ``None`` the parameters start empty.
    rng_key
        PRNG key passed to ``update_inner_kernel_params_fn``.

    Returns
    -------
    The initial ``AdaptiveNSState``.
    """
    base_state = base_init(
        positions, init_state_fn, loglikelihood_birth=loglikelihood_birth
    )
    integrator = init_integrator(base_state.particles)
    inner_kernel_params = {}
    if update_inner_kernel_params_fn is not None:
        inner_kernel_params = update_inner_kernel_params_fn(
            rng_key, base_state, None, {}
        )
    return AdaptiveNSState(
        particles=base_state.particles,
        inner_kernel_params=inner_kernel_params,
        integrator=integrator,
    )


def build_kernel(
    delete_fn: Callable,
    inner_kernel: Callable,
    update_inner_kernel_params_fn: Callable[
        [PRNGKey, NSState, NSInfo, dict[str, ArrayTree]], dict[str, ArrayTree]
    ],
) -> Callable:
    """Build an adaptive Nested Sampling kernel.

    The kernel tunes the inner kernel parameters each step from the current
    state and the information from the previous update.

    Parameters
    ----------
    delete_fn
        Selects which live particles to delete and replace each step.
    inner_kernel
        Inner MCMC kernel used to generate replacement particles, called with
        the current ``inner_kernel_params``.
    update_inner_kernel_params_fn
        ``(rng_key, state, info, params) -> params`` recomputing the inner
        kernel parameters after each step.

    Returns
    -------
    A kernel ``(rng_key, AdaptiveNSState) -> (AdaptiveNSState, NSInfo)``.
    """

    def kernel(
        rng_key: PRNGKey, state: AdaptiveNSState
    ) -> tuple[AdaptiveNSState, NSInfo]:
        """Performs one step of adaptive Nested Sampling."""
        adapted_kernel = base_build_kernel(
            delete_fn,
            partial(inner_kernel, **state.inner_kernel_params),
        )

        new_state, info = adapted_kernel(rng_key, state)
        inner_kernel_update_key, rng_key = jax.random.split(rng_key)
        new_inner_kernel_params = update_inner_kernel_params_fn(
            inner_kernel_update_key, new_state, info, new_state.inner_kernel_params
        )
        new_integrator_state = update_integrator(
            state.integrator, new_state.particles, info.particles
        )
        return (
            AdaptiveNSState(
                particles=new_state.particles,
                inner_kernel_params=new_inner_kernel_params,
                integrator=new_integrator_state,
            ),
            info,
        )

    return kernel
