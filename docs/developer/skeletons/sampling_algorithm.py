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
"""Skeleton for a new MCMC sampling algorithm.

Copy this file to ``blackjax/mcmc/<your_algorithm>.py``, replace every
occurrence of ``MySampler`` / ``my_sampler`` with your algorithm's name, fill
in the blanks, and delete these module-level comments.

See ``docs/developer/new_algorithm_guide.md`` for the complete walkthrough.
"""
from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp

from blackjax.base import SamplingAlgorithm, build_sampling_algorithm
from blackjax.types import ArrayLikeTree, ArrayTree, PRNGKey

__all__ = [
    "MySamplerState",
    "MySamplerInfo",
    "init",
    "build_kernel",
    "as_top_level_api",
]


# ---------------------------------------------------------------------------
# State and info
# ---------------------------------------------------------------------------


class MySamplerState(NamedTuple):
    """State of My Sampler.

    position
        Current position of the chain (a JAX pytree).
    logdensity
        Log-density at the current position.
    """

    position: ArrayTree
    logdensity: float
    # Add fields that the kernel needs to carry forward.
    # Do NOT put tuning parameters or adaptation counters here.


class MySamplerInfo(NamedTuple):
    """Transition information for My Sampler.

    acceptance_rate
        Metropolis–Hastings acceptance probability.
    is_accepted
        Whether the proposal was accepted.
    """

    acceptance_rate: float
    is_accepted: bool
    # Add any per-step diagnostics that do not need to persist.


# ---------------------------------------------------------------------------
# Initializer
# ---------------------------------------------------------------------------


def init(
    position: ArrayLikeTree,
    logdensity_fn: Callable,
    *,
    rng_key: PRNGKey | None = None,
) -> MySamplerState:
    """Initialize My Sampler state.

    Parameters
    ----------
    position
        Initial chain position (array or pytree).
    logdensity_fn
        Log-density of the target distribution.
    rng_key
        Optional PRNG key, used only if your algorithm needs randomness at
        initialization (e.g., to sample initial momentum).

    Returns
    -------
    The initial ``MySamplerState``.
    """
    logdensity = logdensity_fn(position)
    return MySamplerState(position, logdensity)


# ---------------------------------------------------------------------------
# Kernel
# ---------------------------------------------------------------------------


def build_kernel(
    # Add algorithm-level configuration here (integrator choice, threshold, …).
    # These are captured by closure; they do NOT appear in the kernel signature.
) -> Callable:
    """Build My Sampler kernel.

    Returns
    -------
    A kernel ``(rng_key, state, logdensity_fn, step_size) -> (MySamplerState, MySamplerInfo)``.
    """

    def kernel(
        rng_key: PRNGKey,
        state: MySamplerState,
        logdensity_fn: Callable,
        step_size: float,
        # Add per-step parameters (step size, mass matrix, …) here.
    ) -> tuple[MySamplerState, MySamplerInfo]:
        """Generate a new sample with My Sampler."""
        key_proposal, key_accept = jax.random.split(rng_key)

        # 1. Generate a proposal.
        new_position = ...  # replace with your proposal logic

        # 2. Evaluate log-density at the proposal.
        new_logdensity = logdensity_fn(new_position)

        # 3. Accept / reject (Metropolis–Hastings).
        log_p_accept = new_logdensity - state.logdensity
        is_accepted = jnp.log(jax.random.uniform(key_accept)) < log_p_accept
        accepted_position = jax.tree.map(
            lambda p, q: jnp.where(is_accepted, p, q),
            new_position,
            state.position,
        )
        accepted_logdensity = jnp.where(is_accepted, new_logdensity, state.logdensity)

        new_state = MySamplerState(accepted_position, accepted_logdensity)
        info = MySamplerInfo(
            acceptance_rate=jnp.exp(jnp.minimum(log_p_accept, 0.0)),
            is_accepted=is_accepted,
        )
        return new_state, info

    return kernel


# ---------------------------------------------------------------------------
# Top-level API
# ---------------------------------------------------------------------------


def as_top_level_api(
    logdensity_fn: Callable,
    step_size: float,
    # Add user-facing parameters here.
) -> SamplingAlgorithm:
    """My Sampler — user-facing convenience wrapper.

    Examples
    --------

    .. code::

        sampler = blackjax.my_sampler(logdensity_fn, step_size=0.1)
        state = sampler.init(initial_position)
        new_state, info = sampler.step(rng_key, state)

    Parameters
    ----------
    logdensity_fn
        The log-density function of the target distribution.
    step_size
        Proposal step size.

    Returns
    -------
    A ``SamplingAlgorithm``.
    """
    kernel = build_kernel()
    return build_sampling_algorithm(
        kernel,
        init,
        logdensity_fn,
        kernel_args=(step_size,),
        # pass_rng_key_to_init=True  # uncomment if init needs rng_key
    )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------
# Put lower-level proposal generators, energy functions, etc. below.
# Keep them private (no entry in __all__) unless they are genuinely reusable
# building blocks for other algorithms.
