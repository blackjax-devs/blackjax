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
"""Public API for the position-dependant random walk algorithm."""
import operator
from typing import Callable, NamedTuple

import jax
from jax import numpy as jnp

from blackjax.base import SamplingAlgorithm
from blackjax.mcmc import proposal
from blackjax.mcmc.diffusions import DiffusionMetric, logdet, sqrt_multiply, sqrt_solve
from blackjax.mcmc.metrics import _format_covariance
from blackjax.types import ArrayLikeTree, ArrayTree, PRNGKey
from blackjax.util import generate_gaussian_noise

__all__ = [
    "init",
    "build_kernel",
    "PosDepRWMHInfo",
    "PosDepRWMHState",
    "as_top_level_api",
]


class PosDepRWMHState(NamedTuple):
    """State of the position-dependant RW chain.

    position
        Current position of the chain.
    log_density
        Current value of the log-density.
    metric
        Current local metric.

    """

    position: ArrayTree
    logdensity: float
    metric: DiffusionMetric


class PosDepRWMHInfo(NamedTuple):
    """Additional information on position-dependant RW chain.

    This additional information can be used for debugging or computing
    diagnostics.

    acceptance_rate
        The acceptance probability of the transition, linked to the energy
        difference between the original and the proposed states.
    is_accepted
        Whether the proposed position was accepted or the original position
        was returned.
    """

    acceptance_rate: float
    is_accepted: bool


def init(
    position: ArrayLikeTree, logdensity_fn: Callable, mass_matrix_fn: Callable
) -> PosDepRWMHState:
    """Create a chain state from a position.

    Parameters
    ----------
    position: PyTree
        The initial position of the chain
    logdensity_fn: Callable
        Log-probability density function of the distribution we wish to sample
        from.
    mass_matrix_fn: Callable
        A function which computes the mass matrix (not inverse) at a given
        position when drawing a value for the momentum and computing the kinetic
        energy.
    """
    return PosDepRWMHState(position, logdensity_fn(position), mass_matrix_fn(position))


def build_kernel():
    """Build a Rosenbluth-Metropolis-Hastings kernel with position-dependent covariance.

    Returns
    -------
    A kernel that takes a rng_key and a Pytree that contains the current state
    of the chain and that returns a new state of the chain along with
    information about the transition.
    """

    # computes -log p(y)q(x|y) where x is `state` and y is `new_state`
    def transition_energy(state, new_state, step_size):
        """Transition energy to go from `state` to `new_state`"""
        theta = jax.tree_util.tree_map(
            lambda x, y: x - y,
            state.position,
            new_state.position,
        )

        theta_scaled = sqrt_multiply(
            new_state.metric,
            theta,
        )

        theta_dot = jax.tree_util.tree_reduce(
            operator.add, jax.tree_util.tree_map(lambda t: jnp.sum(t * t), theta_scaled)
        )

        log_det_H = logdet(new_state.metric)
        return -new_state.logdensity + (0.25 / step_size) * theta_dot - 0.5 * log_det_H

    compute_acceptance_ratio = proposal.compute_asymmetric_acceptance_ratio(
        transition_energy
    )
    sample_proposal = proposal.static_binomial_sampling

    def kernel(
        rng_key: PRNGKey,
        state: PosDepRWMHState,
        logdensity_fn: Callable,
        mass_matrix_fn: Callable,
        step_size: float,
    ) -> tuple[PosDepRWMHState, PosDepRWMHInfo]:
        """Generate a new sample with the position-dependant RW kernel."""
        position, _, metric = state

        key_noise, key_accept = jax.random.split(rng_key)
        noise = generate_gaussian_noise(rng_key, position)
        noise = sqrt_solve(metric, noise)

        position = jax.tree_util.tree_map(
            lambda p, n: p + jnp.sqrt(2 * step_size) * n,
            position,
            noise,
        )

        logdensity = logdensity_fn(position)
        metric = mass_matrix_fn(position)

        new_state = PosDepRWMHState(position, logdensity, metric)

        log_p_accept = compute_acceptance_ratio(state, new_state, step_size=step_size)
        accepted_state, info = sample_proposal(
            key_accept, log_p_accept, state, new_state
        )
        do_accept, p_accept, _ = info

        info = PosDepRWMHInfo(
            p_accept,
            do_accept,
        )

        return accepted_state, info

    return kernel


def as_top_level_api(
    logdensity_fn: Callable,
    mass_matrix_fn: Callable,
    step_size: float,
    format_covariance: bool = True,
) -> SamplingAlgorithm:
    """Implements the (basic) user interface for the position-dependant RW kernel.

    Parameters
    ----------
    logdensity_fn
        The log-density function we wish to draw samples from.
    mass_matrix_fn
        A function which computes the mass matrix (not inverse) at a given
        position.
    step_size
        The value to use for the step size in the symplectic integrator.
    format_covariance
        If true, `mass_matrix_fn(position)` is expected to return the local mass matrix,
        if false, `mass_matrix_fn(position)` is expected to return a `blackjax.mcmc.diffusions.DiffusionMetric`
        object.

    Returns
    -------
    A ``SamplingAlgorithm``.

    References
    ----------
    .. [1] "Geometric ergodicity of the Random Walk Metropolis with position-dependent proposal covariance"
        (https://arxiv.org/abs/1507.05780)

    """

    kernel = build_kernel()

    if format_covariance:
        _mass_matrix_fn = lambda position: DiffusionMetric(
            *_format_covariance(mass_matrix_fn(position), is_inv=False)[:2]
        )
    else:
        _mass_matrix_fn = mass_matrix_fn

    def init_fn(position: ArrayLikeTree, rng_key=None):
        del rng_key
        return init(position, logdensity_fn, _mass_matrix_fn)

    def step_fn(rng_key: PRNGKey, state):
        return kernel(
            rng_key,
            state,
            logdensity_fn,
            _mass_matrix_fn,
            step_size,
        )

    return SamplingAlgorithm(init_fn, step_fn)
