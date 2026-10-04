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
"""Public API for simplified Manifold Metropolis Adjusted Langevin kernels."""
import operator
from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp

import blackjax.mcmc.diffusions as diffusions
import blackjax.mcmc.proposal as proposal
from blackjax.base import SamplingAlgorithm
from blackjax.mcmc.diffusions import DiffusionMetric, logdet, solve, sqrt_multiply
from blackjax.mcmc.metrics import _format_covariance
from blackjax.types import ArrayLikeTree, ArrayTree, PRNGKey

__all__ = ["SMMALAState", "SMMALAInfo", "init", "build_kernel", "as_top_level_api"]


class SMMALAState(NamedTuple):
    """State of the SMMALA algorithm.

    The SMMALA algorithm takes one position of the chain and returns another
    position. In order to make computations more efficient, we also store
    the current log-probability density as well as the current gradient of the
    log-probability density and the current metric.

    """

    position: ArrayTree
    logdensity: float
    logdensity_grad: ArrayTree
    metric: DiffusionMetric


class SMMALAInfo(NamedTuple):
    """Additional information on the SMMALA transition.

    This additional information can be used for debugging or computing
    diagnostics.

    acceptance_rate
        The acceptance rate of the transition.
    is_accepted
        Whether the proposed position was accepted or the original position
        was returned.

    """

    acceptance_rate: float
    is_accepted: bool


def init(
    position: ArrayLikeTree, logdensity_fn: Callable, mass_matrix_fn: Callable
) -> SMMALAState:
    grad_fn = jax.value_and_grad(logdensity_fn)
    logdensity, grad = grad_fn(position)
    metric = mass_matrix_fn(position)

    grad = solve(metric, grad)  # natural gradient

    return SMMALAState(position, logdensity, grad, metric)


def build_kernel():
    """Build a SMMALA kernel.

    Returns
    -------
    A kernel that takes a rng_key and a Pytree that contains the current state
    of the chain and that returns a new state of the chain along with
    information about the transition.

    """

    # computes -log p(y)q(x|y) where x is `state` and y is `new_state`
    def transition_energy(state, new_state, step_size):
        """Transition energy to go from `state` to `new_state`."""

        theta = jax.tree_util.tree_map(
            lambda x, y, gy: x - y - step_size * gy,
            state.position,
            new_state.position,
            new_state.logdensity_grad,
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
        state: SMMALAState,
        logdensity_fn: Callable,
        mass_matrix_fn: Callable,
        step_size: float,
    ) -> tuple[SMMALAState, SMMALAInfo]:
        """Generate a new sample with the SMMALA kernel."""
        grad_fn = jax.value_and_grad(logdensity_fn)
        integrator = diffusions.overdamped_manifold_langevin(
            grad_fn,
            mass_matrix_fn,
        )

        key_integrator, key_rmh = jax.random.split(rng_key)

        new_state = integrator(key_integrator, state, step_size)
        new_state = SMMALAState(*new_state)

        log_p_accept = compute_acceptance_ratio(state, new_state, step_size=step_size)
        accepted_state, info = sample_proposal(key_rmh, log_p_accept, state, new_state)
        do_accept, p_accept, _ = info

        info = SMMALAInfo(
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
    """Implements the (basic) user interface for the SMMALA kernel.

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
    .. [1] "Riemann manifold Langevin and Hamiltonian Monte Carlo methods"
        (https://rss.onlinelibrary.wiley.com/doi/10.1111/j.1467-9868.2010.00765.x)

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
        return kernel(rng_key, state, logdensity_fn, _mass_matrix_fn, step_size)

    return SamplingAlgorithm(init_fn, step_fn)
