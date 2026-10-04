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
"""Public API for the Elliptical Hit-and-Run (EHR) algorithm."""

from typing import Callable, NamedTuple

import jax.numpy as jnp
from jax import lax
from jax.random import split, uniform

import blackjax.mcmc.proposal as proposal
from blackjax.base import SamplingAlgorithm
from blackjax.mcmc.constraints import build_intersect_fn
from blackjax.mcmc.diffusions import (
    DiffusionMetric,
    logdet,
    solve,
    sqrt_multiply,
    sqrt_solve,
)
from blackjax.mcmc.metrics import _format_covariance
from blackjax.mcmc.step_distributions import normchi
from blackjax.types import ArrayLikeTree, ArrayTree, PRNGKey
from blackjax.util import generate_gaussian_noise

__all__ = [
    "init",
    "build_kernel",
    "build_intersect_fn",
    "EHRState",
    "EHRInfo",
    "as_top_level_api",
]


class EHRState(NamedTuple):
    """State of the EHR algorithm.

    position
        Current position of the chain.
    log_density
        Current value of the log-density.
    logdensity_grad
        Current gradient of the log-density.
    metric
        Current local metric.
    clip
        Current clipped step size along the gradient.
    """

    position: ArrayTree
    logdensity: float
    logdensity_grad: ArrayTree
    metric: DiffusionMetric
    clip: float


class EHRInfo(NamedTuple):
    """Additional information on the EHR transition.

    acceptance_rate
        The acceptance rate of the transition.
    is_accepted
        Whether the proposed position was accepted or the original position
        was returned.
    """

    acceptance_rate: float
    is_accepted: bool


def init(
    position: ArrayLikeTree,
    logdensity_fn: Callable,
    constraint,
    vector_field_fn: Callable,
    mass_matrix_fn: Callable,
    step_size: float,
    grad_clip: float = 0.5,
    eps: float = 1e-8,
    max_iter: int = 100,
) -> EHRState:
    intersect_fn = build_intersect_fn(constraint, eps=eps, max_iter=max_iter)

    logdensity = logdensity_fn(position)
    logdensity_grad = vector_field_fn(position)
    metric = mass_matrix_fn(position)

    logdensity_grad = solve(metric, logdensity_grad)  # natural gradient H^{-1}g

    intersection = intersect_fn(position, logdensity_grad)
    clip = lax.select(
        0.5 * step_size**2 < grad_clip * intersection,
        0.5 * step_size**2,
        grad_clip * intersection,
    )

    return EHRState(position, logdensity, logdensity_grad, metric, clip)


def build_kernel(
    constraint, dim: int, step_dist, eps: float = 1e-8, max_iter: int = 100
):
    """Build an EHR kernel.

    Parameters
    ----------
    constraint
        A ``LinearConstraint``, ``QuadraticConstraint``, ``GeneralConstraint``,
        or ``ComposedConstraint`` describing the feasible region.
    dim
        Dimension of the state space.
    step_dist
        Univariate step-size distribution.
    eps
        Tolerance used by Newton's method (for ``GeneralConstraint``).
    max_iter
        Maximum Newton iterations (for ``GeneralConstraint``).

    Returns
    -------
    A kernel that takes a rng_key and a Pytree that contains the current state
    of the chain and returns a new state along with information about the
    transition.
    """
    compute_intersection = build_intersect_fn(constraint, eps=eps, max_iter=max_iter)

    def truncate(dist):
        p_min = dist.cdf(0.0)

        def sample(key, s_max, n=1):
            p_max = dist.cdf(s_max)
            u = uniform(key)
            p = p_min + u * (p_max - p_min)
            return dist.ppf(p)

        def logpdf(x, s_max, step_size=1.0):
            def _in():
                p_max = dist.cdf(s_max)
                logp = dist.logpdf(x)
                return logp - jnp.log(p_max - p_min)

            logp = lax.cond(
                x > s_max,
                lambda: -jnp.inf,
                lambda: lax.cond(0.0 > x, lambda: -jnp.inf, _in),
            )
            return logp

        return sample, logpdf

    trunc_sample, trunc_logpdf = truncate(step_dist)

    def proposal_logdensity_fn(state, new_state, step_size):
        delta = (
            new_state.position - state.position - state.clip * state.logdensity_grad
        )  # Δ = y − x − g
        step = jnp.linalg.norm(sqrt_multiply(state.metric, delta / step_size))
        direction = delta / step / step_size

        s_max = compute_intersection(
            state.position + state.clip * state.logdensity_grad,
            step_size * direction,
        )

        trunc_logp = trunc_logpdf(step, s_max)
        return trunc_logp + 0.5 * logdet(state.metric) - (dim - 1) * jnp.log(step)

    def transition_energy(state, new_state, step_size):
        """Transition energy to go from `state` to `new_state`."""
        proposal_logdensity = lax.cond(
            jnp.isinf(new_state.logdensity),
            lambda state, new_state, stepsize: 0.0,
            proposal_logdensity_fn,
            state,
            new_state,
            step_size,
        )
        return -new_state.logdensity + proposal_logdensity

    compute_acceptance_ratio = proposal.compute_asymmetric_acceptance_ratio(
        transition_energy
    )
    sample_proposal = proposal.static_binomial_sampling

    def kernel(
        rng_key: PRNGKey,
        state: EHRState,
        logdensity_fn: Callable,
        vector_field_fn: Callable,
        mass_matrix_fn: Callable,
        step_size: float,
        grad_clip: float = 0.5,
    ) -> tuple[EHRState, EHRInfo]:
        """Generate a new sample with the EHR kernel."""
        position, _, logdensity_grad, metric, clip = state
        key_direction, key_step, key_accept = split(rng_key, num=3)

        noise = generate_gaussian_noise(key_direction, position)
        noise = noise / jnp.linalg.norm(noise)
        direction = sqrt_solve(metric, noise)

        intersection = compute_intersection(
            position + clip * logdensity_grad,  # type: ignore[operator]
            step_size * direction,
        )
        step = trunc_sample(key_step, intersection)

        new_position = (
            position + clip * logdensity_grad + step * step_size * direction  # type: ignore[operator]
        )

        new_logdensity = logdensity_fn(new_position)
        new_logdensity_grad = vector_field_fn(new_position)
        new_metric = mass_matrix_fn(new_position)
        new_logdensity_grad = solve(new_metric, new_logdensity_grad)

        intersection = compute_intersection(new_position, new_logdensity_grad)
        new_clip = lax.select(
            0.5 * step_size**2 < grad_clip * intersection,
            0.5 * step_size**2,
            grad_clip * intersection,
        )

        new_state = EHRState(
            new_position, new_logdensity, new_logdensity_grad, new_metric, new_clip
        )

        log_p_accept = compute_acceptance_ratio(state, new_state, step_size=step_size)
        accepted_state, info = sample_proposal(
            key_accept, log_p_accept, state, new_state
        )
        do_accept, p_accept, _ = info

        return accepted_state, EHRInfo(p_accept, do_accept)

    return kernel


def as_top_level_api(
    logdensity_fn: Callable,
    constraint,
    dim: int,
    vector_field_fn: Callable,
    mass_matrix_fn: Callable,
    step_size: float,
    step_dist=None,
    grad_clip: float = 0.5,
    format_covariance: bool = True,
    eps: float = 1e-8,
    max_iter: int = 100,
) -> SamplingAlgorithm:
    """Implements the (basic) user interface for the EHR kernel.

    Parameters
    ----------
    logdensity_fn
        The log-density function we wish to draw samples from.
    constraint
        A ``LinearConstraint``, ``QuadraticConstraint``, ``GeneralConstraint``,
        or ``ComposedConstraint`` from ``blackjax.mcmc.constraints`` describing
        the feasible region.  Use ``ComposedConstraint`` to combine multiple
        constraints of different types.  Each type uses its optimal intersection
        solver:

        * ``LinearConstraint``    — closed-form (exact).
        * ``QuadraticConstraint`` — quadratic formula (exact, one step).
        * ``GeneralConstraint``   — Newton's method (controlled by ``max_iter``
                                    and ``eps``).
    dim
        Dimension of the state space.  Used to set the default ``step_dist``.
    vector_field_fn
        A function which computes the drift at a given position.
    mass_matrix_fn
        A function which computes the mass matrix (not its inverse) at a given
        position.
    step_size
        Step size for the EHR algorithm.
    step_dist
        Univariate distribution for the step magnitude.  Defaults to a normal
        distribution moment-matched to a Chi distribution with ``dim`` degrees
        of freedom.
    grad_clip
        Maximum fraction of the distance to the nearest constraint boundary
        used for the gradient drift step.  Defaults to 0.5.
    format_covariance
        If ``True``, ``mass_matrix_fn(position)`` should return the local mass
        matrix; if ``False`` it should return a
        ``blackjax.mcmc.diffusions.DiffusionMetric``.
    eps
        Convergence tolerance for Newton's method (``GeneralConstraint`` only).
    max_iter
        Maximum Newton iterations (``GeneralConstraint`` only).

    Returns
    -------
    A ``SamplingAlgorithm``.

    Examples
    --------
    Linear constraints only::

        from blackjax.mcmc.constraints import LinearConstraint
        algo = blackjax.ehr(logp, LinearConstraint(A, b), dim, grad_fn, M_fn, 0.2)

    Mixed linear + ellipsoidal::

        from blackjax.mcmc.constraints import ComposedConstraint, LinearConstraint, ellipsoid_constraint
        algo = blackjax.ehr(
            logp,
            ComposedConstraint([LinearConstraint(A, b), ellipsoid_constraint(M, center)]),
            dim, grad_fn, M_fn, 0.2,
        )

    References
    ----------
    .. [1] "Higher-Order Hit-&-Run Samplers for Linearly Constrained Densities"
        (https://arxiv.org/abs/2602.14616)
    """
    if step_dist is None:
        step_dist = normchi(dim)

    kernel = build_kernel(constraint, dim, step_dist, eps=eps, max_iter=max_iter)

    if format_covariance:
        _mass_matrix_fn = lambda position: DiffusionMetric(
            *_format_covariance(mass_matrix_fn(position), is_inv=False)[:2]
        )
    else:
        _mass_matrix_fn = mass_matrix_fn

    def init_fn(position: ArrayLikeTree, rng_key=None):
        del rng_key
        return init(
            position,
            logdensity_fn,
            constraint,
            vector_field_fn,
            _mass_matrix_fn,
            step_size,
            grad_clip,
            eps=eps,
            max_iter=max_iter,
        )

    def step_fn(rng_key: PRNGKey, state):
        return kernel(
            rng_key,
            state,
            logdensity_fn,
            vector_field_fn,
            _mass_matrix_fn,
            step_size,
            grad_clip,
        )

    return SamplingAlgorithm(init_fn, step_fn)
