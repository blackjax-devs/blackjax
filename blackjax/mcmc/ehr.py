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

import operator
from collections.abc import Callable
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import lax
from jax.flatten_util import ravel_pytree
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
    drift_fn: Callable,
    mass_matrix_fn: Callable,
    step_size: float,
    grad_clip: float = 0.5,
    eps: float = 1e-8,
    max_iter: int = 100,
) -> EHRState:
    intersect_fn = build_intersect_fn(constraint, eps=eps, max_iter=max_iter)

    logdensity = logdensity_fn(position)
    logdensity_grad = drift_fn(position)
    metric = mass_matrix_fn(position)

    logdensity_grad = solve(metric, logdensity_grad)  # natural gradient H^{-1}g

    # Flatten pytree position / grad to 1-D arrays for intersection solver.
    pos_flat, _ = ravel_pytree(position)
    grad_flat, _ = ravel_pytree(logdensity_grad)

    intersection = intersect_fn(pos_flat, grad_flat)
    clip = lax.select(
        0.5 * step_size**2 < grad_clip * intersection,
        0.5 * step_size**2,
        grad_clip * intersection,
    )

    return EHRState(position, logdensity, logdensity_grad, metric, clip)


def build_kernel(constraint, step_dist=None, eps: float = 1e-8, max_iter: int = 100):
    """Build an EHR kernel.

    Parameters
    ----------
    constraint
        A ``LinearConstraint``, ``QuadraticConstraint``, ``GeneralConstraint``,
        or ``ComposedConstraint`` describing the feasible region.
    step_dist
        Univariate step-size distribution.  Defaults to a normal distribution
        moment-matched to a Chi distribution whose degrees of freedom are
        inferred from the flattened position at trace time.
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
    sample_proposal = proposal.static_binomial_sampling

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

    def make_transition_energy(step_dim, trunc_logpdf):
        def proposal_logdensity_fn(state, new_state, step_size):
            delta = jax.tree.map(
                lambda y, x, g: y - x - state.clip * g,
                new_state.position,
                state.position,
                state.logdensity_grad,
            )  # Δ = y − x − clip*g  (pytree)

            # Flatten for norm computation; sqrt_multiply unravels internally.
            delta_scaled = sqrt_multiply(
                state.metric,
                jax.tree.map(lambda d: d / step_size, delta),
            )
            delta_scaled_flat, _ = ravel_pytree(delta_scaled)
            step = jnp.linalg.norm(delta_scaled_flat)

            delta_flat, _ = ravel_pytree(delta)
            direction_flat = delta_flat / step / step_size

            pos_flat, _ = ravel_pytree(state.position)
            grad_flat, _ = ravel_pytree(state.logdensity_grad)
            s_max = compute_intersection(
                pos_flat + state.clip * grad_flat,
                step_size * direction_flat,
            )

            trunc_logp = trunc_logpdf(step, s_max)
            return (
                trunc_logp + 0.5 * logdet(state.metric) - (step_dim - 1) * jnp.log(step)
            )

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

        return transition_energy

    def kernel(
        rng_key: PRNGKey,
        state: EHRState,
        logdensity_fn: Callable,
        drift_fn: Callable,
        mass_matrix_fn: Callable,
        step_size: float,
        grad_clip: float = 0.5,
    ) -> tuple[EHRState, EHRInfo]:
        """Generate a new sample with the EHR kernel."""
        position, _, logdensity_grad, metric, clip = state
        key_direction, key_step, key_accept = split(rng_key, num=3)

        # Flatten position to infer the state-space dimension.  This is a static
        # shape, so it can be used when tracing the kernel.
        pos_flat, _ = ravel_pytree(position)
        step_dim = pos_flat.shape[0]
        dist = step_dist if step_dist is not None else normchi(step_dim)
        trunc_sample, trunc_logpdf = truncate(dist)

        transition_energy = make_transition_energy(step_dim, trunc_logpdf)
        compute_acceptance_ratio = proposal.compute_asymmetric_acceptance_ratio(
            transition_energy
        )

        noise = generate_gaussian_noise(key_direction, position)
        # Flatten to 1-D for norm; sqrt_solve will unravel back to pytree.
        noise_flat, unravel_fn = ravel_pytree(noise)
        noise_flat = noise_flat / jnp.linalg.norm(noise_flat)
        noise = unravel_fn(noise_flat)
        direction = sqrt_solve(metric, noise)  # pytree with same structure as position

        # Flatten position, grad, and direction for the intersection solver.
        grad_flat, _ = ravel_pytree(logdensity_grad)
        dir_flat, _ = ravel_pytree(direction)

        intersection = compute_intersection(
            pos_flat + clip * grad_flat,
            step_size * dir_flat,
        )
        step = trunc_sample(key_step, intersection)

        new_position = jax.tree.map(
            lambda p, g, d: p + clip * g + step * step_size * d,
            position,
            logdensity_grad,
            direction,
        )

        new_logdensity = logdensity_fn(new_position)
        new_logdensity_grad = drift_fn(new_position)
        new_metric = mass_matrix_fn(new_position)
        new_logdensity_grad = solve(new_metric, new_logdensity_grad)

        new_pos_flat, _ = ravel_pytree(new_position)
        new_grad_flat, _ = ravel_pytree(new_logdensity_grad)
        intersection = compute_intersection(new_pos_flat, new_grad_flat)
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


def _zero_drift(x):
    return jax.tree.map(jnp.zeros_like, x)


def _eye_mass(x):
    return jnp.eye(jax.tree.reduce(operator.add, jax.tree.map(lambda z: z.shape, x)))


def as_top_level_api(
    logdensity_fn: Callable,
    constraint,
    step_size: float,
    drift_fn: Callable = _zero_drift,
    mass_matrix_fn: Callable = _eye_mass,
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
    drift_fn
        A function which computes the drift at a given position.
    mass_matrix_fn
        A function which computes the mass matrix (not its inverse) at a given
        position.
    step_size
        Step size for the EHR algorithm.
    step_dist
        Univariate distribution for the step magnitude.  Defaults to a normal
        distribution moment-matched to a Chi distribution whose degrees of
        freedom are inferred from the flattened position at trace time.
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
        algo = blackjax.ehr(
            logp,
            LinearConstraint(A, b),
            step_size=0.2,
            drift_fn=grad_fn,
            mass_matrix_fn=M_fn,
        )

    Mixed linear + ellipsoidal::

        from blackjax.mcmc.constraints import ComposedConstraint, LinearConstraint, ellipsoid_constraint
        algo = blackjax.ehr(
            logp,
            ComposedConstraint([LinearConstraint(A, b), ellipsoid_constraint(M, center)]),
            step_size=0.2,
            drift_fn=grad_fn,
            mass_matrix_fn=M_fn,
        )

    References
    ----------
    .. [1] "Higher-Order Hit-&-Run Samplers for Linearly Constrained Densities"
        (https://arxiv.org/abs/2602.14616)
    """
    kernel = build_kernel(constraint, step_dist, eps=eps, max_iter=max_iter)

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
            drift_fn,
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
            drift_fn,
            _mass_matrix_fn,
            step_size,
            grad_clip,
        )

    return SamplingAlgorithm(init_fn, step_fn)


def hr_as_top_level_api(logdensity_fn, constraint, step_size, eps=1e-8, max_iter=100):
    return as_top_level_api(
        logdensity_fn,
        constraint,
        step_size,
        eps=eps,
        max_iter=max_iter,
    )


def lhr_as_top_level_api(logdensity_fn, constraint, step_size, eps=1e-8, max_iter=100):
    return as_top_level_api(
        logdensity_fn,
        constraint,
        step_size,
        drift_fn=jax.grad(logdensity_fn),
        eps=eps,
        max_iter=max_iter,
    )


def smhr_as_top_level_api(
    logdensity_fn, constraint, step_size, mass_matrix_fn, eps=1e-8, max_iter=100
):
    return as_top_level_api(
        logdensity_fn,
        constraint,
        step_size,
        mass_matrix_fn=mass_matrix_fn,
        eps=eps,
        max_iter=max_iter,
    )


def smlhr_as_top_level_api(
    logdensity_fn, constraint, step_size, mass_matrix_fn, eps=1e-8, max_iter=100
):
    return as_top_level_api(
        logdensity_fn,
        constraint,
        step_size,
        drift_fn=jax.grad(logdensity_fn),
        mass_matrix_fn=mass_matrix_fn,
        eps=eps,
        max_iter=max_iter,
    )
