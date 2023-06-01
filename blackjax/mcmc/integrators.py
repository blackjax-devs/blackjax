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
"""Symplectic, time-reversible, integrators for Hamiltonian trajectories."""
from typing import Any, Callable, NamedTuple, Tuple

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

from blackjax.mcmc.metrics import KineticEnergy
from blackjax.types import PyTree

__all__ = ["mclachlan", "velocity_verlet", "yoshida", "implicit_midpoint"]


class IntegratorState(NamedTuple):
    """State of the trajectory integration.

    We keep the gradient of the logdensity function (negative potential energy)
    to speedup computations.
    """

    position: PyTree
    momentum: PyTree
    logdensity: float
    logdensity_grad: PyTree


Integrator = Callable[[IntegratorState, float], IntegratorState]


def new_integrator_state(logdensity_fn, position, momentum):
    logdensity, logdensity_grad = jax.value_and_grad(logdensity_fn)(position)
    return IntegratorState(position, momentum, logdensity, logdensity_grad)


def velocity_verlet(
    logdensity_fn: Callable,
    kinetic_energy_fn: KineticEnergy,
) -> Integrator:
    r"""The velocity Verlet (or Verlet-Störmer) integrator.

    The velocity Verlet is a two-stage palindromic integrator :cite:p:`bou2018geometric`
    of the form (a1, b1, a2, b1, a1) with a1 = 0. It is numerically stable for values of
    the step size that range between 0 and 2 (when the mass matrix is the
    identity).

    While the position (a1 = 0.5) and velocity Verlet are the most commonly used
    in samplers, it is known in the numerical computation literature that the value
    $a1 \approx 0.1932$ leads to a lower integration error
    :cite:p:`mclachlan1995numerical,schlick2010molecular`. The authors of
    :cite:p:`bou2018geometric` show that the value $a1 \approx 0.21132$ leads to an
    even higher step acceptance rate, up to 3 times higher than with the standard
    position verlet (p.22, Fig.4).

    By choosing the velocity verlet we avoid two computations of the gradient
    of the kinetic energy. We are trading accuracy in exchange, and it is not
    clear whether this is the right tradeoff.

    """
    a1 = 0
    b1 = 0.5
    a2 = 1 - 2 * a1

    logdensity_and_grad_fn = jax.value_and_grad(logdensity_fn)
    kinetic_energy_grad_fn = jax.grad(kinetic_energy_fn)

    def one_step(state: IntegratorState, step_size: float) -> IntegratorState:
        position, momentum, _, logdensity_grad = state

        momentum = jax.tree_util.tree_map(
            lambda momentum, logdensity_grad: momentum
            + b1 * step_size * logdensity_grad,
            momentum,
            logdensity_grad,
        )

        kinetic_grad = kinetic_energy_grad_fn(momentum)
        position = jax.tree_util.tree_map(
            lambda position, kinetic_grad: position + a2 * step_size * kinetic_grad,
            position,
            kinetic_grad,
        )

        logdensity, logdensity_grad = logdensity_and_grad_fn(position)
        momentum = jax.tree_util.tree_map(
            lambda momentum, logdensity_grad: momentum
            + b1 * step_size * logdensity_grad,
            momentum,
            logdensity_grad,
        )

        return IntegratorState(position, momentum, logdensity, logdensity_grad)

    return one_step


def mclachlan(
    logdensity_fn: Callable,
    kinetic_energy_fn: KineticEnergy,
) -> Integrator:
    """Two-stage palindromic symplectic integrator derived in
    :cite:p:`blanes2014numerical`.

    The integrator is of the form (b1, a1, b2, a1, b1). The choice of the parameters
    determine both the bound on the integration error and the stability of the
    method with respect to the value of `step_size`. The values used here are
    the ones derived in :cite:p:`mclachlan1995numerical`; note that
    :cite:p:`blanes2014numerical` is more focused on stability and derives different
    values.

    """
    b1 = 0.1932
    a1 = 0.5
    b2 = 1 - 2 * b1

    logdensity_and_grad_fn = jax.value_and_grad(logdensity_fn)
    kinetic_energy_grad_fn = jax.grad(kinetic_energy_fn)

    def one_step(state: IntegratorState, step_size: float) -> IntegratorState:
        position, momentum, _, logdensity_grad = state

        momentum = jax.tree_util.tree_map(
            lambda momentum, logdensity_grad: momentum
            + b1 * step_size * logdensity_grad,
            momentum,
            logdensity_grad,
        )

        kinetic_grad = kinetic_energy_grad_fn(momentum)
        position = jax.tree_util.tree_map(
            lambda position, kinetic_grad: position + a1 * step_size * kinetic_grad,
            position,
            kinetic_grad,
        )

        _, logdensity_grad = logdensity_and_grad_fn(position)
        momentum = jax.tree_util.tree_map(
            lambda momentum, logdensity_grad: momentum
            + b2 * step_size * logdensity_grad,
            momentum,
            logdensity_grad,
        )

        kinetic_grad = kinetic_energy_grad_fn(momentum)
        position = jax.tree_util.tree_map(
            lambda position, kinetic_grad: position + a1 * step_size * kinetic_grad,
            position,
            kinetic_grad,
        )

        logdensity, logdensity_grad = logdensity_and_grad_fn(position)
        momentum = jax.tree_util.tree_map(
            lambda momentum, logdensity_grad: momentum
            + b1 * step_size * logdensity_grad,
            momentum,
            logdensity_grad,
        )

        return IntegratorState(position, momentum, logdensity, logdensity_grad)

    return one_step


def yoshida(
    logdensity_fn: Callable,
    kinetic_energy_fn: KineticEnergy,
) -> Integrator:
    """Three stages palindromic symplectic integrator derived in
    :cite:p:`mclachlan1995numerical`

    The integrator is of the form (b1, a1, b2, a2, b2, a1, b1). The choice of
    the parameters determine both the bound on the integration error and the
    stability of the method with respect to the value of `step_size`. The
    values used here are the ones derived in :cite:p:`mclachlan1995numerical` which
    guarantees a stability interval length approximately equal to 4.67.

    """
    b1 = 0.11888010966548
    a1 = 0.29619504261126
    b2 = 0.5 - b1
    a2 = 1 - 2 * a1

    logdensity_and_grad_fn = jax.value_and_grad(logdensity_fn)
    kinetic_energy_grad_fn = jax.grad(kinetic_energy_fn)

    def one_step(state: IntegratorState, step_size: float) -> IntegratorState:
        position, momentum, _, logdensity_grad = state

        momentum = jax.tree_util.tree_map(
            lambda momentum, logdensity_grad: momentum
            + b1 * step_size * logdensity_grad,
            momentum,
            logdensity_grad,
        )

        kinetic_grad = kinetic_energy_grad_fn(momentum)
        position = jax.tree_util.tree_map(
            lambda position, kinetic_grad: position + a1 * step_size * kinetic_grad,
            position,
            kinetic_grad,
        )

        _, logdensity_grad = logdensity_and_grad_fn(position)
        momentum = jax.tree_util.tree_map(
            lambda momentum, logdensity_grad: momentum
            + b2 * step_size * logdensity_grad,
            momentum,
            logdensity_grad,
        )

        kinetic_grad = kinetic_energy_grad_fn(momentum)
        position = jax.tree_util.tree_map(
            lambda position, kinetic_grad: position + a2 * step_size * kinetic_grad,
            position,
            kinetic_grad,
        )

        _, logdensity_grad = logdensity_and_grad_fn(position)
        momentum = jax.tree_util.tree_map(
            lambda momentum, logdensity_grad: momentum
            + b2 * step_size * logdensity_grad,
            momentum,
            logdensity_grad,
        )

        kinetic_grad = kinetic_energy_grad_fn(momentum)
        position = jax.tree_util.tree_map(
            lambda position, kinetic_grad: position + a1 * step_size * kinetic_grad,
            position,
            kinetic_grad,
        )

        logdensity, logdensity_grad = logdensity_and_grad_fn(position)
        momentum = jax.tree_util.tree_map(
            lambda momentum, logdensity_grad: momentum
            + b1 * step_size * logdensity_grad,
            momentum,
            logdensity_grad,
        )

        return IntegratorState(position, momentum, logdensity, logdensity_grad)

    return one_step


FixedPointSolver = Callable[
    [Callable[[PyTree], Tuple[PyTree, PyTree]], PyTree], Tuple[PyTree, PyTree, Any]
]


class FixedPointIterationInfo(NamedTuple):
    success: bool
    norm: float
    iters: int


def solve_fixed_point_iteration(
    func: Callable[[PyTree], Tuple[PyTree, PyTree]],
    x0: PyTree,
    *,
    convergence_tol: float = 1e-6,
    divergence_tol: float = 1e10,
    max_iters: int = 100,
    norm_fn: Callable[[PyTree], float] = lambda x: jnp.max(jnp.abs(x)),
) -> Tuple[PyTree, PyTree, FixedPointIterationInfo]:
    """Solve for x = func(x) using a fixed point iteration"""

    def compute_norm(x: PyTree, xp: PyTree) -> float:
        return norm_fn(ravel_pytree(jax.tree_util.tree_map(jnp.subtract, x, xp))[0])

    def cond_fn(args: Tuple[int, PyTree, PyTree, float]) -> bool:
        n, _, _, norm = args
        return (
            (n < max_iters)
            & jnp.isfinite(norm)
            & (norm < divergence_tol)
            & (norm > convergence_tol)
        )

    def body_fn(
        args: Tuple[int, PyTree, PyTree, float]
    ) -> Tuple[int, PyTree, PyTree, float]:
        n, x, _, _ = args
        xn, aux = func(x)
        norm = compute_norm(xn, x)
        return n + 1, xn, aux, norm

    x, aux = func(x0)
    iters, x, aux, norm = jax.lax.while_loop(
        cond_fn, body_fn, (0, x, aux, compute_norm(x, x0))
    )
    success = jnp.isfinite(norm) & (norm <= convergence_tol)
    return x, aux, FixedPointIterationInfo(success, norm, iters)


def implicit_midpoint(
    logdensity_fn: Callable,
    kinetic_energy_fn: KineticEnergy,
    *,
    solver: FixedPointSolver = solve_fixed_point_iteration,
    **solver_kwargs: Any,
) -> Integrator:
    """The implicit midpoint integrator with support for non-stationary kinetic energy

    This is an integrator based on :cite:t:`brofos2021evaluating`, which provides
    support for kinetic energies that depend on position. This integrator requires that
    the kinetic energy function takes two arguments: position and momentum.

    The ``solver`` parameter allows overloading of the fixed point solver. By default, a
    simple fixed point iteration is used, but more advanced solvers could be implemented
    in the future.
    """
    logdensity_and_grad_fn = jax.value_and_grad(logdensity_fn)
    kinetic_energy_grad_fn = jax.grad(
        lambda q, p: kinetic_energy_fn(p, position=q), argnums=(0, 1)
    )

    def one_step(state: IntegratorState, step_size: float) -> IntegratorState:
        position, momentum, _, _ = state

        def _update(
            q: PyTree,
            p: PyTree,
            dUdq: PyTree,
            initial: Tuple[PyTree, PyTree] = (position, momentum),
        ) -> Tuple[PyTree, PyTree]:
            dTdq, dHdp = kinetic_energy_grad_fn(q, p)
            dHdq = jax.tree_util.tree_map(jnp.subtract, dTdq, dUdq)

            # Take a step from the _initial coordinates_ using the gradients of the
            # Hamiltonian evaluated at the current guess for the midpoint
            q = jax.tree_util.tree_map(
                lambda q_, d_: q_ + 0.5 * step_size * d_, initial[0], dHdp
            )
            p = jax.tree_util.tree_map(
                lambda p_, d_: p_ - 0.5 * step_size * d_, initial[1], dHdq
            )
            return q, p

        # Solve for the midpoint numerically
        def _step(args: PyTree) -> Tuple[PyTree, PyTree]:
            q, p = args
            _, dLdq = logdensity_and_grad_fn(q)
            return _update(q, p, dLdq), dLdq

        (q, p), dLdq, info = solver(_step, (position, momentum), **solver_kwargs)
        del info  # TODO: Track the returned info

        # Take an explicit update as recommended by Brofos & Lederman
        _, dLdq = logdensity_and_grad_fn(q)
        q, p = _update(q, p, dLdq, initial=(q, p))

        return IntegratorState(q, p, *logdensity_and_grad_fn(q))

    return one_step
