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
from typing import Callable, NamedTuple

import jax

from blackjax.mcmc.metrics import EuclideanKineticEnergy
from blackjax.types import PyTree

__all__ = ["mclachlan", "velocity_verlet", "yoshida"]


class IntegratorState(NamedTuple):
    """State of the trajectory integration.

    We keep the gradient of the potential energy to speedup computations.
    """

    position: PyTree
    momentum: PyTree
    potential_energy: float
    potential_energy_grad: PyTree


EuclideanIntegrator = Callable[[IntegratorState, float], IntegratorState]


def new_integrator_state(potential_fn, position, momentum):
    potential_energy, potential_energy_grad = jax.value_and_grad(potential_fn)(position)
    return IntegratorState(position, momentum, potential_energy, potential_energy_grad)


def velocity_verlet(
    potential_fn: Callable,
    kinetic_energy_fn: EuclideanKineticEnergy,
) -> EuclideanIntegrator:
    """The velocity Verlet (or Verlet-Störmer) integrator.

    The velocity Verlet is a two-stage palindromic integrator [1]_ of the form
    (a1, b1, a2, b1, a1) with a1 = 0. It is numerically stable for values of
    the step size that range between 0 and 2 (when the mass matrix is the
    identity).

    While the position (a1 = 0.5) and velocity Verlet are the most commonly used
    in samplers, it is known in the numerical computation literature that the value
    $a1 \approx 0.1932$ leads to a lower integration error [2,3]_. The authors of [1]_
    show that the value $a1 \approx 0.21132$ leads to an even higher step acceptance
    rate, up to 3 times higher than with the standard position verlet (p.22, Fig.4).

    By choosing the velocity verlet we avoid two computations of the gradient
    of the kinetic energy. We are trading accuracy in exchange, and it is not
    clear whether this is the right tradeoff.


    References
    ----------
    .. [1]: Bou-Rabee, Nawaf, and Jesús Marıa Sanz-Serna. "Geometric
            integrators and the Hamiltonian Monte Carlo method." Acta Numerica 27
            (2018): 113-206.
    .. [2]: McLachlan, Robert I. "On the numerical integration of ordinary
            differential equations by symmetric composition methods." SIAM Journal on
            Scientific Computing 16.1 (1995): 151-168.
    .. [3]: Schlick, Tamar. Molecular modeling and simulation: an
            interdisciplinary guide: Vol. 21. Springer
            Science & Business Media, 2010.

    """
    a1 = 0
    b1 = 0.5
    a2 = 1 - 2 * a1

    potential_and_grad_fn = jax.value_and_grad(potential_fn)
    kinetic_energy_grad_fn = jax.grad(kinetic_energy_fn)

    def one_step(state: IntegratorState, step_size: float) -> IntegratorState:
        position, momentum, _, potential_energy_grad = state

        momentum = jax.tree_util.tree_map(
            lambda momentum, potential_grad: momentum - b1 * step_size * potential_grad,
            momentum,
            potential_energy_grad,
        )

        kinetic_grad = kinetic_energy_grad_fn(momentum)
        position = jax.tree_util.tree_map(
            lambda position, kinetic_grad: position + a2 * step_size * kinetic_grad,
            position,
            kinetic_grad,
        )

        potential_energy, potential_energy_grad = potential_and_grad_fn(position)
        momentum = jax.tree_util.tree_map(
            lambda momentum, potential_grad: momentum - b1 * step_size * potential_grad,
            momentum,
            potential_energy_grad,
        )

        return IntegratorState(
            position, momentum, potential_energy, potential_energy_grad
        )

    return one_step


def mclachlan(
    potential_fn: Callable,
    kinetic_energy_fn: Callable,
) -> EuclideanIntegrator:
    """Two-stage palindromic symplectic integrator derived in [1]_

    The integrator is of the form (b1, a1, b2, a1, b1). The choice of the parameters
    determine both the bound on the integration error and the stability of the
    method with respect to the value of `step_size`. The values used here are
    the ones derived in [2]_; note that [1]_ is more focused on stability
    and derives different values.

    References
    ----------
    .. [1]: Blanes, Sergio, Fernando Casas, and J. M. Sanz-Serna. "Numerical
            integrators for the Hybrid Monte Carlo method." SIAM Journal on Scientific
            Computing 36.4 (2014): A1556-A1580.
    .. [2]: McLachlan, Robert I. "On the numerical integration of ordinary
            differential equations by symmetric composition methods." SIAM Journal on
            Scientific Computing 16.1 (1995): 151-168.

    """
    b1 = 0.1932
    a1 = 0.5
    b2 = 1 - 2 * b1

    potential_and_grad_fn = jax.value_and_grad(potential_fn)
    kinetic_energy_grad_fn = jax.grad(kinetic_energy_fn)

    def one_step(state: IntegratorState, step_size: float) -> IntegratorState:
        position, momentum, _, potential_energy_grad = state

        momentum = jax.tree_util.tree_map(
            lambda momentum, potential_grad: momentum - b1 * step_size * potential_grad,
            momentum,
            potential_energy_grad,
        )

        kinetic_grad = kinetic_energy_grad_fn(momentum)
        position = jax.tree_util.tree_map(
            lambda position, kinetic_grad: position + a1 * step_size * kinetic_grad,
            position,
            kinetic_grad,
        )

        _, potential_energy_grad = potential_and_grad_fn(position)
        momentum = jax.tree_util.tree_map(
            lambda momentum, potential_grad: momentum - b2 * step_size * potential_grad,
            momentum,
            potential_energy_grad,
        )

        kinetic_grad = kinetic_energy_grad_fn(momentum)
        position = jax.tree_util.tree_map(
            lambda position, kinetic_grad: position + a1 * step_size * kinetic_grad,
            position,
            kinetic_grad,
        )

        potential_energy, potential_energy_grad = potential_and_grad_fn(position)
        momentum = jax.tree_util.tree_map(
            lambda momentum, potential_grad: momentum - b1 * step_size * potential_grad,
            momentum,
            potential_energy_grad,
        )

        return IntegratorState(
            position, momentum, potential_energy, potential_energy_grad
        )

    return one_step


def yoshida(
    potential_fn: Callable,
    kinetic_energy_fn: Callable,
) -> EuclideanIntegrator:
    """Three stages palindromic symplectic integrator derived in [1]_

    The integrator is of the form (b1, a1, b2, a2, b2, a1, b1). The choice of
    the parameters determine both the bound on the integration error and the
    stability of the method with respect to the value of `step_size`. The
    values used here are the ones derived in [1]_ which guarantees a stability
    interval length approximately equal to 4.67.

    References
    ----------
    .. [1]: Blanes, Sergio, Fernando Casas, and J. M. Sanz-Serna. "Numerical
            integrators for the Hybrid Monte Carlo method." SIAM Journal on Scientific
            Computing 36.4 (2014): A1556-A1580.

    """
    b1 = 0.11888010966548
    a1 = 0.29619504261126
    b2 = 0.5 - b1
    a2 = 1 - 2 * a1

    potential_and_grad_fn = jax.value_and_grad(potential_fn)
    kinetic_energy_grad_fn = jax.grad(kinetic_energy_fn)

    def one_step(state: IntegratorState, step_size: float) -> IntegratorState:
        position, momentum, _, potential_energy_grad = state

        momentum = jax.tree_util.tree_map(
            lambda momentum, potential_grad: momentum - b1 * step_size * potential_grad,
            momentum,
            potential_energy_grad,
        )

        kinetic_grad = kinetic_energy_grad_fn(momentum)
        position = jax.tree_util.tree_map(
            lambda position, kinetic_grad: position + a1 * step_size * kinetic_grad,
            position,
            kinetic_grad,
        )

        _, potential_energy_grad = potential_and_grad_fn(position)
        momentum = jax.tree_util.tree_map(
            lambda momentum, potential_grad: momentum - b2 * step_size * potential_grad,
            momentum,
            potential_energy_grad,
        )

        kinetic_grad = kinetic_energy_grad_fn(momentum)
        position = jax.tree_util.tree_map(
            lambda position, kinetic_grad: position + a2 * step_size * kinetic_grad,
            position,
            kinetic_grad,
        )

        _, potential_energy_grad = potential_and_grad_fn(position)
        momentum = jax.tree_util.tree_map(
            lambda momentum, potential_grad: momentum - b2 * step_size * potential_grad,
            momentum,
            potential_energy_grad,
        )

        kinetic_grad = kinetic_energy_grad_fn(momentum)
        position = jax.tree_util.tree_map(
            lambda position, kinetic_grad: position + a1 * step_size * kinetic_grad,
            position,
            kinetic_grad,
        )

        potential_energy, potential_energy_grad = potential_and_grad_fn(position)
        momentum = jax.tree_util.tree_map(
            lambda momentum, potential_grad: momentum - b1 * step_size * potential_grad,
            momentum,
            potential_energy_grad,
        )

        return IntegratorState(
            position, momentum, potential_energy, potential_energy_grad
        )

    return one_step
