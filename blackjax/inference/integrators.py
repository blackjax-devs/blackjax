"""Symplectic, time-reversible, integrators for Hamiltonian trajectories."""
from typing import Callable, Dict, List, NamedTuple, Tuple, Union

import jax

from blackjax.inference.metrics import EuclideanKineticEnergy

__all__ = ["velocity_verlet"]

PyTree = Union[Dict, List, Tuple]


class IntegratorState(NamedTuple):
    """State of the trajectory integration.

    We keep the gradient of the potential energy to speedup computations.
    """

    position: PyTree
    momentum: PyTree
    potential_energy: float
    potential_energy_grad: PyTree


def new_integrator_state(potential_fn, position, momentum):
    potential_energy, potential_energy_grad = jax.value_and_grad(potential_fn)(position)
    return IntegratorState(position, momentum, potential_energy, potential_energy_grad)


def velocity_verlet(
    potential_fn: Callable, kinetic_energy_fn: EuclideanKineticEnergy
) -> Callable:
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

    potential_grad_fn = jax.value_and_grad(potential_fn)
    kinetic_energy_grad_fn = jax.grad(kinetic_energy_fn)

    def one_step(state: IntegratorState, step_size: float) -> IntegratorState:
        position, momentum, _, potential_energy_grad = state

        momentum = jax.tree_util.tree_multimap(
            lambda momentum, potential_grad: momentum - b1 * step_size * potential_grad,
            momentum,
            potential_energy_grad,
        )

        kinetic_grad = kinetic_energy_grad_fn(momentum)
        position = jax.tree_util.tree_multimap(
            lambda position, kinetic_grad: position + a2 * step_size * kinetic_grad,
            position,
            kinetic_grad,
        )

        potential_energy, potential_energy_grad = potential_grad_fn(position)
        momentum = jax.tree_util.tree_multimap(
            lambda momentum, potential_grad: momentum - b1 * step_size * potential_grad,
            momentum,
            potential_energy_grad,
        )

        return IntegratorState(
            position, momentum, potential_energy, potential_energy_grad
        )

    return one_step
