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
import functools
import itertools
from typing import Callable, NamedTuple

import jax

from blackjax.mcmc.metrics import EuclideanKineticEnergy
from blackjax.types import ArrayTree

__all__ = ["mclachlan", "velocity_verlet", "yoshida"]


class IntegratorState(NamedTuple):
    """State of the trajectory integration.

    We keep the gradient of the logdensity function (negative potential energy)
    to speedup computations.
    """

    position: ArrayTree
    momentum: ArrayTree
    logdensity: float
    logdensity_grad: ArrayTree


Integrator = Callable[[IntegratorState, float], IntegratorState]


def new_integrator_state(logdensity_fn, position, momentum):
    logdensity, logdensity_grad = jax.value_and_grad(logdensity_fn)(position)
    return IntegratorState(position, momentum, logdensity, logdensity_grad)


def velocity_verlet(
    logdensity_fn: Callable,
    kinetic_energy_fn: EuclideanKineticEnergy,
) -> Integrator:
    """The velocity Verlet (or Verlet-Störmer) integrator.

    The velocity Verlet is a two-stage palindromic integrator :cite:p:`bou2018geometric` of the form
    (a1, b1, a2, b1, a1) with a1 = 0. It is numerically stable for values of
    the step size that range between 0 and 2 (when the mass matrix is the
    identity).

    While the position (a1 = 0.5) and velocity Verlet are the most commonly used
    in samplers, it is known in the numerical computation literature that the value
    $a1 \approx 0.1932$ leads to a lower integration error :cite:p:`mclachlan1995numerical,schlick2010molecular`. The authors of :cite:p:`bou2018geometric`
    show that the value $a1 \approx 0.21132$ leads to an even higher step acceptance
    rate, up to 3 times higher than with the standard position verlet (p.22, Fig.4).

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
    kinetic_energy_fn: Callable,
) -> Integrator:
    """Two-stage palindromic symplectic integrator derived in :cite:p:`blanes2014numerical`.

    The integrator is of the form (b1, a1, b2, a1, b1). The choice of the parameters
    determine both the bound on the integration error and the stability of the
    method with respect to the value of `step_size`. The values used here are
    the ones derived in :cite:p:`mclachlan1995numerical`; note that :cite:p:`blanes2014numerical` is more focused on stability
    and derives different values.

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
    kinetic_energy_fn: Callable,
) -> Integrator:
    """Three stages palindromic symplectic integrator derived in :cite:p:`mclachlan1995numerical`

    The integrator is of the form (b1, a1, b2, a2, b2, a1, b1). The choice of
    the parameters determine both the bound on the integration error and the
    stability of the method with respect to the value of `step_size`. The
    values used here are the ones derived in :cite:p:`mclachlan1995numerical` which guarantees a stability
    interval length approximately equal to 4.67.

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


def palindromic_sequence(s):
    # symetrize
    s = s[:-1] + s[::-1]
    # zip with alternating operators
    return lambda O1, O2 : list(zip(itertools.cycle([O1, O2]), s))

def make_integrator(O1, O2, order):

    sequence = order(O1,O2)
    
    def step(state: IntegratorState, step_size):
        """Integrator from https://arxiv.org/pdf/hep-lat/0505020.pdf, see Equation 20."""

        xx, uu, ll, gg = state
        total_r = 0

        for O,factor in sequence:
            xx, uu, ll, gg, r = jax.tree_util.tree_map(O(step_size*factor), xx, uu, ll, gg)
            total_r += r

        kinetic_change = jax.numpy.sum(total_r)
        return IntegratorState(xx, uu, ll, gg), kinetic_change

    return step

def update_position_mclmc(grad_logp):
    """The position updating map of the esh dynamics (see https://arxiv.org/pdf/2111.02434.pdf)"""

    def update(step_size, x, u, l, g):
        u *= 0.5
        xx = x + step_size * u
        ll, gg = grad_logp(xx)
        u *= 2.
        return xx, u, ll, gg, 0

    return lambda O : functools.partial(update,O)


def update_momentum_mclmc(step_size):
    """The momentum updating map of the esh dynamics (see https://arxiv.org/pdf/2111.02434.pdf)
    similar to the implementation: https://github.com/gregversteeg/esh_dynamics
    There are no exponentials e^delta, which prevents overflows when the gradient norm is large.
    """
    
    def update(x, u, l, g):
        g_norm = jax.numpy.sqrt(jax.numpy.sum(jax.numpy.square(g)))
        e = g / g_norm
        ue = jax.numpy.dot(u, e)
        # jax.debug.print("🤯 {x} inside momentum update 🤯", x=(ue))
        dim = u.shape[0]
        delta = step_size * g_norm / (dim - 1)
        zeta = jax.numpy.exp(-delta)
        uu = e * (1 - zeta) * (1 + zeta + ue * (1 - zeta)) + 2 * zeta * u
        delta_r = delta - jax.numpy.log(2) + jax.numpy.log(1 + ue + (1 - ue) * zeta**2)
        return x, uu / jax.numpy.sqrt(jax.numpy.sum(jax.numpy.square(uu))), l, g, delta_r
    return update

minimal_norm = lambda O1, O2: make_integrator(O1, O2, minimal_norm_sequence)

minimal_norm_sequence = palindromic_sequence([0.1931833275037836, 1., 1.- 2.*0.1931833275037836])
leapfrog_sequence = palindromic_sequence([0.5, 1.])
yoshida_sequence = palindromic_sequence([0.11888010966548, 0.29619504261126, 0.5 - 0.11888010966548, 1 - 2 * 0.29619504261126])


# def minimal_norm(O1, O2):
#     lambda_c = 0.1931833275037836  # critical value of the lambda parameter for the minimal norm integrator

#     def step(state: IntegratorState, step_size):
#         """Integrator from https://arxiv.org/pdf/hep-lat/0505020.pdf, see Equation 20."""

#         # V T V T V
#         # jax.debug.print("🤯 {x} inside integrator 1 🤯", x=(state.momentum, state.logdensity_grad))
#         uu, r1 = jax.tree_util.tree_map(
#             lambda x, u, g: O1(step_size * lambda_c)(x, u, g),
#             state.position,
#             state.momentum,
#             state.logdensity_grad,
#         )
#         # jax.debug.print("🤯 {x} inside integrator 2 🤯", x=(uu))

#         xx, _, ll, gg = jax.tree_util.tree_map(
#             lambda x, u: O2(step_size, x, u), state.position, uu*0.5
#         )
#         uu, r2 = jax.tree_util.tree_map(
#             lambda x, u, g: O1(step_size * (1 - 2 * lambda_c))(x, u, g), xx, uu, gg
#         )
#         xx, _, ll, gg = jax.tree_util.tree_map(
#             lambda x, u: O2(step_size, x, u), xx, uu*0.5
#         )
#         uu, r3 = jax.tree_util.tree_map(
#             lambda x, u, g: O1(step_size * lambda_c)(x, u, g), xx, uu, gg
#         )

#         # kinetic energy change
#         kinetic_change = (r1 + r2 + r3) * (uu.shape[0] - 1)

#         return xx, uu, ll, gg, kinetic_change

#     return step


# def update_position_mclmc(grad_logp):
#     """The position updating map of the esh dynamics (see https://arxiv.org/pdf/2111.02434.pdf)"""

#     def update(step_size, x, u):
#         xx = x + step_size * u
#         ll, gg = grad_logp(xx)
#         return xx, u, ll, gg

#     return update


# def update_momentum_mclmc(step_size):
#     """The momentum updating map of the esh dynamics (see https://arxiv.org/pdf/2111.02434.pdf)
#     similar to the implementation: https://github.com/gregversteeg/esh_dynamics
#     There are no exponentials e^delta, which prevents overflows when the gradient norm is large.
#     """
    
#     def update(x, u, g):
#         g_norm = jax.numpy.sqrt(jax.numpy.sum(jax.numpy.square(g)))
#         e = g / g_norm
#         ue = jax.numpy.dot(u, e)
#         # jax.debug.print("🤯 {x} inside momentum update 🤯", x=(ue))
#         dim = u.shape[0]
#         delta = step_size * g_norm / (dim - 1)
#         zeta = jax.numpy.exp(-delta)
#         uu = e * (1 - zeta) * (1 + zeta + ue * (1 - zeta)) + 2 * zeta * u
#         delta_r = delta - jax.numpy.log(2) + jax.numpy.log(1 + ue + (1 - ue) * zeta**2)
#         return uu / jax.numpy.sqrt(jax.numpy.sum(jax.numpy.square(uu))), delta_r
#     return update