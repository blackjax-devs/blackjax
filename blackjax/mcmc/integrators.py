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
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

from blackjax.mcmc.metrics import EuclideanKineticEnergy
from blackjax.types import ArrayTree

__all__ = [
    "mclachlan",
    "velocity_verlet",
    "yoshida",
    "noneuclidean_leapfrog",
    "noneuclidean_mclachlan",
    "noneuclidean_yoshida",
]


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
GeneralIntegrator = Callable[
    [IntegratorState, float], tuple[IntegratorState, ArrayTree]
]


def generalized_two_stage_integrator(
    operator1: Callable,
    operator2: Callable,
    coefficients: list[float],
    format_output_fn: Callable = lambda x: x,
):
    """Generalized numerical integrator for solving ODEs.

    The generalized integrator performs numerical integration of a ODE system by
    alernating between stage 1 and stage 2 updates.
    The update scheme is decided by the coefficients, The scheme should be palindromic,
    i.e. the coefficients of the update scheme should be symmetric with respect to the
    middle of the scheme.

    For instance, for *any* differential equation of the form:

    .. math:: \\frac{d}{dt}f = (O_1+O_2)f

    The leapfrog operator can be seen as approximating :math:`e^{\\epsilon(O_1 + O_2)}`
    by :math:`e^{\\epsilon O_1/2}e^{\\epsilon O_2}e^{\\epsilon O_1/2}`.

    In a standard Hamiltonian, the forms of :math:`e^{\\epsilon O_2}` and
    :math:`e^{\\epsilon O_1}` are simple, but for other differential equations,
    they may be more complex.

    Parameters
    ----------
    operator1
        Stage 1 operator, a function that updates the momentum.
    operator2
        Stage 2 operator, a function that updates the position.
    coefficients
        Coefficients of the integrator.
    format_output_fn
        Function that formats the output of the integrator.

    Returns
    -------
    integrator
        Integrator function.
    """

    def one_step(state: IntegratorState, step_size: float):
        position, momentum, _, logdensity_grad = state
        # auxiliary infomation generated during integration for diagnostics. It is
        # updated by the operator1 and operator2 at each call.
        momentum_update_info = None
        position_update_info = None
        for i, coef in enumerate(coefficients[:-1]):
            if i % 2 == 0:
                momentum, kinetic_grad, momentum_update_info = operator1(
                    momentum,
                    logdensity_grad,
                    step_size,
                    coef,
                    momentum_update_info,
                    is_last_call=False,
                )
            else:
                (
                    position,
                    logdensity,
                    logdensity_grad,
                    position_update_info,
                ) = operator2(
                    position,
                    kinetic_grad,
                    step_size,
                    coef,
                    position_update_info,
                )
        # Separate the last steps to short circuit the computation of the kinetic_grad.
        momentum, kinetic_grad, momentum_update_info = operator1(
            momentum,
            logdensity_grad,
            step_size,
            coefficients[-1],
            momentum_update_info,
            is_last_call=True,
        )
        return format_output_fn(
            position,
            momentum,
            logdensity,
            logdensity_grad,
            kinetic_grad,
            position_update_info,
            momentum_update_info,
        )

    return one_step


def new_integrator_state(logdensity_fn, position, momentum):
    logdensity, logdensity_grad = jax.value_and_grad(logdensity_fn)(position)
    return IntegratorState(position, momentum, logdensity, logdensity_grad)


def euclidean_position_update_fn(logdensity_fn: Callable):
    logdensity_and_grad_fn = jax.value_and_grad(logdensity_fn)

    def update(
        position: ArrayTree,
        kinetic_grad: ArrayTree,
        step_size: float,
        coef: float,
        auxiliary_info=None,
    ):
        del auxiliary_info
        new_position = jax.tree_util.tree_map(
            lambda x, grad: x + step_size * coef * grad,
            position,
            kinetic_grad,
        )
        logdensity, logdensity_grad = logdensity_and_grad_fn(new_position)
        return new_position, logdensity, logdensity_grad, None

    return update


def euclidean_momentum_update_fn(kinetic_energy_fn: EuclideanKineticEnergy):
    kinetic_energy_grad_fn = jax.grad(kinetic_energy_fn)

    def update(
        momentum: ArrayTree,
        logdensity_grad: ArrayTree,
        step_size: float,
        coef: float,
        auxiliary_info=None,
        is_last_call=False,
    ):
        del auxiliary_info
        new_momentum = jax.tree_util.tree_map(
            lambda x, grad: x + step_size * coef * grad,
            momentum,
            logdensity_grad,
        )
        if is_last_call:
            return new_momentum, None, None
        kinetic_grad = kinetic_energy_grad_fn(new_momentum)
        return new_momentum, kinetic_grad, None

    return update


def format_euclidean_state_output(
    position,
    momentum,
    logdensity,
    logdensity_grad,
    kinetic_grad,
    position_update_info,
    momentum_update_info,
):
    del kinetic_grad, position_update_info, momentum_update_info
    return IntegratorState(position, momentum, logdensity, logdensity_grad)


def generate_euclidean_integrator(cofficients):
    """Generate symplectic integrator for solving a Hamiltonian system.

    The resulting integrator is volume-preserve and preserves the symplectic structure
    of phase space.
    """

    def euclidean_integrator(
        logdensity_fn: Callable, kinetic_energy_fn: EuclideanKineticEnergy
    ) -> Integrator:
        position_update_fn = euclidean_position_update_fn(logdensity_fn)
        momentum_update_fn = euclidean_momentum_update_fn(kinetic_energy_fn)
        one_step = generalized_two_stage_integrator(
            momentum_update_fn,
            position_update_fn,
            cofficients,
            format_output_fn=format_euclidean_state_output,
        )
        return one_step

    return euclidean_integrator


"""
The velocity Verlet (or Verlet-Störmer) integrator.

The velocity Verlet is a two-stage palindromic integrator :cite:p:`bou2018geometric`
of the form (a1, b1, a2, b1, a1) with a1 = 0. It is numerically stable for values of
the step size that range between 0 and 2 (when the mass matrix is the identity).

While the position (a1 = 0.5) and velocity Verlet are the most commonly used
in samplers, it is known in the numerical computation literature that the value
$a1 \approx 0.1932$ leads to a lower integration error :cite:p:`mclachlan1995numerical,schlick2010molecular`.
The authors of :cite:p:`bou2018geometric` show that the value $a1 \approx 0.21132$
leads to an even higher step acceptance rate, up to 3 times higher
than with the standard position verlet (p.22, Fig.4).

By choosing the velocity verlet we avoid two computations of the gradient
of the kinetic energy. We are trading accuracy in exchange, and it is not
clear whether this is the right tradeoff.
"""
velocity_verlet_cofficients = [0.5, 1.0, 0.5]
velocity_verlet = generate_euclidean_integrator(velocity_verlet_cofficients)

"""
Two-stage palindromic symplectic integrator derived in :cite:p:`blanes2014numerical`.

The integrator is of the form (b1, a1, b2, a1, b1). The choice of the parameters
determine both the bound on the integration error and the stability of the
method with respect to the value of `step_size`. The values used here are
the ones derived in :cite:p:`mclachlan1995numerical`; note that :cite:p:`blanes2014numerical`
is more focused on stability and derives different values.

Also known as the minimal norm integrator.
"""
b1 = 0.1931833275037836
a1 = 0.5
b2 = 1 - 2 * b1
mclachlan_cofficients = [b1, a1, b2, a1, b1]
mclachlan = generate_euclidean_integrator(mclachlan_cofficients)

"""
Three stages palindromic symplectic integrator derived in :cite:p:`mclachlan1995numerical`

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
yoshida_cofficients = [b1, a1, b2, a2, b2, a1, b1]
yoshida = generate_euclidean_integrator(yoshida_cofficients)


# Intergrators with non Euclidean updates
def _normalized_flatten_array(x, tol=1e-13):
    norm = jnp.linalg.norm(x)
    return jnp.where(norm > tol, x / norm, x), norm


def esh_dynamics_momentum_update_one_step(
    momentum: ArrayTree,
    logdensity_grad: ArrayTree,
    step_size: float,
    coef: float,
    previous_kinetic_energy_change=None,
    is_last_call=False,
):
    """Momentum update based on Esh dynamics.

    The momentum updating map of the esh dynamics as derived in :cite:p:`steeg2021hamiltonian`
    There are no exponentials e^delta, which prevents overflows when the gradient norm
    is large.
    """

    flatten_grads, unravel_fn = ravel_pytree(logdensity_grad)
    flatten_momentum, _ = ravel_pytree(momentum)
    dims = flatten_momentum.shape[0]
    normalized_gradient, gradient_norm = _normalized_flatten_array(flatten_grads)
    momentum_proj = jnp.dot(flatten_momentum, normalized_gradient)
    delta = step_size * coef * gradient_norm / (dims - 1)
    zeta = jnp.exp(-delta)
    new_momentum_raw = (
        normalized_gradient * (1 - zeta) * (1 + zeta + momentum_proj * (1 - zeta))
        + 2 * zeta * flatten_momentum
    )
    new_momentum_normalized, _ = _normalized_flatten_array(new_momentum_raw)
    next_momentum = unravel_fn(new_momentum_normalized)
    kinetic_energy_change = (
        delta
        - jnp.log(2)
        + jnp.log(1 + momentum_proj + (1 - momentum_proj) * zeta**2)
    )
    if previous_kinetic_energy_change is not None:
        kinetic_energy_change += previous_kinetic_energy_change
    if is_last_call:
        kinetic_energy_change *= dims - 1
    return next_momentum, next_momentum, kinetic_energy_change


def format_noneuclidean_state_output(
    position,
    momentum,
    logdensity,
    logdensity_grad,
    kinetic_grad,
    position_update_info,
    momentum_update_info,
):
    del kinetic_grad, position_update_info
    return (
        IntegratorState(position, momentum, logdensity, logdensity_grad),
        momentum_update_info,
    )


def generate_noneuclidean_integrator(cofficients):
    def noneuclidean_integrator(
        logdensity_fn: Callable, *args, **kwargs
    ) -> GeneralIntegrator:
        position_update_fn = euclidean_position_update_fn(logdensity_fn)
        one_step = generalized_two_stage_integrator(
            esh_dynamics_momentum_update_one_step,
            position_update_fn,
            cofficients,
            format_output_fn=format_noneuclidean_state_output,
        )
        return one_step

    return noneuclidean_integrator


noneuclidean_leapfrog = generate_noneuclidean_integrator(velocity_verlet_cofficients)
noneuclidean_mclachlan = generate_noneuclidean_integrator(mclachlan_cofficients)
noneuclidean_yoshida = generate_noneuclidean_integrator(yoshida_cofficients)
