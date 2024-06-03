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
from jax.random import normal

from blackjax.mcmc.metrics import KineticEnergy
from blackjax.types import ArrayTree

__all__ = [
    "mclachlan",
    "omelyan",
    "velocity_verlet",
    "yoshida",
    "with_isokinetic_maruyama",
    "isokinetic_velocity_verlet",
    "isokinetic_mclachlan",
    "isokinetic_omelyan",
    "isokinetic_yoshida",
    "implicit_midpoint",
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

    The velocity_verlet operator can be seen as approximating :math:`e^{\\epsilon(O_1 + O_2)}`
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


def euclidean_momentum_update_fn(kinetic_energy_fn: KineticEnergy):
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


def generate_euclidean_integrator(coefficients):
    """Generate symplectic integrator for solving a Hamiltonian system.

    The resulting integrator is volume-preserve and preserves the symplectic structure
    of phase space.
    """

    def euclidean_integrator(
        logdensity_fn: Callable, kinetic_energy_fn: KineticEnergy
    ) -> Integrator:
        position_update_fn = euclidean_position_update_fn(logdensity_fn)
        momentum_update_fn = euclidean_momentum_update_fn(kinetic_energy_fn)
        one_step = generalized_two_stage_integrator(
            momentum_update_fn,
            position_update_fn,
            coefficients,
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
velocity_verlet_coefficients = [0.5, 1.0, 0.5]
velocity_verlet = generate_euclidean_integrator(velocity_verlet_coefficients)

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
mclachlan_coefficients = [b1, a1, b2, a1, b1]
mclachlan = generate_euclidean_integrator(mclachlan_coefficients)

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
yoshida_coefficients = [b1, a1, b2, a2, b2, a1, b1]
yoshida = generate_euclidean_integrator(yoshida_coefficients)

"""
Eleven-stage palindromic symplectic integrator derived in :cite:p:`omelyan2003symplectic`.

Popular in LQCD, see also :cite:p:`takaishi2006testing`.
"""
b1 = 0.08398315262876693
a1 = 0.2539785108410595
b2 = 0.6822365335719091
a2 = -0.03230286765269967
b3 = 0.5 - b1 - b2
a3 = 1 - 2 * (a1 + a2)
omelyan_coefficients = [b1, a1, b2, a2, b3, a3, b3, a2, b2, a1, b1]
omelyan = generate_euclidean_integrator(omelyan_coefficients)


# Intergrators with non Euclidean updates
def _normalized_flatten_array(x, tol=1e-13):
    norm = jnp.linalg.norm(x)
    return jnp.where(norm > tol, x / norm, x), norm


def esh_dynamics_momentum_update_one_step(sqrt_diag_cov=1.0):
    def update(
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
        del is_last_call

        logdensity_grad = logdensity_grad
        flatten_grads, unravel_fn = ravel_pytree(logdensity_grad)
        flatten_grads = flatten_grads * sqrt_diag_cov
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
        gr = unravel_fn(new_momentum_normalized * sqrt_diag_cov)
        next_momentum = unravel_fn(new_momentum_normalized)
        kinetic_energy_change = (
            delta
            - jnp.log(2)
            + jnp.log(1 + momentum_proj + (1 - momentum_proj) * zeta**2)
        ) * (dims - 1)
        if previous_kinetic_energy_change is not None:
            kinetic_energy_change += previous_kinetic_energy_change
        return next_momentum, gr, kinetic_energy_change

    return update


def format_isokinetic_state_output(
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


def generate_isokinetic_integrator(coefficients):
    def isokinetic_integrator(
        logdensity_fn: Callable, sqrt_diag_cov: ArrayTree = 1.0
    ) -> GeneralIntegrator:
        position_update_fn = euclidean_position_update_fn(logdensity_fn)
        one_step = generalized_two_stage_integrator(
            esh_dynamics_momentum_update_one_step(sqrt_diag_cov),
            position_update_fn,
            coefficients,
            format_output_fn=format_isokinetic_state_output,
        )
        return one_step

    return isokinetic_integrator


isokinetic_velocity_verlet = generate_isokinetic_integrator(
    velocity_verlet_coefficients
)
isokinetic_yoshida = generate_isokinetic_integrator(yoshida_coefficients)
isokinetic_mclachlan = generate_isokinetic_integrator(mclachlan_coefficients)
isokinetic_omelyan = generate_isokinetic_integrator(omelyan_coefficients)


def partially_refresh_momentum(momentum, rng_key, step_size, L):
    """Adds a small noise to momentum and normalizes.

    Parameters
    ----------
    rng_key
        The pseudo-random number generator key used to generate random numbers.
    momentum
        PyTree that the structure the output should to match.
    step_size
        Step size
    L
        controls rate of momentum change

    Returns
    -------
    momentum with random change in angle
    """
    m, unravel_fn = ravel_pytree(momentum)
    dim = m.shape[0]
    nu = jnp.sqrt((jnp.exp(2 * step_size / L) - 1.0) / dim)
    z = nu * normal(rng_key, shape=m.shape, dtype=m.dtype)
    return unravel_fn((m + z) / jnp.linalg.norm(m + z))


def with_isokinetic_maruyama(integrator):
    def stochastic_integrator(init_state, step_size, L_proposal, rng_key):
        key1, key2 = jax.random.split(rng_key)
        # partial refreshment
        state = init_state._replace(
            momentum=partially_refresh_momentum(
                momentum=init_state.momentum,
                rng_key=key1,
                L=L_proposal,
                step_size=step_size * 0.5,
            )
        )
        # one step of the deterministic dynamics
        state, info = integrator(state, step_size)
        # partial refreshment
        state = state._replace(
            momentum=partially_refresh_momentum(
                momentum=state.momentum,
                rng_key=key2,
                L=L_proposal,
                step_size=step_size * 0.5,
            )
        )
        return state, info

    return stochastic_integrator


FixedPointSolver = Callable[
    [Callable[[ArrayTree], Tuple[ArrayTree, ArrayTree]], ArrayTree],
    Tuple[ArrayTree, ArrayTree, Any],
]


class FixedPointIterationInfo(NamedTuple):
    success: bool
    norm: float
    iters: int


def solve_fixed_point_iteration(
    func: Callable[[ArrayTree], Tuple[ArrayTree, ArrayTree]],
    x0: ArrayTree,
    *,
    convergence_tol: float = 1e-6,
    divergence_tol: float = 1e10,
    max_iters: int = 100,
    norm_fn: Callable[[ArrayTree], float] = lambda x: jnp.max(jnp.abs(x)),
) -> Tuple[ArrayTree, ArrayTree, FixedPointIterationInfo]:
    """Solve for x = func(x) using a fixed point iteration"""

    def compute_norm(x: ArrayTree, xp: ArrayTree) -> float:
        return norm_fn(ravel_pytree(jax.tree_util.tree_map(jnp.subtract, x, xp))[0])

    def cond_fn(args: Tuple[int, ArrayTree, ArrayTree, float]) -> bool:
        n, _, _, norm = args
        return (
            (n < max_iters)
            & jnp.isfinite(norm)
            & (norm < divergence_tol)
            & (norm > convergence_tol)
        )

    def body_fn(
        args: Tuple[int, ArrayTree, ArrayTree, float]
    ) -> Tuple[int, ArrayTree, ArrayTree, float]:
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
            q: ArrayTree,
            p: ArrayTree,
            dUdq: ArrayTree,
            initial: Tuple[ArrayTree, ArrayTree] = (position, momentum),
        ) -> Tuple[ArrayTree, ArrayTree]:
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
        def _step(args: ArrayTree) -> Tuple[ArrayTree, ArrayTree]:
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
