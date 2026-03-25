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
from typing import Any, Callable, NamedTuple

import jax
import jax.numpy as jnp
import jax.random
import optax
from jax import lax
from jax.flatten_util import ravel_pytree

from blackjax.types import Array, ArrayLikeTree

__all__ = [
    "LBFGSHistory",
    "LbfgsState",
    "OptStep",
    "minimize_lbfgs",
    "lbfgs_inverse_hessian_factors",
    "lbfgs_inverse_hessian_formula_1",
    "lbfgs_inverse_hessian_formula_2",
    "bfgs_sample",
]


class LBFGSHistory(NamedTuple):
    """Container for the optimization path of a L-BFGS run

    x
        History of positions
    f
        History of objective values
    g
        History of gradient values
    alpha
        History of the diagonal elements of the inverse Hessian approximation.
    update_mask:
        The indicator of whether the updates of position and gradient are
        included in the inverse-Hessian approximation or not.
        (Xi in the paper)

    """

    x: Array
    f: Array
    g: Array
    alpha: Array
    update_mask: Array


class LbfgsState(NamedTuple):
    """State returned by minimize_lbfgs."""

    iter_num: Array
    value: Array
    grad: Array
    error: Array
    s_history: Array
    y_history: Array
    rho_history: Array
    gamma: Array
    stepsize: Array
    aux: Any


class OptStep(NamedTuple):
    params: Any
    state: LbfgsState


def minimize_lbfgs(
    fun: Callable,
    x0: ArrayLikeTree,
    maxiter: int = 30,
    maxcor: int = 10,
    gtol: float = 1e-08,
    ftol: float = 1e-05,
    maxls: int = 1000,
    **lbfgs_kwargs,
) -> tuple[OptStep, LBFGSHistory]:
    """
    Minimize a function using L-BFGS

    Parameters
    ----------
    fun:
        function of the form f(x) where x is a pytree and returns a real scalar.
        The function should be composed of operations with vjp defined.
    x0:
        initial guess
    maxiter:
        maximum number of iterations
    maxcor:
        maximum number of metric corrections ("history size")
    ftol:
        terminates the minimization when `(f_k - f_{k+1}) < ftol`
    gtol:
        terminates the minimization when `|g_k|_norm < gtol`
    maxls:
        maximum number of line search steps (per iteration)

    Returns
    -------
    Optimization results and optimization path

    """
    # Ravel pytree into flat array.
    x0_raveled, unravel_fn = ravel_pytree(x0)
    unravel_fn_mapped = jax.vmap(unravel_fn)

    # Run LBFGS optimizer on flat input.
    last_step_raveled, history_raveled = _minimize_lbfgs(
        lambda x: fun(unravel_fn(x)),
        x0_raveled,
        maxiter,
        maxcor,
        gtol,
        ftol,
        maxls,
    )

    # Unravel final optimization step.
    last_step = OptStep(
        params=unravel_fn(last_step_raveled.params),
        state=LbfgsState(
            iter_num=last_step_raveled.state.iter_num,
            value=last_step_raveled.state.value,
            grad=unravel_fn(last_step_raveled.state.grad),
            error=last_step_raveled.state.error,
            s_history=unravel_fn_mapped(last_step_raveled.state.s_history),
            y_history=unravel_fn_mapped(last_step_raveled.state.y_history),
            rho_history=last_step_raveled.state.rho_history,
            gamma=last_step_raveled.state.gamma,
            stepsize=last_step_raveled.state.stepsize,
            aux=None,
        ),
    )

    # Unravel optimization path history.
    history = LBFGSHistory(
        x=unravel_fn_mapped(history_raveled.x),
        f=history_raveled.f,
        g=unravel_fn_mapped(history_raveled.g),
        alpha=unravel_fn_mapped(history_raveled.alpha),
        update_mask=jax.tree.map(
            lambda x: x.astype(history_raveled.update_mask.dtype),
            unravel_fn_mapped(history_raveled.update_mask.astype(x0_raveled.dtype)),
        ),
    )

    return last_step, history


def _minimize_lbfgs(
    fun: Callable,
    x0: Array,
    maxiter: int,
    maxcor: int,
    gtol: float,
    ftol: float,
    maxls: int,
) -> tuple[OptStep, LBFGSHistory]:
    linesearch = optax.scale_by_zoom_linesearch(max_linesearch_steps=maxls)
    solver = optax.lbfgs(memory_size=maxcor, linesearch=linesearch)
    value_and_grad_fn = optax.value_and_grad_from_state(fun)

    opt_state = solver.init(x0)
    value0, grad0 = jax.value_and_grad(fun)(x0)

    initial_history = LBFGSHistory(
        x=x0,
        f=value0,
        g=grad0,
        alpha=jnp.ones_like(x0),
        update_mask=jnp.zeros_like(x0, dtype=bool),
    )

    def lbfgs_one_step(carry, i):
        (params, state), previous_history = carry

        # Reuse value/grad cached by the previous linesearch when available.
        value, grad = value_and_grad_fn(params, state=state)

        # One LBFGS step; zoom linesearch runs internally via lax.while_loop.
        updates, new_state = solver.update(
            grad, state, params, value=value, grad=grad, value_fn=fun
        )
        new_params = optax.apply_updates(params, updates)

        # Compute new value/grad directly. optax's internal buffer stores s/y
        # one step late (it writes x_k - x_{k-1} only after step k+1), so we
        # compute s and z from the params and grads we already have.
        new_value, new_grad = jax.value_and_grad(fun)(new_params)
        s_l = new_params - params
        z_l = new_grad - grad

        alpha_lm1 = previous_history.alpha
        alpha_l, mask_l = lbfgs_recover_alpha(alpha_lm1, s_l, z_l)

        history = LBFGSHistory(
            x=new_params,
            f=new_value,
            g=new_grad,
            alpha=alpha_l,
            update_mask=mask_l,
        )

        f_delta = (
            jnp.abs(value - new_value)
            / jnp.asarray([jnp.abs(value), jnp.abs(new_value), 1.0]).max()
        )
        not_converged = (
            (jnp.linalg.norm(grad) > gtol) & (f_delta > ftol) & (i < maxiter)
        )

        return ((new_params, new_state), history), not_converged

    def non_op(carry, it):
        return carry, False

    def scan_body(tup, it):
        carry, not_converged = tup
        next_tup = lax.cond(not_converged, lbfgs_one_step, non_op, carry, it)
        return next_tup, next_tup[0][-1]

    init_carry = ((x0, opt_state), initial_history)

    (((last_params, last_opt_state), _), _), history = lax.scan(
        scan_body, (init_carry, True), jnp.arange(maxiter)
    )

    # Prepend initial state to produce shape (maxiter+1, ...) histories.
    history = jax.tree.map(
        lambda x, y: jnp.concatenate([x[None, ...], y], axis=0),
        initial_history,
        history,
    )

    # Build LbfgsState from the final optax state.
    lbfgs_inner = last_opt_state[0]  # ScaleByLBFGSState
    last_idx = (lbfgs_inner.count - 1) % maxcor
    s_last = lbfgs_inner.diff_params_memory[last_idx]
    y_last = lbfgs_inner.diff_updates_memory[last_idx]
    # gamma = (s^T y) / (y^T y), the Hessian scale estimate.
    gamma = jnp.where(
        jnp.dot(s_last, y_last) > 0,
        jnp.dot(s_last, y_last) / jnp.dot(y_last, y_last),
        1.0,
    )

    status = LbfgsState(
        iter_num=lbfgs_inner.count,
        value=history.f[-1],
        grad=history.g[-1],
        error=jnp.linalg.norm(history.g[-1]),
        s_history=lbfgs_inner.diff_params_memory,
        y_history=lbfgs_inner.diff_updates_memory,
        rho_history=lbfgs_inner.weights_memory,
        gamma=gamma,
        stepsize=jnp.array(1.0),
        aux=None,
    )

    return OptStep(params=last_params, state=status), history


def lbfgs_recover_alpha(alpha_lm1, s_l, z_l, epsilon=1e-12):
    """
    Compute diagonal elements of the inverse Hessian approximation from optimation path.
    It implements the inner loop body of Algorithm 3 in :cite:p:`zhang2022pathfinder`.

    Parameters
    ----------
    alpha_lm1
        The diagonal element of the inverse Hessian approximation of the previous iteration
    s_l
        The update of the position (current position - previous position)
    z_l
        The update of the gradient (current gradient - previous gradient). Note that in :cite:p:`zhang2022pathfinder`
        it is defined as the negative of the update of the gradient, but since we are optimizing
        the negative log prob function taking the update of the gradient is correct here.

    Returns
    -------
    alpha_l
        The diagonal element of the inverse Hessian approximation of the current iteration
    mask_l
        The indicator of whether the update of position and gradient are included in
        the inverse-Hessian approximation or not.

    """

    def compute_next_alpha(s_l, z_l, alpha_lm1):
        a = z_l.T @ jnp.diag(alpha_lm1) @ z_l
        b = z_l.T @ s_l
        c = s_l.T @ jnp.diag(1.0 / alpha_lm1) @ s_l
        inv_alpha_l = (
            a / (b * alpha_lm1)
            + z_l**2 / b
            - (a * s_l**2) / (b * c * alpha_lm1**2)
        )
        return 1.0 / inv_alpha_l

    pred = s_l.T @ z_l > (epsilon * jnp.linalg.norm(z_l, 2))
    alpha_l = lax.cond(
        pred, compute_next_alpha, lambda *_: alpha_lm1, s_l, z_l, alpha_lm1
    )
    mask_l = jnp.where(
        pred,
        jnp.ones_like(alpha_lm1, dtype=bool),
        jnp.zeros_like(alpha_lm1, dtype=bool),
    )
    return alpha_l, mask_l


def lbfgs_inverse_hessian_factors(S, Z, alpha):
    """
    Calculates factors for inverse hessian factored representation.
    It implements formula II.2 of:

    Pathfinder: Parallel quasi-newton variational inference, Lu Zhang et al., arXiv:2108.03782

    """
    param_dims = S.shape[-1]
    StZ = S.T @ Z
    R = jnp.triu(StZ) + jnp.eye(param_dims) * jnp.finfo(S.dtype).eps

    eta = jnp.diag(StZ)

    beta = jnp.hstack([jnp.diag(alpha) @ Z, S])

    minvR = -jnp.linalg.inv(R)
    alphaZ = jnp.diag(jnp.sqrt(alpha)) @ Z
    block_dd = minvR.T @ (alphaZ.T @ alphaZ + jnp.diag(eta)) @ minvR
    gamma = jnp.block(
        [[jnp.zeros((param_dims, param_dims)), minvR], [minvR.T, block_dd]]
    )
    return beta, gamma


def lbfgs_inverse_hessian_formula_1(alpha, beta, gamma):
    """
    Calculates inverse hessian from factors as in formula II.1 of:

    Pathfinder: Parallel quasi-newton variational inference, Lu Zhang et al., arXiv:2108.03782

    """
    return jnp.diag(alpha) + beta @ gamma @ beta.T


def lbfgs_inverse_hessian_formula_2(alpha, beta, gamma):
    """
    Calculates inverse hessian from factors as in formula II.3 of:

    Pathfinder: Parallel quasi-newton variational inference, Lu Zhang et al., arXiv:2108.03782

    """
    param_dims = alpha.shape[0]
    dsqrt_alpha = jnp.diag(jnp.sqrt(alpha))
    idsqrt_alpha = jnp.diag(1 / jnp.sqrt(alpha))
    return (
        dsqrt_alpha
        @ (jnp.eye(param_dims) + idsqrt_alpha @ beta @ gamma @ beta.T @ idsqrt_alpha)
        @ dsqrt_alpha
    )


def bfgs_sample(rng_key, num_samples, position, grad_position, alpha, beta, gamma):
    """
    Draws approximate samples of target distribution.
    It implements Algorithm 4 in:

    Pathfinder: Parallel quasi-newton variational inference, Lu Zhang et al., arXiv:2108.03782

    """
    if not isinstance(num_samples, tuple):
        num_samples = (num_samples,)

    Q, R = jnp.linalg.qr(jnp.diag(jnp.sqrt(1 / alpha)) @ beta)
    param_dims = beta.shape[0]
    Id = jnp.identity(R.shape[0])
    L = jnp.linalg.cholesky(Id + R @ gamma @ R.T)

    logdet = jnp.log(jnp.prod(alpha)) + 2 * jnp.log(jnp.linalg.det(L))
    mu = (
        position
        + jnp.diag(alpha) @ grad_position
        + beta @ gamma @ beta.T @ grad_position
    )

    u = jax.random.normal(rng_key, num_samples + (param_dims, 1))
    phi = mu[..., None] + jnp.diag(jnp.sqrt(alpha)) @ (Q @ (L - Id) @ (Q.T @ u) + u)

    logdensity = -0.5 * (
        logdet
        + jnp.einsum("...ji,...ji->...", u, u)
        + param_dims * jnp.log(2.0 * jnp.pi)
    )
    return phi[..., 0], logdensity
