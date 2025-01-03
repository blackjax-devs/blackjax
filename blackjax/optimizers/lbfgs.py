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
from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp
import jax.random
import jaxopt
import optax
import optax.tree_utils as otu
from jax import lax
from jax.flatten_util import ravel_pytree
from jaxopt._src.lbfgs import LbfgsState
from jaxopt.base import OptStep

from blackjax.types import Array, ArrayLikeTree

__all__ = [
    "LBFGSHistory",
    "minimize_lbfgs",
    "lbfgs_inverse_hessian_factors",
    "lbfgs_inverse_hessian_formula_1",
    "lbfgs_inverse_hessian_formula_2",
    "bfgs_sample",
]

INIT_STEP_SIZE = 1.0
MIN_STEP_SIZE = 1e-3


class _OptaxLBFGSHistory(NamedTuple):
    x: Array
    f: Array
    g: Array
    alpha: Array
    update_mask: Array
    # store intermediate values to perform checks
    not_converged: Array
    s: Array
    z: Array
    s_l: Array
    z_l: Array
    last: Array
    iter: Array


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
    not_converged: Array  # for clipping history for shorter inverse hessian calcs and bfgs sampling


def minimize_lbfgs(
    fun: Callable,
    x0: ArrayLikeTree,
    maxiter: int = 30,
    maxcor: float = 10,
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
    **lbfgs_kwargs
        other keyword arguments passed to `jaxopt.LBFGS`.

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
        **lbfgs_kwargs,
    )

    # Unravel final optimization step.
    last_step = OptStep(
        params=unravel_fn(last_step_raveled.params),
        state=LbfgsState(
            iter_num=last_step_raveled.state.iter_num,
            value=last_step_raveled.state.value,
            grad=unravel_fn(last_step_raveled.state.grad),
            stepsize=last_step_raveled.state.stepsize,
            error=last_step_raveled.state.error,
            s_history=unravel_fn_mapped(last_step_raveled.state.s_history),
            y_history=unravel_fn_mapped(last_step_raveled.state.y_history),
            rho_history=last_step_raveled.state.rho_history,
            gamma=last_step_raveled.state.gamma,
            aux=last_step_raveled.state.aux,
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


def optax_lbfgs(
    fun: Callable,
    x0: Array,
    maxiter: int,
    maxcor: float,
    gtol: float,
    ftol: float,
    maxls: int,
    # **lbfgs_kwargs, # TODO: insert kwargs to optax.scale_by_zoom_linesearch and optax.value_and_grad_from_state
):
    linesearch = optax.scale_by_zoom_linesearch(
        max_linesearch_steps=maxls,
        verbose=True,
    )
    solver = optax.lbfgs(
        memory_size=maxcor,
        linesearch=linesearch,
    )
    value_and_grad_fun = optax.value_and_grad_from_state(fun)

    def lbfgs_one_step(carry, i):
        # state is a 3-dim tuple
        (params, state), previous_history = carry
        value, grad = value_and_grad_fun(params, state=state)
        updates, next_state = solver.update(
            grad, state, params, value=value, grad=grad, value_fn=fun
        )

        # ensure num_linesearch_steps is of the same type
        info = next_state[2].info._replace(
            num_linesearch_steps=jnp.asarray(
                next_state[2].info.num_linesearch_steps, dtype=jnp.int32
            )
        )

        next_state = (next_state[0], next_state[1], next_state[2]._replace(info=info))

        # LBFGS use a rolling history, getting the correct index here.
        iter = state[0].count
        # last variable for getting the correct index where updates occur
        last = jnp.max(jnp.array([iter - 1, 0], dtype=jnp.int32)) % maxcor
        next_params = optax.apply_updates(params, updates)

        # Recover alpha and update mask
        s_l = next_state[0].diff_params_memory[last]
        z_l = next_state[0].diff_updates_memory[last]
        alpha_lm1 = previous_history.alpha
        alpha_l, mask_l = lbfgs_recover_alpha(alpha_lm1, s_l, z_l)

        # TODO: check correct calc for g
        # g = next_state[2].grad
        # g = state[2].grad
        # g = grad
        # g = previous_history.g
        # g = previous_history.g + z_l
        # g = state[2].grad + z_l

        not_converged = check_convergence(state, next_state, iter)
        history = _OptaxLBFGSHistory(
            x=next_params,
            f=next_state[2].value,
            g=next_state[2].grad,
            alpha=alpha_l,
            update_mask=mask_l,
            not_converged=not_converged,
            s=next_state[0].diff_params_memory,
            z=next_state[0].diff_updates_memory,
            s_l=s_l,
            z_l=z_l,
            last=jnp.asarray(last, dtype=jnp.int32),
            iter=jnp.asarray(iter, dtype=jnp.int32),
        )
        return ((next_params, next_state), history), not_converged

    def check_convergence(state, next_state, iter):
        f_delta = (
            jnp.abs(state[2].value - next_state[2].value)
            / jnp.asarray(
                [jnp.abs(state[2].value), jnp.abs(next_state[2].value), 1.0]
            ).max()
        )
        next_state_grad = otu.tree_get(next_state[2], "grad")
        error = otu.tree_l2_norm(next_state_grad)
        return jnp.array(
            (iter == 0) | (error > gtol) & (f_delta > ftol) & (iter < maxiter),
            dtype=bool,
        )

    def non_op(carry, i):
        (params, state), previous_history = carry

        info = state[2].info._replace(
            num_linesearch_steps=jnp.asarray(
                state[2].info.num_linesearch_steps, dtype=jnp.int32
            )
        )
        state = (state[0], state[1], state[2]._replace(info=info))

        return ((params, state), previous_history), jnp.array(False, dtype=bool)

    def scan_body(tup, i):
        carry, not_converged = tup
        next_tup = jax.lax.cond(not_converged, lbfgs_one_step, non_op, carry, i)
        return next_tup, next_tup[0][-1]

    x0, init_state = (x0, solver.init(x0))
    init_history = _OptaxLBFGSHistory(
        x=init_state[0].params,
        f=init_state[2].value,
        g=init_state[2].grad,
        alpha=jnp.ones_like(x0),
        update_mask=jnp.zeros_like(x0, dtype=bool),
        not_converged=jnp.array(True, dtype=bool),
        s=init_state[0].diff_params_memory,
        z=init_state[0].diff_updates_memory,
        s_l=jnp.zeros_like(x0),
        z_l=jnp.zeros_like(x0),
        last=jnp.asarray(-1, dtype=jnp.int32),
        iter=jnp.asarray(-1, dtype=jnp.int32),
    )

    # Use lax.scan to accumulate history
    (((final_params, final_state), _), _), history = jax.lax.scan(
        scan_body,
        (((x0, init_state), init_history), True),
        jnp.arange(maxiter),
        length=maxiter,
    )

    history = jax.tree.map(
        lambda x, y: jnp.concatenate([x[None, ...], y], axis=0),
        init_history,
        history,
    )
    return (final_params, final_state), history


def _minimize_lbfgs(
    fun: Callable,
    x0: Array,
    maxiter: int,
    maxcor: float,
    gtol: float,
    ftol: float,
    maxls: int,
    **lbfgs_kwargs,
) -> tuple[OptStep, LBFGSHistory]:
    def lbfgs_one_step(carry, i):
        (params, state), previous_history = carry

        # this is to help optimization when using log-likelihoods, especially for float 32
        # it resets stepsize of the line search algorithm back to stating value (INIT_STEP_SIZE) if
        # it get stuck in very small values
        state = state._replace(
            stepsize=jnp.where(
                state.stepsize < MIN_STEP_SIZE, INIT_STEP_SIZE, state.stepsize
            )
        )
        # LBFGS use a rolling history, getting the correct index here.
        last = (state.iter_num % maxcor + maxcor) % maxcor
        next_params, next_state = solver.update(params, state)

        # Recover alpha and update mask
        s_l = next_state.s_history[last]
        z_l = next_state.y_history[last]
        alpha_lm1 = previous_history.alpha

        alpha_l, mask_l = lbfgs_recover_alpha(alpha_lm1, s_l, z_l)

        current_grad = previous_history.g + z_l

        # check convergence
        f_delta = (
            jnp.abs(state.value - next_state.value)
            / jnp.asarray([jnp.abs(state.value), jnp.abs(next_state.value), 1.0]).max()
        )
        not_converged = (next_state.error > gtol) & (f_delta > ftol) & (i < maxiter)
        history = LBFGSHistory(
            x=next_params,
            f=next_state.value,
            g=current_grad,
            alpha=alpha_l,
            update_mask=mask_l,
            not_converged=jnp.array(not_converged, dtype=bool),
        )
        return (OptStep(params=next_params, state=next_state), history), not_converged

    def non_op(carry, it):
        return carry, False

    def scan_body(tup, it):
        carry, not_converged = tup
        # When cond is met, we start doing no-ops.
        next_tup = lax.cond(not_converged, lbfgs_one_step, non_op, carry, it)
        return next_tup, next_tup[0][-1]

    solver = jaxopt.LBFGS(
        fun=fun,
        maxiter=maxiter,
        maxls=maxls,
        history_size=maxcor,
        **lbfgs_kwargs,
    )
    state = solver.init_state(x0)

    value0, grad0 = jax.value_and_grad(fun)(x0)
    # LBFGS update overwrite value internally, here is to set the value for checking condition
    state = state._replace(value=value0)
    init_step = OptStep(params=x0, state=state)
    initial_history = LBFGSHistory(
        x=x0,
        f=value0,
        g=grad0,
        alpha=jnp.ones_like(x0),
        update_mask=jnp.zeros_like(x0, dtype=bool),
        not_converged=jnp.array(True, dtype=bool),
    )

    ((last_step, _), _), history = lax.scan(
        scan_body, ((init_step, initial_history), True), jnp.arange(maxiter)
    )
    # Append initial state to history.
    history = jax.tree.map(
        lambda x, y: jnp.concatenate([x[None, ...], y], axis=0),
        initial_history,
        history,
    )
    return last_step, history


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
