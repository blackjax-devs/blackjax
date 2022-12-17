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
from typing import Callable, NamedTuple, Tuple

import jax
import jax.numpy as jnp
import jax.random
import jaxopt
from jax import lax
from jax.flatten_util import ravel_pytree
from jaxopt._src.lbfgs import LbfgsState
from jaxopt.base import OptStep

from blackjax.types import Array, PyTree

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


class LBFGSHistory(NamedTuple):
    """Container for the optimization path of a L-BFGS run

    Attributes
    ---------
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


def minimize_lbfgs(
    fun: Callable,
    x0: PyTree,
    maxiter: int = 30,
    maxcor: float = 10,
    gtol: float = 1e-08,
    ftol: float = 1e-05,
    maxls: int = 1000,
) -> Tuple[OptStep, LBFGSHistory]:
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
        update_mask=jax.tree_map(
            lambda x: x.astype(history_raveled.update_mask.dtype),
            unravel_fn_mapped(history_raveled.update_mask.astype(x0_raveled.dtype)),
        ),
    )

    return last_step, history


def _minimize_lbfgs(
    fun: Callable,
    x0: Array,
    maxiter: int,
    maxcor: float,
    gtol: float,
    ftol: float,
    maxls: int,
) -> Tuple[OptStep, LBFGSHistory]:
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
        history = LBFGSHistory(
            x=next_params,
            f=next_state.value,
            g=current_grad,
            alpha=alpha_l,
            update_mask=mask_l,
        )
        # check convergence
        f_delta = (
            jnp.abs(state.value - next_state.value)
            / jnp.asarray([jnp.abs(state.value), jnp.abs(next_state.value), 1.0]).max()
        )
        not_converged = (next_state.error > gtol) & (f_delta > ftol) & (i < maxiter)
        return (OptStep(params=next_params, state=next_state), history), not_converged

    def non_op(carry, it):
        return carry, False

    def scan_body(tup, it):
        carry, not_converged = tup
        # When cond is met, we start doing no-ops.
        next_tup = lax.cond(not_converged, lbfgs_one_step, non_op, carry, it)
        return next_tup, next_tup[0][-1]

    solver = jaxopt.LBFGS(fun=fun, maxiter=maxiter, maxls=maxls, history_size=maxcor)
    value0, grad0 = jax.value_and_grad(fun)(x0)
    state = solver.init_state(x0)

    value0, grad0 = jax.value_and_grad(fun)(x0)
    # LBFGS update overwirte value internally, here is to set the value for checking condition
    state = state._replace(value=value0)
    init_step = OptStep(params=x0, state=state)
    initial_history = LBFGSHistory(
        x=x0,
        f=value0,
        g=grad0,
        alpha=jnp.ones_like(x0),
        update_mask=jnp.zeros_like(x0, dtype=bool),
    )

    ((last_step, _), _), history = lax.scan(
        scan_body, ((init_step, initial_history), True), jnp.arange(maxiter)
    )
    # Append initial state to history.
    history = jax.tree_map(
        lambda x, y: jnp.concatenate([x[None, ...], y], axis=0),
        initial_history,
        history,
    )
    return last_step, history


def lbfgs_recover_alpha(alpha_lm1, s_l, z_l, epsilon=1e-12):
    """
    Compute diagonal elements of the inverse Hessian approximation from optimation path.
    It implements the inner loop body of Algorithm 3 in [1].

    Parameters
    ----------
    alpha_lm1
        The diagonal element of the inverse Hessian approximation of the previous iteration
    s_l
        The update of the position (current position - previous position)
    z_l
        The update of the gradient (current gradient - previous gradient). Note that in [1]
        it is defined as the negative of the update of the gradient, but since we are optimizing
        the negative log prob function taking the update of the gradient is correct here.

    Returns
    -------
    alpha_l
        The diagonal element of the inverse Hessian approximation of the current iteration
    mask_l
        The indicator of whether the update of position and gradient are included in
        the inverse-Hessian approximation or not.

    References
    ----------

    .. [1]: Pathfinder: Parallel quasi-newton variational inference,
            Lu Zhang et al., arXiv:2108.03782

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
