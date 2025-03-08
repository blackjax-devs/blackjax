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
import functools
import logging
from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp
import jax.random
import jax.scipy as jsp
import optax
import optax.tree_utils as otu

from blackjax.types import Array

jax.config.update("jax_enable_x64", True)


logger = logging.getLogger(__name__)

__all__ = [
    "LBFGSHistory",
    "minimize_lbfgs",
    "lbfgs_diff_history_matrix",
    "lbfgs_inverse_hessian_factors",
    "lbfgs_inverse_hessian_formula_1",
    "lbfgs_inverse_hessian_formula_2",
    "bfgs_sample",
    "lbfgs_recover_alpha",
]

INIT_STEP_SIZE = 1.0
MIN_STEP_SIZE = 1e-3


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
    converged: Array  # for clipping history for shorter inverse hessian calcs and bfgs sampling
    iter: Array  # TODO: remove iter. not needed


def minimize_lbfgs(
    fun: Callable,
    x0: Array,
    maxiter: int = 100,
    maxcor: int = 6,
    gtol: float = 1e-8,
    ftol: float = 1e-5,
    maxls: int = 100,
    **lbfgs_kwargs,
):
    def lbfgs_one_step(carry, i):
        # state is a 3-dim tuple
        (params, state), _ = carry
        lbfgs_state, _, _ = state

        value, grad = value_grad_fn(params)
        updates, next_state = solver.update(
            grad, state, params, value=value, grad=grad, value_fn=fun
        )
        _, _, next_ls_state = next_state

        # LBFGS use a rolling history, getting the correct index here.
        iter = lbfgs_state.count
        next_params = params + updates

        converged = check_convergence(state, next_state, iter)
        history = LBFGSHistory(
            x=next_params,
            f=next_ls_state.value,
            g=next_ls_state.grad,
            converged=converged,
            iter=jnp.asarray(iter, dtype=jnp.int32),
        )
        return ((next_params, next_state), history), converged

    def check_convergence(state, next_state, iter):
        _, _, ls_state = state
        _, _, next_ls_state = next_state
        f_delta = (
            jnp.abs(ls_state.value - next_ls_state.value)
            / jnp.asarray(
                [jnp.abs(ls_state.value), jnp.abs(next_ls_state.value), 1.0]
            ).max()
        )
        error = otu.tree_l2_norm(next_ls_state.grad)
        return jnp.array(
            (iter > 0) & ((error <= gtol) | (f_delta <= ftol) | (iter >= maxiter)),
            dtype=bool,
        )

    def state_type_handler(state):
        # ensure num_linesearch_steps is of the same type
        lbfgs_state, _, ls_state = state
        info = ls_state.info._replace(
            num_linesearch_steps=jnp.asarray(
                ls_state.info.num_linesearch_steps, dtype=jnp.int32
            )
        )
        return lbfgs_state, _, ls_state._replace(info=info)

    def non_op(carry, i):
        (params, state), previous_history = carry

        return ((params, state), previous_history), jnp.array(True, dtype=bool)

    def scan_body(tup, i):
        carry, converged = tup
        next_tup = jax.lax.cond(converged, non_op, lbfgs_one_step, carry, i)
        return next_tup, next_tup[0][-1]

    linesearch = optax.scale_by_zoom_linesearch(
        max_linesearch_steps=maxls,
        verbose=False,
        **lbfgs_kwargs,
    )
    solver = optax.lbfgs(
        memory_size=maxcor,
        linesearch=linesearch,
    )
    value_grad_fn = jax.value_and_grad(fun)

    init_state = solver.init(x0)
    init_state = state_type_handler(init_state)

    value, grad = value_grad_fn(x0)
    init_history = LBFGSHistory(
        x=x0,
        f=value,
        g=grad,
        converged=jnp.array(False, dtype=bool),
        iter=jnp.asarray(0, dtype=jnp.int32),
    )

    # Use lax.scan to accumulate history
    (((last_params, last_state), _), _), history = jax.lax.scan(
        scan_body,
        (((x0, init_state), init_history), False),
        jnp.arange(maxiter),
    )
    last_lbfgs_state, _, last_ls_state = last_state

    history = jax.tree.map(
        lambda x, y: jnp.concatenate([x[None, ...], y], axis=0),
        init_history,
        history,
    )
    return (last_params, (last_lbfgs_state, last_ls_state)), history


def lbfgs_recover_alpha(
    position: Array,
    grad_position: Array,
    not_converged_mask: Array,
    epsilon=1e-12,
):
    """
    Compute diagonal elements of the inverse Hessian approximation from optimation path.
    It implements the inner loop body of Algorithm 3 in :cite:p:`zhang2022pathfinder`.

    Parameters
    ----------
    position
        shape (L+1, N)
        The position at the current iteration
    grad_position
        shape (L+1, N)
        The gradient at the current iteration
    not_converged_mask
        shape (L, N)
        The indicator of whether the update of position and gradient are included in the inverse-Hessian approximation or not.
    epsilon
        The threshold for filtering updates based on inner product of position
        and gradient differences

    Returns
    -------
    alpha
        shape (L+1, N)
        The diagonal element of the inverse Hessian approximation of the current iteration
    s
        shape (L, N)
        The update of the position (current position - previous position)
    z
        shape (L, N)
        The update of the gradient (current gradient - previous gradient). Note that in :cite:p:`zhang2022pathfinder` it is defined as the negative of the update of the gradient, but since we are optimizing the negative log prob function taking the update of the gradient is correct here.
    update_mask
        shape (L, N)
        The indicator of whether the update of position and gradient are included in the inverse-Hessian approximation or not.

    Notes
    -----
    shapes: N=num_params
    """

    def compute_next_alpha(alpha_lm1, s_l, z_l):
        a = z_l.T @ jnp.diag(alpha_lm1) @ z_l
        b = z_l.T @ s_l
        c = s_l.T @ jnp.diag(1.0 / alpha_lm1) @ s_l
        inv_alpha_l = (
            a / (b * alpha_lm1)
            + z_l**2 / b
            - (a * s_l**2) / (b * c * alpha_lm1**2)
        )
        return 1.0 / inv_alpha_l

    def non_op(alpha_lm1, s_l, z_l):
        return alpha_lm1

    def scan_body(alpha_init, tup):
        update_mask_l, s_l, z_l = tup
        next_tup = jax.lax.cond(
            update_mask_l,
            compute_next_alpha,
            non_op,
            alpha_init,
            s_l,
            z_l,
        )
        return next_tup, next_tup

    nan_pos_mask = jnp.any(~jnp.isfinite(position.at[1:].get()), axis=-1)
    nan_grad_mask = jnp.any(~jnp.isfinite(grad_position.at[1:].get()), axis=-1)
    nan_mask = jnp.logical_not(nan_pos_mask | nan_grad_mask)

    param_dims = position.shape[-1]
    s = jnp.diff(position, axis=0)
    z = jnp.diff(grad_position, axis=0)
    sz = jnp.sum(s * z, axis=-1)
    update_mask = (
        (sz > epsilon * jnp.sqrt(jnp.sum(z**2, axis=-1)))
        & not_converged_mask
        & nan_mask
    )
    alpha_init = jnp.ones((param_dims,))
    tup = (update_mask, s, z)

    _, alpha = jax.lax.scan(scan_body, alpha_init, tup)

    return alpha, s, z, update_mask


def lbfgs_diff_history_matrix(diff: Array, update_mask: Array, J: int):
    """
    Construct an NxJ matrix that stores the previous J differences for position or gradient updates in L-BFGS. Storage is based on the update mask.

    Parameters
    ----------
    diff : Array
        shape (L, N)
        array of differences, where L is the number of iterations
        and N is the number of parameters.
    update_mask : Array
        shape (L, N)
        boolean array indicating which differences to include.
    J : int
        history size, the number of past differences to store.

    Returns
    -------
    chi_mat
        shape (L, N, J)
        history matrix of differences.
    """
    L, N = diff.shape
    j_last = jnp.array(J - 1)  # since indexing starts at 0

    def chi_update(chi_lm1, diff_l):
        chi_l = jnp.roll(chi_lm1, -1, axis=0)
        return chi_l.at[j_last].set(diff_l)

    def non_op(chi_lm1, diff_l):
        return chi_lm1

    def scan_body(chi_init, tup):
        update_mask_l, diff_l = tup
        next_tup = jax.lax.cond(update_mask_l, chi_update, non_op, chi_init, diff_l)
        return next_tup, next_tup

    chi_init = jnp.zeros((J, N))
    _, chi_mat = jax.lax.scan(scan_body, chi_init, (update_mask, diff))

    chi_mat = jnp.matrix_transpose(chi_mat)

    # (L, N, J)
    return chi_mat


def lbfgs_inverse_hessian_factors(S, Z, alpha):
    """

    Calculates factors for inverse hessian factored representation.
    It implements formula II.2 of:

    Pathfinder: Parallel quasi-newton variational inference, Lu Zhang et al., arXiv:2108.03782

    """

    param_dims, J = S.shape

    StZ = S.T @ Z
    Ij = jnp.eye(J)

    # TODO: uncomment this
    # R = jnp.triu(StZ) + Ij * jnp.finfo(S.dtype).eps
    # TODO: delete this
    REGULARISATION_TERM = 1e-8
    R = jnp.triu(StZ) + Ij * REGULARISATION_TERM

    eta = jnp.diag(R)

    beta = jnp.hstack([jnp.diag(alpha) @ Z, S])

    # jsp.linalg.solve is more stable than jnp.linalg.inv
    minvR = -jsp.linalg.solve_triangular(R, Ij)
    alphaZ = jnp.diag(jnp.sqrt(alpha)) @ Z
    block_dd = minvR.T @ (alphaZ.T @ alphaZ + jnp.diag(eta)) @ minvR
    gamma = jnp.block([[jnp.zeros((J, J)), minvR], [minvR.T, block_dd]])

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


def bfgs_sample(
    rng_key,
    num_samples,
    position,
    grad_position,
    alpha,
    beta,
    gamma,
    sparse: bool | None = None,
):
    """
    Draws approximate samples of target distribution.
    It implements Algorithm 4 in:

    Pathfinder: Parallel quasi-newton variational inference, Lu Zhang et al., arXiv:2108.03782

    parameters
    ----------
    rng_key : array
        prng key
    num_samples : int
        number of samples to draw
    position : array
        current position in parameter space
    grad_position : array
        gradient at current position
    alpha : array
        diagonal elements of inverse hessian approximation
    beta : array
        first factor of inverse hessian approximation
    gamma : array
        second factor of inverse hessian approximation
    sparse : bool | none, optional
        whether to use sparse computation, by default none
        if none, automatically determined based on problem size

    returns
    -------
    phi : array
        samples drawn from approximate distribution
    logdensity : array
        log density of samples
    """
    param_dims = position.shape[-1]
    J = beta.shape[-1] // 2  # beta has shape (param_dims, 2*J)

    def _bfgs_sample_sparse(args):
        rng_key, position, grad_position, alpha, beta, gamma = args
        sqrt_alpha = jnp.sqrt(alpha)
        Q, R = jnp.linalg.qr(jnp.diag(1.0 / sqrt_alpha) @ beta)
        Id = jnp.identity(R.shape[0])
        L = jnp.linalg.cholesky(Id + R @ gamma @ R.T)

        u = jax.random.normal(rng_key, (num_samples, param_dims, 1))
        logdet = jnp.sum(jnp.log(alpha), axis=-1) + 2.0 * jnp.sum(
            jnp.log(jnp.abs(jnp.diagonal(L))), axis=-1
        )
        mu = position - (
            (jnp.diag(alpha) @ grad_position) + (beta @ gamma @ beta.T @ grad_position)
        )
        phi = jnp.squeeze(
            mu[..., None] + jnp.diag(sqrt_alpha) @ (Q @ (L - Id) @ (Q.T @ u) + u),
            axis=-1,
        )
        logdensity = -0.5 * (
            logdet
            + jnp.einsum("...ji,...ji->...", u, u)
            + param_dims * jnp.log(2.0 * jnp.pi)
        )
        return phi, logdensity

    def _bfgs_sample_dense(args):
        rng_key, position, grad_position, alpha, beta, gamma = args
        sqrt_alpha = jnp.sqrt(alpha)
        sqrt_alpha_diag = jnp.diag(sqrt_alpha)
        inv_sqrt_alpha_diag = jnp.diag(1.0 / sqrt_alpha)
        Id = jnp.identity(param_dims)

        H_inv = (
            sqrt_alpha_diag
            @ (Id + inv_sqrt_alpha_diag @ beta @ gamma @ beta.T @ inv_sqrt_alpha_diag)
            @ sqrt_alpha_diag
        )

        u = jax.random.normal(rng_key, (num_samples, param_dims, 1))
        Lchol = jnp.linalg.cholesky(H_inv)
        logdet = 2.0 * jnp.sum(
            jnp.log(jnp.abs(jnp.diagonal(Lchol, axis1=-2, axis2=-1))), axis=-1
        )
        mu = position - (H_inv @ grad_position)
        phi = jnp.squeeze(mu[..., None] + (Lchol @ u), axis=-1)
        logdensity = -0.5 * (
            logdet
            + jnp.einsum("...ji,...ji->...", u, u)
            + param_dims * jnp.log(2.0 * jnp.pi)
        )
        return phi, logdensity

    # pack args to avoid excessive parameter passing
    args = (rng_key, position, grad_position, alpha, beta, gamma)

    sparse = jax.lax.cond(
        sparse is None, lambda _: param_dims < 2 * J, lambda _: sparse, None
    )

    phi, logdensity = jax.lax.cond(
        sparse, _bfgs_sample_sparse, _bfgs_sample_dense, args
    )

    nan_phi_mask = jnp.any(~jnp.isfinite(phi), axis=-1)
    nan_logdensity_mask = ~jnp.isfinite(logdensity)
    nan_mask = nan_phi_mask | nan_logdensity_mask

    logdensity = jnp.where(nan_mask, jnp.inf, logdensity)
    return phi, logdensity


bfgs_sample_sparse = functools.partial(bfgs_sample, sparse=True)
bfgs_sample_dense = functools.partial(bfgs_sample, sparse=False)
