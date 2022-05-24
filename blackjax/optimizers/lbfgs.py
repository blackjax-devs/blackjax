from typing import Callable, NamedTuple, Tuple, Union

import jax
import jax.numpy as jnp
import jax.random
import jaxopt
from jax import lax
from jaxopt._src.lbfgs import compute_gamma
from jaxopt.base import OptStep

from blackjax.types import Array

__all__ = ["LBFGSHistory", "minimize_lbfgs", "lbfgs_inverse_hessian_factors", "lbfgs_inverse_hessian_formula_1", "lbfgs_inverse_hessian_formula_2", "lbfgs_sample"]

class LBFGSHistory(NamedTuple):
    "Container for the optimization path of a L-BFGS run"
    x: Array
    f: Array
    g: Array
    gamma: Array


_update_history_vectors = (
    lambda history, new: jnp.roll(history, -1, axis=0).at[-1, :].set(new)
)
_update_history_scalar = (
    lambda history, new: jnp.roll(history, -1, axis=0).at[-1].set(new)
)


def minimize_lbfgs(
    fun: Callable,
    x0: Array,
    maxiter: int = 30,
    maxcor: float = 10,
    gtol: float = 1e-03,
    ftol: float = 1e-02,
    maxls: int = 1000,
) -> Tuple[OptStep, LBFGSHistory]:
    """
    Minimize a function using L-BFGS

    Parameters
    ----------
    fun:
        function of the form f(x) where x is a flat ndarray and returns a real scalar.
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
    d = len(x0)
    dtype = jnp.dtype(x0)
    grad_fun = jax.grad(fun)

    def cond_fun(inputs):
        (_, state), i, history = inputs
        return (
            (state.error > gtol)
            & (((history.f[-2] - history.f[-1]) > ftol) | (i == 0))
            & (i < maxiter)
        ).all()

    def body_fun(inputs):
        (params, state), i, history = inputs

        # this is to help optimization when using log-likelihoods, especially for float 32
        # it resets stepsize of the line search algorithm back to stating value (1.0) if
        # it get stuck in very small values
        state = state._replace(
            stepsize=jnp.where(state.stepsize < 1e-3, 1.0, state.stepsize)
        )
        last = (state.iter_num % maxcor + maxcor) % maxcor
        gamma = compute_gamma(state.s_history, state.y_history, last)
        new_opt = solver.update(params, state)

        i += 1

        history = history._replace(
            x=_update_history_vectors(history=history.x, new=new_opt.params),
            f=_update_history_scalar(history=history.f, new=fun(new_opt.params)),
            g=_update_history_vectors(history=history.g, new=grad_fun(new_opt.params)),
            gamma=_update_history_scalar(history=history.gamma, new=gamma),
        )
        return (new_opt, i, history)

    history_initial = LBFGSHistory(
        x=_update_history_vectors(
            jnp.zeros((maxiter + maxcor + 2, d), dtype=dtype), x0
        ),
        f=_update_history_scalar(
            jnp.zeros((maxiter + maxcor + 2), dtype=dtype), fun(x0)
        ),
        g=_update_history_vectors(
            jnp.zeros((maxiter + maxcor + 2, d), dtype=dtype), grad_fun(x0)
        ),
        gamma=_update_history_scalar(jnp.zeros(maxiter + maxcor + 2, dtype=dtype), 1.0),
    )

    solver = jaxopt.LBFGS(fun=fun, maxiter=maxiter, maxls=maxls, history_size=maxcor)
    state = solver.init_state(x0)
    state = state._replace(stepsize=state.stepsize / 1.5)
    zero_step = OptStep(params=x0, state=state)
    init_val = (zero_step, 0, history_initial)

    state, _, history = lax.while_loop(
        cond_fun=cond_fun, body_fun=body_fun, init_val=init_val
    )

    return state, history


def lbfgs_inverse_hessian_factors(S, Z, alpha):
    """
    Calculates factors for inverse hessian factored representation.
    It implements algorithm of figure 7 in:

    Pathfinder: Parallel quasi-newton variational inference, Lu Zhang et al., arXiv:2108.03782
    """
    param_dims = S.shape[1]
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
    Calculates inverse hessian from factors as in figure 7 of:

    Pathfinder: Parallel quasi-newton variational inference, Lu Zhang et al., arXiv:2108.03782
    """
    return jnp.diag(alpha) + beta @ gamma @ beta.T


def lbfgs_inverse_hessian_formula_2(alpha, beta, gamma):
    """
    Calculates inverse hessian from factors as in formula II.1 of:

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


def lbfgs_sample(rng_key, num_samples, position, grad_position, alpha, beta, gamma):
    """
    Draws approximate samples of target distribution.
    It implements algorithm of figure 8 in:

    Pathfinder: Parallel quasi-newton variational inference, Lu Zhang et al., arXiv:2108.03782
    """
    if not isinstance(num_samples, tuple):
        num_samples = (num_samples,)

    Q, R = jnp.linalg.qr(jnp.diag(jnp.sqrt(1 / alpha)) @ beta)
    param_dims = beta.shape[0]
    L = jnp.linalg.cholesky(jnp.eye(param_dims) + R @ gamma @ R.T)

    logdet = jnp.log(jnp.prod(alpha)) + 2 * jnp.log(jnp.linalg.det(L))
    mu = (
        position
        + jnp.diag(alpha) @ grad_position
        + beta @ gamma @ beta.T @ grad_position
    )

    u = jax.random.normal(rng_key, num_samples + (param_dims, 1))
    phi = (
        mu[..., None]
        + jnp.diag(jnp.sqrt(alpha)) @ (Q @ L @ Q.T @ u + u - (Q @ Q.T @ u))
    )[..., 0]

    logq = -0.5 * (
        logdet
        + jnp.einsum("...ji,...ji->...", u, u)
        + param_dims * jnp.log(2.0 * jnp.pi)
    )
    return phi, logq
