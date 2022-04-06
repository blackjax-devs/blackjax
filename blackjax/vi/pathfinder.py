from pdb import set_trace
from typing import NamedTuple, Callable, Tuple
import jax
import jax.random
import jax.numpy as jnp
from jax import lax
from jax._src.scipy.optimize.line_search import line_search
from jax._src.scipy.optimize._lbfgs import (
    LBFGSResults,
    _dot,
    _two_loop_recursion,
    _update_history_vectors,
    _update_history_scalars,
)
from jax.flatten_util import ravel_pytree
from blackjax.types import Array, PRNGKey, PyTree

__all__ = ["PathfinderState", "init", "kernel", "sample_from_state"]


class PathfinderState(NamedTuple):
    """State of the Pathfinder algorithm

    Pathfinder locates normal approximations to the target density along a
    quasi-Newton optimization path, with local covariance estimated using
    the inverse Hessian estimates produced by the L-BFGS optimizer.
    PathfinderState stores for an interation fo the L-BFGS optimizer the
    resulting ELBO and all factors needed to sample from the approximated
    target density.

    elbo:
        ELBO of approximation wrt target distribution
    i:
        iteration of the L-BFGS optimizer
    position:
        position
    grad_position:
        gradient of target distribution wrt position
    alpha, beta, gamma:
        factored rappresentation of the inverse hessian
    """

    elbo: Array
    i: int
    position: PyTree
    grad_position: PyTree
    alpha: Array
    beta: Array
    gamma: Array


def init(
    rng_key: PRNGKey,
    logprob_fn: Callable,
    initial_position: PyTree,
    num_samples: int = 200,
    return_path: bool = False,
    **lbfgs_kwargs
) -> PathfinderState:
    """
    Pathfinder variational inference algorithm:
    pathfinder locates normal approximations to the target density along a
    quasi-Newton optimization path, with local covariance estimated using
    the inverse Hessian estimates produced by the L-BFGS optimizer.

    function implements algorithm in figure 3 of:

    Pathfinder: Parallel quasi-newton variational inference, Lu Zhang et al., arXiv:2108.03782

    Parameters
    ----------
    rng_key
        PRPNG key
    logprob_fn
        (un-normalized) log densify function of target distribution to take
        approximate samples from
    initial_position
        starting point of the L-BFGS optimization routine
    num_samples
        number of samples to draw to estimate ELBO
    return_path
        if False output only iteration that maximize ELBO, otherwise output
        all iterations
    lbfgs_kwargs:
        kwargs passed to the internal call to lbfgs_minimize

    Returns
    -------
    if return_path=True a PathfinderState with full information
    on the optimization path
    if return_path=False a PathfinderState with information on the iteration
    in the optimization path whose approximate samples yields the highest ELBO
    """

    initial_position_flatten, unravel_fn = ravel_pytree(initial_position)
    objective_fn = lambda x: -logprob_fn(unravel_fn(x))
    param_dims = initial_position_flatten.shape[0]

    if "maxiter" not in lbfgs_kwargs:
        lbfgs_kwargs["maxiter"] = 10
    if "maxcor" not in lbfgs_kwargs:
        lbfgs_kwargs["maxcor"] = 10

    status, history = minimize_lbfgs(
        objective_fn, initial_position_flatten, **lbfgs_kwargs
    )

    position, grad_position, alpha_scalar = history.x, history.g, history.gamma

    # set difference between empty zero states and initial point x0 to zero
    # this is beacuse here we are working with static shapes and keeping all
    # the zero states in the arrays
    s = jnp.diff(position, axis=0).at[status.k - 1].set(0.0)
    z = jnp.diff(grad_position, axis=0).at[status.k - 1].set(0.0)

    maxiter = lbfgs_kwargs["maxiter"]
    maxcor = lbfgs_kwargs["maxcor"]

    rng_keys = jax.random.split(rng_key, maxiter)

    def pathfinder_inner_step(i):
        i_offset = maxcor + i
        S = lax.dynamic_slice(s, (i_offset - maxcor, 0), (maxcor, param_dims)).T
        Z = lax.dynamic_slice(z, (i_offset - maxcor, 0), (maxcor, param_dims)).T
        alpha = alpha_scalar[i_offset] * jnp.ones(param_dims)
        beta, gamma = lbfgs_inverse_hessian_factors(S, Z, alpha)

        phi, logq = lbfgs_sample(
            rng_key=rng_keys[i],
            num_samples=num_samples,
            position=position[i_offset],
            grad_position=grad_position[i_offset],
            alpha=alpha,
            beta=beta,
            gamma=gamma,
        )

        logp = -jax.lax.map(objective_fn, phi)  # cannot vmap pytrees
        elbo = (logp - logq).mean()  # algorithm of figure 9 of the paper

        state = PathfinderState(
            i=jnp.where(i - maxiter + status.k < 0, -1, i - maxiter + status.k),
            elbo=elbo,
            position=unravel_fn(position[i_offset]),
            grad_position=unravel_fn(grad_position[i_offset]),
            alpha=alpha,
            beta=beta,
            gamma=gamma,
        )
        return state

    # LBFGS maximum iteration condition is always checked after running a step
    # hence given max iter, max iter+1 LBFGS steps are run at maximum
    out = lax.map(pathfinder_inner_step, jnp.arange(maxiter + 1))
    if return_path:
        return out
    else:
        best_i = jnp.argmax(jnp.where(out.i > 0, out.elbo, -jnp.inf))
        return jax.tree_map(lambda x: x[best_i], out)


def kernel():
    """
    Builds a pathfinder kernel.

    Returns:
    a kernel that takes rng_key and the pathfinder state and returns
    the pathfinder state and a draw from the approximate distribution
    """

    def one_step(rng_key: PRNGKey, state: PathfinderState):

        sample = sample_from_state(rng_key, state, num_samples=1)
        sample_no_leading_dim = jax.tree_map(lambda x: x[0], sample)
        return state, sample_no_leading_dim

    return one_step


def sample_from_state(rng_key: PRNGKey, state: PathfinderState, num_samples: int):
    """
    Draws samples of the target distribution using approixmation from
    pathfinder algorithm.

    Parameters
    ----------
    rng_key
        PRNG key
    state
        PathfinderState containing information for sampling
    num_samples
        Number of samples to draw

    Returns
    -------
    samples drawn from the approximate Pathfinder distribution
    """

    position_flatten, unravel_fn = ravel_pytree(state.position)
    grad_position_flatten, _ = ravel_pytree(state.grad_position)

    phi, _ = lbfgs_sample(
        rng_key,
        num_samples,
        position_flatten,
        grad_position_flatten,
        state.alpha,
        state.beta,
        state.gamma,
    )

    return jax.lax.map(unravel_fn, phi)


class LBFGSHistory(NamedTuple):
    "Container for the optimization path of a L-BFGS run"
    x: Array
    f: Array
    g: Array
    gamma: Array


def minimize_lbfgs(
        fun: Callable,
        x0: Array,
        maxiter: int = 100,
        norm: float = jnp.inf,
        maxcor: float = 10,
        ftol: float = 2.220446049250313e-09,
        gtol: float = 1e-05,
        maxfun: int = None,
        maxgrad: int = None,
        maxls: int = 20,
        ) -> Tuple[LBFGSResults, LBFGSHistory]:
    """
    Licensed under the Apache License, Version 2.0 (the "License");
    TAKEN FROM https://github.com/google/jax/blob/main/jax/_src/scipy/optimize/_lbfgs.py
    and added functionality to return optimization path

    Minimize a function using L-BFGS

    Implements the L-BFGS algorithm from
      Algorithm 7.5 from Wright and Nocedal, 'Numerical Optimization', 1999, pg. 176-185
    And generalizes to complex variables from
       Sorber, L., Barel, M.V. and Lathauwer, L.D., 2012.
       "Unconstrained optimization of real functions in complex variables"
       SIAM Journal on Optimization, 22(3), pp.879-898.

    Parameters
    ----------
    fun:
        function of the form f(x) where x is a flat ndarray and returns a real scalar.
        The function should be composed of operations with vjp defined.
    x0:
        initial guess
    maxiter:
        maximum number of iterations
    norm:
        order of norm for convergence check. Default inf.
    maxcor:
        maximum number of metric corrections ("history size")
    ftol:
        terminates the minimization when `(f_k - f_{k+1}) < ftol`
    gtol:
        terminates the minimization when `|g_k|_norm < gtol`
    maxfun:
        maximum number of function evaluations
    maxgrad:
        maximum number of gradient evaluations
    maxls:
        maximum number of line search steps (per iteration)

    Returns
    -------
    Optimization results and optimization path
    """
    d = len(x0)
    dtype = jnp.dtype(x0)

    # ensure there is at least one termination condition
    if (maxiter is None) and (maxfun is None) and (maxgrad is None):
        maxiter = d * 200

    # set others to inf, such that >= is supported
    if maxfun is None:
        maxfun = jnp.inf
    if maxgrad is None:
        maxgrad = jnp.inf

    # initial evaluation
    f_0, g_0 = jax.value_and_grad(fun)(x0)
    state_initial = LBFGSResults(
      converged=False,
      failed=False,
      k=0,
      nfev=1,
      ngev=1,
      x_k=x0,
      f_k=f_0,
      g_k=g_0,
      s_history=jnp.zeros((maxcor, d), dtype=dtype),
      y_history=jnp.zeros((maxcor, d), dtype=dtype),
      rho_history=jnp.zeros((maxcor,), dtype=dtype),
      gamma=1.,
      status=0,
      ls_status=0,
    )

    history_initial = LBFGSHistory(
      x=_update_history_vectors(
          jnp.zeros((maxiter + maxcor + 1, d), dtype=dtype),
          x0),
      f=_update_history_scalars(
          jnp.zeros(maxiter + maxcor + 1, dtype=dtype),
          f_0),
      g=_update_history_vectors(
          jnp.zeros((maxiter + maxcor + 1, d), dtype=dtype),
          g_0),
      gamma=_update_history_scalars(
          jnp.zeros(maxiter + maxcor + 1, dtype=dtype),
          state_initial.gamma)
      )

    def cond_fun(args):
        state, history = args
        return (~(state.converged)) & (~(state.failed))

    def body_fun(args):
        state, history = args
        # find search direction
        p_k = _two_loop_recursion(state)

        # line search
        ls_results = line_search(
          f=fun,
          xk=state.x_k,
          pk=p_k,
          old_fval=state.f_k,
          gfk=state.g_k,
          maxiter=maxls,
        )

        # evaluate at next iterate
        s_k = ls_results.a_k * p_k
        x_kp1 = state.x_k + s_k
        f_kp1 = ls_results.f_k
        g_kp1 = ls_results.g_k
        y_k = g_kp1 - state.g_k
        rho_k_inv = jnp.real(_dot(y_k, s_k))
        rho_k = jnp.reciprocal(rho_k_inv)
        gamma = rho_k_inv / jnp.real(_dot(jnp.conj(y_k), y_k))

        # replacements for next iteration
        status = 0
        status = jnp.where(state.f_k - f_kp1 < ftol, 4, status)
        status = jnp.where(state.ngev >= maxgrad, 3, status)  # type: ignore
        status = jnp.where(state.nfev >= maxfun, 2, status)  # type: ignore
        status = jnp.where(state.k >= maxiter, 1, status)  # type: ignore
        status = jnp.where(ls_results.failed, 5, status)

        converged = jnp.linalg.norm(g_kp1, ord=norm) < gtol

        state = state._replace(
          converged=converged,
          failed=(status > 0) & (~converged),
          k=state.k + 1,
          nfev=state.nfev + ls_results.nfev,
          ngev=state.ngev + ls_results.ngev,
          x_k=x_kp1,
          f_k=f_kp1,
          g_k=g_kp1,
          s_history=_update_history_vectors(history=state.s_history, new=s_k),
          y_history=_update_history_vectors(history=state.y_history, new=y_k),
          rho_history=_update_history_scalars(history=state.rho_history,
                                              new=rho_k),
          gamma=gamma,
          status=jnp.where(converged, 0, status),
          ls_status=ls_results.status,
        )

        history = history._replace(
          x=_update_history_vectors(history=history.x, new=x_kp1),
          f=_update_history_scalars(history=history.f, new=f_kp1),
          g=_update_history_vectors(history=history.g, new=g_kp1),
          gamma=_update_history_scalars(history=history.gamma, new=gamma)
          )

        return state, history

    state, history = lax.while_loop(cond_fun, body_fun,
                                    (state_initial, history_initial))
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

    Q, R = jnp.linalg.qr(jnp.diag(jnp.sqrt(1 / alpha)) @ beta)
    param_dims = beta.shape[0]
    L = jnp.linalg.cholesky(jnp.eye(param_dims) + R @ gamma @ R.T)

    logdet = jnp.log(jnp.prod(alpha)) + 2 * jnp.log(jnp.linalg.det(L))
    mu = (
        position
        + jnp.diag(alpha) @ grad_position
        + beta @ gamma @ beta.T @ grad_position
    )

    u = jax.random.normal(rng_key, (num_samples, param_dims, 1))
    phi = (
        mu[..., :, None]
        + jnp.diag(jnp.sqrt(alpha)) @ (Q @ L @ Q.T @ u + u - (Q @ Q.T @ u))
    )[..., 0]

    logq = -0.5 * (
        logdet + jnp.einsum("mji,mji->m", u, u) + param_dims * jnp.log(2.0 * jnp.pi)
    )
    return phi, logq
