from pdb import set_trace
from typing import NamedTuple, Callable, Tuple
import jax
import jax.random
import jax.numpy as jnp
from jax import lax
from jax.flatten_util import ravel_pytree
import jaxopt
from jaxopt._src.lbfgs import compute_gamma
from jaxopt.base import OptStep

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
        kwargs passed to the internal call to lbfgs_minimize, avaiable params:
            maxiter: maximum number of iterations
            maxcor: maximum number of metric corrections ("history size")
            ftol: terminates the minimization when `(f_k - f_{k+1}) < ftol`
            gtol: terminates the minimization when `|g_k|_norm < gtol`
            maxls: maximum number of line search steps (per iteration)

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
        lbfgs_kwargs["maxiter"] = 30
    if "maxcor" not in lbfgs_kwargs:
        lbfgs_kwargs["maxcor"] = 10
    if "maxls" not in lbfgs_kwargs:
        # high max line search steps helps optimizing negative log likelihoods
        # that are sums over (large number of) observations' likelihood
        lbfgs_kwargs["maxls"] = 1000

    (result, status), history = minimize_lbfgs(
        objective_fn, initial_position_flatten, **lbfgs_kwargs
    )

    position, grad_position, alpha_scalar = history.x, history.g, history.gamma

    # set difference between empty zero states and initial point x0 to zero
    # this is beacuse here we are working with static shapes and keeping all
    # the zero states in the arrays
    s = jnp.diff(position, axis=0).at[-status.iter_num - 1].set(0.0)
    z = jnp.diff(grad_position, axis=0).at[-status.iter_num - 1].set(0.0)

    maxiter = lbfgs_kwargs["maxiter"]
    maxcor = lbfgs_kwargs["maxcor"]

    rng_keys = jax.random.split(rng_key, maxiter)

    def pathfinder_inner_step(i):

        i_offset = position.shape[0] + i - 2 - maxiter
        S = lax.dynamic_slice(s, (i_offset - maxcor, 0 + 1), (maxcor, param_dims)).T
        Z = lax.dynamic_slice(z, (i_offset - maxcor, 0 + 1), (maxcor, param_dims)).T
        alpha = alpha_scalar[i_offset + 1] * jnp.ones(param_dims)
        beta, gamma = lbfgs_inverse_hessian_factors(S, Z, alpha)

        phi, logq = lbfgs_sample(
            rng_key=rng_keys[i],
            num_samples=num_samples,
            position=position[i_offset + 1],
            grad_position=grad_position[i_offset + 1],
            alpha=alpha,
            beta=beta,
            gamma=gamma,
        )

        logp = -jax.lax.map(objective_fn, phi)  # cannot vmap pytrees
        elbo = (logp - logq).mean()  # algorithm of figure 9 of the paper

        state = PathfinderState(
            i=jnp.where(
                i - maxiter + status.iter_num < 0, -1, i - maxiter + status.iter_num
            ),
            elbo=elbo,
            position=unravel_fn(position[i_offset + 1]),
            grad_position=unravel_fn(grad_position[i_offset + 1]),
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
        # sample is a pyteee with leaf arrays' shape (1, param_dims)
        # we use tree_map and indexing to remove the leading dimension
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
