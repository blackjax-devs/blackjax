from typing import Callable, NamedTuple, Tuple, Union

import jax
import jax.numpy as jnp
import jax.random
from jax import lax
from jax.flatten_util import ravel_pytree

from blackjax.optimizers.lbfgs import (
    lbfgs_inverse_hessian_factors,
    lbfgs_sample,
    minimize_lbfgs,
)
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

        sample = sample_from_state(rng_key, state, num_samples=())
        return state, sample

    return one_step


def sample_from_state(
    rng_key: PRNGKey,
    state: PathfinderState,
    num_samples: Union[int, Tuple[()], Tuple[int]] = (),
) -> PyTree:
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
