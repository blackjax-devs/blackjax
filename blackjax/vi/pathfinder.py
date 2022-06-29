from typing import Callable, NamedTuple, Tuple, Union

import jax
import jax.numpy as jnp
import jax.random
from jax.flatten_util import ravel_pytree

from blackjax.optimizers.lbfgs import (
    _minimize_lbfgs,
    bfgs_sample,
    lbfgs_inverse_hessian_factors,
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
    position:
        position
    grad_position:
        gradient of target distribution wrt position
    alpha, beta, gamma:
        factored rappresentation of the inverse hessian
    """

    elbo: Array
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

    Function implements Algorithm 3 in [1]:

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
        kwargs passed to the internal call to lbfgs_minimize, available params:
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

    References
    ----------

    .. [1]: Pathfinder: Parallel quasi-newton variational inference,
            Lu Zhang et al., arXiv:2108.03782
    """

    initial_position_flatten, unravel_fn = ravel_pytree(initial_position)
    objective_fn = lambda x: -logprob_fn(unravel_fn(x))

    if "maxiter" not in lbfgs_kwargs:
        lbfgs_kwargs["maxiter"] = 30
    if "maxcor" not in lbfgs_kwargs:
        lbfgs_kwargs["maxcor"] = 10
    if "maxls" not in lbfgs_kwargs:
        # high max line search steps helps optimizing negative log likelihoods
        # that are sums over (large number of) observations' likelihood
        lbfgs_kwargs["maxls"] = 1000
    if "gtol" not in lbfgs_kwargs:
        lbfgs_kwargs["gtol"] = 1e-08
    if "ftol" not in lbfgs_kwargs:
        lbfgs_kwargs["ftol"] = 1e-05

    maxiter = lbfgs_kwargs["maxiter"]
    maxcor = lbfgs_kwargs["maxcor"]
    (_, status), history = _minimize_lbfgs(
        objective_fn,
        initial_position_flatten,
        lbfgs_kwargs["maxiter"],
        lbfgs_kwargs["maxcor"],
        lbfgs_kwargs["gtol"],
        lbfgs_kwargs["ftol"],
        lbfgs_kwargs["maxls"],
    )

    # Get postions and gradients of the optimization path (including the starting point).
    position = history.x
    grad_position = history.g
    alpha = history.alpha
    # Get the update of position and gradient.
    update_mask = history.update_mask[1:]
    s = jnp.diff(position, axis=0)
    z = jnp.diff(grad_position, axis=0)
    # Account for the mask
    s_masked = jnp.where(update_mask, s, jnp.zeros_like(s))
    z_masked = jnp.where(update_mask, z, jnp.zeros_like(z))
    # Pad 0 to leading dimension so we have constant shape output
    s_padded = jnp.pad(s_masked, ((maxcor, 0), (0, 0)), mode="constant")
    z_padded = jnp.pad(z_masked, ((maxcor, 0), (0, 0)), mode="constant")

    def path_finder_body_fn(rng_key, S, Z, alpha_l, theta, theta_grad):
        """The for loop body in Algorithm 1 of the Pathfinder paper."""
        beta, gamma = lbfgs_inverse_hessian_factors(S.T, Z.T, alpha_l)
        phi, logq = bfgs_sample(
            rng_key=rng_key,
            num_samples=num_samples,
            position=theta,
            grad_position=theta_grad,
            alpha=alpha_l,
            beta=beta,
            gamma=gamma,
        )
        logp = -jax.vmap(objective_fn)(phi)
        elbo = (logp - logq).mean()  # Algorithm 7 of the paper
        return elbo, beta, gamma

    # Index and reshape S and Z to be sliding window view shape=(maxiter, maxcor, param_dim),
    # so we can vmap over all the iterations.
    # This is in effect numpy.lib.stride_tricks.sliding_window_view
    path_size = maxiter + 1
    index = jnp.arange(path_size)[:, None] + jnp.arange(maxcor)[None, :]
    s_j = s_padded[index.reshape(path_size, maxcor)].reshape(path_size, maxcor, -1)
    z_j = z_padded[index.reshape(path_size, maxcor)].reshape(path_size, maxcor, -1)
    rng_keys = jax.random.split(rng_key, path_size)
    elbo, beta, gamma = jax.vmap(path_finder_body_fn)(
        rng_keys, s_j, z_j, alpha, position, grad_position
    )
    elbo = jnp.where(
        (jnp.arange(path_size) < (status.iter_num)) & jnp.isfinite(elbo),
        elbo,
        -jnp.inf,
    )

    unravel_fn_mapped = jax.vmap(unravel_fn)
    pathfinder_result = PathfinderState(
        elbo=elbo,
        position=unravel_fn_mapped(position),
        grad_position=unravel_fn_mapped(grad_position),
        alpha=alpha,
        beta=beta,
        gamma=gamma,
    )

    if return_path:
        return pathfinder_result
    else:
        best_i = jnp.argmax(elbo)
        return jax.tree_map(lambda x: x[best_i], pathfinder_result)


def kernel():
    """
    Builds a pathfinder kernel.

    Returns:
    a kernel that takes rng_key and the pathfinder state and returns
    the pathfinder state and a draw from the approximate distribution
    """

    def one_step(rng_key: PRNGKey, state: PathfinderState):

        sample, _ = sample_from_state(rng_key, state, num_samples=())
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

    phi, logq = bfgs_sample(
        rng_key,
        num_samples,
        position_flatten,
        grad_position_flatten,
        state.alpha,
        state.beta,
        state.gamma,
    )

    if num_samples == ():
        return unravel_fn(phi), logq
    else:
        return jax.vmap(unravel_fn)(phi), logq
