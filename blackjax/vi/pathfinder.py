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

__all__ = ["PathfinderState", "approximate", "sample"]


class PathfinderState(NamedTuple):
    """State of the Pathfinder algorithm.

    Pathfinder locates normal approximations to the target density along a
    quasi-Newton optimization path, with local covariance estimated using
    the inverse Hessian estimates produced by the L-BFGS optimizer.
    PathfinderState stores for an interation fo the L-BFGS optimizer the
    resulting ELBO and all factors needed to sample from the approximated
    target density.

    position:
        position
    grad_position:
        gradient of target distribution wrt position
    alpha, beta, gamma:
        factored rappresentation of the inverse hessian
    elbo:
        ELBO of approximation wrt target distribution

    """

    elbo: Array
    position: PyTree
    grad_position: PyTree
    alpha: Array
    beta: Array
    gamma: Array


class PathfinderInfo(NamedTuple):
    """Extra information returned by the Pathfinder algorithm."""

    path: PathfinderState


def approximate(
    rng_key: PRNGKey,
    logprob_fn: Callable,
    initial_position: PyTree,
    num_samples: int = 200,
    *,  # lgbfs parameters
    maxiter=30,
    maxcor=10,
    maxls=1000,
    gtol=1e-08,
    ftol=1e-05,
) -> Tuple[PathfinderState, PathfinderInfo]:
    """Pathfinder variational inference algorithm.

    Pathfinder locates normal approximations to the target density along a
    quasi-Newton optimization path, with local covariance estimated using
    the inverse Hessian estimates produced by the L-BFGS optimizer.

    Function implements the algorithm 3 in [1]:

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
    maxiter
        Maximum number of iterations of the LGBFS algorithm.
    maxcor
        Maximum number of metric corrections of the LGBFS algorithm ("history
        size")
    ftol
        The LGBFS algorithm terminates the minimization when `(f_k - f_{k+1}) <
        ftol`
    gtol
        The LGBFS algorithm terminates the minimization when `|g_k|_norm < gtol`
    maxls
        The maximum number of line search steps (per iteration) for the LGBFS
        algorithm

    Returns
    -------
    A PathfinderState with information on the iteration in the optimization path
    whose approximate samples yields the highest ELBO, and PathfinderInfo that
    contains all the states traversed.

    References
    ----------

    .. [1]: Pathfinder: Parallel quasi-newton variational inference,
            Lu Zhang et al., arXiv:2108.03782

    """
    initial_position_flatten, unravel_fn = ravel_pytree(initial_position)
    objective_fn = lambda x: -logprob_fn(unravel_fn(x))

    (_, status), history = _minimize_lbfgs(
        objective_fn,
        initial_position_flatten,
        maxiter,
        maxcor,
        gtol,
        ftol,
        maxls,
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

    # Index and reshape S and Z to be sliding window view shape=(maxiter,
    # maxcor, param_dim), so we can vmap over all the iterations.
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
        elbo,
        unravel_fn_mapped(position),
        unravel_fn_mapped(grad_position),
        alpha,
        beta,
        gamma,
    )

    max_elbo_idx = jnp.argmax(elbo)
    return jax.tree_map(lambda x: x[max_elbo_idx], pathfinder_result), PathfinderInfo(
        pathfinder_result
    )


def sample(
    rng_key: PRNGKey,
    state: PathfinderState,
    num_samples: Union[int, Tuple[()], Tuple[int]] = (),
) -> PyTree:
    """Draw from the Pathfinder approximation of the target distribution.

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
    Samples drawn from the approximate Pathfinder distribution

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
