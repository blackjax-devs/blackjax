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

import logging
import warnings
from enum import IntEnum
from typing import Callable, Literal, NamedTuple, Optional

import arviz as az
import jax
import jax.numpy as jnp
import jax.random
import numpy as np
from jax.flatten_util import ravel_pytree
from jax.scipy.special import logsumexp

from blackjax.optimizers.lbfgs import (
    bfgs_sample,
    lbfgs_diff_history_matrix,
    lbfgs_inverse_hessian_factors,
    lbfgs_recover_alpha,
    minimize_lbfgs,
)
from blackjax.types import Array, PRNGKey

logger = logging.getLogger(__name__)

__all__ = [
    "MultiPathfinderAlgorithm",
    "multi_pathfinder",
    "as_top_level_api",
]

# TODO: set jnp.arrays to float64


class SinglePathfinderState(NamedTuple):
    """State of the Pathfinder algorithm.

    Pathfinder locates normal approximations to the target density along a
    quasi-Newton optimization path, with local covariance estimated using
    the inverse Hessian estimates produced by the L-BFGS optimizer.
    PathfinderState stores for an iteration of the L-BFGS optimizer the
    resulting ELBO and all factors needed to sample from the approximated
    target density.

    Parameters
    ----------
    initial_position : Array
        initial position for the optimization
    position : Array, optional
        current position in parameter space
    grad_position : Array, optional
        gradient of target distribution at current position
    alpha : Array, optional
        first factor of the inverse hessian representation
    beta : Array, optional
        second factor of the inverse hessian representation
    gamma : Array, optional
        third factor of the inverse hessian representation
    elbo : Array, optional
        evidence lower bound of approximation to target distribution
    sparse : bool
        whether to use sparse representation of the inverse hessian
    """

    initial_position: Array
    position: Optional[Array] = None
    grad_position: Optional[Array] = None
    alpha: Optional[Array] = None
    beta: Optional[Array] = None
    gamma: Optional[Array] = None
    elbo: Optional[Array] = None
    sparse: bool = False


class SinglePathfinderInfo(NamedTuple):
    """Extra information returned by the Pathfinder algorithm."""

    path: SinglePathfinderState
    update_mask: Array


class ImpSamplingMethod(IntEnum):
    PSIS = 0
    PSIR = 1
    IDENTITY = 2
    NONE = 3


class ImportanceSamplingState(NamedTuple):
    """Container for importance sampling results.

    This class stores the results of importance sampling from multiple Pathfinder
    approximations, including diagnostic information about the quality of the
    importance sampling.

    Parameters
    ----------
    num_paths : int
        number of paths used in multi-pathfinder
    samples : Array
        importance sampled draws from the approximate distribution
    pareto_k : float, optional
        Pareto k diagnostic value from importance sampling
    method : str, optional
        importance sampling method used
    """

    num_paths: int
    samples: Array
    pareto_k: Optional[float] = None
    method: ImpSamplingMethod = ImpSamplingMethod.PSIS


class MultiPathfinderAlgorithm(NamedTuple):
    init: Callable
    pathfinder: Callable
    logp: Callable
    importance_sampling: Callable


def init(
    rng_key: Optional[PRNGKey] = None,
    base_position: Optional[Array] = None,
    num_paths: Optional[int] = None,
    jitter_amount: Optional[float] = None,
    initial_position: Optional[Array] = None,
) -> Array:
    """Initialize positions for multi-pathfinder.

    This function either returns the provided initial positions or generates them
    by adding jitter to a base position.

    Parameters
    ----------
    rng_key : PRNGKey, optional
        JAX PRNG key for generating jittered positions
    base_position : Array, optional
        Unflattened base position to jitter around
    num_paths : int, optional
        Number of paths to generate
    jitter_amount : float, optional
        Scale of the uniform jitter to add to the base position
    initial_position : Array, optional
        Pre-specified initial positions to use.

    Returns
    -------
    Array
        Initial positions for multi-pathfinder
    """

    if num_paths is None:
        raise ValueError("num_paths must be provided")

    if initial_position is not None:
        # Check if initial_position is a single position or multiple positions
        batch_size = jax.tree.leaves(initial_position)[0].shape[0]
        if initial_position.ndim == 1:
            logger.warning(
                "Initial position is a single position, repeating for all paths. "
                "This is likely to lead to poor performance. "
                "Consider providing a batch of initial positions, or use base_position and jitter_amount."
            )
            return jax.tree.map(
                lambda x: jnp.repeat(x[jnp.newaxis, ...], num_paths, axis=0),
                initial_position,
            )
        else:
            if num_paths == batch_size:
                return initial_position
            else:
                raise ValueError(
                    f"num_paths ({num_paths}) must match batch_size ({batch_size})."
                )

    if base_position is None:
        raise ValueError(
            "base_position must be provided if initial_position is not provided"
        )
    if jitter_amount is None:
        raise ValueError(
            "jitter_amount must be provided if initial_position is not provided"
        )
    if rng_key is None:
        raise ValueError("rng_key must be provided if initial_position is not provided")

    # Generate jittered positions from base position
    base_position_flatten, unravel_fn = ravel_pytree(base_position)

    jitter_value = jax.random.uniform(
        rng_key,
        shape=(num_paths, base_position_flatten.shape[0]),
        minval=-jitter_amount,
        maxval=jitter_amount,
    )
    jittered_positions = base_position_flatten + jitter_value

    return jax.vmap(unravel_fn)(jittered_positions)


def approximate(
    rng_key: PRNGKey,
    logdensity_fn: Callable,
    initial_position: Array,
    num_elbo_draws: int = 15,
    maxcor: int = 6,
    maxiter: int = 100,
    maxls: int = 100,
    gtol: float = 1e-08,
    ftol: float = 1e-05,
    epsilon: float = 1e-8,
    **lbfgs_kwargs,
) -> tuple[SinglePathfinderState, SinglePathfinderInfo]:
    """Pathfinder variational inference algorithm.

    Pathfinder locates normal approximations to the target density along a
    quasi-Newton optimization path, with local covariance estimated using
    the inverse Hessian estimates produced by the L-BFGS optimizer.

    Function implements the algorithm 3 in :cite:p:`zhang2022pathfinder`:

    Parameters
    ----------
    rng_key : PRNGKey
        JAX PRNG key
    logdensity_fn : Callable
        (un-normalized) log density function of target distribution to take
        approximate samples from
    initial_position : Array
        starting point of the L-BFGS optimization routine
    maxcor : int
        Maximum number of metric corrections of the LGBFS algorithm ("history
        size")
    num_elbo_draws : int
        number of samples to draw to estimate ELBO
    maxiter : int
        Maximum number of iterations of the LGBFS algorithm.
    ftol : float
        The LGBFS algorithm terminates the minimization when `(f_k - f_{k+1}) <
        ftol`
    gtol : float
        The LGBFS algorithm terminates the minimization when `|g_k|_norm < gtol`
    maxls : int
        The maximum number of line search steps (per iteration) for the LGBFS
        algorithm
    **lbfgs_kwargs
        other keyword arguments passed to `jaxopt.LBFGS`.


    Returns
    -------
    A PathfinderState with information on the iteration in the optimization path
    whose approximate samples yields the highest ELBO, and PathfinderInfo that
    contains all the states traversed.

    """

    initial_position_flatten, unravel_fn = ravel_pytree(initial_position)
    objective_fn = lambda x: -logdensity_fn(unravel_fn(x))

    (_, _), history = minimize_lbfgs(
        objective_fn,
        initial_position_flatten,
        maxiter,
        maxcor,
        gtol,
        ftol,
        maxls,
        **lbfgs_kwargs,
    )

    not_converged_mask = jnp.logical_not(history.converged.at[1:].get())

    # jax.jit would not work with truncated history, so we keep the full history
    position = history.x
    grad_position = history.g

    alpha, s, z, update_mask = lbfgs_recover_alpha(
        position, grad_position, not_converged_mask, epsilon
    )

    s = jnp.diff(position, axis=0)
    z = jnp.diff(grad_position, axis=0)
    S = lbfgs_diff_history_matrix(s, update_mask, maxcor)
    Z = lbfgs_diff_history_matrix(z, update_mask, maxcor)

    position = position.at[1:].get()
    grad_position = grad_position.at[1:].get()

    param_dims = position.shape[-1]
    sparse = param_dims < 2 * maxcor

    def pathfinder_body_fn(
        rng_key, update_mask_l, S_l, Z_l, alpha_l, theta, theta_grad
    ):
        """The for loop body in Algorithm 1 of the Pathfinder paper."""

        def _pathfinder_body_fn(args):
            beta, gamma = lbfgs_inverse_hessian_factors(S_l, Z_l, alpha_l)
            phi, logq = bfgs_sample(
                rng_key=rng_key,
                num_samples=num_elbo_draws,
                position=theta,
                grad_position=theta_grad,
                alpha=alpha_l,
                beta=beta,
                gamma=gamma,
                sparse=sparse,
            )
            logp = jax.vmap(logdensity_fn)(phi)
            logp = jnp.where(~jnp.isfinite(logp), jnp.inf, logp)
            elbo = jnp.mean(logp - logq)
            return elbo, beta, gamma

        def _nan_op(args):
            elbo = jnp.asarray(jnp.nan, dtype=jnp.float64)
            beta = jnp.ones((param_dims, 2 * maxcor), dtype=jnp.float64) * jnp.nan
            gamma = jnp.ones((2 * maxcor, 2 * maxcor), dtype=jnp.float64) * jnp.nan
            return elbo, beta, gamma

        args = (rng_key, S_l, Z_l, alpha_l, theta, theta_grad)
        return jax.lax.cond(update_mask_l, _pathfinder_body_fn, _nan_op, args)

    rng_keys = jax.random.split(rng_key, maxiter)
    elbo, beta, gamma = jax.vmap(pathfinder_body_fn)(
        rng_keys, update_mask, S, Z, alpha, position, grad_position
    )

    unravel_fn_mapped = jax.vmap(unravel_fn)
    res_argmax = (
        unravel_fn_mapped(position),
        unravel_fn_mapped(grad_position),
        alpha,
        beta,
        gamma,
        elbo,
    )

    max_elbo_idx = jnp.nanargmax(elbo)

    # keep all of PathfinderInfo, including masked info to make approximate jittable.
    return SinglePathfinderState(
        initial_position,
        *jax.tree.map(lambda x: x.at[max_elbo_idx].get(), res_argmax),
        sparse,
    ), SinglePathfinderInfo(
        SinglePathfinderState(initial_position, *res_argmax, sparse), update_mask
    )


def sample(
    rng_key: PRNGKey,
    state: SinglePathfinderState,
    num_samples_per_path: int = 1000,
) -> tuple[Array, Array]:
    """Draw from the Pathfinder approximation of the target distribution.

    Parameters
    ----------
    rng_key : PRNGKey
        JAX PRNG key
    state : PathfinderState
        PathfinderState containing information for sampling
    num_samples_per_path : int
        Number of samples to draw

    Returns
    -------
    tuple
        Samples drawn from the approximate Pathfinder distribution and their log probabilities

    Raises
    ------
    ValueError
        If the state contains invalid values (NaN or Inf) or is not properly initialized
    """

    position_flatten, unravel_fn = ravel_pytree(state.position)
    grad_position_flatten, _ = ravel_pytree(state.grad_position)

    psi, logq = bfgs_sample(
        rng_key,
        num_samples_per_path,
        position_flatten,
        grad_position_flatten,
        state.alpha,
        state.beta,
        state.gamma,
        state.sparse,
    )

    return jax.vmap(unravel_fn)(psi), logq


def logp(logdensity_fn: Callable, samples: Array):
    return logdensity_fn(samples)


def importance_sampling(
    rng_key: PRNGKey,
    samples: Array,
    logP: Array,
    logQ: Array,
    num_paths: int,
    num_samples: int = 1000,
    method: int = 0,
) -> ImportanceSamplingState:
    """Pareto Smoothed Importance Resampling (PSIR)

    This implements the Pareto Smooth Importance Resampling (PSIR) method, as described in
    Algorithm 5 of Zhang et al. (2022). The PSIR follows a similar approach to Algorithm 1
    PSIS diagnostic from Yao et al., (2018). However, before computing the importance ratio r_s,
    the logP and logQ are adjusted to account for the number of multiple estimators (or paths).
    The process involves resampling from the original sample with replacement, with probabilities
    proportional to the computed importance weights from PSIS.

    Parameters
    ----------
    rng_key : PRNGKey
        JAX random key for sampling
    logdensity_fn : Callable
        log density function of the target distribution
    samples : Array
        samples from proposal distribution, shape (L, M, N)
    logQ : Array
        log probability values of proposal distribution, shape (L, M)
    num_paths : int
        number of paths used in multi-pathfinder
    num_samples : int
        number of draws to return where num_samples <= samples.shape[0]
    method : str, optional
        importance sampling method to use. Options are "psis" (default), "psir", "identity", None.
        Pareto Smoothed Importance Sampling (psis) is recommended in many cases for more stable
        results than Pareto Smoothed Importance Resampling (psir). identity applies the log
        importance weights directly without resampling. None applies no importance sampling
        weights and returns the samples as is.

    Returns
    -------
    ImportanceSamplingState
        importance sampled draws and other info based on the specified method
    """

    # TODO: make this function jax.jit compatible

    if num_paths != samples.shape[0]:
        raise ValueError(
            f"num_paths ({num_paths}) must be equal to the number of rows in samples ({samples.shape[0]})"
        )

    if samples.ndim != 3:
        raise ValueError(f"Samples must be a 3D array, got shape {samples.shape}")

    batch_shape, param_shape = samples.shape[:2], samples.shape[2:]
    batch_size = np.prod(batch_shape).item()

    if method == ImpSamplingMethod.NONE:
        logger.warning("No importance sampling method specified, returning raw samples")
        return ImportanceSamplingState(
            num_paths=num_paths, samples=samples, method=method
        )
    else:
        log_I = jnp.log(num_paths)
        logP = logP.ravel() - log_I
        logP = jnp.where(~jnp.isfinite(logP), -jnp.inf, logP)
        samples = samples.reshape(batch_size, *param_shape)

        logQ = logQ.ravel() - log_I

        logiw = logP - logQ

        def psislw_wrapper(logiw_array):
            def psislw(logiw_array):
                result_logiw, result_k = az.psislw(np.array(logiw_array))
                return np.array(result_logiw), np.array(result_k)

            return jax.pure_callback(
                psislw,
                (jnp.zeros_like(logiw_array), jnp.zeros((), dtype=jnp.float64)),
                logiw_array,
            )

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", category=RuntimeWarning, message="overflow encountered in exp"
            )
            match method:
                case ImpSamplingMethod.PSIS:
                    replace = False
                    logiw, pareto_k = psislw_wrapper(logiw)
                case ImpSamplingMethod.PSIR:
                    replace = True
                    logiw, pareto_k = psislw_wrapper(logiw)
                case ImpSamplingMethod.IDENTITY:
                    replace = False
                    pareto_k = None
                    logger.info("Identity importance sampling (no smoothing)")
                case _:
                    raise ValueError(f"Invalid importance sampling method: {method}. ")

    # NOTE: Pareto k is normally bad for Pathfinder even when the posterior is close to the NUTS posterior or closer to NUTS than ADVI.
    # Pareto k may not be a good diagnostic for Pathfinder.
    p = jnp.exp(logiw - logsumexp(logiw))

    non_infinite = ~jnp.isfinite(p)

    def handle_non_infinite(p, non_infinite):
        logger.warning(
            "Detected NaN or Inf values in importance weights. "
            "This may indicate numerical instability in the target or proposal distributions."
        )
        p = jnp.where(non_infinite, 0.0, p)
        return p / jnp.sum(p)

    p = jax.lax.cond(
        jnp.any(non_infinite), handle_non_infinite, lambda p, _: p, p, non_infinite
    )

    try:
        resampled = jax.random.choice(
            rng_key,
            samples,
            shape=(num_samples,),
            replace=replace,
            p=p,
            axis=0,
        )
        return ImportanceSamplingState(
            num_paths=num_paths,
            samples=resampled,
            pareto_k=pareto_k,
            method=method,
        )
    except ValueError as e1:
        if "Cannot take a larger sample" in str(e1):
            num_nonzero = jnp.sum(p > 0)
            logger.warning(
                f"Not enough valid samples: {num_nonzero} available out of {num_samples} requested. "
                f"Switching to psir importance sampling with replacement."
            )
            try:
                resampled = jax.random.choice(
                    rng_key,
                    samples,
                    shape=(num_samples,),
                    replace=True,
                    p=p,
                    axis=0,
                )
                return ImportanceSamplingState(
                    num_paths=num_paths,
                    samples=resampled,
                    pareto_k=pareto_k,
                    method="psir",
                )
            except ValueError as e2:
                logger.error(f"Importance sampling failed: {str(e2)}")
                raise ValueError(
                    "Importance sampling failed for both with and without replacement. "
                )


def pathfinder(
    rng_key: PRNGKey,
    logdensity_fn: Callable,
    initial_position: Array,
    num_samples_per_path: int = 1000,
    num_elbo_draws: int = 15,
    maxcor: int = 6,
    maxiter: int = 100,
    maxls: int = 100,
    gtol: float = 1e-08,
    ftol: float = 1e-05,
    epsilon: float = 1e-8,
    **lbfgs_kwargs,
):
    approx_key, sample_key = jax.random.split(rng_key, 2)

    state, _ = approximate(
        rng_key=approx_key,
        logdensity_fn=logdensity_fn,
        initial_position=initial_position,
        num_elbo_draws=num_elbo_draws,
        maxcor=maxcor,
        maxiter=maxiter,
        maxls=maxls,
        gtol=gtol,
        ftol=ftol,
        epsilon=epsilon,
        **lbfgs_kwargs,
    )

    samples, logq = sample(
        rng_key=sample_key, state=state, num_samples_per_path=num_samples_per_path
    )

    return samples, logq


def _shape_handler_for_parallel(
    batch_shape: tuple,
    parallel_method: Literal["parallel", "vectorize"] = "parallel",
):
    batch_size = np.prod(batch_shape).item()
    if parallel_method == "vectorize":
        return batch_shape
    elif parallel_method == "parallel":
        num_devices = len(jax.devices())
        if num_devices >= batch_size:
            return batch_shape
        elif batch_size % num_devices == 0:
            return (num_devices, batch_size // num_devices)
        else:
            raise ValueError(
                f"The batch size must be divisible by the number of devices ({num_devices}). Received batch size ({batch_size}) from the batch shape ({batch_shape})."
            )
    else:
        raise ValueError(f"Unsupported parallel method: {parallel_method}")


def multi_pathfinder(
    rng_key: PRNGKey,
    logdensity_fn: Callable,
    base_position: Optional[Array] = None,
    jitter_amount: Optional[float] = None,
    num_paths: int = None,
    num_samples: int = 1000,
    num_samples_per_path: int = 1000,
    num_elbo_draws: int = 15,
    maxcor: int = 6,
    maxiter: int = 100,
    maxls: int = 100,
    gtol: float = 1e-08,
    ftol: float = 1e-05,
    epsilon: float = 1e-8,
    importance_sampling_method: Literal["psis", "psir", "identity"] = "psis",
    initial_position: Optional[Array] = None,
    parallel_method: Literal["parallel", "vectorize"] = "parallel",
    **lbfgs_kwargs,
) -> ImportanceSamplingState:
    """Run the multi-pathfinder algorithm.

    This function runs multiple instances of Pathfinder in parallel and combines
    the results using importance sampling.

    Parameters
    ----------
    rng_key : PRNGKey
        JAX PRNG key
    logdensity_fn : Callable
        log density function of the target distribution
    base_position : Array, optional
        Unflattened base position to jitter around
    jitter_amount : float, optional
        scale of the uniform jitter to add to the base position
    num_paths : int
        number of parallel Pathfinder instances to run
    num_samples : int
        number of final draws to return after importance sampling
    num_samples_per_path : int
        number of samples to draw from each Pathfinder instance
    num_elbo_draws : int
        number of samples to draw for ELBO estimation
    maxcor : int
        maximum number of metric corrections for L-BFGS
    maxiter : int
        maximum number of iterations for L-BFGS
    maxls : int
        maximum number of line search steps for L-BFGS
    gtol : float
        gradient tolerance for L-BFGS
    ftol : float
        function value tolerance for L-BFGS
    importance_sampling_method : str
        importance sampling method to use
    initial_position : Array, optional
        pre-specified initial positions to use

    Returns
    -------
    ImportanceSamplingState
        Result of importance sampling

    Raises
    ------
    ValueError
        If the inputs are inconsistent or insufficient
    """

    path_batch_shape = _shape_handler_for_parallel(
        (num_paths,), parallel_method=parallel_method
    )
    logp_batch_shape = _shape_handler_for_parallel(
        (num_paths, num_samples_per_path), parallel_method=parallel_method
    )

    # Split the random key for initialization and sampling
    init_key, rng_path_key, choice_key = jax.random.split(rng_key, 3)

    # Create the multi-pathfinder algorithm
    multi_pathfinder = as_top_level_api(
        logdensity_fn=logdensity_fn,
        num_paths=num_paths,
        num_samples=num_samples,
        num_samples_per_path=num_samples_per_path,
        num_elbo_draws=num_elbo_draws,
        maxcor=maxcor,
        maxiter=maxiter,
        maxls=maxls,
        gtol=gtol,
        ftol=ftol,
        epsilon=epsilon,
        importance_sampling_method=importance_sampling_method,
        **lbfgs_kwargs,
    )

    # Initialize the positions
    initial_positions = multi_pathfinder.init(
        rng_key=init_key,
        base_position=base_position,
        jitter_amount=jitter_amount,
        initial_position=initial_position,
    )

    param_shape = initial_positions.shape[1:]

    path_keys = jax.random.split(rng_path_key, path_batch_shape)

    path_batch_size = np.prod(path_batch_shape).item()

    if parallel_method == "vectorize":
        pathfinder_pmap = jax.jit(jax.vmap(multi_pathfinder.pathfinder))
    else:  # parallel_method == "parallel"
        num_devices = len(jax.devices())
        if num_devices >= path_batch_size:
            pathfinder_pmap = jax.pmap(multi_pathfinder.pathfinder)
        elif path_batch_size % num_devices == 0:
            initial_positions = initial_positions.reshape(
                (*path_batch_shape, *param_shape)
            )
            initial_positions = jax.pmap(jax.vmap(lambda x: x))(initial_positions)

            path_keys = jax.pmap(jax.vmap(lambda r: r))(path_keys)

            pathfinder_pmap = jax.pmap(jax.vmap(multi_pathfinder.pathfinder))
        else:
            raise ValueError(
                f"The batch size must be divisible by the number of devices ({num_devices}). Received batch size ({path_batch_size}) from the batch shape ({path_batch_shape})."
            )

    # Run Pathfinder on each path
    samples, logq = pathfinder_pmap(
        rng_key=path_keys, initial_position=initial_positions
    )

    logp_batch_size = np.prod(logp_batch_shape).item()

    if parallel_method == "vectorize":
        logp_pmap = jax.jit(jax.vmap(multi_pathfinder.logp))
        samples = samples.reshape((-1, *param_shape))
    else:  # parallel_method == "parallel"
        if num_devices >= logp_batch_size:
            logp_pmap = jax.pmap(multi_pathfinder.logp)
        elif logp_batch_size % num_devices == 0:
            new_batch_shape = _shape_handler_for_parallel(
                logp_batch_shape,
            )

            samples = samples.reshape((*new_batch_shape, *param_shape))
            samples = jax.pmap(jax.vmap(lambda x: x))(samples)
            logp_pmap = jax.pmap(jax.vmap(multi_pathfinder.logp))
        else:
            raise ValueError(
                f"The batch size must be divisible by the number of devices ({num_devices}). Received batch size ({logp_batch_size}) from the batch shape ({logp_batch_shape})."
            )

    logp = logp_pmap(samples)
    samples = samples.reshape((num_paths, num_samples_per_path, *param_shape))

    # Perform importance sampling
    result = jax.jit(multi_pathfinder.importance_sampling)(
        rng_key=choice_key,
        samples=samples,
        logP=logp,
        logQ=logq,
    )

    return result


def as_top_level_api(
    logdensity_fn: Callable,
    num_paths: int,
    num_samples: int = 1000,
    num_samples_per_path: int = 1000,
    num_elbo_draws: int = 15,
    maxcor: int = 6,
    maxiter: int = 100,
    maxls: int = 100,
    gtol: float = 1e-08,
    ftol: float = 1e-05,
    epsilon: float = 1e-8,
    importance_sampling_method: Literal["psis", "psir", "identity"] | None = "psis",
    **lbfgs_kwargs,
) -> MultiPathfinderAlgorithm:
    """Implements the (basic) user interface for the pathfinder kernel.

    Pathfinder locates normal approximations to the target density along a
    quasi-Newton optimization path, with local covariance estimated using
    the inverse Hessian estimates produced by the L-BFGS optimizer.
    Pathfinder returns draws from the approximation with the lowest estimated
    Kullback-Leibler (KL) divergence to the true posterior.

    Note: all the heavy processing in performed in the init function, step
    function is just a drawing a sample from a normal distribution

    Parameters
    ----------
    logdensity_fn
        A function that represents the log-density of the model we want
        to sample from.

    Returns
    -------
    A ``VISamplingAlgorithm``.

    """

    valid_isamp = {"psis", "psir", "identity", None}
    if importance_sampling_method not in valid_isamp:
        raise ValueError(
            f"Invalid importance sampling method: {importance_sampling_method}. "
        )

    # jittable
    def init_fn(
        rng_key: Optional[PRNGKey] = None,
        base_position: Optional[Array] = None,
        jitter_amount: float = None,
        initial_position: Optional[Array] = None,
    ):
        return init(
            rng_key=rng_key,
            base_position=base_position,
            num_paths=num_paths,
            jitter_amount=jitter_amount,
            initial_position=initial_position,
        )

    # jittable
    def pathfinder_fn(rng_key: PRNGKey, initial_position: Array):
        return pathfinder(
            rng_key=rng_key,
            logdensity_fn=logdensity_fn,
            initial_position=initial_position,
            num_samples_per_path=num_samples_per_path,
            num_elbo_draws=num_elbo_draws,
            maxcor=maxcor,
            maxiter=maxiter,
            maxls=maxls,
            gtol=gtol,
            ftol=ftol,
            epsilon=epsilon,
            **lbfgs_kwargs,
        )

    # jittable
    def logp_fn(
        samples: Array,
    ):
        return logp(logdensity_fn=logdensity_fn, samples=samples)

    # jittable
    def importance_sampling_fn(
        rng_key: PRNGKey,
        samples: Array,
        logP: Array,
        logQ: Array,
    ) -> ImportanceSamplingState:
        # Inside the function, convert the enum back to the string if needed
        method = {
            "psis": ImpSamplingMethod.PSIS,
            "psir": ImpSamplingMethod.PSIR,
            "identity": ImpSamplingMethod.IDENTITY,
            None: ImpSamplingMethod.NONE,
        }.get(importance_sampling_method, ImpSamplingMethod.PSIS)
        return importance_sampling(
            rng_key=rng_key,
            samples=samples,
            logP=logP,
            logQ=logQ,
            num_paths=num_paths,
            num_samples=num_samples,
            method=method,
        )

    return MultiPathfinderAlgorithm(
        init_fn, pathfinder_fn, logp_fn, importance_sampling_fn
    )
