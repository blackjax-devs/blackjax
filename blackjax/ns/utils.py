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
"""Utility functions for Nested Sampling.

This module provides helper functions for common tasks associated with Nested
Sampling, such as calculating log-volumes, log-weights, effective sample sizes,
and post-processing of results.
"""

import functools
from typing import Callable, List, Tuple, Any, Dict  # Added Dict

import jax
import jax.numpy as jnp

from blackjax.ns.base import NSInfo, NSState
from blackjax.types import Array, ArrayTree, PRNGKey
from blackjax.smc.inner_kernel_tuning import StateWithParameterOverride  # Added import


def log1mexp(x: Array) -> Array:
    """Computes log(1 - exp(x)) in a numerically stable way.

    This function implements an algorithm for computing log(1 - exp(x))
    while avoiding precision issues, especially when x is close to 0.
    The implementation chooses between `log(-expm1(x))` and `log1p(-exp(x))`
    based on the value of `x` to maintain accuracy.

    Parameters
    ----------
    x
        Input array or scalar. Values in x should be less than or equal to 0;
        the function returns `jnp.nan` for `x > 0`.

    Returns
    -------
    Array
        The value of log(1 - exp(x)).

    References
    ----------
    .. [1] Mächler, M. (2012). Accurately computing log(1-exp(-|a|)).
           CRAN R project, package Rmpfr, vignette log1mexp-note.pdf.
           https://cran.r-project.org/web/packages/Rmpfr/vignettes/log1mexp-note.pdf
    """
    return jnp.where(
        x > -0.6931472,
        jnp.log(-jnp.expm1(x)),
        jnp.log1p(-jnp.exp(x)),
    )


def compute_nlive(info: NSInfo) -> Array:
    """Compute the effective number of live points at each conceptual death contour.

    In Nested Sampling, especially with batch deletions (k > 1 particle deleted per step),
    the conceptual number of live points, denoted `m*_i` (or `nlive_i`), changes
    with each individual particle `i` considered "dead" within that batch. This
    function calculates this `m*_i` sequence, which is crucial for unbiased
    estimation of prior volume elements when $k>1$.

    The method involves:
    1. Creating "birth" events (+1 count) at `loglikelihood_birth` for each particle
       and "death" events (-1 count) at `loglikelihood` (death likelihood) for each particle.
    2. Sorting all these $2N_{total}$ events primarily by their log-likelihood values.
       Ties can be broken (e.g., births before deaths at same logL), though the
       primary sort key is log-likelihood. The `lexsort` handles this.
    3. Computing the cumulative sum of these +1/-1 counts along the sorted events.
       This sum, at any point, represents the number of particles whose birth
       likelihood is less than or equal to the current event's logL AND whose death
       likelihood is greater than the current event's logL.
    4. For each actual death event in the sorted list, this cumulative sum (adjusted by +1,
       as the particle itself was live just before its conceptual "death" at that logL)
       gives the effective number of live points `m*_i` at that specific death contour.

    Parameters
    ----------
    info
        An `NSInfo` object (or a PyTree with compatible `loglikelihood_birth`
        and `loglikelihood` fields, typically from a concatenated history of
        all dead particles from an NS run).

    Returns
    -------
    Array
        An array where each element `nlive[j]` is the effective number of live
        points `m*_i` when the j-th particle (in the sorted list of conceptual
        death events) was considered "dead". The length of this array is equal
        to the number of particles in `info`.
    """
    birth_logL = info.loglikelihood_birth
    death_logL = info.loglikelihood

    # Create event entries: (logL, type_of_event (+1 for birth, -1 for death))
    birth_events = jnp.column_stack(
        (birth_logL, jnp.ones_like(birth_logL, dtype=jnp.int32))
    )
    death_events = jnp.column_stack(
        (death_logL, -jnp.ones_like(death_logL, dtype=jnp.int32))
    )
    combined = jnp.concatenate([birth_events, death_events], axis=0)
    logL_col = combined[:, 0]
    n_col = combined[:, 1]
    not_nan_sort_key = ~jnp.isnan(logL_col)
    logL_sort_key = logL_col
    n_sort_key = n_col
    sorted_indices = jnp.lexsort((n_sort_key, logL_sort_key, not_nan_sort_key))
    sorted_n_col = n_col[sorted_indices]
    cumsum = jnp.cumsum(sorted_n_col)
    cumsum = jnp.maximum(cumsum, 0)
    death_mask_sorted = sorted_n_col == -1
    nlive = cumsum[death_mask_sorted] + 1

    return nlive


def logX(
    rng_key: PRNGKey, dead_info: NSInfo, shape: int = 100
) -> Tuple[Array, Array]:  # Renamed shape to num_samples
    """Simulate the stochastic evolution of log prior volumes.

    This function estimates the sequence of log prior volumes `log(X_i)` and the
    log prior volume elements `log(dX_i)` associated with each dead particle.
    The input `dead_info` is internally sorted by death log-likelihood before
    processing. For each conceptual dead particle `i` in this sorted sequence,
    the change in log volume is modeled as `delta_logX_i = log(u_i) / m*_i`,
    where `u_i` is a standard uniform random variable and `m*_i` is the
    effective number of live points when particle `i` died (obtained from
    `compute_nlive`).

    Parameters
    ----------
    rng_key
        A JAX PRNG key for generating uniform random variates.
    dead_info
        An `NSInfo` object (or compatible PyTree) containing `loglikelihood_birth`
        and `loglikelihood` for all dead particles accumulated during an NS run.
        This data will be sorted internally by death log-likelihood.
    shape
        The shape of Monte Carlo samples (simulated paths of volume shrinkage)
        to generate for the stochastic log-volume sequence. Default is 100.

    Returns
    -------
    tuple[Array, Array]
        - `logX_cumulative_samples`: An array of shape `(num_dead_particles, num_samples)`
          containing `num_samples` simulated sequences of cumulative log prior
          volumes `log(X_i)`. These are sorted by original death log-likelihood.
        - `log_dX_elements_samples`: An array of shape `(num_dead_particles, num_samples)`
          containing `num_samples` simulated sequences of log prior volume elements
          `log(dX_i)`, where `dX_i = (X_{i-1} - X_{i+1}) / 2` (trapezoidal rule width).
          These are also sorted by original death log-likelihood.
    """
    rng_key, subkey = jax.random.split(rng_key)
    min_val = jnp.finfo(dead_info.loglikelihood.dtype).tiny
    r = jnp.log(
        jax.random.uniform(
            subkey, shape=(dead_info.loglikelihood.shape[0], shape)
        ).clip(min_val, 1 - min_val)
    )

    nlive = compute_nlive(dead_info)
    t = r / nlive[:, jnp.newaxis]
    logX = jnp.cumsum(t, axis=0)

    logXp = jnp.concatenate([jnp.zeros((1, logX.shape[1])), logX[:-1]], axis=0)
    logXm = jnp.concatenate([logX[1:], jnp.full((1, logX.shape[1]), -jnp.inf)], axis=0)
    log_diff = logXm - logXp
    logdX = log1mexp(log_diff) + logXp - jnp.log(2)
    return logX, logdX


def log_weights(
    rng_key: PRNGKey, dead_info: NSInfo, shape: int = 100, beta: float = 1.0
) -> Array:
    """Calculate the log importance weights for Nested Sampling results.

    The importance weight for each dead particle `i` is `w_i = dX_i * L_i^beta`,
    where `dX_i` is the prior volume element associated with the particle and
    `L_i` is its likelihood. This function computes `log(w_i)` using stochastically
    simulated `log(dX_i)` values obtained from `logX`. The input `dead_info`
    is handled internally regarding sorting for `logX`.

    Parameters
    ----------
    rng_key
        A JAX PRNG key for simulating `log(dX_i)` via `logX`.
    dead_info
        An `NSInfo` object (or compatible PyTree) containing `loglikelihood_birth`
        and `loglikelihood` for all dead particles.
    shape
        The shape of Monte Carlo samples to use for simulating `log(dX_i)`.
        Default is 100.
    beta
        The inverse temperature. Typically 1.0 for standard evidence calculation.
        Allows for reweighting to different temperatures.

    Returns
    -------
    Array
        An array of log importance weights, shape `(num_dead_particles, num_samples)`.
        The order of particles corresponds to their original order in `dead_info`.
    """
    sort_indices = jnp.argsort(dead_info.loglikelihood)
    unsort_indices = jnp.empty_like(sort_indices)
    unsort_indices = unsort_indices.at[sort_indices].set(jnp.arange(len(sort_indices)))
    dead_info_sorted = jax.tree.map(lambda x: x[sort_indices], dead_info)
    _, log_dX = logX(rng_key, dead_info_sorted, shape)
    log_w = log_dX + beta * dead_info_sorted.loglikelihood[..., jnp.newaxis]
    return log_w[unsort_indices]


def finalise(
    state: StateWithParameterOverride[NSState, Dict[str, Any]],
    dead_info_history: List[NSInfo],
) -> NSInfo:
    """Combines the history of dead particle information with the final live points.

    At the end of a Nested Sampling run, the remaining live points in the
    `state.sampler_state` are treated as if they were the next set
    of "dead" points. This is done to complete the evidence integral and provide
    a full set of samples for posterior estimation. This function concatenates
    the `NSInfo` objects accumulated for dead particles throughout the run
    (in `dead_info_history`) with a new `NSInfo` object created from these
    final live particles.

    Parameters
    ----------
    state
        The final state of the Nested Sampler, typically a `StateWithParameterOverride`
        object where `state.sampler_state` is the `NSState`
        containing the live particles. The code accesses this via `state[0]`.
    dead_info_history
        A list of `NSInfo` objects, where each object contains information
        about the particles that "died" at one step of the NS algorithm.

    Returns
    -------
    NSInfo
        A single `NSInfo` object where all fields are concatenations of the
        corresponding fields from `dead_info_history` and the final live points.
        The `update_info` for the final live points is taken from the last
        element of `dead_info_history` as a placeholder.
    """

    all_pytrees_to_combine = dead_info_history + [
        NSInfo(  # Assuming NSInfo is your constructor
            state[0].particles,  # type: ignore
            state[0].loglikelihood,  # type: ignore
            state[0].loglikelihood_birth,  # type: ignore
            state[0].logprior,  # type: ignore
            dead_info_history[-1].update_info,
        )
    ]
    combined_dead_info = jax.tree.map(
        lambda *args: jnp.concatenate(args),
        all_pytrees_to_combine[0],
        *all_pytrees_to_combine[1:],
    )
    return combined_dead_info


def ess(rng_key: PRNGKey, dead_info_map: NSInfo) -> Array:
    """Computes the Effective Sample Size (ESS) from log-weights.

    The ESS is a measure of the quality of importance samples, indicating
    how many independent samples the weighted set is equivalent to.
    It's calculated as `(sum w_i)^2 / sum (w_i^2)`. This function computes
    the ESS based on log-weights, averaging over multiple stochastic log-weight
    samples if `log_weights` provides them.

    Parameters
    ----------
    rng_key
        A JAX PRNG key, used by `log_weights` to simulate volume elements.
    dead_info_map
        An `NSInfo` object containing information for the full set of dead
        (and final live) particles, typically the output of `finalise`.

    Returns
    -------
    Array
        The mean Effective Sample Size, a scalar float.
    """
    logw = log_weights(rng_key, dead_info_map).mean(axis=-1)
    logw -= logw.max()
    l_sum_w = jax.scipy.special.logsumexp(logw)
    l_sum_w_sq = jax.scipy.special.logsumexp(2 * logw)
    ess = jnp.exp(2 * l_sum_w - l_sum_w_sq)
    return ess


def sample(
    rng_key: PRNGKey, dead_info_map: NSInfo, shape: int = 1000
) -> ArrayTree:  # Renamed params
    """Resamples particles according to their importance weights.

    This function takes the full set of dead (and final live) particles and
    their computed importance weights, and draws `num_final_samples` particles
    with replacement. The probability of drawing each particle is proportional
    to its (exponentiated and normalized) importance weight. This produces an
    approximately unweighted sample from the target posterior distribution.

    Parameters
    ----------
    rng_key
        A JAX PRNG key, used for both `log_weights` and `jax.random.choice`.
    dead_info_map
        An `NSInfo` object containing information for the full set of dead
        (and final live) particles, typically the output of `finalise`.
    shape
        The number of posterior samples to draw. Defaults to 1000.

    Returns
    -------
    ArrayTree
        A PyTree of resampled particles. Each leaf array in the PyTree will
        have a leading dimension of size `num_final_samples`.
    """
    logw = log_weights(rng_key, dead_info_map).mean(axis=-1)
    indices = jax.random.choice(
        rng_key,
        dead_info_map.loglikelihood.shape[0],
        p=jnp.exp(logw.squeeze() - jnp.max(logw)),
        shape=(shape,),
        replace=True,
    )
    return jax.tree.map(lambda leaf: leaf[indices], dead_info_map.particles)


def get_first_row(x: ArrayTree) -> ArrayTree:
    """Extracts the first "row" (element along the leading axis) of each leaf in a PyTree.

    This is typically used to get a single particle's structure or values from
    a PyTree representing a collection of particles, where the leading dimension
    of each leaf array corresponds to the particle index (or batch index).

    Parameters
    ----------
    x
        A PyTree of arrays, where each leaf array has a leading dimension.

    Returns
    -------
    ArrayTree
        A PyTree with the same structure as `x`, but where each leaf is the
        first slice `leaf[0]` of the corresponding leaf in `x`.
    """
    return jax.tree.map(lambda x: x[0], x)


def repeat_kernel(num_repeats: int) -> Callable[[Callable], Callable]:
    """Decorator to repeat a kernel function multiple times."""

    def decorator(kernel):
        @functools.wraps(kernel)
        def repeated_kernel(rng_key: PRNGKey, state, *args, **kwargs):
            def body_fn(state, rng_key):
                return kernel(rng_key, state, *args, **kwargs)

            keys = jax.random.split(rng_key, num_repeats)
            return jax.lax.scan(body_fn, state, keys)

        return repeated_kernel

    return decorator
