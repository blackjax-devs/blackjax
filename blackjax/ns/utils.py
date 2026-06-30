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
"""Utility functions for setting up and post-processing Nested Sampling runs.

References
----------
.. [1] Skilling, J. (2006). "Nested sampling for general Bayesian computation."
       Bayesian Analysis, 1(4), 833-859. https://doi.org/10.1214/06-BA127
.. [2] Fowlie, A., Handley, W., & Su, L. (2021). "Nested sampling with plateaus."
       Monthly Notices of the Royal Astronomical Society, 503(1), 1199-1205.
       https://doi.org/10.1093/mnras/stab590
"""

from typing import Callable

import jax
import jax.numpy as jnp

from blackjax.ns.base import NSInfo, NSState
from blackjax.types import Array, ArrayTree, PRNGKey


def log1mexp(x: Array) -> Array:
    """Computes log(1 - exp(x)) in a numerically stable way."""
    # precision hack: clamp x <= -eps so float32 cumsum drift in logX
    # can't push the argument positive and produce NaN downstream.
    x = jnp.minimum(x, -jnp.finfo(x.dtype).eps)
    return jnp.where(
        x > -0.6931472,  # approx log(2)
        jnp.log(-jnp.expm1(x)),
        jnp.log1p(-jnp.exp(x)),
    )


def compute_num_live(info: NSInfo) -> Array:
    """Compute the effective number of live points at each death contour (Fowlie, Handley & Su, 2021).

    When doing batch deletions, the jump in energy level can be smoothed by
    transforming 1 jump of size k into k jumps of size 1. This function computes
    the effective population size associated with this transformation.

    Expects the **complete finalised output** -- the dead points together with the
    final live particles (e.g. from :func:`finalise`): it relies on every
    particle's birth event being present. Called on a dead-only subset that omits
    the initial live particles' births, the live counts are wrong (1 instead of
    ``N`` for a standard run).

    Returns
    -------
    Array
        An array where each element `num_live[j]` is the effective number of live
        points `m*_i` when the j-th particle (in the sorted list of dead particles)
        was considered "dead".
    """
    birth_logL = info.particles.loglikelihood_birth
    death_logL = info.particles.loglikelihood

    birth_events = jnp.column_stack((birth_logL, jnp.ones_like(birth_logL, dtype=int)))
    death_events = jnp.column_stack((death_logL, -jnp.ones_like(death_logL, dtype=int)))
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
    num_live = cumsum[death_mask_sorted] + 1
    return num_live


def logX(rng_key: PRNGKey, dead_info: NSInfo, shape: int = 100) -> tuple[Array, Array]:
    """Simulate the stochastic evolution of log prior volumes (Skilling, 2006).

    Wraps the effective population size in `compute_num_live`, along with stochastic
    simulation of the log prior shrinkage associated with each deleted particle.


    Parameters
    ----------
    rng_key
        A JAX PRNG key for generating uniform random variates.
    dead_info
        An `NSInfo` object (or compatible PyTree) containing `loglikelihood_birth`
        and `loglikelihood` for all dead particles accumulated during an NS run.
        It's assumed these particles are already sorted by their death log-likelihood.
    shape
        The shape of Monte Carlo samples to generate for the stochastic
        log-volume sequence. Each sample represents one possible path of
        volume shrinkage. Default is 100.

    Returns
    -------
    tuple[Array, Array]
        - `logX_cumulative`: An array of shape `(num_dead_particles, shape)`
          containing `shape` simulated sequences of cumulative log prior volumes `log(X_i)`.
        - `log_dX_elements`: An array of shape `(num_dead_particles, shape)`
          containing `shape` simulated sequences of log prior volume elements `log(dX_i)`.
          `dX_i` is the trapezoidal volume element `(X_{i-1} - X_{i+1}) / 2`.
    """
    rng_key, subkey = jax.random.split(rng_key)
    u = jax.random.uniform(
        subkey,
        shape=(dead_info.particles.loglikelihood.shape[0], shape),
    )
    r = jax.lax.log1p(jax.lax.neg(u))
    num_live = compute_num_live(dead_info)
    t = r / num_live[:, jnp.newaxis]
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

    Parameters
    ----------
    rng_key
        A JAX PRNG key for simulating `log(dX_i)`.
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
        An array of log importance weights, shape `(num_dead_particles, *shape)`.
        The original order of particles in `dead_info` is preserved.
    """
    sort_indices = jnp.argsort(dead_info.particles.loglikelihood)
    unsort_indices = jnp.empty_like(sort_indices)
    unsort_indices = unsort_indices.at[sort_indices].set(jnp.arange(len(sort_indices)))
    dead_info_sorted = jax.tree.map(lambda x: x[sort_indices], dead_info)
    _, log_dX = logX(rng_key, dead_info_sorted, shape)
    log_w = log_dX + beta * dead_info_sorted.particles.loglikelihood[..., jnp.newaxis]
    return log_w[unsort_indices]


def finalise(live: NSState, dead: list[NSInfo], update_info: bool = True) -> NSInfo:
    """Combines the history of dead particle information with the final live points.

    Parameters
    ----------
    live
        The final `NSState` of the Nested Sampler, containing the live particles.
    dead
        A list of `NSInfo` objects, where each object contains information
        about the particles that "died" at one step of the NS algorithm.
    update_info
        Whether to concatenate the `update_info` from each element of `dead`.
        If False, the returned `update_info` is None. Default is True.

    Returns
    -------
    NSInfo
        A single `NSInfo` whose `particles` field concatenates all dead
        particles with the final live particles. When ``update_info=True`` the
        `update_info` field concatenates the `update_info` from each element of
        `dead` only -- no entry is added for the final live points, so it is
        shorter than `particles` by the number of live points. When
        ``update_info=False`` the `update_info` field is None.
    """

    if update_info:
        update_infos = [d.update_info for d in dead]
        final_update_info = jax.tree.map(
            lambda *xs: jnp.concatenate(xs, axis=0), *update_infos
        )
    else:
        final_update_info = None

    particles = [d.particles for d in dead] + [live.particles]
    final_particles = jax.tree.map(lambda *xs: jnp.concatenate(xs, axis=0), *particles)
    return NSInfo(final_particles, final_update_info)


def ess(rng_key: PRNGKey, dead: NSInfo) -> Array:
    """Computes the Effective Sample Size (ESS) from log-weights.

    Parameters
    ----------
    rng_key
        A JAX PRNG key, used by `log_weights`.
    dead
        An `NSInfo` object containing the full set of dead (and final live)
        particles, typically the output of `finalise`.

    Returns
    -------
    Array
        The mean Effective Sample Size, a scalar float.
    """
    logw = log_weights(rng_key, dead).mean(axis=-1)
    logw -= logw.max()
    l_sum_w = jax.scipy.special.logsumexp(logw)
    l_sum_w_sq = jax.scipy.special.logsumexp(2 * logw)
    ess = jnp.exp(2 * l_sum_w - l_sum_w_sq)
    return ess


def sample(rng_key: PRNGKey, dead: NSInfo, shape: int = 1000) -> ArrayTree:
    """Resamples particles according to their importance weights.

    Parameters
    ----------
    rng_key
        A JAX PRNG key, used by `log_weights` and for resampling.
    dead
        An `NSInfo` object containing the full set of dead (and final live)
        particles, typically the output of `finalise`.
    shape
        The number of resampled particles to draw. Default is 1000.

    Returns
    -------
    ArrayTree
        A PyTree of resampled particles, where each leaf has `shape`.
    """
    logw = log_weights(rng_key, dead).mean(axis=-1)
    indices = jax.random.choice(
        rng_key,
        dead.particles.loglikelihood.shape[0],
        p=jnp.exp(logw.squeeze() - jnp.max(logw)),
        shape=(shape,),
        replace=True,
    )
    return jax.tree.map(lambda leaf: leaf[indices], dead.particles)


def get_first_row(x: ArrayTree) -> ArrayTree:
    """Extracts the first "row" (element along the leading axis) of each leaf in a PyTree.

    This is typically used to get a single particle's structure or values from
    a PyTree representing a collection of particles, where the leading dimension
    of each leaf array corresponds to the particle index.

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


def uniform_prior(
    rng_key: PRNGKey, num_live: int, bounds: dict[str, tuple[float, float]]
) -> tuple[ArrayTree, Callable]:
    """Sample initial particles and build a log-prior for a box-uniform prior.

    Parameters
    ----------
    rng_key
        A JAX PRNG key for random number generation.
    num_live
        The number of live particles to sample.
    bounds
        A dictionary mapping parameter names to their bounds (tuples of min and max).
        Each parameter will be sampled uniformly within these bounds.
        Example: {'param1': (0.0, 1.0), 'param2': (-5.0, 5.0)}

    Returns
    -------
    tuple
        - `particles`: A PyTree of sampled parameters, where each leaf has shape `(num_live,)`.
        - `logprior_fn`: A function that computes the log-prior probability
          for a given set of parameters.
    """

    def logprior_fn(params):
        logprior = 0.0
        for p, (a, b) in bounds.items():
            x = params[p]
            logprior += jax.scipy.stats.uniform.logpdf(x, a, b - a)
        return logprior

    def prior_sample(rng_key):
        init_keys = jax.random.split(rng_key, len(bounds))
        params = {}
        for rng_key, (p, (a, b)) in zip(init_keys, bounds.items()):
            params[p] = jax.random.uniform(rng_key, minval=a, maxval=b)
        return params

    init_keys = jax.random.split(rng_key, num_live)
    particles = jax.vmap(prior_sample)(init_keys)

    return particles, logprior_fn
