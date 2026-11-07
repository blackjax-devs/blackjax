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
"""All things resampling."""

from collections.abc import Callable
from functools import partial

import jax
import jax.numpy as jnp

from blackjax.types import Array, PRNGKey


def _resampling_func(
    func: Callable,
    name: str,
    desc: str = "",
    additional_params: str = "",
) -> Callable:
    # Decorator for resampling function

    doc = f"""
    {name} resampling. {desc}

    Parameters
    ----------
    key: Array
        PRNGKey to use in resampling
    weights: Array
        Weights to resample
    num_samples: int
        Number of particles to sample

    Returns
    -------
    idx: Array
        Array of size `num_samples` to use for resampling
    """

    func.__doc__ = doc
    return func


@partial(_resampling_func, name="Systematic")
def systematic(rng_key: PRNGKey, weights: Array, num_samples: int) -> Array:
    return _systematic_or_stratified(rng_key, weights, num_samples, True)


@partial(_resampling_func, name="Stratified")
def stratified(rng_key: PRNGKey, weights: Array, num_samples: int) -> Array:
    return _systematic_or_stratified(rng_key, weights, num_samples, False)


@partial(
    _resampling_func,
    name="Multinomial",
    desc="""
    This has higher variance than other resampling schemes,
    and should only be used for illustration purposes,
    or if your algorithm *REALLY* needs independent samples.""",
)
def multinomial(rng_key: PRNGKey, weights: Array, num_samples: int) -> Array:
    # In practice we don't have to sort the generated uniforms, but searchsorted
    # works faster and is more stable if both inputs are sorted, so we use the
    # _sorted_uniforms from N. Chopin, but still use searchsorted instead of his
    # O(N) loop as our code is meant to work on GPU where searchsorted is
    # O(log(N)) anyway.

    queries = _sorted_uniforms(rng_key, num_samples)
    return _inverse_cdf(weights, queries)


@partial(
    _resampling_func,
    name="Residual",
    desc="""
    This code is adapted from https://github.com/nchopin/particles, but made to
    be compatible with JAX static shape jitting that would not have supported
    the dynamic slicing implementation of Nicolas.  The below will be (slightly)
    less efficient on CPU but has the benefit of being all XLA-devices
    compatible. The main difference with Nicolas Chopin's code lies in the
    introduction of N+1 in the array as a 'sink state' for unused indices.""",
)
def residual(rng_key: PRNGKey, weights: Array, num_samples: int) -> Array:
    key1, key2 = jax.random.split(rng_key)
    N = weights.shape[0]
    N_sample_weights = num_samples * weights
    idx = jnp.arange(num_samples)

    integer_part = jnp.floor(N_sample_weights).astype(jnp.int32)
    sum_integer_part = jnp.sum(integer_part)

    residual_part = N_sample_weights - integer_part
    residual_sample = multinomial(
        key1, residual_part / (num_samples - sum_integer_part), num_samples
    )

    # Permutation is needed due to the concatenation happening at the last step.
    #
    # I am pretty sure we can use lower variance resamplers inside here instead
    # of multinomial, but I am not sure yet due to the loss of exchangeability,
    # and as a consequence I am playing it safe.
    residual_sample = jax.random.permutation(key2, residual_sample)

    integer_idx = jnp.repeat(
        jnp.arange(N + 1),
        jnp.concatenate([integer_part, jnp.array([num_samples - sum_integer_part])], 0),
        total_repeat_length=num_samples,
    )

    idx = jnp.where(idx >= sum_integer_part, residual_sample, integer_idx)

    return idx


def _systematic_or_stratified(
    rng_key: PRNGKey, weights: Array, num_samples: int, is_systematic: bool
) -> Array:
    """Helper function for systematic and stratified resampling.

    Parameters
    ----------
    rng_key: PRNGKey
        PRNGKey to use in resampling.
    weights: Array
        Weights to resample.
    num_samples: int
        Number of particles to sample.
    is_systematic: bool
        If True, use systematic resampling; otherwise use stratified resampling.

    Returns
    -------
    idx: Array
        Array of size `num_samples` to use for resampling.
    """
    if is_systematic:
        u = jax.random.uniform(rng_key, ())
    else:
        u = jax.random.uniform(rng_key, (num_samples,))
    return _inverse_cdf_indices(weights, u, num_samples)


def _inverse_cdf_indices(weights: Array, u: Array, num_samples: int) -> Array:
    """Map explicit systematic or stratified uniforms to particle indices."""
    queries = (jnp.arange(num_samples, dtype=weights.dtype) + u) / num_samples
    return _inverse_cdf(weights, queries)


def _sorted_uniforms_from(us: Array) -> Array:
    """Compute sorted uniforms from explicit uniform draws.

    Given uniform draws in [0, 1), computes sorted uniforms via exponential-order
    statistics without numerical issues. Uses -log1p(-u) instead of -log(u) to
    handle u == 0.0 gracefully (which occurs at probability 2^-23 per draw).

    Parameters
    ----------
    us: Array
        Array of uniform random variables in [0, 1).

    Returns
    -------
    Array
        Array of size len(us) - 1 containing sorted uniform random variables in [0, 1).
    """
    # -log1p(-u) ~ Exp(1) and is finite on u in [0, 1);
    # -log(u) is inf at u == 0
    z = jnp.cumsum(-jnp.log1p(-us))
    return z[:-1] / z[-1]


def _sorted_uniforms(rng_key: PRNGKey, n: int) -> Array:
    """Generate sorted uniform random variables.

    Credit goes to Nicolas Chopin.

    Parameters
    ----------
    rng_key: PRNGKey
        PRNGKey to use for random number generation.
    n: int
        Number of sorted uniform random variables to generate.

    Returns
    -------
    Array
        Array of size n containing sorted uniform random variables.
    """
    us = jax.random.uniform(rng_key, (n + 1,))
    return _sorted_uniforms_from(us)


def _inverse_cdf(weights: Array, queries: Array) -> Array:
    """Map queries to particle indices via inverse CDF.

    Given normalized weights and query values in [0, 1), returns indices via
    the inverse CDF on half-open intervals. Ensures a zero-weight particle is
    never selected by clipping queries strictly below the total mass.

    Parameters
    ----------
    weights: Array
        Normalized weights (non-negative, sum to 1).
    queries: Array
        Query values in [0, 1).

    Returns
    -------
    Array
        Particle indices corresponding to each query.
    """
    cumsum = jnp.cumsum(weights)
    # Keep queries strictly below the total mass so a rounded-up query cannot
    # be clipped onto a trailing zero-weight particle.
    queries = jnp.minimum(queries, jnp.nextafter(cumsum[-1], -jnp.inf))
    idx = jnp.searchsorted(cumsum, queries, side="right")
    return jnp.clip(idx, 0, weights.shape[0] - 1)
