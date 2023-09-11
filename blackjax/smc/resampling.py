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
""" All things resampling. """
from functools import partial
from typing import Callable

import jax
import jax.numpy as jnp

from blackjax.types import Array, PRNGKey


def _resampling_func(func, name, desc="", additional_params="") -> Callable:
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

    n = weights.shape[0]
    linspace = _sorted_uniforms(rng_key, num_samples)
    cumsum = jnp.cumsum(weights)
    idx = jnp.searchsorted(cumsum, linspace)
    return jnp.clip(idx, 0, n - 1)


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
    n = weights.shape[0]
    if is_systematic:
        u = jax.random.uniform(rng_key, ())
    else:
        u = jax.random.uniform(rng_key, (num_samples,))
    cumsum = jnp.cumsum(weights)
    linspace = (jnp.arange(num_samples, dtype=weights.dtype) + u) / num_samples
    idx = jnp.searchsorted(cumsum, linspace)
    return jnp.clip(idx, 0, n - 1)


def _sorted_uniforms(rng_key: PRNGKey, n) -> Array:
    # Credit goes to Nicolas Chopin
    us = jax.random.uniform(rng_key, (n + 1,))
    z = jnp.cumsum(-jnp.log(us))
    return z[:-1] / z[-1]
