""" All things resampling. """
from functools import partial
from typing import Callable

import jax
import jax.numpy as jnp

from blackjax.types import PRNGKey


def _resampling_func(func, name, desc="", additional_params="") -> Callable:
    doc = f"""
    {name} resampling. {desc}

    Parameters
    ----------
    weights: jnp.ndarray
        Weights to resample
    key: jnp.ndarray
        PRNGKey to use in resampling
    n_samples: int
        Total number of particles to sample
    {additional_params}

    Returns
    -------
    idx: jnp.ndarray
        Array of integers fo size `n_samples` to use for resampling
    """

    func.__doc__ = doc
    return func


@partial(_resampling_func, name="Systematic")
def systematic(weights: jnp.ndarray, rng_key: PRNGKey, n_samples: int) -> jnp.ndarray:
    return _systematic_or_stratified(weights, rng_key, n_samples, True)


@partial(_resampling_func, name="Stratified")
def stratified(weights: jnp.ndarray, rng_key: PRNGKey, n_samples: int) -> jnp.ndarray:
    return _systematic_or_stratified(weights, rng_key, n_samples, False)


@partial(
    _resampling_func,
    name="Multinomial",
    desc="This has higher variance than other resampling schemes, "
    "and should only be used for illustration purposes, "
    "or if your algorithm *REALLY* needs independent samples.",
)
def multinomial(weights: jnp.ndarray, rng_key: PRNGKey, n_samples: int) -> jnp.ndarray:
    # In practice we don't have to sort the generated uniforms, but searchsorted works faster and is more stable
    # if both inputs are sorted, so we use the _sorted_uniforms from N. Chopin, but still use searchsorted instead of
    # his O(N) loop as our code is meant to work on GPU where searchsorted is O(log(N)) anyway.

    n = weights.shape[0]
    linspace = _sorted_uniforms(n_samples, rng_key)
    cumsum = jnp.cumsum(weights)
    idx = jnp.searchsorted(cumsum, linspace)
    return jnp.clip(idx, 0, n - 1)


@partial(_resampling_func, name="Residual")
def residual(weights: jnp.ndarray, rng_key: PRNGKey, n_samples: int) -> jnp.ndarray:
    key1, key2 = jax.random.split(rng_key)
    N = weights.shape[0]
    n_samples_weights = n_samples * weights
    idx = jnp.arange(n_samples)

    integer_part = jnp.floor(n_samples_weights).astype(jnp.int32)
    sum_integer_part = jnp.sum(integer_part)

    residual_part = n_samples_weights - integer_part
    residual_sample = multinomial(
        residual_part / (n_samples - sum_integer_part), key1, n_samples
    )

    # Permutation is needed due to the concatenation happening at the last step.
    # I am pretty sure we can use lower variance resamplers inside here instead of multinomial,
    # but I am not sure yet due to the loss of exchangeability, and as a consequence I am playing it safe.
    residual_sample = jax.random.permutation(key2, residual_sample)

    integer_idx = jnp.repeat(
        jnp.arange(N + 1),
        jnp.concatenate([integer_part, jnp.array([n_samples - sum_integer_part])], 0),
        total_repeat_length=n_samples,
    )

    idx = jnp.where(idx >= sum_integer_part, residual_sample, integer_idx)

    return idx


def _systematic_or_stratified(
    weights: jnp.ndarray, rng_key: PRNGKey, n_sampled: int, is_systematic: bool
) -> jnp.ndarray:
    n = weights.shape[0]
    if is_systematic:
        u = jax.random.uniform(rng_key, ())
    else:
        u = jax.random.uniform(rng_key, (n_sampled,))
    cumsum = jnp.cumsum(weights)
    linspace = (jnp.arange(n_sampled, dtype=weights.dtype) + u) / n_sampled
    idx = jnp.searchsorted(cumsum, linspace)
    return jnp.clip(idx, 0, n - 1)


def _sorted_uniforms(n, rng_key: PRNGKey) -> jnp.ndarray:
    # Credit goes to Nicolas Chopin
    us = jax.random.uniform(rng_key, (n + 1,))
    z = jnp.cumsum(-jnp.log(us))
    return z[:-1] / z[-1]
