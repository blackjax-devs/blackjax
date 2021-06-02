""" All things resampling. """
from functools import partial

import jax
import jax.numpy as jnp


def _resampling_func(func, name, desc="", additional_params=""):
    # Decorator for resampling function

    doc = f""" {name} resampling. {desc}

    Parameters
    ----------
    weights: jnp.ndarray
        Weights to resample
    key: jnp.ndarray
        PRNGKey to use in resampling
    {additional_params}

    Returns
    -------
    idx: jnp.ndarray
        Array of integers to use for resampling
    """

    func.__doc__ = doc
    return func


@partial(_resampling_func, name="Systematic")
def systematic(weights: jnp.ndarray, key: jnp.ndarray):
    return _systematic_or_stratified(weights, key, True)


@partial(_resampling_func, name="Stratified")
def stratified(weights: jnp.ndarray, key: jnp.ndarray):
    return _systematic_or_stratified(weights, key, False)


@partial(
    _resampling_func,
    name="Multinomial",
    desc="This has higher variance than other resampling schemes, "
    "and should only be used for illustration purposes, "
    "or if your algorithm *REALLY* needs independent samples. Does it?",
)
def multinomial(weights: jnp.ndarray, key: jnp.ndarray):
    # In practice we don't have to sort the generated uniforms, but searchsorted works faster and is more stable
    # if both inputs are sorted, so we use the _sorted_uniforms from N. Chopin, but still use searchsorted instead of
    # his O(N) loop as our code is meant to work on GPU where searchsorted is O(log(N)) anyway.

    n = weights.shape[0]
    linspace = _sorted_uniforms(n, key)
    cumsum = jnp.cumsum(weights)
    idx = jnp.searchsorted(cumsum, linspace)
    return jnp.clip(idx, 0, n - 1)


@partial(_resampling_func, name="Residual")
def residual(weights: jnp.ndarray, key: jnp.ndarray):
    # This code is adapted from nchopin/particles library, but made to be compatible with JAX static shape jitting that
    # would not have supported the dynamic slicing implementation of Nicolas. The below will be (slightly) less
    # efficient on CPU but has the benefit of being all XLA-devices compatible. The main difference with Nicolas's code
    # lies in the introduction of N+1 in the array as a "sink state" for unused indices. Sadly this can't reuse the code
    # for low variance resampling methods as it is not compatible with the sorted approach taken.

    import warnings

    warnings.warn(
        "Residual resampling typically has a low variance. However the JAX implementation of categorical"
        "sampling is memory consuming and the program may fail for a large number of samples."
    )

    N = weights.shape[0]
    N_weights = N * weights
    idx = jnp.arange(N)

    integer_part = jnp.floor(N_weights).astype(jnp.int32)
    sum_integer_part = jnp.sum(integer_part)

    residual_part = N_weights - integer_part
    residual_sample = jax.random.categorical(
        key, jnp.log(residual_part / (N - sum_integer_part)), shape=(N,)
    )
    integer_idx = jnp.repeat(
        jnp.arange(N + 1),
        jnp.concatenate([integer_part, jnp.array([N - sum_integer_part])], 0),
        total_repeat_length=N,
    )

    idx = jnp.where(idx >= sum_integer_part, residual_sample, integer_idx)

    return idx


def _systematic_or_stratified(
    weights: jnp.ndarray, key: jnp.ndarray, is_systematic: bool
):
    n = weights.shape[0]
    if is_systematic:
        u = jax.random.uniform(key, ())
    else:
        u = jax.random.uniform(key, (n,))
    cumsum = jnp.cumsum(weights)
    linspace = (jnp.arange(n, dtype=weights.dtype) + u) / n
    idx = jnp.searchsorted(cumsum, linspace)
    return jnp.clip(idx, 0, n - 1)


def _sorted_uniforms(n, key):
    # Credit goes to Nicolas Chopin
    us = jax.random.uniform(key, (n + 1,))
    z = jnp.cumsum(-jnp.log(us))
    return z[:-1] / z[-1]
