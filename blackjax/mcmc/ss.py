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
"""Hit-and-Run Slice Sampling.

This module implements the Hit-and-Run Slice Sampling algorithm as described by
Neal (2003) [1]_. Slice sampling is an MCMC method that adapts its step size
automatically and can be efficient for sampling from distributions with complex
geometries. The "hit-and-run" variant involves proposing a direction and then
finding an acceptable point along that direction within a slice defined by the
current auxiliary variable.

References
----------
.. [1] Neal, R. M. (2003). Slice sampling. The Annals of Statistics, 31(3), 705-767.

"""

from functools import partial
from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp

from blackjax.base import SamplingAlgorithm
from blackjax.types import Array, ArrayLikeTree, ArrayTree, PRNGKey

__all__ = [
    "SliceState",
    "SliceInfo",
    "init",
    "build_kernel",
    "build_hrss_kernel",
    "hrss_as_top_level_api",
]


class SliceState(NamedTuple):
    """State of the Slice Sampling algorithm.

    Attributes
    ----------
    position
        The current position of the chain.
    logdensity
        The log-density of the target distribution at the current position.
    """

    position: ArrayLikeTree
    logdensity: float


class SliceInfo(NamedTuple):
    """Additional information about the Slice Sampling transition.

    This information can be used for diagnostics and monitoring the sampler's
    performance.

    Attributes
    ----------
    is_accepted
        A boolean indicating whether the proposed sample was accepted.
    num_steps
        The number of steps taken to expand the interval during the "stepping-out" phase.
    num_shrink
        The number of steps taken during the "shrinking" phase to find an
        acceptable sample.
    """

    is_accepted: bool
    num_steps: int
    num_shrink: int


def init(position: ArrayTree, logdensity_fn: Callable) -> SliceState:
    """Initialize the Slice Sampler state.

    Parameters
    ----------
    position
        The initial position of the chain.
    logdensity_fn
        A function that computes the log-density of the target distribution.

    Returns
    -------
    SliceState
        The initial state of the Slice Sampler.
    """
    return SliceState(position, logdensity_fn(position))


def build_kernel(
    slice_fn: Callable[[float], tuple[SliceState, bool]],
    max_steps: int = 10,
    max_shrinkage: int = 100,
) -> Callable:
    """Build a Slice Sampling kernel.

    This kernel performs one step of Slice Sampling algorithm, which involves
    defining a vertical slice, stepping out to define an interval, and then
    shrinking that interval to find an acceptable new sample.

    Parameters
    ----------
    slice_fn
        A function that takes a scalar parameter `t` and returns a tuple
        (SliceState, is_accepted) indicating the state at that parameter value
        and whether it satisfies acceptance criteria.
    max_steps
        The maximum number of steps to take when expanding the interval in
        each direction during the stepping-out phase.
    max_shrinkage
        The maximum number of shrinking steps to perform to avoid infinite loops.

    Returns
    -------
    Callable
        A kernel function that takes a PRNG key and the current `SliceState`,
        and returns a new `SliceState` and `SliceInfo`.

    References
    ----------
    .. [1] Neal, R. M. (2003). Slice sampling. The Annals of Statistics, 31(3), 705-767.
    """

    def kernel(
        rng_key: PRNGKey,
        state: SliceState,
    ) -> tuple[SliceState, SliceInfo]:
        vs_key, hs_key = jax.random.split(rng_key)
        u = jax.random.uniform(vs_key)
        logslice = state.logdensity + jnp.log(u)
        vertical_is_accepted = logslice < state.logdensity

        def _slice_fn(t):
            new_state, is_accepted = slice_fn(t)
            in_slice = new_state.logdensity >= logslice
            return new_state, is_accepted & in_slice

        new_state, info = horizontal_slice(
            hs_key, state, _slice_fn, max_steps, max_shrinkage
        )
        info = info._replace(is_accepted=info.is_accepted & vertical_is_accepted)
        return new_state, info

    return kernel


def horizontal_slice(
    rng_key: PRNGKey,
    state: SliceState,
    slice_fn: Callable[[float], tuple[SliceState, bool]],
    m: int,
    max_shrinkage: int,
) -> tuple[SliceState, SliceInfo]:
    """Propose a new sample using the stepping-out and shrinking procedures.

    This function implements the core of the Hit-and-Run Slice Sampling algorithm.
    It first expands an interval (`[l, r]`) along a one-dimensional parameterization
    until both ends are outside the slice defined by `logslice` (stepping-out).
    Then, it samples points uniformly from this interval and shrinks the interval
    until a point is found that lies within the slice (shrinking).

    Parameters
    ----------
    rng_key
        A JAX PRNG key.
    state
        The current slice sampling state.
    slice_fn
        A function that takes a scalar parameter `t` and returns a tuple
        (SliceState, is_accepted) indicating the state at that parameter value
        and whether it satisfies acceptance criteria.
    m
        The maximum number of steps to take when expanding the interval in
        each direction during the stepping-out phase.
    max_shrinkage
        The maximum number of shrinking steps to perform to avoid infinite loops.

    Returns
    -------
    tuple[SliceState, SliceInfo]
        A tuple containing the new state (with the accepted sample and its
        log-density) and information about the sampling process (number of
        expansion and shrinkage steps).
    """
    # Initial bounds
    rng_key, subkey = jax.random.split(rng_key)
    u, v = jax.random.uniform(subkey, 2)
    j = jnp.floor(m * v).astype(jnp.int32)
    k = (m - 1) - j

    # Expand
    def step_body_fun(carry):
        i, s, t, _ = carry
        t += s
        _, is_accepted = slice_fn(t)
        i -= 1
        return i, s, t, is_accepted

    def step_cond_fun(carry):
        i, _, _, is_accepted = carry
        return is_accepted & (i > 0)

    j, _, left, _ = jax.lax.while_loop(
        step_cond_fun, step_body_fun, (j + 1, -1, 1 - u, True)
    )
    k, _, right, _ = jax.lax.while_loop(
        step_cond_fun, step_body_fun, (k + 1, +1, -u, True)
    )

    # Shrink
    def shrink_body_fun(carry):
        n, rng_key, left, right, state, is_accepted = carry

        rng_key, subkey = jax.random.split(rng_key)
        u = jax.random.uniform(subkey, minval=left, maxval=right)

        new_state, is_accepted = slice_fn(u)
        n += 1

        left = jnp.where(u < 0, u, left)
        right = jnp.where(u > 0, u, right)

        return n, rng_key, left, right, new_state, is_accepted

    def shrink_cond_fun(carry):
        n, _, _, _, _, is_accepted = carry
        return ~is_accepted & (n < max_shrinkage)

    carry = 0, rng_key, left, right, state, False
    carry = jax.lax.while_loop(shrink_cond_fun, shrink_body_fun, carry)
    n, _, _, _, new_state, is_accepted = carry
    new_state = jax.tree.map(
        lambda new, old: jnp.where(is_accepted, new, old), new_state, state
    )
    slice_info = SliceInfo(is_accepted, m + 1 - j - k, n)
    return new_state, slice_info


def build_hrss_kernel(
    cov: Array,
    init_fn: Callable = init,
    max_steps: int = 10,
    max_shrinkage: int = 100,
) -> Callable:
    """Build a Hit-and-Run Slice Sampling kernel.

    This kernel performs one step of the Hit-and-Run Slice Sampling algorithm,
    which involves defining a vertical slice, proposing a direction, stepping out
    to define an interval, and then shrinking that interval to find an acceptable
    new sample.

    Parameters
    ----------
    cov
        The covariance matrix used by the direction proposal function
    init_fn
        A function initializing a SliceState
    max_steps
        The maximum number of steps to take when expanding the interval in
        each direction during the stepping-out phase.
    max_shrinkage
        The maximum number of shrinking steps to perform to avoid infinite loops.

    Returns
    -------
    Callable
        A kernel function that takes a PRNG key, the current `SliceState`, and
        the log-density function, and returns a new `SliceState` and `SliceInfo`.
    """

    def kernel(
        rng_key: PRNGKey, state: SliceState, logdensity_fn: Callable
    ) -> tuple[SliceState, SliceInfo]:
        rng_key, prop_key = jax.random.split(rng_key, 2)
        d = sample_direction_from_covariance(prop_key, state.position, cov)

        def slice_fn(t):
            x = jax.tree.map(lambda x, d: x + t * d, state.position, d)
            is_accepted = True
            new_state = init_fn(x, logdensity_fn)
            return new_state, is_accepted

        slice_kernel = build_kernel(slice_fn, max_steps, max_shrinkage)
        return slice_kernel(rng_key, state)

    return kernel


def sample_direction_from_covariance(
    rng_key: PRNGKey, position: ArrayLikeTree, cov: Array
) -> Array:
    """Generates a random direction vector, normalized, from a multivariate Gaussian.


    Parameters
    ----------
    rng_key
        A JAX PRNG key.
    position
        The current position of the chain (used for extracting shape).
    cov
        The covariance matrix.
    Returns
    -------
    Array
        A normalized direction vector (1D array).
    """
    p, unravel_fn = jax.flatten_util.ravel_pytree(position)
    d = jax.random.multivariate_normal(rng_key, mean=jnp.zeros(cov.shape[0]), cov=cov)
    invcov = jnp.linalg.inv(cov)
    norm = jnp.sqrt(jnp.einsum("...i,...ij,...j", d, invcov, d))
    d = d / norm[..., None]
    d *= 2
    return unravel_fn(d)


def hrss_as_top_level_api(
    logdensity_fn: Callable,
    cov: Array,
    init_fn: Callable = init,
    max_steps: int = 10,
    max_shrinkage: int = 100,
) -> SamplingAlgorithm:
    """Creates a Hit-and-Run Slice Sampling algorithm.

    This function serves as a convenience wrapper to easily construct a
    Hit-and-Run Slice Sampler using a default direction proposal mechanism
    based on a multivariate Gaussian distribution with the provided covariance.

    Parameters
    ----------
    logdensity_fn
        The log-density function of the target distribution to sample from.
    cov
        The covariance matrix used by the direction proposal function
        (`sample_direction_from_covariance`). This matrix shapes the random
        directions proposed for the slice sampling steps.
    init_fn
        A function initializing a SliceState
    max_steps
        The maximum number of steps to take when expanding the interval in
        each direction during the stepping-out phase.
    max_shrinkage
        The maximum number of shrinking steps to perform to avoid infinite loops.

    Returns
    -------
    SamplingAlgorithm
        A `SamplingAlgorithm` tuple containing `init` and `step` functions for
        the configured Hit-and-Run Slice Sampler.
    """
    kernel = build_hrss_kernel(cov, init_fn, max_steps, max_shrinkage)
    init_fn = partial(init_fn, logdensity_fn=logdensity_fn)
    step_fn = partial(kernel, logdensity_fn=logdensity_fn)
    return SamplingAlgorithm(init_fn, step_fn)
