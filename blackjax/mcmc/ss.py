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

    position: ArrayLikeTree
        The current position of the chain.
    logdensity: float
        The log-density of the target distribution at the current position.
    """

    position: ArrayLikeTree
    logdensity: float


class SliceInfo(NamedTuple):
    """Additional information about the Slice Sampling transition.

    This information can be used for diagnostics and monitoring the sampler's
    performance.

    l_steps: Array
        The number of steps taken to expand the interval to the left during the
        "stepping-out" phase.
    r_steps: Array
        The number of steps taken to expand the interval to the right during the
        "stepping-out" phase.
    s_steps: Array
        The number of steps taken during the "shrinking" phase to find an
        acceptable sample.
    evals: Array
        The total number of log-density evaluations performed during the step.
    """

    l_steps: Array
    r_steps: Array
    s_steps: Array
    evals: Array


def init(position: ArrayTree, logdensity_fn: Callable):
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
    stepper_fn: Callable,
) -> Callable:
    """Build a Slice Sampling kernel.

    This kernel performs one step of Slice Sampling algorithm, which involves
    defining a vertical slice, stepping out to define an interval, and then
    shrinking that interval to find an acceptable new sample.

    Parameters
    ----------
    stepper_fn
        A function that computes a new position given an initial position,
        direction `d` and a slice parameter `t`.
        `(x0, d, t) -> x_new` where e.g. `x_new = x0 + t * d`.

    Returns
    -------
    Callable
        A kernel function that takes a PRNG key, the current `SliceState`, and
        the log-density function, and returns a new `SliceState` and `SliceInfo`.

    References
    ----------
    .. [1] Neal, R. M. (2003). Slice sampling. The Annals of Statistics, 31(3), 705-767.
    """

    def kernel(
        rng_key: PRNGKey, state: SliceState, logdensity_fn: Callable, d: ArrayTree
    ) -> tuple[SliceState, SliceInfo]:
        rng_key, vs_key, hs_key = jax.random.split(rng_key, 3)
        logdensity = vertical_slice(vs_key, logdensity_fn, state.position)
        slice_state, slice_info = horizontal_slice_proposal(
            hs_key, state.position, d, stepper_fn, logdensity_fn, logdensity
        )

        return slice_state, slice_info

    return kernel


def vertical_slice(
    rng_key: PRNGKey, logdensity_fn: Callable, position: ArrayTree
) -> float:
    """Define the vertical slice for the Slice Sampling algorithm.

    This function determines the height `y` for the horizontal slice by sampling
    uniformly from `[0, p(x)]`, where `p(x)` is the target density at the current
    position `x`. This is equivalent to sampling `log_y` uniformly from
    `(-inf, log_p(x)]`.

    Parameters
    ----------
    rng_key
        A JAX PRNG key.
    logdensity_fn
        The log-density function of the target distribution.
    position
        The current position of the chain.

    Returns
    -------
    Array
        The log-height `log_y` defining the lower bound for the horizontal slice.
        A new sample `x'` will be accepted if `logdensity_fn(x') >= log_y`.
    """
    logdensity = logdensity_fn(position)
    return logdensity + jnp.log(jax.random.uniform(rng_key))


def horizontal_slice_proposal(
    rng_key: PRNGKey,
    x0: ArrayTree,
    d: ArrayTree,
    stepper_fn: Callable,
    logdensity_fn: Callable,
    log_slice_height: Array,
) -> tuple[SliceState, SliceInfo]:
    """Propose a new sample using the stepping-out and shrinking procedures.

    This function implements the core of the Hit-and-Run Slice Sampling algorithm.
    It first expands an interval (`[l, r]`) along the slice starting
    from `x0` and proceeding along direction `d` until both ends are outside
    the slice defined by `log_slice_height` (stepping-out). Then, it samples
    points uniformly from this interval and shrinks the interval until a point
    is found that lies within the slice (shrinking).

    Parameters
    ----------
    rng_key
        A JAX PRNG key.
    x0
        The current position (PyTree).
    d
        The direction (PyTree) for proposing moves.
    stepper_fn
        A function `(x0, t) -> x_new` that computes a new point by
        moving `t` units along from `x0`.
    logdensity_fn
        The log-density function of the target distribution.
    log_slice_height
        The log-height defining the current slice, typically obtained from
        `vertical_slice`. A new sample `x_new` is accepted if
        `logdensity_fn(x_new) >= log_slice_height`.

    Returns
    -------
    tuple[SliceState, SliceInfo]
        A tuple containing the new state (with the accepted sample and its
        log-density) and information about the sampling process (number of
        expansion and shrinkage steps).
    """
    # Initial bounds
    rng_key, subkey = jax.random.split(rng_key)
    u = jax.random.uniform(subkey)

    def body_fun(carry):
        _, s, t, count = carry
        t += s
        x = stepper_fn(x0, d, t)
        within = logdensity_fn(x) >= log_slice_height
        count += 1
        return within, s, t, count

    def cond_fun(carry):
        within = carry[0]
        return within

    # Expand
    _, _, l, count_l = jax.lax.while_loop(cond_fun, body_fun, (True, -1, -u, 0))
    _, _, r, count_r = jax.lax.while_loop(cond_fun, body_fun, (True, +1, 1 - u, 0))

    # Shrink
    def body_fun(carry):
        _, l, r, _, _, rng_key, count = carry
        count += 1

        rng_key, subkey = jax.random.split(rng_key)
        u = jax.random.uniform(subkey, minval=l, maxval=r)
        x = stepper_fn(x0, d, u)

        logdensity_x = logdensity_fn(x)
        within = logdensity_x >= log_slice_height

        l = jnp.where(u < 0, u, l)
        r = jnp.where(u > 0, u, r)

        return within, l, r, x, logdensity_x, rng_key, count

    def cond_fun(carry):
        within = carry[0]
        return ~within

    carry = (False, l, r, x0, -jnp.inf, rng_key, 0)
    carry = jax.lax.while_loop(cond_fun, body_fun, carry)
    _, l, r, x, logdensity_x, rng_key, count = carry
    slice_state = SliceState(x, logdensity_x)
    slice_info = SliceInfo(count_l, count_r, count, (count_l + count_r + count))
    return slice_state, slice_info


def build_hrss_kernel(
    generate_slice_direction_fn: Callable,
    stepper_fn: Callable,
) -> Callable:
    """Build a Hit-and-Run Slice Sampling kernel.

    This kernel performs one step of the Hit-and-Run Slice Sampling algorithm,
    which involves defining a vertical slice, proposing a direction, stepping out
    to define an interval, and then shrinking that interval to find an acceptable
    new sample.

    Parameters
    ----------
    generate_slice_direction_fn
        A function that, given a PRNG key, generates a direction vector (PyTree
        with the same structure as the position) for the "hit-and-run" part of
        the algorithm. This direction is typically normalized.

    stepper_fn
        A function that computes a new position given an initial position, a
        direction, and a step size `t`. It should implement something analogous
        to `x_new = x_initial + t * direction`.

    Returns
    -------
    Callable
        A kernel function that takes a PRNG key, the current `SliceState`, and
        the log-density function, and returns a new `SliceState` and `SliceInfo`.
    """
    slice_kernel = build_kernel(stepper_fn)

    def kernel(
        rng_key: PRNGKey, state: SliceState, logdensity_fn: Callable
    ) -> tuple[SliceState, SliceInfo]:
        rng_key, prop_key = jax.random.split(rng_key, 2)
        d = generate_slice_direction_fn(prop_key)
        return slice_kernel(rng_key, state, logdensity_fn, d)

    return kernel


def default_stepper_fn(x, d, t):
    """A simple stepper function that moves from `x` along direction `d` by `t` units.

    Implements the operation: `x_new = x + t * d`.

    Parameters
    ----------
    x
        The starting position (PyTree).
    d
        The direction of movement (PyTree, same structure as `x`).
    t
        The scalar step size or distance along the direction.

    Returns
    -------
    ArrayTree
        The new position.
    """
    return jax.tree.map(lambda x, d: x + t * d, x, d)


def default_generate_slice_direction_fn(rng_key: PRNGKey, cov: Array) -> Array:
    """Generates a random direction vector, normalized, from a multivariate Gaussian.

    This function samples a direction `d` from a zero-mean multivariate Gaussian
    distribution with covariance matrix `cov`, and then normalizes `d` to be a
    unit vector with respect to the Mahalanobis norm defined by `inv(cov)`.
    That is, `d_normalized^T @ inv(cov) @ d_normalized = 1`.

    Parameters
    ----------
    rng_key
        A JAX PRNG key.
    cov
        The covariance matrix for the multivariate Gaussian distribution from which
        the initial direction is sampled. Assumed to be a 2D array.

    Returns
    -------
    Array
        A normalized direction vector (1D array).
    """
    d = jax.random.multivariate_normal(rng_key, mean=jnp.zeros(cov.shape[0]), cov=cov)
    invcov = jnp.linalg.inv(cov)
    norm = jnp.sqrt(jnp.einsum("...i,...ij,...j", d, invcov, d))
    d = d / norm[..., None]
    return d


def hrss_as_top_level_api(
    logdensity_fn: Callable,
    cov: Array,
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
        The covariance matrix used by the default direction proposal function
        (`default_proposal_distribution`). This matrix shapes the random
        directions proposed for the slice sampling steps.

    Returns
    -------
    SamplingAlgorithm
        A `SamplingAlgorithm` tuple containing `init` and `step` functions for
        the configured Hit-and-Run Slice Sampler.
    """
    generate_slice_direction_fn = partial(default_generate_slice_direction_fn, cov=cov)
    kernel = build_kernel(generate_slice_direction_fn, default_stepper_fn)
    init_fn = partial(init, logdensity_fn=logdensity_fn)
    step_fn = partial(kernel, logdensity_fn=logdensity_fn)
    return SamplingAlgorithm(init_fn, step_fn)
