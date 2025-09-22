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
from blackjax.mcmc.proposal import static_binomial_sampling

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
    constraint: Array


class SliceInfo(NamedTuple):
    """Additional information about the Slice Sampling transition.

    This information can be used for diagnostics and monitoring the sampler's
    performance.

    Attributes
    ----------
    is_accepted
        A boolean indicating whether the proposed sample was accepted.
    constraint
        The constraint values at the final accepted position.
    num_steps
        The number of steps taken to expand the interval during the "stepping-out" phase.
    num_shrink
        The number of steps taken during the "shrinking" phase to find an
        acceptable sample.
    """

    is_accepted: bool
    num_steps: int
    num_shrink: int


def init(position: ArrayTree, logdensity_fn: Callable, constraint_fn: Callable) -> SliceState:
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
    return SliceState(position, logdensity_fn(position), constraint_fn(position))


def build_kernel(
    stepper_fn: Callable,
    max_steps: int = 10,
    max_shrinkage: int = 100,
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
        A kernel function that takes a PRNG key, the current `SliceState`,
        the log-density function, direction `d`, constraint function, constraint
        values, and strict flags, and returns a new `SliceState` and `SliceInfo`.

    References
    ----------
    .. [1] Neal, R. M. (2003). Slice sampling. The Annals of Statistics, 31(3), 705-767.
    """

    def kernel(
        rng_key: PRNGKey,
        state: SliceState,
        logdensity_fn: Callable,
        d: ArrayTree,
        constraint_fn: Callable,
        constraint: Array,
        strict: Array,
    ) -> tuple[SliceState, SliceInfo]:

        vs_key, hs_key = jax.random.split(rng_key)
        logslice = state.logdensity + jnp.log(jax.random.uniform(vs_key))
        vertical_is_accepted = logslice < state.logdensity

        def slicer(t) -> tuple[SliceState, SliceInfo]:
            x, step_accepted = stepper_fn(state.position, d, t)
            new_state = init(x, logdensity_fn, constraint_fn)
            constraints_ok = jnp.all(
                jnp.where(
                    strict,
                    new_state.constraint > constraint,
                    new_state.constraint >= constraint
                )
            )
            in_slice = new_state.logdensity >= logslice
            is_accepted = in_slice & constraints_ok & step_accepted
            return new_state, is_accepted

        new_state, info = horizontal_slice(hs_key, slicer, state, max_steps, max_shrinkage)
        info = info._replace(is_accepted=info.is_accepted & vertical_is_accepted)
        return new_state, info

    return kernel


def horizontal_slice(
    rng_key: PRNGKey,
    slicer: Callable,
    state: SliceState,
    m: int,
    max_shrinkage: int,
) -> tuple[SliceState, SliceInfo]:
    """Propose a new sample using the stepping-out and shrinking procedures.

    This function implements the core of the Hit-and-Run Slice Sampling algorithm.
    It first expands an interval (`[l, r]`) along the slice starting
    from `x0` and proceeding along direction `d` until both ends are outside
    the slice defined by `logslice` (stepping-out). Then, it samples
    points uniformly from this interval and shrinks the interval until a point
    is found that lies within the slice (shrinking).

    Parameters
    ----------
    rng_key
        A JAX PRNG key.
    slicer
        A function that takes a scalar `t` and returns a state and info on the
        slice.
    state
        The current slice sampling state.
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
    j = jnp.floor(m * v).astype(int)
    k = (m - 1) - j

    # Expand
    def step_body_fun(carry):
        i, s, t, _ = carry
        t += s
        _, is_accepted = slicer(t)
        i -= 1
        return i, s, t, is_accepted

    def step_cond_fun(carry):
        i, _, _, is_accepted = carry
        return is_accepted & (i > 0)

    j, _, l, _ = jax.lax.while_loop(step_cond_fun, step_body_fun, (j+1, -1, 1 - u, True))
    k, _, r, _ = jax.lax.while_loop(step_cond_fun, step_body_fun, (k+1, +1, -u, True))

    # Shrink
    def shrink_body_fun(carry):
        n, rng_key, l, r, state, is_accepted = carry

        rng_key, subkey = jax.random.split(rng_key)
        u = jax.random.uniform(subkey, minval=l, maxval=r)

        new_state, is_accepted = slicer(u)
        n += 1

        l = jnp.where(u < 0, u, l)
        r = jnp.where(u > 0, u, r)

        return n, rng_key, l, r, new_state, is_accepted

    def shrink_cond_fun(carry):
        n, _, _, _, _, is_accepted = carry
        return ~is_accepted & (n < max_shrinkage)

    carry = 0, rng_key, l, r, state, False
    carry = jax.lax.while_loop(shrink_cond_fun, shrink_body_fun, carry)
    n, _, _, _, new_state, is_accepted = carry
    new_state, (is_accepted, _, _) = static_binomial_sampling(
        rng_key, jnp.log(is_accepted), state, new_state
    )
    slice_info = SliceInfo(is_accepted, m + 1 - j - k, n)
    return new_state, slice_info


def build_hrss_kernel(
    generate_slice_direction_fn: Callable,
    stepper_fn: Callable,
    max_steps: int = 10,
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
    slice_kernel = build_kernel(stepper_fn, max_steps)

    def kernel(
        rng_key: PRNGKey, state: SliceState, logdensity_fn: Callable
    ) -> tuple[SliceState, SliceInfo]:
        rng_key, prop_key = jax.random.split(rng_key, 2)
        d = generate_slice_direction_fn(prop_key)
        constraint_fn = lambda x: jnp.array([])
        constraint = jnp.array([])
        strict = jnp.array([], dtype=bool)
        return slice_kernel(
            rng_key, state, logdensity_fn, d, constraint_fn, constraint, strict
        )

    return kernel


def default_stepper_fn(x: ArrayTree, d: ArrayTree, t: float) -> ArrayTree:
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
    position, is_accepted
    """
    return jax.tree.map(lambda x, d: x + t * d, x, d), True


def sample_direction_from_covariance(rng_key: PRNGKey, cov: Array) -> Array:
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
    generate_slice_direction_fn = partial(sample_direction_from_covariance, cov=cov)
    kernel = build_hrss_kernel(generate_slice_direction_fn, default_stepper_fn)
    init_fn = partial(init, logdensity_fn=logdensity_fn)
    step_fn = partial(kernel, logdensity_fn=logdensity_fn)
    return SamplingAlgorithm(init_fn, step_fn)
