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
    logslice
        The log-height defining the slice for sampling. Defaults to infinity.
    """

    position: ArrayLikeTree
    logdensity: float
    logslice: float = jnp.inf


class SliceInfo(NamedTuple):
    """Additional information about the Slice Sampling transition.

    This information can be used for diagnostics and monitoring the sampler's
    performance.

    Attributes
    ----------
    constraint
        The constraint values at the final accepted position.
    l_steps
        The number of steps taken to expand the interval to the left during the
        "stepping-out" phase.
    r_steps
        The number of steps taken to expand the interval to the right during the
        "stepping-out" phase.
    s_steps
        The number of steps taken during the "shrinking" phase to find an
        acceptable sample.
    is_accepted
        A boolean indicating whether the proposed sample was accepted.
    """

    d: ArrayLikeTree = jnp.array([])
    constraint: Array = jnp.array([])
    l_steps: int = 0
    r_steps: int = 0
    s_steps: int = 0
    is_accepted: bool = True


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
        rng_key, vs_key, hs_key = jax.random.split(rng_key, 3)
        intermediate_state, vs_info = vertical_slice(vs_key, state)
        new_state, hs_info = horizontal_slice(
            hs_key,
            intermediate_state,
            d,
            stepper_fn,
            logdensity_fn,
            constraint_fn,
            constraint,
            strict,
        )

        info = SliceInfo(
            d=d,
            constraint=hs_info.constraint,
            l_steps=hs_info.l_steps,
            r_steps=hs_info.r_steps,
            s_steps=hs_info.s_steps,
            is_accepted=True,
        )

        return new_state, info

    return kernel


def vertical_slice(rng_key: PRNGKey, state: SliceState) -> tuple[SliceState, SliceInfo]:
    """Define the vertical slice for the Slice Sampling algorithm.

    This function determines the height `y` for the horizontal slice by sampling
    uniformly from `[0, p(x)]`, where `p(x)` is the target density at the current
    position `x`. This is equivalent to sampling `log_y` uniformly from
    `(-inf, log_p(x)]`.

    Parameters
    ----------
    rng_key
        A JAX PRNG key.
    state
        The current slice sampling state.

    Returns
    -------
    tuple[SliceState, SliceInfo]
        A tuple containing the updated state with the slice height set and
        info about the vertical slice step.
    """
    logslice = state.logdensity + jnp.log(jax.random.uniform(rng_key))
    new_state = state._replace(logslice=logslice)
    info = SliceInfo()
    return new_state, info


def horizontal_slice(
    rng_key: PRNGKey,
    state: SliceState,
    d: ArrayTree,
    stepper_fn: Callable,
    logdensity_fn: Callable,
    constraint_fn: Callable,
    constraint: Array,
    strict: Array,
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
    x0
        The current position (PyTree).
    d
        The direction (PyTree) for proposing moves.
    stepper_fn
        A function `(x0, d, t) -> x_new` that computes a new point by
        moving `t` units along direction `d` from `x0`.
    logdensity_fn
        The log-density function of the target distribution.
    constraint_fn
        A function that evaluates additional constraints on the position beyond
        the target distribution. Takes a position (PyTree) and returns an array
        of constraint values. These values are compared against `constraint`
        thresholds to determine if a position is acceptable. For example, in
        nested sampling, this could evaluate the log-likelihood to ensure it
        exceeds a minimum threshold.
    constraint
        An array of constraint threshold values that must be satisfied.
        Each constraint value from `constraint_fn(x)` is compared against the
        corresponding threshold in this array.
    strict
        An array of boolean flags indicating whether each constraint should be
        strict (constraint_fn(x) > constraint) or non-strict
        (constraint_fn(x) >= constraint).

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
    x0 = state.position

    def body_fun(carry):
        _, s, t, n = carry
        t += s
        x = stepper_fn(x0, d, t)
        logdensity_x = logdensity_fn(x)
        constraint_x = constraint_fn(x)
        constraints = jnp.where(
            strict, constraint_x > constraint, constraint_x >= constraint
        )
        constraints = jnp.append(constraints, logdensity_x >= state.logslice)
        within = jnp.all(constraints)
        n += 1
        return within, s, t, n

    def cond_fun(carry):
        within = carry[0]
        return within

    # Expand
    _, _, l, l_steps = jax.lax.while_loop(cond_fun, body_fun, (True, -1, -u, 0))
    _, _, r, r_steps = jax.lax.while_loop(cond_fun, body_fun, (True, +1, 1 - u, 0))

    # Shrink
    def shrink_body_fun(carry):
        _, l, r, _, _, _, rng_key, s_steps = carry
        s_steps += 1

        rng_key, subkey = jax.random.split(rng_key)
        u = jax.random.uniform(subkey, minval=l, maxval=r)
        x = stepper_fn(x0, d, u)

        logdensity_x = logdensity_fn(x)
        constraint_x = constraint_fn(x)
        constraints = jnp.where(
            strict, constraint_x > constraint, constraint_x >= constraint
        )
        constraints = jnp.append(constraints, logdensity_x >= state.logslice)
        within = jnp.all(constraints)

        l = jnp.where(u < 0, u, l)
        r = jnp.where(u > 0, u, r)

        return within, l, r, x, logdensity_x, constraint_x, rng_key, s_steps

    def shrink_cond_fun(carry):
        within = carry[0]
        return ~within

    carry = (False, l, r, x0, -jnp.inf, constraint, rng_key, 0)
    carry = jax.lax.while_loop(shrink_cond_fun, shrink_body_fun, carry)
    _, l, r, x, logdensity_x, constraint_x, rng_key, s_steps = carry
    slice_state = SliceState(x, logdensity_x)
    slice_info = SliceInfo(d, constraint_x, l_steps, r_steps, s_steps)
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
        constraint_fn = lambda x: jnp.array([])
        constraint = jnp.array([])
        strict = jnp.array([])
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
    ArrayTree
        The new position.
    """
    return jax.tree.map(lambda x, d: x + t * d, x, d)


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
