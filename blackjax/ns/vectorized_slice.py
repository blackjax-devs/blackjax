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
"""Public API for the  Slice sampling Kernel"""
from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp
from jax import random

from blackjax.base import SamplingAlgorithm
from blackjax.types import ArrayLikeTree, ArrayTree, PRNGKey, Array

__all__ = [
    "SliceState",
    "SliceInfo",
    "init",
    "build_kernel",
    # "as_top_level_api",
]


# def as_top_level_api(
#     loglikelihood_fn: Callable,
#     *,
#     n_doublings: int = 5,
# ) -> SamplingAlgorithm:
#     """Implements the (basic) user interface for the Slice sampling kernel.

#     Examples
#     --------

#     A slice sampling kernel can be initialized like this:

#     .. code::

#         slice = blackjax.slice(logdensity_fn, n_doublings)
#         state = slice.init(position)
#         new_state, info = slice.step(rng_key, state)

#     We can JIT-compile the step function for better performance

#     .. code::

#         step = jax.jit(slice.step)
#         new_state, info = step(rng_key, state)

#     Parameters
#     ----------
#     logdensity_fn: Callable
#         the unnormalized posterior distribution we wish to sample from.
#     n_doublings: int
#         maximal number of slice expansions.

#     Returns
#     -------
#     A ``MCMCSamplingAlgorithm``.
#     """

#     kernel = build_kernel(n_doublings)

#     def init_fn(position: ArrayLikeTree, rng_key=None):
#         del rng_key
#         return init(position, loglikelihood_fn)

#     def step_fn(rng_key: PRNGKey, state):
#         return kernel(rng_key, state, loglikelihood_fn)

#     return SamplingAlgorithm(init_fn, step_fn)


class SliceState(NamedTuple):
    position: ArrayTree
    logdensity: ArrayTree
    loglikelihood: ArrayTree


class SliceInfo(NamedTuple):
    l_steps: Array
    r_steps: Array
    s_steps: Array


def init(position: ArrayTree, logdensity_fn: Callable, loglikelihood: Array):
    logdensity = logdensity_fn(position)
    return SliceState(position, logdensity, loglikelihood)


def build_kernel(
    cov: Array,
    logL0: Array,
    logprior_fn: Callable,
    loglikelihood_fn: Callable,
) -> Callable:
    """Instantiate a slice sampling kernel.

    Implementation according to [1]. Doubling implementation inspired
    by Tensorflow probability's implementation. Performs a univariate update in
    each dimension.

    Parameters
    ----------
    n_doublings: int
        maximal number of slice expansions

    References
    -------
    [1] Radford M. Neal "Slice sampling",
    The Annals of Statistics, Ann. Statist. 31(3), 705-767, (June 2003)
    """

    def one_step(rng_key: PRNGKey, state: SliceState):
        rng_key, vertical_slice_key = jax.random.split(rng_key)
        logpi0, logpi = vertical_slice(
            vertical_slice_key, logprior_fn, state.position
        )
        rng_key, horizontal_slice_key = jax.random.split(rng_key)
        slice_state, slice_info = horizontal_slice_proposal(
            horizontal_slice_key,
            state.position,
            cov,
            loglikelihood_fn,
            logL0,
            logprior_fn,
            logpi0,
        )

        return slice_state, slice_info

    return one_step


def vertical_slice(rng, logdensity, positions):
    logpi = logdensity(positions)
    logpi0 = logpi + jnp.log(
        jax.random.uniform(rng, shape=(positions.shape[0],))
    )
    return logpi0, logpi


def horizontal_slice_proposal(key, x0, cov, logL, logL0, logpi, logpi0):
    # Ensure cov_ and x0 are JAX arrays
    cov = jnp.asarray(cov)
    x0 = jnp.atleast_2d(x0)

    # Random direction
    key, subkey = jax.random.split(key)
    n = jax.random.multivariate_normal(
        subkey, jnp.zeros(x0.shape[1]), cov, shape=(x0.shape[0],)
    )  # Standard normal samples

    # Compute Mahalanobis norms and normalize n
    invcov = jnp.linalg.inv(cov)
    dim = n.shape[-1]  # Get the dimension of the vector
    norm = jnp.sqrt(dim * jnp.einsum("...i,...ij,...j", n, invcov, n))
    n = n / norm[..., None]


    # Initial bounds
    key, subkey = jax.random.split(key)
    w = jax.random.uniform(subkey, shape=(x0.shape[0],))
    l = x0 + w[:, None] * n
    r = x0 + (w[:, None] - 1) * n

    # Expand l
    def expand_l(carry):
        l, within, counter = carry
        counter += 1
        l = l + within[:, None] * n
        within = jnp.logical_and(logL(l) > logL0, logpi(l) > logpi0)
        return l, within, counter

    def cond_fun_l(carry):
        within = carry[1]
        return jnp.any(within)

    within = jnp.ones(x0.shape[0], dtype=bool)
    carry = (l, within, 0)
    l, _, l_i = jax.lax.while_loop(cond_fun_l, expand_l, carry)
    # Expand r
    def expand_r(carry):
        r, within, counter = carry
        counter += 1
        r = r - within[:, None] * n
        within = jnp.logical_and(logL(r) > logL0, logpi(r) > logpi0)
        return r, within, counter

    def cond_fun_r(carry):
        within = carry[1]
        return jnp.any(within)

    within = jnp.ones(x0.shape[0], dtype=bool)
    carry = (r, within, 0)
    r, _, r_i = jax.lax.while_loop(cond_fun_r, expand_r, carry)

    # Shrink
    def shrink_step(carry):
        l, r, _, _, key, within, counter = carry
        counter += 1
        key, subkey = jax.random.split(key)
        u = jax.random.uniform(subkey, shape=(x0.shape[0],))
        x1 = l + u[:, None] * (r - l)
        logLx1 = logL(x1)
        within_new = jnp.logical_and(logLx1 > logL0, logpi(x1) > logpi0)
        s = jnp.sum((x1 - x0) * (r - l), axis=-1) > 0
        condition_l = (~within_new) & (~s)
        l = jnp.where(condition_l[:, None], x1, l)
        condition_r = (~within_new) & s
        r = jnp.where(condition_r[:, None], x1, r)
        return l, r, x1, logLx1, key, within_new, counter

    def cond_fun(carry):
        within = carry[-2]
        return ~jnp.all(within)

    within = jnp.zeros(x0.shape[0], dtype=bool)
    carry = (l, r, x0, jnp.zeros(x0.shape[0]), key, within, 0)
    l, r, x1, logl, key, within, s_i = jax.lax.while_loop(
        cond_fun, shrink_step, carry
    )
    slice_state = SliceState(x1,logpi(x1), logl)
    slice_info = SliceInfo(
        l_i, r_i, s_i
    )
    return slice_state, slice_info
