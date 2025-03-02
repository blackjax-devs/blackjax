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
from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp

from blackjax.types import Array, ArrayTree, PRNGKey

__all__ = [
    "SliceState",
    "SliceInfo",
    "init",
    "build_slice_kernel",
]


class SliceState(NamedTuple):
    position: ArrayTree
    logdensity: ArrayTree
    loglikelihood: ArrayTree


class SliceInfo(NamedTuple):
    l_steps: Array
    r_steps: Array
    s_steps: Array
    evals: Array


def init(position: ArrayTree, logdensity_fn: Callable, loglikelihood: Array):
    logdensity = logdensity_fn(position)
    return SliceState(position, logdensity, loglikelihood)


def build_slice_kernel(
    logprior_fn: Callable,
    loglikelihood_fn: Callable,
    logL0: Array,
    proposal_distribution: Callable,
    stepper: Callable,
) -> Callable:
    """Instantiate a (constrained) hit and run slice sampling kernel.

    Parameters
    ----------
    logprior_fn : Callable
        Function to compute the log prior probability.
    loglikelihood_fn : Callable
        Function to compute the log likelihood.
    logL0 : Array
        Initial log-likelihood values.
    proposal_distribution : Callable
        Function to generate a proposal PyTree for slice stepping.
    stepper : Callable
        Function to compute the next step in the slice stepping.

    Returns
    -------
    Callable
        A slice sampling kernel function.
    References
    -------
    [1] Radford M. Neal "Slice sampling",
    The Annals of Statistics, Ann. Statist. 31(3), 705-767, (June 2003)
    """

    def kernel(rng_key: PRNGKey, state: SliceState):
        rng_key, vertical_slice_key = jax.random.split(rng_key)
        logpi0, _ = vertical_slice(vertical_slice_key, logprior_fn, state.position)
        rng_key, horizontal_slice_key = jax.random.split(rng_key)
        slice_state, slice_info = horizontal_slice_proposal(
            horizontal_slice_key,
            state.position,
            proposal_distribution,
            stepper,
            loglikelihood_fn,
            logL0,
            logprior_fn,
            logpi0,
        )

        return slice_state, slice_info

    return kernel


def vertical_slice(rng, logdensity, positions):
    logpi = logdensity(positions)
    logpi0 = logpi + jnp.log(jax.random.uniform(rng))
    return logpi0, logpi


def horizontal_slice_proposal(key, x0, proposal, step, logL, logL0, logpi, logpi0):
    key, proposal_key = jax.random.split(key)
    n = proposal(proposal_key)

    # # Initial bounds
    key, subkey = jax.random.split(key)
    w = jax.random.uniform(subkey)

    def body_fun(carry):
        _, s, t, count = carry
        t += s
        x = step(x0, n, t)
        within = jnp.logical_and(logL(x) > logL0, logpi(x) >= logpi0)
        count += 1
        return within, s, t, count

    def cond_fun(carry):
        within = carry[0]
        return within

    _, _, l, count_l = jax.lax.while_loop(cond_fun, body_fun, (True, +1, w-1, 0))
    _, _, r, count_r = jax.lax.while_loop(cond_fun, body_fun, (True, -1, w,   0))

    # Shrink
    def body_fun(carry):
        _, l, r, _, _, _, key, count = carry
        count += 1

        key, subkey = jax.random.split(key)
        u = jax.random.uniform(subkey, minval=r, maxval=l)
        x = step(x0, n, u)
        # check for nan values

        logL_x = logL(x)
        logpi_x = logpi(x)
        within = jnp.logical_and(logL_x > logL0, logpi_x >= logpi0)

        l = jnp.where(u>0, u, l)
        r = jnp.where(u<0, u, r)

        return within, l, r, x, logL_x, logpi_x, key, count

    def cond_fun(carry):
        within = carry[0]
        return ~within

    carry = (False, l, r, x0, -jnp.inf, -jnp.inf, key, 0)
    carry = jax.lax.while_loop(cond_fun, body_fun, carry)
    _, l, r, x, logL_x, logpi_x, key, count = carry
    slice_state = SliceState(x, logpi_x, logL_x)
    slice_info = SliceInfo(count_l, count_r, count, (count_l + count_r + count))
    return slice_state, slice_info
