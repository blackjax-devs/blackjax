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
    l = step(x0, n*w)
    r = step(x0, n*(w-1))

    # Expand l
    def expand_l(carry):
        l0, within, counter = carry
        counter += 1
        l = step(l0, n)
        within = jnp.logical_and(logL(l) > logL0, logpi(l) >= logpi0)
        return l, within, counter

    def cond_fun_l(carry):
        within = carry[1]
        return within

    within = True
    carry = (l, within, 0)
    l, _, l_i = jax.lax.while_loop(cond_fun_l, expand_l, carry)

    # Expand r
    def expand_r(carry):
        r0, within, counter = carry
        counter += 1
        r = step(r0, -n)
        within = jnp.logical_and(logL(r) > logL0, logpi(r) >= logpi0)
        return r, within, counter

    def cond_fun_r(carry):
        within = carry[1]
        return within

    within = True
    carry = (r, within, 0)
    r, _, r_i = jax.lax.while_loop(cond_fun_r, expand_r, carry)

    # Shrink
    def shrink_step(carry):
        l, r, xminus1, _, key, within, counter = carry
        counter += 1

        key, subkey = jax.random.split(key)
        u = jax.random.uniform(subkey)

        x1 = step(l, u*(r-l))

        logLx1 = logL(x1)
        within_new = jnp.logical_and(logLx1 > logL0, logpi(x1) >= logpi0)

        s_vec = jax.tree_util.tree_map(
            lambda x1, x0, r, l: (x1 - x0) * (r - l),
            x1,
            x0,
            r,
            l,
        )
        s_flat, _ = jax.flatten_util.ravel_pytree(s_vec)
        s = jnp.sum(s_flat) > 0
        # this is slow and should be optimized, dont need to check all the elements!
        l = jax.tree_map(lambda l_i, x1_i: jnp.where(s, l_i, x1_i), l, x1)
        r = jax.tree_map(lambda r_i, x1_i: jnp.where(s, x1_i, r_i), r, x1)
        return l, r, x1, logLx1, key, within_new, counter

    def cond_fun(carry):
        within = carry[-2]
        return ~within

    within = False
    carry = (l, r, x0, 0.0, key, within, 0)
    l, r, x1, logl, key, within, s_i = jax.lax.while_loop(cond_fun, shrink_step, carry)
    slice_state = SliceState(x1, logpi(x1), logl)
    slice_info = SliceInfo(l_i, r_i, s_i, (l_i + r_i + s_i))
    return slice_state, slice_info
