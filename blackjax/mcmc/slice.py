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

from blackjax.types import PRNGKey, PyTree

__all__ = ["SliceState", "init", "kernel"]


class SliceState(NamedTuple):
    position: PyTree
    logdensity: PyTree
    widths: PyTree
    n: jnp.ndarray


def init(position: PyTree, logdensity_fn: Callable):
    logdensity = logdensity_fn(position)
    widths = jax.tree_map(lambda x: jnp.full(x.shape, 0.01), position)
    return SliceState(position, jnp.atleast_1d(logdensity), widths, jnp.atleast_1d(0.0))


def kernel(n_doublings: int) -> Callable:
    """Instantiate a slice sampling kernel.

    Implementation according to [1]. Doubling implementation inspired
    by Tensorflow probability's implementation

    Parameters
    ----------
    n_doublings: int
        maximal number of slice expansions

    References
    -------
    [1] Radford M. Neal "Slice sampling",
    The Annals of Statistics, Ann. Statist. 31(3), 705-767, (June 2003)
    """

    def one_step(rng_key: PRNGKey, state: SliceState, logdensity_fn: Callable):
        proposal_generator = slice_proposal(logdensity_fn, n_doublings)
        return proposal_generator(rng_key, state)

    return one_step


def slice_proposal(logdensity_fn, n_doublings) -> Callable:
    def generate(rng_key, state):
        order_key, rng_key = random.split(rng_key)
        n = state.n[0]
        positions, unravel_fn = jax.flatten_util.ravel_pytree(state.position)
        widths, _ = jax.flatten_util.ravel_pytree(state.widths)

        def conditional_proposal(rng_key, idx):
            return _sample_conditionally(
                rng_key, logdensity_fn, idx, positions, widths, n_doublings
            )

        def body_fn(carry, rn):
            seed, idx = rn
            positions, widths = carry
            xi, wi = conditional_proposal(seed, idx)
            positions = positions.at[idx].set(xi)
            nw = widths[idx] + (wi - widths[idx]) / (n + 1)
            widths = widths.at[idx].set(nw)
            return (positions, widths), (positions, widths)

        order = random.choice(
            order_key,
            jnp.arange(len(positions)),
            shape=(len(positions),),
            replace=False,
        )

        keys = random.split(rng_key, len(positions))
        (new_positions, new_widths), _ = jax.lax.scan(
            body_fn, (positions, widths), (keys, order)
        )

        new_positions = unravel_fn(new_positions)
        new_widths = unravel_fn(new_widths)
        new_state = SliceState(
            new_positions,
            jnp.atleast_1d(logdensity_fn(new_positions)),
            new_widths,
            jnp.atleast_1d(n + 1.0),
        )
        return new_state

    return generate


def _sample_conditionally(seed, logdensity_fn, idx, positions, widths, n_doublings):
    def cond_lp_fn(xi_to_set):
        return logdensity_fn({"theta": positions.at[idx].set(xi_to_set)})

    key, seed1, seed2 = random.split(seed, 3)
    x0, w0 = positions[idx], widths[idx]
    y = cond_lp_fn(x0) - random.exponential(key)
    left, right, _ = _doubling_fn(seed1, y, x0, cond_lp_fn, w0, n_doublings)
    x1 = _shrinkage_fn(seed2, y, x0, cond_lp_fn, left, right, w0)
    return x1, right - left


def _doubling_fn(rng, y, x0, cond_lp_fn, w, n_doublings):
    key1, key2, key3, key4 = random.split(rng, 4)
    left = x0 - w * random.uniform(key1)

    K = n_doublings + 1
    left_expands = random.bernoulli(key2, 0.5, (K,))
    width_multipliers = 2 ** jnp.arange(0, K, dtype=jnp.int32)
    widths = width_multipliers * w
    left_increments = jnp.cumsum(widths * left_expands)

    lefts = left - left_increments
    rights = left + widths
    left_lps = jax.vmap(cond_lp_fn)(lefts)
    right_lps = jax.vmap(cond_lp_fn)(rights)

    both_ok = jnp.logical_and(left_lps < y, right_lps < y)
    best_interval_idx = _best_interval(both_ok.astype(jnp.int32))

    return (
        lefts[best_interval_idx],
        rights[best_interval_idx],
        both_ok[best_interval_idx],
    )


def _best_interval(x):
    k = x.shape[0]
    mults = jnp.arange(2 * k, k, -1, dtype=x.dtype)
    shifts = jnp.arange(k, dtype=x.dtype)
    indices = jnp.argmax(mults * x + shifts).astype(x.dtype)
    return indices


def _shrinkage_fn(seed, y, x0, cond_lp_fn, left, right, w):
    def cond_fn(state):
        *_, found = state
        return jnp.logical_not(found)

    def body_fn(state):
        x1, left, right, seed, _ = state
        key, seed = random.split(seed)
        v = random.uniform(key)
        x1 = left + v * (right - left)

        found = jnp.logical_and(
            y < cond_lp_fn(x1), _accept_fn(y, x1, x0, cond_lp_fn, left, right, w)
        )

        left = jnp.where(x1 < x0, x1, left)
        right = jnp.where(x1 >= x0, x1, right)

        return x1, left, right, seed, found

    key, seed = random.split(seed)
    v = random.uniform(key)
    x1 = left + v * (right - left)
    x1, left, right, seed, _ = jax.lax.while_loop(
        cond_fn, body_fn, (x1, left, right, seed, False)
    )
    return x1


def _accept_fn(y, x1, x0, cond_lp_fn, left, right, w):
    def cond_fn(state):
        _, _, left, right, w, _, is_acceptable = state
        return jnp.logical_and(right - left > 1.1 * w, is_acceptable)

    def body_fn(state):
        x1, x0, left, right, w, D, _ = state
        mid = (left + right) / 2
        D = jnp.logical_or(
            jnp.logical_or(
                jnp.logical_and(x0 < mid, x1 >= mid),
                jnp.logical_and(x0 >= mid, x1 < mid),
            ),
            D,
        )
        right = jnp.where(x1 < mid, mid, right)
        left = jnp.where(x1 >= mid, mid, left)

        left_is_not_acceptable = y >= cond_lp_fn(left)
        right_is_not_acceptable = y >= cond_lp_fn(right)
        interval_is_not_acceptable = jnp.logical_and(
            left_is_not_acceptable, right_is_not_acceptable
        )
        is_still_acceptable = jnp.logical_not(
            jnp.logical_and(D, interval_is_not_acceptable)
        )
        return x1, x0, left, right, w, D, is_still_acceptable

    *_, is_acceptable = jax.lax.while_loop(
        cond_fn, body_fn, (x1, x0, left, right, w, False, True)
    )
    return is_acceptable
