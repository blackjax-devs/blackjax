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
"""Public API for the Slice sampling kernel."""

from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp
from jax import random

from blackjax.base import SamplingAlgorithm, build_sampling_algorithm
from blackjax.types import Array, ArrayLikeTree, ArrayTree, PRNGKey

__all__ = ["SliceState", "SliceInfo", "init", "build_kernel", "as_top_level_api"]


class SliceState(NamedTuple):
    """State of the Slice sampling chain.

    position
        Current position of the chain.
    logdensity
        Current value of the log-density.

    """

    position: ArrayTree
    logdensity: float


class SliceInfo(NamedTuple):
    """Additional information on the Slice sampling transition.

    bracket_widths
        Per-dimension realized bracket widths produced by the doubling
        procedure on this step.  Useful for diagnosing proposal efficiency
        (very wide brackets relative to the posterior suggest the initial
        width is too small; very narrow ones suggest it may be too large).

    """

    bracket_widths: ArrayTree


def init(position: ArrayLikeTree, logdensity_fn: Callable) -> SliceState:
    """Create an initial state from a position and log-density function.

    Parameters
    ----------
    position
        Initial position of the chain.
    logdensity_fn
        Log-probability density function of the target distribution.

    Returns
    -------
    The initial state of the Slice sampling chain.
    """
    logdensity = logdensity_fn(position)
    return SliceState(position, jnp.atleast_1d(logdensity))


def build_kernel(
    n_doublings: int = 10, initial_widths: float | Array = 1.0
) -> Callable:
    """Build a Slice sampling kernel.

    Implementation according to [1]. Doubling implementation inspired
    by TensorFlow Probability's implementation.

    Parameters
    ----------
    n_doublings
        Maximum number of slice interval doublings.
    initial_widths
        Fixed bracket width(s) used as the starting interval for the doubling
        procedure.  Accepts either a scalar (applied uniformly to every
        coordinate, default: 1.0) or a 1-D array of length equal to the
        total flattened position dimension ``D``, giving a per-coordinate
        width — mirroring TFP's per-dimension ``step_size``.  The value 1.0
        is a reasonable default for posterior scales in the range 0.1–10:
        the doubling procedure rapidly expands the bracket when the width is
        too small, so correctness is insensitive to the exact value.  Pass
        a smaller value (or per-dimension array) for very narrow posteriors,
        or a larger one for very diffuse ones, to improve efficiency.

    Returns
    -------
    A kernel that takes a rng_key and a Pytree that contains the current state
    of the chain and returns a new state of the chain along with information
    about the transition.

    References
    ----------
    .. [1] Radford M. Neal, "Slice sampling", The Annals of Statistics,
       Ann. Statist. 31(3), 705-767, (June 2003).
    """

    def kernel(
        rng_key: PRNGKey, state: SliceState, logdensity_fn: Callable
    ) -> tuple[SliceState, SliceInfo]:
        proposal_generator = _slice_proposal(logdensity_fn, n_doublings, initial_widths)
        new_state, bracket_widths = proposal_generator(rng_key, state)
        return new_state, SliceInfo(bracket_widths)

    return kernel


def as_top_level_api(
    logdensity_fn: Callable,
    *,
    n_doublings: int = 10,
    initial_widths: float | Array = 1.0,
) -> SamplingAlgorithm:
    """Implements the user interface for the Slice sampling kernel.

    Examples
    --------

    A new Slice sampling kernel can be initialized and used with the following
    code:

    .. code::

        slice_sampling = blackjax.slice_sampling(logdensity_fn, n_doublings=10)
        state = slice_sampling.init(position)
        new_state, info = slice_sampling.step(rng_key, state)

    We can JIT-compile the step function for better performance:

    .. code::

        step = jax.jit(slice_sampling.step)
        new_state, info = step(rng_key, state)

    Parameters
    ----------
    logdensity_fn
        The log-density function of the distribution we wish to sample from.
    n_doublings
        Maximum number of slice interval doublings (default: 10).
    initial_widths
        Fixed bracket width(s) used as the starting interval for the doubling
        procedure.  A scalar applies to all coordinates; a 1-D array of
        length ``D`` (total flattened position dimension) gives per-coordinate
        widths, mirroring TFP's per-dimension ``step_size`` (default: 1.0).

    Returns
    -------
    A ``SamplingAlgorithm``.
    """
    kernel = build_kernel(n_doublings, initial_widths)
    return build_sampling_algorithm(kernel, init, logdensity_fn)


def _slice_proposal(
    logdensity_fn: Callable, n_doublings: int, initial_widths: float | Array
) -> Callable:
    def generate(rng_key: PRNGKey, state: SliceState) -> tuple[SliceState, ArrayTree]:
        order_key, rng_key = random.split(rng_key)
        positions, unravel_fn = jax.flatten_util.ravel_pytree(state.position)
        # Scalar → uniform width; array → per-coordinate widths (length D).
        # jnp.broadcast_to handles both: a scalar broadcasts to (D,) cleanly,
        # and an array of shape (D,) is validated by broadcast semantics
        # (a length mismatch raises an error at build time, before any JIT).
        widths = jnp.broadcast_to(jnp.asarray(initial_widths).ravel(), positions.shape)

        def body_fn(
            carry: tuple[Array, Array], rn: tuple[PRNGKey, Array]
        ) -> tuple[tuple[Array, Array], tuple[Array, Array]]:
            seed, idx = rn
            positions, bracket_widths = carry
            xi, bi = _sample_conditionally(
                seed, logdensity_fn, unravel_fn, idx, positions, widths, n_doublings
            )
            positions = positions.at[idx].set(xi)
            bracket_widths = bracket_widths.at[idx].set(bi)
            return (positions, bracket_widths), (positions, bracket_widths)

        order = random.choice(
            order_key,
            jnp.arange(len(positions)),
            shape=(len(positions),),
            replace=False,
        )

        keys = random.split(rng_key, len(positions))
        bracket_widths_init = jnp.zeros_like(positions)
        (new_positions, new_bracket_widths), _ = jax.lax.scan(
            body_fn, (positions, bracket_widths_init), (keys, order)
        )

        new_positions = unravel_fn(new_positions)
        new_bracket_widths = unravel_fn(new_bracket_widths)
        new_state = SliceState(
            new_positions,
            jnp.atleast_1d(logdensity_fn(new_positions)),
        )
        return new_state, new_bracket_widths

    return generate


def _sample_conditionally(
    seed: PRNGKey,
    logdensity_fn: Callable,
    unravel_fn: Callable,
    idx: Array,
    positions: Array,
    widths: Array,
    n_doublings: int,
) -> tuple[Array, Array]:
    def conditional_logdensity_fn(xi_to_set: Array) -> Array:
        return logdensity_fn(unravel_fn(positions.at[idx].set(xi_to_set)))

    key, seed1, seed2 = random.split(seed, 3)
    x0, w0 = positions[idx], widths[idx]
    logdensity = conditional_logdensity_fn(x0) - random.exponential(key)
    left, right, _ = _doubling_fn(
        seed1, logdensity, x0, conditional_logdensity_fn, w0, n_doublings
    )
    x1 = _shrinkage_fn(
        seed2, logdensity, x0, conditional_logdensity_fn, left, right, w0
    )
    return x1, right - left


# --- Doubling ---


def _doubling_fn(
    rng: PRNGKey,
    logdensity: Array,
    x0: Array,
    conditional_logdensity_fn: Callable,
    w: Array,
    n_doublings: int,
) -> tuple[Array, Array, Array]:
    key1, key2 = random.split(rng, 2)
    initial_left = x0 - w * random.uniform(key1)
    initial_right = initial_left + w

    K = n_doublings + 1
    left_expands = random.bernoulli(key2, 0.5, (K,))
    right_expands = 1 - left_expands.astype(jnp.int32)
    step_widths = w * (2 ** jnp.arange(0, K, dtype=jnp.float32))

    # Exclusive cumsum: increment at level k = sum of expansions at steps 0..k-1.
    # rights[0] and lefts[0] are the initial interval (0 doublings).
    left_increments = jnp.concatenate(
        [jnp.zeros(1), jnp.cumsum(step_widths * left_expands)[:-1]]
    )
    right_increments = jnp.concatenate(
        [jnp.zeros(1), jnp.cumsum(step_widths * right_expands)[:-1]]
    )

    lefts = initial_left - left_increments
    rights = initial_right + right_increments
    left_lps = jax.vmap(conditional_logdensity_fn)(lefts)
    right_lps = jax.vmap(conditional_logdensity_fn)(rights)

    both_ok = jnp.logical_and(left_lps < logdensity, right_lps < logdensity)
    best_interval_idx = _best_interval(both_ok.astype(jnp.int32))

    return (
        lefts[best_interval_idx],
        rights[best_interval_idx],
        both_ok[best_interval_idx],
    )


def _best_interval(x: Array) -> Array:
    k = x.shape[0]
    mults = jnp.arange(2 * k, k, -1, dtype=x.dtype)
    shifts = jnp.arange(k, dtype=x.dtype)
    indices = jnp.argmax(mults * x + shifts).astype(x.dtype)
    return indices


# --- Shrinkage ---


def _shrinkage_fn(
    seed: PRNGKey,
    logdensity: Array,
    x0: Array,
    conditional_logdensity_fn: Callable,
    left: Array,
    right: Array,
    w: Array,
) -> Array:
    def cond_fn(state: tuple) -> Array:
        *_, found = state
        return jnp.logical_not(found)

    def body_fn(state: tuple) -> tuple:
        x1, left, right, seed, _ = state
        key, seed = random.split(seed)
        v = random.uniform(key)
        x1 = left + v * (right - left)

        found = jnp.logical_and(
            logdensity < conditional_logdensity_fn(x1),
            _accept_fn(logdensity, x1, x0, conditional_logdensity_fn, left, right, w),
        )

        left = jnp.where(x1 < x0, x1, left)
        right = jnp.where(x1 >= x0, x1, right)

        return x1, left, right, seed, found

    x1, left, right, seed, _ = jax.lax.while_loop(
        cond_fn, body_fn, (x0, left, right, seed, False)
    )
    return x1


# --- Acceptance test ---


def _accept_fn(
    logdensity: Array,
    x1: Array,
    x0: Array,
    conditional_logdensity_fn: Callable,
    left: Array,
    right: Array,
    w: Array,
) -> Array:
    # The 1.1 * w termination threshold is from Neal 2003 Fig. 6: the
    # acceptance test bisects the interval until it is within 10% of the
    # original width w, at which point no further refinement is needed.
    def cond_fn(state: tuple) -> Array:
        _, _, left, right, w, _, is_acceptable = state
        return jnp.logical_and(right - left > 1.1 * w, is_acceptable)

    def body_fn(state: tuple) -> tuple:
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

        left_is_not_acceptable = logdensity >= conditional_logdensity_fn(left)
        right_is_not_acceptable = logdensity >= conditional_logdensity_fn(right)
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
