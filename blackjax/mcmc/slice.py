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
"""Public API for the Slice sampling family.

Every slice update is univariate: a one-dimensional slice through the current
point. Multivariate behaviour is determined entirely by the proposal generator
that produces the line, ``proposal_generator(rng_key, position, logdensity_fn)
-> slice_fn`` with ``slice_fn(t) -> (state, is_valid)``. The candidate state is
threaded straight through the kernel, so a proposal can record extra quantities
on it.

Two samplers are built on this spine:

1. Multivariate slice: one univariate slice along a random direction. This is
   the top-level ``blackjax.slice_sampling`` (:func:`as_top_level_api`), with
   the direction drawn by :func:`direction_proposal` (a ``scale``-shaped random
   direction, unit by default). Chaining such random-direction moves is the
   hit-and-run strategy.

2. Coordinate-wise (slice-within-Gibbs, :func:`coordinate_slice`): sweep the
   coordinate axes in turn, updating each full conditional with a univariate
   slice.

The one-dimensional interval is built by the stepping-out or doubling procedure
of Neal (2003), passed as a callable (``interval=stepping_out`` or
``interval=doubling``), then narrowed by shrinkage to draw the new point.
Doubling additionally applies the Fig. 6 acceptance test. Additional
constraints are not built in but added downstream by overriding the proposal,
which gates on ``is_valid`` and may record extra quantities on the state.

References
----------
.. [1] Radford M. Neal, "Slice sampling", The Annals of Statistics,
   Ann. Statist. 31(3), 705-767, (June 2003).
"""

from typing import Callable, NamedTuple, TypeAlias

import jax
import jax.flatten_util
import jax.numpy as jnp
from jax import random

from blackjax.base import SamplingAlgorithm, build_sampling_algorithm
from blackjax.types import Array, ArrayLikeTree, ArrayTree, PRNGKey
from blackjax.util import generate_gaussian_noise

AcceptFn: TypeAlias = Callable[[Array], Array]

__all__ = [
    "SliceState",
    "SliceInfo",
    "init",
    "build_kernel",
    "build_coordinate_kernel",
    "as_top_level_api",
    "coordinate_slice",
    "direction_proposal",
    "coordinate_proposal",
    "sample_direction",
    "stepping_out",
    "doubling",
    "random_order",
    "fixed_order",
]


class SliceState(NamedTuple):
    """State of the Slice sampling chain.

    position
        Current position of the chain.
    logdensity
        Log-density of the target at ``position``.

    """

    position: ArrayTree
    logdensity: float


class SliceInfo(NamedTuple):
    """Additional information on a Slice sampling transition.

    is_accepted
        Whether shrinkage found a valid point within ``max_shrinkage`` steps.
        Always ``True`` for an unconstrained target (the slice always contains
        the current point); can be ``False`` when the proposal gates a
        constraint into ``is_valid`` and the budget is exhausted, leaving the
        chain in place.
        For the coordinate sweep it is ``True`` only if every coordinate
        succeeded.
    num_expansions
        Number of interval expansions (stepping-out steps or doublings).
        Summed over coordinates for the sweep.
    num_shrink
        Number of shrinkage evaluations taken to find the new point.
        Summed over coordinates for the sweep.
    bracket_left, bracket_right
        The realized slice bracket in the 1-D slice coordinate ``t``, where the
        current point sits at ``t = 0`` (so typically
        ``bracket_left <= 0 <= bracket_right``). For the multivariate slice these
        are scalars; for the coordinate sweep they are per-axis ``t`` values, a
        PyTree aligned with ``position``. The bracket width is
        ``bracket_right - bracket_left``.

    """

    is_accepted: Array
    num_expansions: Array
    num_shrink: Array
    bracket_left: ArrayTree
    bracket_right: ArrayTree


def init(position: ArrayLikeTree, logdensity_fn: Callable) -> SliceState:
    """Create an initial state from a position and log-density function."""
    return SliceState(position, logdensity_fn(position))


def stepping_out(
    rng_key: PRNGKey, in_slice: Callable, width: float, max_expansions: int
) -> tuple[Array, Array, Array, AcceptFn]:
    """Neal (2003) Fig. 3 stepping-out interval, in t-space (x0 at t=0).

    An interval procedure is a pluggable callable (pass it as
    ``interval=stepping_out``). It returns its own acceptance test so the kernel
    never branches on a name; stepping-out needs none, so ``accept_fn`` always
    returns ``True``.

    Returns
    -------
    The tuple ``(left, right, num_expansions, accept_fn)``: the bracket
    endpoints, the number of expansions, and the acceptance test.
    """
    u_key, jk_key = random.split(rng_key)
    u = random.uniform(u_key)
    left = -width * u
    right = left + width

    v = random.uniform(jk_key)
    j = jnp.floor(max_expansions * v).astype(int)
    k = (max_expansions - 1) - j

    def left_cond(carry):
        left, n = carry
        return in_slice(left) & (n > 0)

    def left_body(carry):
        left, n = carry
        return left - width, n - 1

    left, jl = jax.lax.while_loop(left_cond, left_body, (left, j))

    def right_cond(carry):
        right, n = carry
        return in_slice(right) & (n > 0)

    def right_body(carry):
        right, n = carry
        return right + width, n - 1

    right, kr = jax.lax.while_loop(right_cond, right_body, (right, k))
    num_expansions = (j - jl) + (k - kr)
    accept_fn = lambda t: jnp.asarray(True)  # noqa: E731  (stepping-out: no test)
    return left, right, num_expansions, accept_fn


def _best_interval(x: Array) -> Array:
    """Index of the first ``True`` (else the last), for the doubling cap."""
    k = x.shape[0]
    mults = jnp.arange(2 * k, k, -1, dtype=x.dtype)
    shifts = jnp.arange(k, dtype=x.dtype)
    return jnp.argmax(mults * x + shifts).astype(int)


def doubling(
    rng_key: PRNGKey, in_slice: Callable, width: float, max_expansions: int
) -> tuple[Array, Array, Array, AcceptFn]:
    """Neal (2003) Fig. 4 doubling interval, vectorized, in t-space.

    Expands one (randomly chosen) side at a time, doubling the bracket each
    step, until both ends are outside the slice or ``max_expansions`` is hit. A
    pluggable interval callable, like :func:`stepping_out`. Its ``accept_fn`` is
    Neal's Fig. 6 acceptance test bound to this (original) bracket, which is
    required for doubling's reversibility.

    Returns
    -------
    The tuple ``(left, right, num_expansions, accept_fn)``: the bracket
    endpoints, the number of expansions, and the acceptance test.
    """
    key1, key2 = random.split(rng_key)
    initial_left = -width * random.uniform(key1)
    initial_right = initial_left + width

    k = max_expansions + 1
    left_expands = random.bernoulli(key2, 0.5, (k,))
    right_expands = 1 - left_expands.astype(int)
    step_widths = width * (2.0 ** jnp.arange(k))

    # Exclusive cumsum: level j reflects expansions at steps 0..j-1; index 0 is
    # the initial (un-doubled) interval.
    left_inc = jnp.concatenate(
        [jnp.zeros(1), jnp.cumsum(step_widths * left_expands)[:-1]]
    )
    right_inc = jnp.concatenate(
        [jnp.zeros(1), jnp.cumsum(step_widths * right_expands)[:-1]]
    )
    lefts = initial_left - left_inc
    rights = initial_right + right_inc

    both_out = jnp.logical_and(
        jnp.logical_not(jax.vmap(in_slice)(lefts)),
        jnp.logical_not(jax.vmap(in_slice)(rights)),
    )
    idx = _best_interval(both_out.astype(int))
    left, right = lefts[idx], rights[idx]
    accept_fn = lambda t: _doubling_accept(
        in_slice, t, left, right, width
    )  # noqa: E731
    return left, right, idx, accept_fn


def _doubling_accept(
    in_slice: Callable, t: Array, left: Array, right: Array, width: float
) -> Array:
    """Neal (2003) Fig. 6 acceptance test for a doubling-found interval.

    Works on the original bracket ``(left, right)`` (not the shrunk one),
    bisecting toward ``t`` until the sub-interval is within ~``width`` (the
    1.1 factor guards round-off), rejecting if the doubling started from ``t``
    would have stopped earlier. ``x0`` is at ``t = 0``.
    """

    def cond(carry):
        left, right, _, ok = carry
        return (right - left > 1.1 * width) & ok

    def body(carry):
        left, right, d, _ = carry
        mid = 0.5 * (left + right)
        d = d | ((0.0 < mid) & (t >= mid)) | ((0.0 >= mid) & (t < mid))
        right = jnp.where(t < mid, mid, right)
        left = jnp.where(t >= mid, mid, left)
        both_out = jnp.logical_not(in_slice(left)) & jnp.logical_not(in_slice(right))
        ok = jnp.logical_not(d & both_out)
        return left, right, d, ok

    _, _, _, ok = jax.lax.while_loop(
        cond, body, (left, right, jnp.asarray(False), jnp.asarray(True))
    )
    return ok


def _shrink(
    rng_key: PRNGKey,
    slice_fn: Callable,
    level: Array,
    accept_fn: Callable,
    left: Array,
    right: Array,
    current_state,
    max_shrinkage: int,
):
    """Neal (2003) Fig. 5 shrinkage, in t-space (x0 at t=0).

    Threads the accepted candidate state through the loop, so whatever the
    proposal records on it (e.g. a nested-sampling ``loglikelihood``) is carried
    out rather than recomputed. Budget-capped; on exhaustion the chain stays put
    (``current_state`` is returned).
    """

    def cond(carry):
        _, _, _, _, n, _, found = carry
        return jnp.logical_not(found) & (n < max_shrinkage)

    def body(carry):
        _, left, right, key, n, state, _ = carry
        key, subkey = random.split(key)
        t = left + random.uniform(subkey) * (right - left)
        candidate, is_valid = slice_fn(t)
        found = (candidate.logdensity >= level) & is_valid & accept_fn(t)
        left = jnp.where(t < 0.0, t, left)
        right = jnp.where(t >= 0.0, t, right)
        state = jax.tree.map(
            lambda new, old: jnp.where(found, new, old), candidate, state
        )
        return t, left, right, key, n + 1, state, found

    init = (
        0.0,
        left,
        right,
        rng_key,
        jnp.asarray(0),
        current_state,
        jnp.asarray(False),
    )
    _, _, _, _, n, state, found = jax.lax.while_loop(cond, body, init)
    return state, n, found


def _univariate_slice(
    rng_key: PRNGKey,
    slice_fn: Callable,
    current_state,
    width: float,
    interval: Callable,
    max_expansions: int,
    max_shrinkage: int,
):
    """One univariate slice through the current point.

    ``slice_fn(t) -> (state, is_valid)`` produces the candidate at coordinate
    ``t``.
    """
    level_key, interval_key, shrink_key = random.split(rng_key, 3)
    level = current_state.logdensity + jnp.log(random.uniform(level_key))

    # ``slice_fn(t) -> (state, is_valid)`` is the slice function: it builds the
    # candidate state at coordinate ``t`` (computing whatever it records) and
    # reports whether it is admissible (where a constraint is consumed). A point
    # is in the slice iff its density is above the level and it is valid.
    def in_slice(t):
        candidate, is_valid = slice_fn(t)
        return (candidate.logdensity >= level) & is_valid

    # ``interval`` is a pluggable procedure (doubling / stepping_out) that
    # returns the bracket and its own acceptance test, so there is no name match.
    left, right, num_expansions, accept_fn = interval(
        interval_key, in_slice, width, max_expansions
    )

    new_state, num_shrink, is_accepted = _shrink(
        shrink_key,
        slice_fn,
        level,
        accept_fn,
        left,
        right,
        current_state,
        max_shrinkage,
    )
    info = SliceInfo(
        is_accepted=is_accepted,
        num_expansions=num_expansions,
        num_shrink=num_shrink,
        bracket_left=left,
        bracket_right=right,
    )
    return new_state, info


def build_kernel(
    interval: Callable = doubling,
    max_expansions: int = 10,
    max_shrinkage: int = 100,
) -> Callable:
    """Build a slice kernel driven by a proposal generator.

    The kernel performs one univariate slice using ``proposal_generator``, a
    callable ``(rng_key, position, logdensity_fn) -> slice_fn`` where
    ``slice_fn(t) -> (state, is_valid)`` builds the candidate state at coordinate
    ``t`` and reports whether it is admissible. Because the candidate state is
    threaded straight out, the proposal can record extra quantities on it and
    consume a constraint through ``is_valid``. To sample under a constraint,
    override the proposal generator rather than the kernel.

    Parameters
    ----------
    interval
        Interval-finding procedure, passed directly as a callable. Use
        :func:`doubling` (the default, Neal Fig. 4 with the Fig. 6 acceptance
        test) or :func:`stepping_out` (Neal Fig. 3).
    max_expansions
        Cap on interval expansions (doublings or stepping-out steps).
    max_shrinkage
        Cap on shrinkage evaluations. Bounds the loop; on exhaustion the chain
        stays put.

    Returns
    -------
    A kernel that takes a rng_key, the current state, a log-density function, a
    proposal generator and a bracket width, and returns a new state along with
    information about the transition.
    """

    def kernel(
        rng_key: PRNGKey,
        state: SliceState,
        logdensity_fn: Callable,
        proposal_generator: Callable,
        width: float = 1.0,
    ) -> tuple[SliceState, SliceInfo]:
        prop_key, slice_key = random.split(rng_key)
        slice_fn = proposal_generator(prop_key, state.position, logdensity_fn)
        new_state, info = _univariate_slice(
            slice_key, slice_fn, state, width, interval, max_expansions, max_shrinkage
        )
        return new_state, info

    return kernel


def random_order(rng_key: PRNGKey, d: int) -> Array:
    """A fresh random permutation of the ``d`` coordinate indices (the default)."""
    return random.permutation(rng_key, d)


def fixed_order(rng_key: PRNGKey, d: int) -> Array:
    """Sweep the coordinates in fixed natural order ``0, 1, ..., d - 1``."""
    del rng_key
    return jnp.arange(d)


def coordinate_proposal(
    rng_key: PRNGKey,
    position: ArrayLikeTree,
    logdensity_fn: Callable,
    i: int,
) -> Callable:
    """Default per-axis proposal for the coordinate sweep.

    The coordinate analogue of :func:`direction_proposal`: a unit step along
    flattened axis ``i`` (the one-hot direction ``e_i``), so ``x(t)`` is
    ``position`` with ``flat[i] += t`` and the current point sits at ``t = 0``.
    Shares the ``slice_fn(t) -> (state, is_valid)`` contract of the multivariate
    proposals.

    A constraint is added the same way as on the multivariate path -- by
    overriding the proposal (``axis_proposal``) to gate ``is_valid``; there is no
    built-in constraint argument.
    """
    del rng_key  # the coordinate move is deterministic given the axis
    flat, unravel_fn = jax.flatten_util.ravel_pytree(position)

    def slice_fn(t):
        x = unravel_fn(flat.at[i].add(t))
        return SliceState(x, logdensity_fn(x)), True

    return slice_fn


def build_coordinate_kernel(
    interval: Callable = doubling,
    axis_proposal: Callable = coordinate_proposal,
    coordinate_order: Callable = random_order,
    initial_widths: float | Array = 1.0,
    max_expansions: int = 10,
    max_shrinkage: int = 100,
) -> Callable:
    """Build a coordinate-wise (slice-within-Gibbs) kernel.

    One step updates each scalar coordinate's full conditional with a univariate
    slice, in the order given by ``coordinate_order``, the choice function
    ``(rng_key, d) -> indices`` (:func:`random_order`, the default, or
    :func:`fixed_order`). Each coordinate move is drawn by ``axis_proposal``, the
    per-axis analogue of the multivariate ``proposal_generator``
    (:func:`coordinate_proposal` by default); override it to gate a constraint
    into ``is_valid``. ``initial_widths`` is a scalar (applied to every
    coordinate) or a length-``D`` array of per-coordinate bracket widths.

    Returns
    -------
    A kernel that takes a rng_key, the current state and a log-density function,
    and returns a new state along with information about the transition.
    """

    def kernel(
        rng_key: PRNGKey, state: SliceState, logdensity_fn: Callable
    ) -> tuple[SliceState, SliceInfo]:
        flat0, unravel_fn = jax.flatten_util.ravel_pytree(state.position)
        d = flat0.shape[0]
        widths = jnp.broadcast_to(jnp.asarray(initial_widths, float).ravel(), (d,))

        order_key, scan_key = random.split(rng_key)
        # ``coordinate_order`` is the choice function ``(rng_key, d) -> indices``
        # (random_order by default, or fixed_order): which coordinates to update
        # and in what order.
        order = coordinate_order(order_key, d)
        m = order.shape[0]

        def body(carry, inp):
            position, logdensity = carry
            key, i, w = inp
            prop_key, slice_key = random.split(key)
            # Same two steps as the multivariate kernel: build the proposal for
            # this axis, then run the univariate slice along it.
            slice_fn = axis_proposal(prop_key, position, logdensity_fn, i)
            new_state, info = _univariate_slice(
                slice_key,
                slice_fn,
                SliceState(position, logdensity),
                w,
                interval,
                max_expansions,
                max_shrinkage,
            )
            return (new_state.position, new_state.logdensity), info

        keys = random.split(scan_key, m)
        (pos_final, ld_final), swept = jax.lax.scan(
            body, (state.position, state.logdensity), (keys, order, widths[order])
        )
        # The sweep does D univariate slices. Summarise the counters over the
        # sweep, and stitch the per-coordinate bracket endpoints back into the
        # position structure (scatter to natural order, then unravel) so that
        # ``info.bracket_left``/``bracket_right`` align with ``position``.
        stitch = lambda v: unravel_fn(  # noqa: E731
            jnp.zeros(d, v.dtype).at[order].set(v)
        )
        info = SliceInfo(
            is_accepted=jnp.all(swept.is_accepted),
            num_expansions=jnp.sum(swept.num_expansions),
            num_shrink=jnp.sum(swept.num_shrink),
            bracket_left=stitch(swept.bracket_left),
            bracket_right=stitch(swept.bracket_right),
        )
        return SliceState(pos_final, ld_final), info

    return kernel


def sample_direction(
    rng_key: PRNGKey, position: ArrayLikeTree, scale: float | Array = 1.0
) -> ArrayTree:
    """A random slice direction shaped by ``scale`` and normalized to unit length.

    ``scale`` is a scalar (isotropic), a vector (per-coordinate / diagonal) or a
    dense matrix (a full preconditioner, applied as a linear map to
    standard-normal noise, so its covariance is ``scale @ scale.T``). Defaults
    to ``1.0`` (uniformly random unit directions).
    """
    noise = generate_gaussian_noise(rng_key, position, sigma=scale)
    flat, unravel_fn = jax.flatten_util.ravel_pytree(noise)
    return unravel_fn(flat / jnp.linalg.norm(flat))


def direction_proposal(scale: float | Array = 1.0) -> Callable:
    """Proposal-generator factory: slice along a random ``scale``-shaped direction.

    See :func:`sample_direction` for ``scale`` (scalar / vector / dense, unit by
    default). Pass as
    ``slice_sampling(logp, proposal_generator=direction_proposal(scale))``.
    """

    def proposal_generator(rng_key, position, logdensity_fn):
        direction = sample_direction(rng_key, position, scale)

        def slice_fn(t):
            x = jax.tree.map(lambda p, d: p + t * d, position, direction)
            return SliceState(x, logdensity_fn(x)), True

        return slice_fn

    return proposal_generator


def as_top_level_api(
    logdensity_fn: Callable,
    *,
    proposal_generator: Callable = direction_proposal(),
    width: float = 1.0,
    interval: Callable = doubling,
    max_expansions: int = 10,
    max_shrinkage: int = 100,
) -> SamplingAlgorithm:
    """Multivariate slice sampler, ``blackjax.slice_sampling``.

    Each step takes one univariate slice along a random direction (chaining such
    moves is the hit-and-run strategy) drawn by ``proposal_generator``. The
    default :func:`direction_proposal` draws a uniformly random direction; pass
    ``direction_proposal(scale)`` to precondition, or override with your own
    proposal to gate a constraint or record extra quantities on the state, as
    nested sampling does. For coordinate-wise slice-within-Gibbs, use
    :func:`coordinate_slice`.

    Examples
    --------

    A new slice sampling kernel can be initialized and used with the following
    code:

    .. code::

        slice_sampling = blackjax.slice_sampling(logdensity_fn)
        state = slice_sampling.init(position)
        new_state, info = slice_sampling.step(rng_key, state)

    Parameters
    ----------
    logdensity_fn
        Log-density of the distribution to sample from.
    proposal_generator
        Proposal generator ``(rng_key, position, logdensity_fn) -> slice_fn``,
        where ``slice_fn(t) -> (state, is_valid)``. Defaults to
        ``direction_proposal()`` (isotropic unit directions).
    width
        Initial bracket width along the direction (default 1.0).
    interval
        Interval procedure :func:`doubling` (default) or :func:`stepping_out`,
        passed as a callable.
    max_expansions, max_shrinkage
        Caps on interval expansion and shrinkage.

    Returns
    -------
    A ``SamplingAlgorithm``.
    """
    kernel = build_kernel(interval, max_expansions, max_shrinkage)
    return build_sampling_algorithm(
        kernel, init, logdensity_fn, kernel_args=(proposal_generator, width)
    )


def coordinate_slice(
    logdensity_fn: Callable,
    *,
    max_expansions: int = 10,
    initial_widths: float | Array = 1.0,
    interval: Callable = doubling,
    coordinate_order: Callable = random_order,
    axis_proposal: Callable = coordinate_proposal,
    max_shrinkage: int = 100,
) -> SamplingAlgorithm:
    """Coordinate-wise (slice-within-Gibbs) slice sampler.

    Updates each scalar coordinate's full conditional with a univariate slice,
    swept in the order given by ``coordinate_order``. The single-variable
    counterpart to the multivariate :func:`as_top_level_api`.

    Parameters
    ----------
    logdensity_fn
        Log-density of the distribution to sample from.
    max_expansions
        Cap on interval expansions per coordinate (default 10).
    initial_widths
        Scalar or per-coordinate initial bracket width(s) (default 1.0).
    interval
        Interval procedure :func:`doubling` (default) or :func:`stepping_out`,
        passed as a callable.
    coordinate_order
        Choice function ``(rng_key, d) -> indices``, either :func:`random_order`
        (default) or :func:`fixed_order`.
    axis_proposal
        Per-axis proposal ``(rng_key, position, logdensity_fn, i) -> slice_fn``
        (:func:`coordinate_proposal` by default). Override to gate a constraint
        into ``is_valid``, as a custom ``proposal_generator`` does for
        :func:`as_top_level_api`.
    max_shrinkage
        Cap on shrinkage evaluations per coordinate.

    Returns
    -------
    A ``SamplingAlgorithm``.
    """
    kernel = build_coordinate_kernel(
        interval=interval,
        axis_proposal=axis_proposal,
        coordinate_order=coordinate_order,
        initial_widths=initial_widths,
        max_expansions=max_expansions,
        max_shrinkage=max_shrinkage,
    )
    return build_sampling_algorithm(kernel, init, logdensity_fn)
