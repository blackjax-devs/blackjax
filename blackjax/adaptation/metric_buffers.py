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
"""Data-feeding (buffer) layer for metric adaptation.

Each of the four policies below is a state machine with **fixed-shape,
scan-carry-safe state** that feeds Chan-mergeable moment blocks to metric
estimators.  The shared interface across all policies::

    init(...)       -> PolicyState
    update(state, batch)           -> PolicyState   # batch: (d,) or (nc, d)
    push_split(state)              -> PolicyState   # finalise current accumulation
    get_moments(state)             -> MomentBlock   # merged sufficient stats
    get_diag_reference(state)      -> Array         # (d,) diagonal for ε-proxy
    get_support(state)             -> tuple[Array, Array]  # (total, per_block)

**Block representation.** Buffers store per-split Chan-mergeable *moment
blocks* — O(d) or O(d²) sufficient statistics (count, mean, M2 matrix)
rather than raw draws.  Merging blocks gives the current estimate inputs;
dropping the oldest block and re-merging the rest implements exact
split-granular forgetting with no raw-draw ring.

**D-layer contract (A1 — ensemble split semantics).** For ``(nc, d)``-block
consumers a "split" is a **draw-axis partition** (step-ranges; all chains
fold into the block via Chan merge) — NEVER a chain-subset.  This is
enforced by the ``ensemble_batch_buffer`` factory and documented on every
``push_split`` operation.

**D-layer contract (A2 — diag_reference).** The running diagonal is
derivable from block moments: ``diag_ref = diag(M2_merged) / max(n-1, 1)``.
A single accumulator serves both the adapted metric and the ε-proxy
diagonal channel; step-size proxies read ``get_diag_reference``, not the
adapted low-rank metric (L3 decoupling contract).

**``requires_draws`` is opt-in, default OFF (A4).** Raw draws exist only
behind a ``requires_draws`` capability flag needed by the draws-SVD
estimator family.  Allocating an ``(n_chains, steps, d)`` raw-draw ring is
prohibitive for ensemble consumers.  All four policies default to
``requires_draws=False``; passing ``True`` raises ``NotImplementedError``
(the draw-ring variant is a follow-up work item).

**Fisher estimator block-moments handoff.** The current Fisher E-layer
function (``fisher_score_low_rank``) takes raw draws and gradients.
``FisherMomentBlock`` below accumulates the gradient moments that a
future moments-consuming Fisher variant would need.  The call-site wiring
is a follow-up work item; the data type is here so the D-layer can
accumulate gradient moments alongside position moments when needed.
"""

from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp

from blackjax.types import Array

__all__ = [
    "MomentBlock",
    "FisherMomentBlock",
    "chan_merge_two",
    "chan_update_batch",
    "merge_block_ring",
    "diag_from_moment_block",
    "ResetWindowState",
    "AccumulatingSplitPopState",
    "reset_window_buffer",
    "accumulating_split_pop_buffer",
    "ensemble_batch_buffer",
    "late_start",
]


# ---------------------------------------------------------------------------
# Moment block types
# ---------------------------------------------------------------------------


class MomentBlock(NamedTuple):
    """Chan-mergeable sufficient statistics for covariance estimation.

    Carries exactly the fields consumed by
    :func:`~blackjax.adaptation.metric_estimators.sample_covariance_eigh_low_rank`:

    .. code-block:: python

        metric = sample_covariance_eigh_low_rank(m2=block.m2, count=block.count, k)

    The ``m2`` field is the **accumulated sum of squared deviations** (Chan's
    M₂ matrix), NOT the covariance.  For a dense block ``m2`` has shape
    ``(d, d)``; for a diagonal block it has shape ``(d,)`` (the diagonal).

    Parameters
    ----------
    count
        Number of samples accumulated, scalar ``()``.
    mean
        Sample mean, shape ``(d,)``.
    m2
        Sum of squared deviations, shape ``(d, d)`` (dense) or ``(d,)``
        (diagonal).

    Notes
    -----
    An empty (zero-initialised) block has ``count=0``.  The Chan merge
    formula is safe for empty blocks — the result equals the non-empty
    partner when one block is empty.
    """

    count: Array  # ()
    mean: Array  # (d,)
    m2: Array  # (d, d) or (d,)


class FisherMomentBlock(NamedTuple):
    """Chan-mergeable sufficient statistics for a future moments-consuming
    Fisher estimator (position + gradient moments).

    The current Fisher E-layer function (``fisher_score_low_rank``) takes raw
    draws and gradients.  This block type accumulates the sufficient statistics
    that a moments-based Fisher variant would consume instead:
    ``cov(x)`` from ``(mean_x, m2_x, count)`` and ``cov(∇ log p)`` from
    ``(mean_g, m2_g, count)``.

    **This block type is here for design completeness; the call-site wiring
    (replacing ``fisher_score_low_rank``'s raw-array signature with a
    moments-based one) is a follow-up work item.**  Use ``MomentBlock`` for
    currently-wired consumers (``sample_covariance_eigh_low_rank``).

    Parameters
    ----------
    count
        Number of samples accumulated, scalar ``()``.
    mean_x
        Position mean, shape ``(d,)``.
    m2_x
        Position sum of squared deviations, shape ``(d, d)``.
    mean_g
        Gradient mean, shape ``(d,)``.
    m2_g
        Gradient sum of squared deviations, shape ``(d, d)``.
    """

    count: Array  # ()
    mean_x: Array  # (d,)
    m2_x: Array  # (d, d)
    mean_g: Array  # (d,)
    m2_g: Array  # (d, d)


# ---------------------------------------------------------------------------
# Chan merge core
# ---------------------------------------------------------------------------


def chan_merge_two(block_a: MomentBlock, block_b: MomentBlock) -> MomentBlock:
    r"""Chan-merge two pre-accumulated moment blocks.

    Combines ``(n_a, mean_a, M2_a)`` and ``(n_b, mean_b, M2_b)`` into
    ``(n_ab, mean_ab, M2_ab)`` using the parallel Chan et al. recurrence:

    .. math::

        n_{ab} &= n_a + n_b \\
        \delta &= \bar{x}_b - \bar{x}_a \\
        \bar{x}_{ab} &= \bar{x}_a + \delta \cdot \frac{n_b}{n_{ab}} \\
        M2_{ab} &= M2_a + M2_b + \delta\delta^\top \cdot \frac{n_a n_b}{n_{ab}}

    This is exact in exact arithmetic.  When either block is empty
    (``count=0``), the result equals the non-empty partner — the ``n_ab=0``
    division is guarded via ``jnp.where``.

    This is the building block for every pop/merge operation in the buffer
    layer.  For merging a NEW batch (raw draws) into an existing block, use
    :func:`chan_update_batch`.

    Parameters
    ----------
    block_a, block_b
        Two :class:`MomentBlock` instances to merge.  Must have the same
        ``m2`` shape (both dense ``(d, d)`` or both diagonal ``(d,)``).

    Returns
    -------
    MomentBlock
        Merged block with combined statistics.
    """
    n_a = block_a.count
    n_b = block_b.count
    mean_a = block_a.mean
    mean_b = block_b.mean
    m2_a = block_a.m2
    m2_b = block_b.m2

    n_ab = n_a + n_b
    delta = mean_b - mean_a

    # Guard division by zero: when n_ab=0 both n_a=n_b=0 so delta=0 and
    # the zero-guards below produce the correct zero block.
    safe_n_ab = jnp.where(n_ab > 0, n_ab, jnp.ones_like(n_ab))

    mean_ab = mean_a + delta * (n_b / safe_n_ab)
    # For diagonal m2: delta*(delta) is element-wise; for dense: outer product.
    if m2_a.ndim == 1:
        cross = delta * delta * (n_a * n_b / safe_n_ab)
    else:
        cross = jnp.outer(delta, delta) * (n_a * n_b / safe_n_ab)
    m2_ab = m2_a + m2_b + cross

    # When n_ab=0, both inputs were empty — return the zero block.
    mean_ab = jnp.where(n_ab > 0, mean_ab, jnp.zeros_like(mean_a))
    m2_ab = jnp.where(n_ab > 0, m2_ab, jnp.zeros_like(m2_a))

    return MomentBlock(count=n_ab, mean=mean_ab, m2=m2_ab)


def chan_update_batch(block: MomentBlock, batch: Array) -> MomentBlock:
    r"""Chan-merge an existing moment block with a new batch of raw draws.

    Equivalent to computing a temporary block from ``batch`` and calling
    :func:`chan_merge_two`, but avoids the intermediate allocation by
    computing the batch statistics inline.

    ``batch`` has shape ``(n_b, d)`` — a batch of ``n_b`` draw vectors.
    Single draws should be reshaped to ``(1, d)`` before calling.

    **Ensemble (nc, d) feeds**: when ``batch`` is a ``(n_chains, d)``
    ensemble snapshot, all chains fold into the block's Chan merge — this
    is how :func:`ensemble_batch_buffer` satisfies the A1 draw-axis
    semantics: a "split" is a time-range partition across all chains, not a
    chain-subset partition.

    Parameters
    ----------
    block
        Existing :class:`MomentBlock` (may be empty, ``count=0``).
    batch
        New raw draws, shape ``(n_b, d)``.

    Returns
    -------
    MomentBlock
        Updated block merging the previous statistics with the new batch.
    """
    n_a = block.count
    mean_a = block.mean
    m2_a = block.m2

    n_b = jnp.asarray(batch.shape[0], dtype=n_a.dtype)
    mean_b = jnp.mean(batch, axis=0)  # (d,)
    centered_b = batch - mean_b[None, :]  # (n_b, d)

    if m2_a.ndim == 1:
        m2_b = jnp.mean(centered_b**2, axis=0) * n_b  # sum-of-sq-dev, (d,)
    else:
        m2_b = centered_b.T @ centered_b  # (d, d)

    n_ab = n_a + n_b
    delta = mean_b - mean_a
    safe_n_ab = jnp.where(n_ab > 0, n_ab, jnp.ones_like(n_ab))

    mean_ab = mean_a + delta * (n_b / safe_n_ab)
    if m2_a.ndim == 1:
        cross = delta * delta * (n_a * n_b / safe_n_ab)
    else:
        cross = jnp.outer(delta, delta) * (n_a * n_b / safe_n_ab)
    m2_ab = m2_a + m2_b + cross

    mean_ab = jnp.where(n_ab > 0, mean_ab, jnp.zeros_like(mean_a))
    m2_ab = jnp.where(n_ab > 0, m2_ab, jnp.zeros_like(m2_a))

    return MomentBlock(count=n_ab, mean=mean_ab, m2=m2_ab)


def merge_block_ring(
    counts: Array, means: Array, m2s: Array
) -> MomentBlock:
    """Reduce a ring of ``k`` moment blocks into a single merged block.

    Iterates through the ring with :func:`~jax.lax.scan`, Chan-merging
    each slot in turn.  Empty slots (``count=0``) contribute nothing to
    the merged result — the Chan merge formula handles them correctly via
    the zero-guard in :func:`chan_merge_two`.

    Parameters
    ----------
    counts
        Shape ``(k,)``.  Per-block sample counts.  Zero entries indicate
        empty (unfilled) slots.
    means
        Shape ``(k, d)``.
    m2s
        Shape ``(k, d, d)`` or ``(k, d)``.

    Returns
    -------
    MomentBlock
        Chan-merged result across all ``k`` blocks (or the zero block if
        all slots are empty).
    """
    k = counts.shape[0]
    d = means.shape[1]
    is_diag = m2s.ndim == 2  # (k, d) vs (k, d, d)

    if is_diag:
        init = MomentBlock(
            count=jnp.zeros((), dtype=counts.dtype),
            mean=jnp.zeros((d,), dtype=means.dtype),
            m2=jnp.zeros((d,), dtype=m2s.dtype),
        )
    else:
        init = MomentBlock(
            count=jnp.zeros((), dtype=counts.dtype),
            mean=jnp.zeros((d,), dtype=means.dtype),
            m2=jnp.zeros((d, d), dtype=m2s.dtype),
        )

    def _step(acc: MomentBlock, i: Array) -> tuple[MomentBlock, None]:
        slot = MomentBlock(count=counts[i], mean=means[i], m2=m2s[i])
        merged = chan_merge_two(acc, slot)
        return merged, None

    merged, _ = jax.lax.scan(_step, init, jnp.arange(k))
    return merged


def diag_from_moment_block(block: MomentBlock) -> Array:
    """Bessel-corrected per-coordinate variance from a :class:`MomentBlock`.

    Implements the A2 diagonal-reference contract: one accumulator serves
    both the adapted metric and the ε-proxy diagonal channel.  Step-size
    proxies read this diagonal view; they never read the adapted low-rank
    metric directly (L3 decoupling).

    The formula is ``diag_ref = diag(M2) / max(count - 1, 1)``.  For an
    empty block (``count=0``), returns ones (isotropic default).

    Parameters
    ----------
    block
        A :class:`MomentBlock`.  Dense blocks (``m2`` shape ``(d, d)``)
        extract the diagonal; diagonal blocks (``m2`` shape ``(d,)``) use
        ``m2`` directly.

    Returns
    -------
    Array, shape ``(d,)``
        Per-coordinate Bessel-corrected variance.
    """
    m2 = block.m2
    n = block.count
    denom = jnp.maximum(n - 1.0, 1.0)
    if m2.ndim == 2:
        var = jnp.diag(m2) / denom
    else:
        var = m2 / denom
    # For empty blocks return ones (isotropic default).
    return jnp.where(n > 0, var, jnp.ones_like(var))


# ---------------------------------------------------------------------------
# Policy state types
# ---------------------------------------------------------------------------


class ResetWindowState(NamedTuple):
    """State for the hard-reset (Stan-style) window adaptation buffer.

    A single accumulator block that is zeroed out at every window boundary
    (``push_split``).  This is the Stan default behaviour expressed in the
    moment-block representation.

    Parameters
    ----------
    count
        Number of samples accumulated in the current window, scalar ``()``.
    mean
        Running mean, shape ``(d,)``.
    m2
        Running M2 matrix, shape ``(d, d)`` (dense) or ``(d,)`` (diagonal).
    """

    count: Array  # ()
    mean: Array  # (d,)
    m2: Array  # (d, d) or (d,)


class AccumulatingSplitPopState(NamedTuple):
    """State for the split-based rolling-window buffer.

    Maintains a ring of ``k`` moment blocks (one active + up to ``k-1``
    completed).  At each ``push_split``, the active block is "finalized"
    (its index becomes a completed slot) and the ring pointer advances to a
    freshly-zeroed slot for the next accumulation.  When the ring wraps
    around, the oldest completed block is overwritten — exactly the
    nuts-rs-faithful ``background_split`` pop at split granularity.

    This state type is shared between :func:`accumulating_split_pop_buffer`
    (single-draw updates) and :func:`ensemble_batch_buffer` (ensemble-batch
    updates), which differ only in the ``update`` function's batch shape.

    Parameters
    ----------
    counts
        Per-block sample counts, shape ``(k,)``.  Zero entries are empty.
    means
        Per-block running means, shape ``(k, d)``.
    m2s
        Per-block M2 matrices, shape ``(k, d, d)`` or ``(k, d)``.
    write_pos
        Index of the currently-active (in-progress) block, scalar ``()``.
    num_valid
        Number of blocks with ``count > 0`` (includes the active block),
        scalar ``()``.  Saturates at ``k``.

    Notes
    -----
    Split semantics (A1 for ensemble consumers): a "split" is always a
    **draw-axis time partition** — all chains in a batch fold into the
    active block's Chan merge.  Splits are never chain-subset partitions.
    """

    counts: Array  # (k,)
    means: Array  # (k, d)
    m2s: Array  # (k, d, d) or (k, d)
    write_pos: Array  # ()
    num_valid: Array  # ()


# ---------------------------------------------------------------------------
# Policy 1: reset_window
# ---------------------------------------------------------------------------


def reset_window_buffer(
    d: int,
    *,
    diagonal: bool = False,
    requires_draws: bool = False,
) -> tuple[Callable, Callable, Callable, Callable, Callable, Callable]:
    """Stan-style hard-reset window adaptation buffer.

    Maintains a single accumulator block.  On ``push_split`` (= window
    boundary) the accumulator is zeroed out; the next window starts fresh.

    This is the **default** feeding policy used by ``window_adaptation_low_rank``
    with ``buffer_policy="reset"``, expressed in the moment-block representation.
    A :class:`ResetWindowState` accumulates during a slow window;
    ``push_split`` zeroes it; the next window accumulates again.

    The estimator call at window-end::

        metric = sample_covariance_eigh_low_rank(
            m2=get_moments(state).m2,
            count=get_moments(state).count,
            max_rank=k,
        )

    Parameters
    ----------
    d
        Dimension of the position space.
    diagonal
        If ``True``, the M2 field has shape ``(d,)`` (diagonal sufficient
        statistics); if ``False`` (default), shape ``(d, d)``.
    requires_draws
        If ``True``, attach a raw-draw ring to the state for draw-SVD
        estimators.  **Currently not implemented** (raises
        ``NotImplementedError``); default ``False`` is the A4 contract.

    Returns
    -------
    init, update, push_split, get_moments, get_support, get_diag_reference
        Six callables with the shared buffer-policy interface.

        - ``init()`` → :class:`ResetWindowState`
        - ``update(state, batch)`` → :class:`ResetWindowState`
        - ``push_split(state)`` → :class:`ResetWindowState`  (zeroes accumulator)
        - ``get_moments(state)`` → :class:`MomentBlock`
        - ``get_support(state)`` → ``(total_count, per_block_counts)``
        - ``get_diag_reference(state)`` → ``Array`` shape ``(d,)``
    """
    if requires_draws:
        raise NotImplementedError(
            "requires_draws=True (raw-draw ring for the draws-SVD estimator) "
            "is not yet implemented.  Set requires_draws=False (the default)."
        )

    _m2_shape = (d,) if diagonal else (d, d)

    def init() -> ResetWindowState:
        return ResetWindowState(
            count=jnp.zeros((), dtype=jnp.float32),
            mean=jnp.zeros((d,)),
            m2=jnp.zeros(_m2_shape),
        )

    def update(state: ResetWindowState, batch: Array) -> ResetWindowState:
        """Chan-merge a new batch of draws into the accumulator.

        Parameters
        ----------
        state
            Current :class:`ResetWindowState`.
        batch
            New draws, shape ``(n_b, d)`` or ``(d,)`` (reshaped to ``(1, d)``
            internally).  For ensemble consumers ``batch`` is ``(n_chains, d)``
            and all chains fold into the block via Chan merge (A1).
        """
        if batch.ndim == 1:
            batch = batch[None, :]  # (1, d)
        block = MomentBlock(count=state.count, mean=state.mean, m2=state.m2)
        updated = chan_update_batch(block, batch)
        return ResetWindowState(
            count=updated.count, mean=updated.mean, m2=updated.m2
        )

    def push_split(state: ResetWindowState) -> ResetWindowState:
        """Zero the accumulator (Stan-style hard reset at window boundary)."""
        return ResetWindowState(
            count=jnp.zeros_like(state.count),
            mean=jnp.zeros_like(state.mean),
            m2=jnp.zeros_like(state.m2),
        )

    def get_moments(state: ResetWindowState) -> MomentBlock:
        """Return the current accumulator as a :class:`MomentBlock`."""
        return MomentBlock(count=state.count, mean=state.mean, m2=state.m2)

    def get_support(state: ResetWindowState) -> tuple[Array, Array]:
        """Return ``(total_count, per_block_counts)``."""
        per_block = jnp.array([state.count])
        return state.count, per_block

    def get_diag_reference(state: ResetWindowState) -> Array:
        """Bessel-corrected per-coordinate variance (A2 diagonal channel).

        Step-size proxies read this; they do not read the adapted low-rank
        metric (L3 decoupling contract).
        """
        return diag_from_moment_block(get_moments(state))

    return init, update, push_split, get_moments, get_support, get_diag_reference


# ---------------------------------------------------------------------------
# Policy 2: accumulating_split_pop (and Policy 3: ensemble_batch)
# ---------------------------------------------------------------------------


def _make_split_pop_fns(
    d: int,
    k: int,
    diagonal: bool,
    n_chains_per_update: int | None,
    requires_draws: bool,
) -> tuple[Callable, Callable, Callable, Callable, Callable, Callable]:
    """Shared factory for ``accumulating_split_pop_buffer`` and
    ``ensemble_batch_buffer``.

    Both policies use :class:`AccumulatingSplitPopState` and share all
    operations except the effective batch size per ``update`` call.
    ``n_chains_per_update`` is purely informational (passed for documentation
    purposes); the actual Chan-merge update is batch-size-agnostic.
    """
    if requires_draws:
        raise NotImplementedError(
            "requires_draws=True (raw-draw ring for the draws-SVD estimator) "
            "is not yet implemented.  Set requires_draws=False (the default)."
        )
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")

    _m2_shape = (d,) if diagonal else (d, d)

    def init() -> AccumulatingSplitPopState:
        return AccumulatingSplitPopState(
            counts=jnp.zeros((k,), dtype=jnp.float32),
            means=jnp.zeros((k, d)),
            m2s=jnp.zeros((k,) + _m2_shape),
            write_pos=jnp.zeros((), dtype=jnp.int32),
            num_valid=jnp.zeros((), dtype=jnp.int32),
        )

    def update(
        state: AccumulatingSplitPopState, batch: Array
    ) -> AccumulatingSplitPopState:
        """Chan-merge a new batch into the active block (``state[write_pos]``).

        All samples in ``batch`` fold into the active block via
        :func:`chan_update_batch`.  For ensemble consumers, ``batch`` is
        ``(n_chains, d)`` and all chains are treated as a single temporal
        batch — a "split" is a time-range partition, not a chain-subset
        partition (A1 semantics).

        Parameters
        ----------
        state
            Current :class:`AccumulatingSplitPopState`.
        batch
            New draws, shape ``(n_b, d)`` or ``(d,)`` (reshaped to ``(1, d)``
            internally).
        """
        if batch.ndim == 1:
            batch = batch[None, :]

        wp = state.write_pos
        cur_block = MomentBlock(
            count=state.counts[wp],
            mean=state.means[wp],
            m2=state.m2s[wp],
        )
        updated = chan_update_batch(cur_block, batch)

        new_counts = state.counts.at[wp].set(updated.count)
        new_means = state.means.at[wp].set(updated.mean)
        new_m2s = state.m2s.at[wp].set(updated.m2)

        # num_valid: if this is the first write to write_pos (previously count
        # was 0), the active block becomes a valid block.
        was_empty = state.counts[wp] == 0
        new_num_valid = jnp.where(
            was_empty, jnp.minimum(state.num_valid + 1, k), state.num_valid
        )

        return AccumulatingSplitPopState(
            counts=new_counts,
            means=new_means,
            m2s=new_m2s,
            write_pos=wp,
            num_valid=new_num_valid,
        )

    def push_split(
        state: AccumulatingSplitPopState,
    ) -> AccumulatingSplitPopState:
        """Finalise the active block, advance to a fresh slot.

        Advances ``write_pos`` by 1 (mod ``k``), zeroing out the newly
        active slot.  This implements the split-granular forgetting:
        when the ring wraps, the oldest completed block's slot is
        overwritten — exact "pop oldest split" semantics with O(d²)
        memory rather than a raw-draw ring.

        **A1 clarification**: the split here is a time-range boundary
        (a "step count" partition).  For ensemble consumers all chains
        participated in the now-completed block; the split never partitions
        chains into subsets.
        """
        old_wp = state.write_pos
        new_wp = (old_wp + 1) % k

        # Zero the newly-active slot (overwriting the oldest when the ring
        # has wrapped k times).
        new_counts = state.counts.at[new_wp].set(jnp.zeros((), dtype=state.counts.dtype))
        new_means = state.means.at[new_wp].set(jnp.zeros((d,), dtype=state.means.dtype))
        new_m2s = state.m2s.at[new_wp].set(
            jnp.zeros(_m2_shape, dtype=state.m2s.dtype)
        )

        # When we overwrite slot new_wp and it was previously valid (has data
        # from a full wrap-around), num_valid decreases by 1 before the
        # forthcoming update() increments it back.  We handle this by
        # decrementing if new_wp was non-empty ONLY when the ring was full.
        ring_full = state.num_valid >= k
        slot_was_valid = state.counts[new_wp] > 0
        new_num_valid = jnp.where(
            ring_full & slot_was_valid,
            state.num_valid - 1,
            state.num_valid,
        )

        return AccumulatingSplitPopState(
            counts=new_counts,
            means=new_means,
            m2s=new_m2s,
            write_pos=new_wp,
            num_valid=new_num_valid,
        )

    def get_moments(state: AccumulatingSplitPopState) -> MomentBlock:
        """Chan-merge all ``k`` slots (including the active block) into one.

        Empty slots (``count=0``) contribute nothing — the Chan merge
        formula handles them correctly.
        """
        return merge_block_ring(state.counts, state.means, state.m2s)

    def get_support(
        state: AccumulatingSplitPopState,
    ) -> tuple[Array, Array]:
        """Return ``(total_count, per_block_counts)``.

        ``total_count`` is the sum of all valid block counts (the merged
        effective support).  ``per_block_counts`` is the raw ``(k,)``
        counts array including zeros for empty slots.
        """
        total = jnp.sum(state.counts)
        return total, state.counts

    def get_diag_reference(state: AccumulatingSplitPopState) -> Array:
        """Bessel-corrected per-coordinate variance (A2 diagonal channel).

        Derived from the MERGED moments across all blocks; step-size
        proxies read this, never the adapted low-rank metric (L3).
        """
        return diag_from_moment_block(get_moments(state))

    return init, update, push_split, get_moments, get_support, get_diag_reference


def accumulating_split_pop_buffer(
    d: int,
    k: int,
    *,
    diagonal: bool = False,
    requires_draws: bool = False,
) -> tuple[Callable, Callable, Callable, Callable, Callable, Callable]:
    """Rolling-window buffer with exact oldest-split forgetting.

    Maintains ``k`` Chan-mergeable moment blocks in a ring.  Each call to
    ``push_split`` finalises the currently-active block and advances the
    ring pointer to a fresh slot; when the ring is full the advance
    overwrites the oldest completed block.  This gives exact
    **split-granular forgetting** with O(k·d²) moment memory rather than
    O(k·n_per_split·d) raw draws.

    This is the nuts-rs-faithful forgetting policy: in nuts-rs, the
    ``background_split`` (oldest split) is popped at each window switch and
    the rest of the buffer is retained.  That is exactly what
    ``push_split`` does here, at the block-granularity level.

    **Split semantics (A1)**: for this policy a "split" is a time-range
    partition — the caller decides when to call ``push_split`` (e.g., at
    each adaptation window boundary).  For ensemble consumers all chains in
    each update batch fold into the SAME active block (A1 contract); use
    :func:`ensemble_batch_buffer` which documents this explicitly.

    Parameters
    ----------
    d
        Dimension of the position space.
    k
        Number of splits (moment blocks) in the rolling window.  The oldest
        block is dropped when more than ``k`` blocks have accumulated.
        Must be ≥ 1.
    diagonal
        If ``True``, M2 fields are shape ``(d,)`` (diagonal); if ``False``
        (default), shape ``(d, d)``.
    requires_draws
        Default ``False`` (A4).  ``True`` raises ``NotImplementedError``.

    Returns
    -------
    init, update, push_split, get_moments, get_support, get_diag_reference
        Six callables with the shared buffer-policy interface.
    """
    return _make_split_pop_fns(
        d=d, k=k, diagonal=diagonal, n_chains_per_update=None, requires_draws=requires_draws
    )


def ensemble_batch_buffer(
    d: int,
    n_chains: int,
    k: int,
    *,
    diagonal: bool = False,
    requires_draws: bool = False,
) -> tuple[Callable, Callable, Callable, Callable, Callable, Callable]:
    """Rolling-window buffer for ensemble (multi-chain) consumers.

    A specialisation of :func:`accumulating_split_pop_buffer` for
    ``(n_chains, d)`` batch inputs, with explicit A1-semantics
    documentation.

    **A1 — ensemble split semantics**: for ``(nc, d)``-block consumers a
    "split" is a **draw-axis partition** (step-ranges; all chains fold into
    the active block via Chan merge) — NEVER a chain-subset.  Concretely,
    calling ``update(state, batch)`` with ``batch`` shape ``(n_chains, d)``
    folds all ``n_chains`` positions into the single active block's
    sufficient statistics; calling ``push_split`` advances the ring pointer
    to start a new time-range block (all chains still folded together).

    This policy is the intended feeding backend for MEADS-LRD and
    ChEES-metric consumers.

    Parameters
    ----------
    d
        Dimension of the position space.
    n_chains
        Number of chains per ensemble update.  This parameter is used for
        documentation and precondition checking only; the actual Chan merge
        is shape-agnostic.
    k
        Number of splits (moment blocks) in the rolling window.
    diagonal
        If ``True``, M2 fields are shape ``(d,)``; if ``False`` (default),
        shape ``(d, d)``.
    requires_draws
        Default ``False`` (A4).  ``True`` raises ``NotImplementedError``.

    Returns
    -------
    init, update, push_split, get_moments, get_support, get_diag_reference
        Six callables with the shared buffer-policy interface.
    """
    if n_chains < 1:
        raise ValueError(f"n_chains must be >= 1, got {n_chains}")
    return _make_split_pop_fns(
        d=d,
        k=k,
        diagonal=diagonal,
        n_chains_per_update=n_chains,
        requires_draws=requires_draws,
    )


# ---------------------------------------------------------------------------
# Policy 4: late_start (composable offset wrapper)
# ---------------------------------------------------------------------------


class LateStartState(NamedTuple):
    """State for the late-start offset policy.

    Wraps an inner policy state (any of the three above) and suppresses
    updates for the first ``offset_steps`` calls to ``update``.  After
    ``offset_steps`` draws have been skipped the inner policy receives all
    subsequent updates normally.

    This implements the MEADS fraction-style transient skipping (skip the
    first ``low_rank_window_fraction`` fraction of a window's draws, then
    accumulate in the second half).

    Parameters
    ----------
    inner
        The wrapped policy state (e.g., :class:`ResetWindowState` or
        :class:`AccumulatingSplitPopState`).
    num_skipped
        Number of draws that have been skipped so far, scalar ``()``.
        Saturates at ``offset_steps`` (so the carry never grows unboundedly
        and the shape is static).
    """

    inner: NamedTuple  # inner policy state
    num_skipped: Array  # ()


def late_start(
    inner_fns: tuple[Callable, ...],
    offset_steps: int,
) -> tuple[Callable, Callable, Callable, Callable, Callable, Callable]:
    """Offset policy: skip the first ``offset_steps`` draws, then accumulate.

    Wraps any of the three policies above (or another ``late_start``) with a
    transient-skip period.  The first ``offset_steps`` calls to ``update``
    are suppressed; all subsequent calls delegate to the inner policy.

    This implements the MEADS-style late-window accumulation — in MEADS,
    only draws in the second half of each adaptation window are accumulated
    into the covariance estimate (``low_rank_window_fraction=0.5``, so
    ``offset_steps = window_size // 2``).

    **Composability**: ``late_start`` can wrap any of the four policies.
    It delegates ``push_split``, ``get_moments``, ``get_support``, and
    ``get_diag_reference`` directly to the inner policy; only ``update`` has
    the skip logic.

    Parameters
    ----------
    inner_fns
        A tuple ``(init, update, push_split, get_moments, get_support,
        get_diag_reference)`` as returned by one of the three policy
        factories.
    offset_steps
        Number of draws to skip before starting accumulation.  Must be ≥ 0.

    Returns
    -------
    init, update, push_split, get_moments, get_support, get_diag_reference
        Six callables with the shared buffer-policy interface.  The wrapped
        state type is :class:`LateStartState`.
    """
    if offset_steps < 0:
        raise ValueError(f"offset_steps must be >= 0, got {offset_steps}")

    (
        inner_init,
        inner_update,
        inner_push_split,
        inner_get_moments,
        inner_get_support,
        inner_get_diag_reference,
    ) = inner_fns

    def init(*args, **kwargs) -> LateStartState:
        return LateStartState(
            inner=inner_init(*args, **kwargs),
            num_skipped=jnp.zeros((), dtype=jnp.int32),
        )

    def update(state: LateStartState, batch: Array) -> LateStartState:
        """Update the inner policy, or skip if still in the offset period.

        During the first ``offset_steps`` calls, ``batch`` is dropped and
        ``num_skipped`` increments.  After that, every call delegates to
        the inner ``update``.
        """
        still_skipping = state.num_skipped < offset_steps

        new_inner = jax.lax.cond(
            still_skipping,
            lambda: state.inner,
            lambda: inner_update(state.inner, batch),
        )
        new_skipped = jnp.minimum(
            state.num_skipped + 1, jnp.asarray(offset_steps, dtype=jnp.int32)
        )
        return LateStartState(inner=new_inner, num_skipped=new_skipped)

    def push_split(state: LateStartState) -> LateStartState:
        """Delegate to inner policy and reset the skip counter."""
        return LateStartState(
            inner=inner_push_split(state.inner),
            num_skipped=jnp.zeros((), dtype=state.num_skipped.dtype),
        )

    def get_moments(state: LateStartState) -> MomentBlock:
        return inner_get_moments(state.inner)

    def get_support(state: LateStartState) -> tuple[Array, Array]:
        return inner_get_support(state.inner)

    def get_diag_reference(state: LateStartState) -> Array:
        return inner_get_diag_reference(state.inner)

    return init, update, push_split, get_moments, get_support, get_diag_reference
