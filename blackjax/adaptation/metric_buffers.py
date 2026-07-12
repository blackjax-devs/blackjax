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
scan-carry-safe state** that feeds CGL-mergeable moment blocks to metric
estimators.  The shared interface across all policies is returned as a
:class:`MetricBuffer` (a NamedTuple of callables, house-style)::

    buf.init(...)                    -> PolicyState
    buf.update(state, batch)         -> PolicyState   # batch: (d,) or (nc, d)
    buf.push_split(state)            -> PolicyState   # finalise current accumulation
    buf.get_moments(state)           -> MomentBlock   # merged sufficient stats
    buf.get_diag_reference(state)    -> Array         # (d,) diagonal for ε-proxy
    buf.get_support(state)           -> tuple[Array, Array]  # (total, per_block)

**Block representation.** Buffers store per-split CGL-mergeable *moment
blocks* — O(d) or O(d²) sufficient statistics (count, mean, M2 matrix)
rather than raw draws :cite:p:`chan1983algorithms`.  Merging blocks gives
the current estimate inputs; dropping the oldest block and re-merging the
rest implements exact split-granular forgetting with no raw-draw ring.

**Memory tradeoff.** O(k·d²) blocks win when ``n_per_split > d`` (ensemble
folding almost always satisfies this; with 128 chains and any ``d``, each
push accumulates 128 draws, far exceeding ``d`` for ``d < 128``).  For
single-chain adaptation with ``d > n_per_split`` a raw-draw ring is more
memory-efficient — the opt-in draw ring is the intended design for high-d
single-chain use, not an optional optimization.

**Ensemble split semantics.** For ``(nc, d)``-block consumers a "split" is
a **draw-axis partition** (step-ranges; all chains fold into the block via
CGL merge) — NEVER a chain-subset.  This is enforced by the
``ensemble_batch_buffer`` factory and documented on every ``push_split``
operation.

**Ensemble pooling note.** Moments are POOLED across chains and steps —
between-chain dispersion enters the covariance by design (unconverged
ensembles inflate it by a factor of roughly ``1 + between/within``).  A
between/within decomposition is NOT recoverable from the folded blocks;
callers should be aware of this when the ensemble has not yet converged.

**Diagonal-reference contract.** The running diagonal is derivable from
block moments: ``diag_ref = diag(M2_merged) / max(n-1, 1)``.  A single
accumulator serves both the adapted metric and the step-size proxy channel;
step-size proxies read ``get_diag_reference``, never the adapted low-rank
metric directly.

**``requires_draws`` is opt-in, default off.** Raw draws exist only behind
a ``requires_draws`` capability flag needed by the draws-SVD estimator
family.  Allocating an ``(n_chains, steps, d)`` raw-draw ring is
prohibitive for ensemble consumers.  All four policies default to
``requires_draws=False``; passing ``True`` raises ``NotImplementedError``
(the draw-ring variant is a follow-up work item).

**Read-before-push ordering.** Callers MUST read ``get_moments`` (and
``get_diag_reference``) BEFORE calling ``push_split``.  The ordering
contract is::

    block   = buf.get_moments(state)           # read first
    diag    = buf.get_diag_reference(state)    # read first
    state   = buf.push_split(state)            # then advance

Violation consequences differ by policy — see ``push_split`` docstrings.

**``reset_window_buffer`` scope.** This policy replaces the
sample-covariance estimator path (``sample_covariance_eigh_low_rank``).
It is NOT a drop-in for the Fisher-score path (``fisher_score_low_rank``),
which still takes raw draws; that wiring is deliberate follow-up work.
Additionally, in the current in-tree ``window_adaptation`` the wrap-buffer
normalizes by the full ``n`` even when ``n > B`` (buffer size); this
module's accumulate-all semantics is the correct Stan-reset behavior —
a future consumer swap is behavior-improving (not bit-identical) in
the wrap regime.

**Cross-context dtype.** Create buffer state in the same x64 regime you
sample in.  Calling ``init`` outside a ``jax.enable_x64()`` context and
then calling ``update`` inside one silently degrades merge weights to f32
precision (eager mode) or raises a shape-mismatch error under ``lax.scan``
(the scan carries the original dtype from init).

**f32 accuracy for far-from-origin positions.** The CGL recurrence is
subject to catastrophic cancellation when ``|mean| ≳ 1e5``; accuracy
degrades to ``O(ε_mach × |mean|²)`` absolute.  Prefer x64 or centering
positions when working at large scales in f32.

**Fisher estimator block-moments handoff.** The current Fisher estimator
function (``fisher_score_low_rank``) takes raw draws and gradients.
``_FisherMomentBlock`` below accumulates the gradient moments that a
future moments-consuming Fisher variant would need.  The call-site wiring
is a follow-up work item; the data type is here so the D-layer can
accumulate gradient moments alongside position moments when needed.
"""

from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp

from blackjax.types import Array

__all__ = [
    "MetricBuffer",
    "MomentBlock",
    "cgl_merge_two",
    "cgl_update_batch",
    "merge_block_ring",
    "diag_from_moment_block",
    "ResetWindowState",
    "AccumulatingSplitPopState",
    "LateStartState",
    "reset_window_buffer",
    "accumulating_split_pop_buffer",
    "ensemble_batch_buffer",
    "late_start",
]


# ---------------------------------------------------------------------------
# MetricBuffer — house-style interface NamedTuple
# ---------------------------------------------------------------------------


class MetricBuffer(NamedTuple):
    """Buffer policy as a NamedTuple of callables (house-style).

    Follows the :class:`~blackjax.base.SamplingAlgorithm` convention of
    bundling a policy's callables into a named container so consumers can
    access them by name (``buf.get_moments(state)``) rather than by
    positional destructuring.  Still positionally compatible with tuple
    unpacking (``NamedTuple`` IS a ``tuple``).

    Parameters
    ----------
    init
        ``() -> PolicyState``
    update
        ``(state, batch) -> PolicyState``
    push_split
        ``(state) -> PolicyState``
    get_moments
        ``(state) -> MomentBlock``
    get_support
        ``(state) -> tuple[Array, Array]``
    get_diag_reference
        ``(state) -> Array``
    """

    init: Callable
    update: Callable
    push_split: Callable
    get_moments: Callable
    get_support: Callable
    get_diag_reference: Callable


# ---------------------------------------------------------------------------
# Moment block types
# ---------------------------------------------------------------------------


class MomentBlock(NamedTuple):
    """CGL-mergeable sufficient statistics for covariance estimation.

    Carries exactly the fields consumed by
    :func:`~blackjax.adaptation.metric_estimators.sample_covariance_eigh_low_rank`:

    .. code-block:: python

        metric = sample_covariance_eigh_low_rank(m2=block.m2, count=block.count, k)

    The ``m2`` field is the **accumulated sum of squared deviations** (CGL's
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
    An empty (zero-initialised) block has ``count=0``.  The CGL merge
    formula is safe for empty blocks — the result equals the non-empty
    partner when one block is empty.
    """

    count: Array  # ()
    mean: Array  # (d,)
    m2: Array  # (d, d) or (d,)


class _FisherMomentBlock(NamedTuple):
    """CGL-mergeable sufficient statistics for a future moments-consuming
    Fisher estimator (position + gradient moments).

    Companion type for a moments-based Fisher variant that would replace
    ``fisher_score_low_rank``'s raw-array signature.  The call-site wiring
    is deliberate follow-up work; use :class:`MomentBlock` for currently-wired
    consumers (``sample_covariance_eigh_low_rank``).

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
# CGL merge core
# ---------------------------------------------------------------------------


def cgl_merge_two(block_a: MomentBlock, block_b: MomentBlock) -> MomentBlock:
    r"""CGL-merge two pre-accumulated moment blocks.

    Combines ``(n_a, mean_a, M2_a)`` and ``(n_b, mean_b, M2_b)`` into
    ``(n_ab, mean_ab, M2_ab)`` using the parallel Chan–Golub–LeVeque
    recurrence :cite:p:`chan1983algorithms`:

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
    :func:`cgl_update_batch`.

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


def cgl_update_batch(block: MomentBlock, batch: Array) -> MomentBlock:
    r"""CGL-merge an existing moment block with a new batch of raw draws.

    Equivalent to computing a temporary block from ``batch`` and calling
    :func:`cgl_merge_two`, but avoids the intermediate allocation by
    computing the batch statistics inline.

    ``batch`` has shape ``(n_b, d)`` — a batch of ``n_b`` draw vectors.
    Single draws should be reshaped to ``(1, d)`` before calling.

    **Ensemble (nc, d) feeds**: when ``batch`` is a ``(n_chains, d)``
    ensemble snapshot, all chains fold into the block's CGL merge — this
    is how :func:`ensemble_batch_buffer` satisfies the draw-axis split
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


def merge_block_ring(counts: Array, means: Array, m2s: Array) -> MomentBlock:
    """Reduce a ring of ``k`` moment blocks into a single merged block.

    For ``k == 1`` uses a direct-slice short-circuit (no ``lax.scan``
    compiled) that is bit-identical to the scan path while avoiding the
    ~1.6× compile overhead at large ``d``.  For ``k > 1`` iterates with
    :func:`~jax.lax.scan`, CGL-merging each slot in turn.  Empty slots
    (``count=0``) contribute nothing to the merged result — the CGL merge
    formula handles them correctly via the zero-count guard in
    :func:`cgl_merge_two`.

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
        CGL-merged result across all ``k`` blocks (or the zero block if
        all slots are empty).
    """
    k = counts.shape[0]

    # k=1 short-circuit: direct slice avoids scan compile overhead (~1.6×
    # at d=400); bit-identical to the scan path (verified).
    if k == 1:
        return MomentBlock(count=counts[0], mean=means[0], m2=m2s[0])

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
        merged = cgl_merge_two(acc, slot)
        return merged, None

    merged, _ = jax.lax.scan(_step, init, jnp.arange(k))
    return merged


def diag_from_moment_block(block: MomentBlock) -> Array:
    """Bessel-corrected per-coordinate variance from a :class:`MomentBlock`.

    One accumulator serves both the adapted metric and the step-size proxy
    channel.  Step-size proxies read this diagonal view; they never read
    the adapted low-rank metric directly.

    The formula is ``diag_ref = diag(M2) / max(count - 1, 1)``, returning
    ones (isotropic default) when ``count < 2``.  This guards two
    degenerate cases: ``count=0`` (empty block, M2=0) and ``count=1``
    (single point, M2=0 by definition — dividing by ``max(0,1)=1`` would
    return zeros, which is wrong as a step-size proxy).  The in-tree Welford
    accumulator returns NaN at ``count=1`` (``M2/(n-1) = 0/0``); neither
    zero nor NaN is useful for a step-size proxy, so ones is the correct
    isotropic fallback.

    Parameters
    ----------
    block
        A :class:`MomentBlock`.  Dense blocks (``m2`` shape ``(d, d)``)
        extract the diagonal; diagonal blocks (``m2`` shape ``(d,)``) use
        ``m2`` directly.

    Returns
    -------
    Array, shape ``(d,)``
        Per-coordinate Bessel-corrected variance (denominator ``count - 1``),
        or ones when ``count < 2``.

    Notes
    -----
    This returns the **Bessel-corrected** (unbiased) sample variance with
    denominator ``count - 1``.  This matches the convention of
    :func:`~blackjax.adaptation.metric_estimators.sample_covariance_eigh_low_rank`.
    Consumers expecting **population variance** (denominator ``count``) will
    observe an upward shift of ``count / (count - 1)`` relative to their
    expectation; callers using this output as a population-variance proxy
    must account for that factor.
    """
    m2 = block.m2
    n = block.count
    denom = jnp.maximum(n - 1.0, 1.0)
    if m2.ndim == 2:
        var = jnp.diag(m2) / denom
    else:
        var = m2 / denom
    # n >= 2 guard: return ones for n=0 AND n=1 (both degenerate for Bessel).
    return jnp.where(n >= 2, var, jnp.ones_like(var))


# ---------------------------------------------------------------------------
# Policy state types
# ---------------------------------------------------------------------------


class ResetWindowState(NamedTuple):
    """Legacy state type for the hard-reset window buffer.

    Kept for import compatibility.  ``reset_window_buffer`` now returns
    :class:`AccumulatingSplitPopState` (via the ``k=1`` ring path), so
    ``isinstance(state, ResetWindowState)`` will be ``False`` for states
    returned by the current factory.

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
    """State for the split-based rolling-window buffer (``k >= 1``).

    Maintains a ring of ``k`` moment blocks (one active + up to ``k-1``
    completed).  At each ``push_split``, the ring pointer advances to a
    freshly-zeroed slot; when the ring wraps, the oldest slot is zeroed
    (overwritten by the new active slot, implementing exact
    split-granular forgetting).

    This state type is shared by all ring-based policies:
    :func:`reset_window_buffer` (``k=1``), :func:`accumulating_split_pop_buffer`,
    and :func:`ensemble_batch_buffer`.  For ``k=1`` the ring trivially
    implements hard-reset semantics: ``push_split`` advances ``write_pos``
    from 0 to ``(0+1)%1 = 0`` and zeroes slot 0, leaving the block empty.

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

    Notes
    -----
    The number of non-empty slots is recomputable as
    ``jnp.sum(counts > 0)`` and is not stored as a carried field —
    storing it would risk staleness under consecutive empty pushes.

    Split semantics (for ensemble consumers): a "split" is always a
    **draw-axis time partition** — all chains in a batch fold into the
    active block's CGL merge.  Splits are never chain-subset partitions.
    """

    counts: Array  # (k,)
    means: Array  # (k, d)
    m2s: Array  # (k, d, d) or (k, d)
    write_pos: Array  # ()  int32


class LateStartState(NamedTuple):
    """State for the late-start offset policy.

    Wraps an inner policy state (any of the three ring-based policies) and
    suppresses updates for the first ``offset_steps`` calls to ``update``.
    After ``offset_steps`` calls have been skipped the inner policy receives
    all subsequent updates normally.

    Parameters
    ----------
    inner
        The wrapped policy state (e.g., :class:`AccumulatingSplitPopState`).
    num_skipped
        Number of update calls that have been skipped so far, scalar ``()``.
        Saturates at ``offset_steps`` (so the carry never grows unboundedly
        and the shape is static).  Reset to zero on every ``push_split``.
    """

    inner: NamedTuple  # inner policy state
    num_skipped: Array  # ()


# ---------------------------------------------------------------------------
# Single unified ring-buffer factory
# ---------------------------------------------------------------------------


def _make_split_pop_fns(
    d: int,
    k: int,
    diagonal: bool,
    n_chains_per_update: int | None,
    requires_draws: bool,
) -> MetricBuffer:
    """Unified ring-buffer core for all ring-based policies.

    ``n_chains_per_update``:
      - ``None`` → :func:`accumulating_split_pop_buffer` /
        :func:`reset_window_buffer` (no per-call shape contract)
      - ``int`` → :func:`ensemble_batch_buffer` (trace-time shape guard
        in ``update``: ``batch.shape[0] != n_chains_per_update`` raises
        ``ValueError``; fires at JIT trace time, free at runtime)
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
        # counts uses the ambient float dtype (matches means/m2s) so CGL
        # merge weights (n_b/n_ab) are computed at full available precision.
        # write_pos is always int32 (array index).
        return AccumulatingSplitPopState(
            counts=jnp.zeros((k,)),
            means=jnp.zeros((k, d)),
            m2s=jnp.zeros((k,) + _m2_shape),
            write_pos=jnp.zeros((), dtype=jnp.int32),
        )

    def update(
        state: AccumulatingSplitPopState, batch: Array
    ) -> AccumulatingSplitPopState:
        """CGL-merge a new batch into the active block (``state[write_pos]``).

        All samples in ``batch`` fold into the active block via
        :func:`cgl_update_batch`.  For ensemble consumers, ``batch`` is
        ``(n_chains, d)`` and all chains are treated as a single temporal
        batch — a "split" is a time-range partition, not a chain-subset
        partition (draw-axis split semantics).

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

        # Trace-time shape guard for ensemble consumers (n_chains_per_update set).
        if n_chains_per_update is not None and batch.shape[0] != n_chains_per_update:
            raise ValueError(
                f"ensemble_batch_buffer: expected batch.shape[0]={n_chains_per_update}, "
                f"got {batch.shape[0]}.  Partial batches are not supported -- "
                f"all n_chains must participate in each update call."
            )

        wp = state.write_pos
        cur_block = MomentBlock(
            count=state.counts[wp],
            mean=state.means[wp],
            m2=state.m2s[wp],
        )
        updated = cgl_update_batch(cur_block, batch)

        new_counts = state.counts.at[wp].set(updated.count)
        new_means = state.means.at[wp].set(updated.mean)
        new_m2s = state.m2s.at[wp].set(updated.m2)

        return AccumulatingSplitPopState(
            counts=new_counts,
            means=new_means,
            m2s=new_m2s,
            write_pos=wp,
        )

    def push_split(
        state: AccumulatingSplitPopState,
    ) -> AccumulatingSplitPopState:
        """Finalise the active block, advance to a fresh slot.

        Advances ``write_pos`` by 1 (mod ``k``), zeroing out the newly
        active slot.  This implements the split-granular forgetting:
        when the ring wraps, the oldest completed block's slot is
        overwritten (zeroed) — exact "pop oldest split" semantics with O(d²)
        memory rather than a raw-draw ring.

        The split here is a time-range boundary (a "step count" partition).
        For ensemble consumers all chains participated in the now-completed
        block; the split never partitions chains into subsets.

        **Read-before-push ordering violation consequences:**

        - *k=1 (reset_window):* push_split BEFORE get_moments zeroes the
          single accumulator immediately — all accumulated data is lost.
          The subsequent get_moments call returns an empty block (count=0).
          This is the catastrophic failure mode.
        - *k>1 (rolling ring):* push_split BEFORE get_moments advances the
          ring pointer and zeroes what was the oldest completed slot — one
          split of data is silently lost.  The subsequent get_moments call
          returns k-1 retained splits (not the full k-split window).
        """
        old_wp = state.write_pos
        new_wp = (old_wp + 1) % k

        # Zero the newly-active slot (overwriting the oldest when the ring
        # has wrapped k times).
        new_counts = state.counts.at[new_wp].set(
            jnp.zeros((), dtype=state.counts.dtype)
        )
        new_means = state.means.at[new_wp].set(jnp.zeros((d,), dtype=state.means.dtype))
        new_m2s = state.m2s.at[new_wp].set(jnp.zeros(_m2_shape, dtype=state.m2s.dtype))

        return AccumulatingSplitPopState(
            counts=new_counts,
            means=new_means,
            m2s=new_m2s,
            write_pos=new_wp,
        )

    def get_moments(state: AccumulatingSplitPopState) -> MomentBlock:
        """CGL-merge all ``k`` slots (including the active block) into one.

        Empty slots (``count=0``) contribute nothing — the CGL merge
        formula handles them correctly.  For ``k=1``, uses the direct-slice
        short-circuit in :func:`merge_block_ring`.
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
        """Bessel-corrected per-coordinate variance for the step-size proxy.

        Derived from the merged moments across all blocks; step-size
        proxies read this, never the adapted low-rank metric directly.
        Returns ones when fewer than 2 samples have been accumulated
        (isotropic default — see :func:`diag_from_moment_block`).
        """
        return diag_from_moment_block(get_moments(state))

    return MetricBuffer(
        init, update, push_split, get_moments, get_support, get_diag_reference
    )


# ---------------------------------------------------------------------------
# Public policy factories
# ---------------------------------------------------------------------------


def reset_window_buffer(
    d: int,
    *,
    diagonal: bool = False,
    requires_draws: bool = False,
) -> MetricBuffer:
    """Stan-style hard-reset window adaptation buffer.

    Implemented as :func:`_make_split_pop_fns` with ``k=1``.  With a ring
    of size 1, ``push_split`` advances ``write_pos`` from 0 to
    ``(0+1) % 1 = 0`` and zeroes slot 0 — exactly the hard-reset semantics
    of the Stan default.  The ``k=1`` short-circuit in
    :func:`merge_block_ring` ensures no scan is compiled (no overhead vs
    the former single-accumulator implementation).

    State returned by ``init`` is :class:`AccumulatingSplitPopState` (not
    :class:`ResetWindowState`).  The former ``ResetWindowState`` is kept
    as a legacy export for import compatibility.

    The estimator call at window-end::

        metric = sample_covariance_eigh_low_rank(
            m2=buf.get_moments(state).m2,
            count=buf.get_moments(state).count,
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
        ``NotImplementedError``); default ``False`` (opt-in, off by default).

    Returns
    -------
    MetricBuffer
        Named bundle of ``(init, update, push_split, get_moments,
        get_support, get_diag_reference)``.
    """
    return _make_split_pop_fns(
        d=d,
        k=1,
        diagonal=diagonal,
        n_chains_per_update=None,
        requires_draws=requires_draws,
    )


def accumulating_split_pop_buffer(
    d: int,
    k: int,
    *,
    diagonal: bool = False,
    requires_draws: bool = False,
) -> MetricBuffer:
    """Rolling-window buffer with exact oldest-split forgetting.

    Maintains ``k`` CGL-mergeable moment blocks in a ring.  Each call to
    ``push_split`` finalises the currently-active block and advances the
    ring pointer to a fresh slot; when the ring is full the advance
    overwrites the oldest completed block.  This gives exact
    **split-granular forgetting** with O(k·d²) moment memory rather than
    O(k·n_per_split·d) raw draws.

    This is the nuts-rs-faithful forgetting policy: in nuts-rs, the
    ``background_split`` (oldest split) is popped at each window switch and
    the rest of the buffer is retained.  That is exactly what
    ``push_split`` does here, at the block-granularity level.

    **Split semantics**: for this policy a "split" is a time-range
    partition — the caller decides when to call ``push_split`` (e.g., at
    each adaptation window boundary).  For ensemble consumers all chains in
    each update batch fold into the same active block; use
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
        Default ``False`` (raw-draw ring is opt-in).  ``True`` raises
        ``NotImplementedError``.

    Returns
    -------
    MetricBuffer
        Named bundle of ``(init, update, push_split, get_moments,
        get_support, get_diag_reference)``.
    """
    return _make_split_pop_fns(
        d=d,
        k=k,
        diagonal=diagonal,
        n_chains_per_update=None,
        requires_draws=requires_draws,
    )


def ensemble_batch_buffer(
    d: int,
    n_chains: int,
    k: int,
    *,
    diagonal: bool = False,
    requires_draws: bool = False,
) -> MetricBuffer:
    """Rolling-window buffer for ensemble (multi-chain) consumers.

    A specialisation of :func:`accumulating_split_pop_buffer` for
    ``(n_chains, d)`` batch inputs, with explicit draw-axis split semantics
    and a trace-time shape guard on ``update``.

    **Ensemble split semantics**: for ``(nc, d)``-block consumers a
    "split" is a **draw-axis partition** (step-ranges; all chains fold into
    the active block via CGL merge) — NEVER a chain-subset.  Concretely,
    calling ``update(state, batch)`` with ``batch`` shape ``(n_chains, d)``
    folds all ``n_chains`` positions into the single active block's
    sufficient statistics; calling ``push_split`` advances the ring pointer
    to start a new time-range block (all chains still folded together).

    **Pooling note**: moments are pooled across all chains and steps.
    Between-chain dispersion enters the covariance by design — unconverged
    ensembles inflate the estimated covariance by a factor of roughly
    ``1 + between/within``.  A between/within decomposition is NOT
    recoverable from the folded blocks.

    **Shape guard**: ``n_chains`` is checked to be ≥ 1 at factory-creation
    time.  A trace-time guard in ``update`` raises ``ValueError`` if
    ``batch.shape[0] != n_chains`` (fires at JIT trace time, free at
    runtime), turning the shape contract from decorative to enforced.

    This policy is the intended feeding backend for MEADS-LRD and
    ChEES-metric consumers.

    Parameters
    ----------
    d
        Dimension of the position space.
    n_chains
        Number of chains per ensemble update.  Checked ≥ 1 at factory
        creation; enforced at trace time on each ``update`` call.
    k
        Number of splits (moment blocks) in the rolling window.
    diagonal
        If ``True``, M2 fields are shape ``(d,)``; if ``False`` (default),
        shape ``(d, d)``.
    requires_draws
        Default ``False`` (raw-draw ring is opt-in).  ``True`` raises
        ``NotImplementedError``.

    Returns
    -------
    MetricBuffer
        Named bundle of ``(init, update, push_split, get_moments,
        get_support, get_diag_reference)``.
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


def late_start(
    inner_fns: MetricBuffer | tuple,
    offset_steps: int,
) -> MetricBuffer:
    """Offset policy: skip the first ``offset_steps`` update calls, then accumulate.

    Wraps any of the three ring-based policies (or another ``late_start``)
    with a transient-skip period.  The first ``offset_steps`` calls to
    ``update`` are suppressed; all subsequent calls delegate to the inner
    policy.

    ``offset_steps`` counts **update calls**, not individual draws.  For
    ensemble consumers (``ensemble_batch_buffer``) each call feeds
    ``n_chains`` draws, so ``offset_steps`` skips
    ``offset_steps × n_chains`` individual draws.

    The skip counter (``num_skipped`` in :class:`LateStartState`) resets to
    zero on every ``push_split`` call, so each adaptation window has its
    own independent ``offset_steps`` skip period — the late-start is NOT
    cumulative across windows.

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
        A :class:`MetricBuffer` (or legacy 6-tuple) as returned by one of
        the three policy factories.
    offset_steps
        Number of update calls to skip before starting accumulation.
        Must be ≥ 0.

    Returns
    -------
    MetricBuffer
        Named bundle with :class:`LateStartState` as the state type.
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
    ) = inner_fns  # works for MetricBuffer (NamedTuple IS tuple) and legacy 6-tuples

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
        """Delegate to inner policy and reset the skip counter.

        The skip counter resets to 0 so the new window has a fresh,
        independent offset period (not cumulative across windows).
        """
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

    return MetricBuffer(
        init, update, push_split, get_moments, get_support, get_diag_reference
    )
