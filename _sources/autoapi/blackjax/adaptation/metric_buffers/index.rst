blackjax.adaptation.metric_buffers
==================================

.. py:module:: blackjax.adaptation.metric_buffers

.. autoapi-nested-parse::

   Data-feeding (buffer) layer for metric adaptation.

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



Classes
-------

.. autoapisummary::

   blackjax.adaptation.metric_buffers.MetricBuffer
   blackjax.adaptation.metric_buffers.MomentBlock
   blackjax.adaptation.metric_buffers.AccumulatingSplitPopState
   blackjax.adaptation.metric_buffers.LateStartState


Functions
---------

.. autoapisummary::

   blackjax.adaptation.metric_buffers.cgl_merge_two
   blackjax.adaptation.metric_buffers.cgl_update_batch
   blackjax.adaptation.metric_buffers.merge_block_ring
   blackjax.adaptation.metric_buffers.diag_from_moment_block
   blackjax.adaptation.metric_buffers.reset_window_buffer
   blackjax.adaptation.metric_buffers.accumulating_split_pop_buffer
   blackjax.adaptation.metric_buffers.ensemble_batch_buffer
   blackjax.adaptation.metric_buffers.late_start


Module Contents
---------------

.. py:class:: MetricBuffer



   Buffer policy as a NamedTuple of callables (house-style).

   Follows the :class:`~blackjax.base.SamplingAlgorithm` convention of
   bundling a policy's callables into a named container so consumers can
   access them by name (``buf.get_moments(state)``) rather than by
   positional destructuring.  Still positionally compatible with tuple
   unpacking (``NamedTuple`` IS a ``tuple``).

   :param init: ``() -> PolicyState``
   :param update: ``(state, batch) -> PolicyState``
   :param push_split: ``(state) -> PolicyState``
   :param get_moments: ``(state) -> MomentBlock``
   :param get_support: ``(state) -> tuple[Array, Array]``
   :param get_diag_reference: ``(state) -> Array``


   .. py:attribute:: init
      :type:  Callable


   .. py:attribute:: update
      :type:  Callable


   .. py:attribute:: push_split
      :type:  Callable


   .. py:attribute:: get_moments
      :type:  Callable


   .. py:attribute:: get_support
      :type:  Callable


   .. py:attribute:: get_diag_reference
      :type:  Callable


.. py:class:: MomentBlock



   CGL-mergeable sufficient statistics for covariance estimation.

   Carries exactly the fields consumed by
   :func:`~blackjax.adaptation.metric_estimators.sample_covariance_eigh_low_rank`:

   .. code-block:: python

       metric = sample_covariance_eigh_low_rank(m2=block.m2, count=block.count, k)

   The ``m2`` field is the **accumulated sum of squared deviations** (CGL's
   M₂ matrix), NOT the covariance.  For a dense block ``m2`` has shape
   ``(d, d)``; for a diagonal block it has shape ``(d,)`` (the diagonal).

   :param count: Number of samples accumulated, scalar ``()``.
   :param mean: Sample mean, shape ``(d,)``.
   :param m2: Sum of squared deviations, shape ``(d, d)`` (dense) or ``(d,)``
              (diagonal).

   .. rubric:: Notes

   An empty (zero-initialised) block has ``count=0``.  The CGL merge
   formula is safe for empty blocks — the result equals the non-empty
   partner when one block is empty.


   .. py:attribute:: count
      :type:  blackjax.types.Array


   .. py:attribute:: mean
      :type:  blackjax.types.Array


   .. py:attribute:: m2
      :type:  blackjax.types.Array


.. py:function:: cgl_merge_two(block_a: MomentBlock, block_b: MomentBlock) -> MomentBlock

   CGL-merge two pre-accumulated moment blocks.

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

   :param block_a: Two :class:`MomentBlock` instances to merge.  Must have the same
                   ``m2`` shape (both dense ``(d, d)`` or both diagonal ``(d,)``).
   :param block_b: Two :class:`MomentBlock` instances to merge.  Must have the same
                   ``m2`` shape (both dense ``(d, d)`` or both diagonal ``(d,)``).

   :returns: Merged block with combined statistics.
   :rtype: MomentBlock


.. py:function:: cgl_update_batch(block: MomentBlock, batch: blackjax.types.Array) -> MomentBlock

   CGL-merge an existing moment block with a new batch of raw draws.

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

   :param block: Existing :class:`MomentBlock` (may be empty, ``count=0``).
   :param batch: New raw draws, shape ``(n_b, d)``.

   :returns: Updated block merging the previous statistics with the new batch.
   :rtype: MomentBlock


.. py:function:: merge_block_ring(counts: blackjax.types.Array, means: blackjax.types.Array, m2s: blackjax.types.Array) -> MomentBlock

   Reduce a ring of ``k`` moment blocks into a single merged block.

   For ``k == 1`` uses a direct-slice short-circuit (no ``lax.scan``
   compiled) that is bit-identical to the scan path while avoiding the
   ~1.6× compile overhead at large ``d``.  For ``k > 1`` iterates with
   :func:`~jax.lax.scan`, CGL-merging each slot in turn.  Empty slots
   (``count=0``) contribute nothing to the merged result — the CGL merge
   formula handles them correctly via the zero-count guard in
   :func:`cgl_merge_two`.

   :param counts: Shape ``(k,)``.  Per-block sample counts.  Zero entries indicate
                  empty (unfilled) slots.
   :param means: Shape ``(k, d)``.
   :param m2s: Shape ``(k, d, d)`` or ``(k, d)``.

   :returns: CGL-merged result across all ``k`` blocks (or the zero block if
             all slots are empty).
   :rtype: MomentBlock


.. py:function:: diag_from_moment_block(block: MomentBlock) -> blackjax.types.Array

   Bessel-corrected per-coordinate variance from a :class:`MomentBlock`.

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

   :param block: A :class:`MomentBlock`.  Dense blocks (``m2`` shape ``(d, d)``)
                 extract the diagonal; diagonal blocks (``m2`` shape ``(d,)``) use
                 ``m2`` directly.

   :returns: Per-coordinate Bessel-corrected variance (denominator ``count - 1``),
             or ones when ``count < 2``.
   :rtype: Array, shape ``(d,)``

   .. rubric:: Notes

   This returns the **Bessel-corrected** (unbiased) sample variance with
   denominator ``count - 1``.  This matches the convention of
   :func:`~blackjax.adaptation.metric_estimators.sample_covariance_eigh_low_rank`.
   Consumers expecting **population variance** (denominator ``count``) will
   observe an upward shift of ``count / (count - 1)`` relative to their
   expectation; callers using this output as a population-variance proxy
   must account for that factor.


.. py:class:: AccumulatingSplitPopState



   State for the split-based rolling-window buffer (``k >= 1``).

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

   :param counts: Per-block sample counts, shape ``(k,)``.  Zero entries are empty.
   :param means: Per-block running means, shape ``(k, d)``.
   :param m2s: Per-block M2 matrices, shape ``(k, d, d)`` or ``(k, d)``.
   :param write_pos: Index of the currently-active (in-progress) block, scalar ``()``.

   .. rubric:: Notes

   The number of non-empty slots is recomputable as
   ``jnp.sum(counts > 0)`` and is not stored as a carried field —
   storing it would risk staleness under consecutive empty pushes.

   Split semantics (for ensemble consumers): a "split" is always a
   **draw-axis time partition** — all chains in a batch fold into the
   active block's CGL merge.  Splits are never chain-subset partitions.


   .. py:attribute:: counts
      :type:  blackjax.types.Array


   .. py:attribute:: means
      :type:  blackjax.types.Array


   .. py:attribute:: m2s
      :type:  blackjax.types.Array


   .. py:attribute:: write_pos
      :type:  blackjax.types.Array


.. py:class:: LateStartState



   State for the late-start offset policy.

   Wraps an inner policy state (any of the three ring-based policies) and
   suppresses updates for the first ``offset_steps`` calls to ``update``.
   After ``offset_steps`` calls have been skipped the inner policy receives
   all subsequent updates normally.

   :param inner: The wrapped policy state (e.g., :class:`AccumulatingSplitPopState`).
   :param num_skipped: Number of update calls that have been skipped so far, scalar ``()``.
                       Saturates at ``offset_steps`` (so the carry never grows unboundedly
                       and the shape is static).  Reset to zero on every ``push_split``.


   .. py:attribute:: inner
      :type:  NamedTuple


   .. py:attribute:: num_skipped
      :type:  blackjax.types.Array


.. py:function:: reset_window_buffer(d: int, *, diagonal: bool = False, requires_draws: bool = False) -> MetricBuffer

   Stan-style hard-reset window adaptation buffer.

   Implemented as :func:`_make_split_pop_fns` with ``k=1``.  With a ring
   of size 1, ``push_split`` advances ``write_pos`` from 0 to
   ``(0+1) % 1 = 0`` and zeroes slot 0 — exactly the hard-reset semantics
   of the Stan default.  The ``k=1`` short-circuit in
   :func:`merge_block_ring` ensures no scan is compiled (no overhead vs
   the former single-accumulator implementation).

   State returned by ``init`` is :class:`AccumulatingSplitPopState`.

   The estimator call at window-end::

       metric = sample_covariance_eigh_low_rank(
           m2=buf.get_moments(state).m2,
           count=buf.get_moments(state).count,
           max_rank=k,
       )

   :param d: Dimension of the position space.
   :param diagonal: If ``True``, the M2 field has shape ``(d,)`` (diagonal sufficient
                    statistics); if ``False`` (default), shape ``(d, d)``.
   :param requires_draws: If ``True``, attach a raw-draw ring to the state for draw-SVD
                          estimators.  **Currently not implemented** (raises
                          ``NotImplementedError``); default ``False`` (opt-in, off by default).

   :returns: Named bundle of ``(init, update, push_split, get_moments,
             get_support, get_diag_reference)``.
   :rtype: MetricBuffer


.. py:function:: accumulating_split_pop_buffer(d: int, k: int, *, diagonal: bool = False, requires_draws: bool = False) -> MetricBuffer

   Rolling-window buffer with exact oldest-split forgetting.

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

   :param d: Dimension of the position space.
   :param k: Number of splits (moment blocks) in the rolling window.  The oldest
             block is dropped when more than ``k`` blocks have accumulated.
             Must be ≥ 1.
   :param diagonal: If ``True``, M2 fields are shape ``(d,)`` (diagonal); if ``False``
                    (default), shape ``(d, d)``.
   :param requires_draws: Default ``False`` (raw-draw ring is opt-in).  ``True`` raises
                          ``NotImplementedError``.

   :returns: Named bundle of ``(init, update, push_split, get_moments,
             get_support, get_diag_reference)``.
   :rtype: MetricBuffer


.. py:function:: ensemble_batch_buffer(d: int, n_chains: int, k: int, *, diagonal: bool = False, requires_draws: bool = False) -> MetricBuffer

   Rolling-window buffer for ensemble (multi-chain) consumers.

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

   :param d: Dimension of the position space.
   :param n_chains: Number of chains per ensemble update.  Checked ≥ 1 at factory
                    creation; enforced at trace time on each ``update`` call.
   :param k: Number of splits (moment blocks) in the rolling window.
   :param diagonal: If ``True``, M2 fields are shape ``(d,)``; if ``False`` (default),
                    shape ``(d, d)``.
   :param requires_draws: Default ``False`` (raw-draw ring is opt-in).  ``True`` raises
                          ``NotImplementedError``.

   :returns: Named bundle of ``(init, update, push_split, get_moments,
             get_support, get_diag_reference)``.
   :rtype: MetricBuffer


.. py:function:: late_start(inner_fns: MetricBuffer | tuple, offset_steps: int) -> MetricBuffer

   Offset policy: skip the first ``offset_steps`` update calls, then accumulate.

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

   :param inner_fns: A :class:`MetricBuffer` (or legacy 6-tuple) as returned by one of
                     the three policy factories.
   :param offset_steps: Number of update calls to skip before starting accumulation.
                        Must be ≥ 0.

   :returns: Named bundle with :class:`LateStartState` as the state type.
   :rtype: MetricBuffer


