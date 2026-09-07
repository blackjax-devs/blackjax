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
"""Opt-in observation of metric-publication decisions at slow-window boundaries.

The meta-adaptation controller's ``final()`` computes a candidate low-rank
metric, a set of escalation gate predicates and the actual number of draws it
consumed, then resets its buffers and returns only the new state.  Everything
except the deployed metric is discarded, so a caller watching the adaptation
info stream sees the *outcome* of a publication decision but none of its
*inputs*.

This module adds a read-only record of those inputs.  It changes no threshold,
no gate, no ordering and no published metric: with telemetry disabled nothing
here is constructed or traced.

Scope
-----
**Single-chain only.**  :func:`~blackjax.adaptation.meta.builders.build_multi_chain_meta_core`
raises when telemetry is requested.  The multi-chain controller has distinct
raw-W and routed-T ``R²`` gates, branch-specific support, two candidate metrics
before routing, a three-way unimodality rule and a *historical*
``detection_branch``; a record that flattens those into the single-chain shape
would misreport them.  Its schema is deliberately not settled here.

What this adds, and what it does not
------------------------------------
Not every field here is information that only telemetry can supply, and the
distinction matters when judging the feature's cost.

``deployed_effective_rank`` is *derivable* from the low-rank payload the warmup
already returns, given a declared rank convention -- here, the count of
``|lam - 1| > _LAM_NONTRIVIAL_TOL`` (``1e-6``), the same convention
:func:`~blackjax.adaptation.meta.verdict.extract_meta_verdict` uses for its
``effective_rank``.  It is carried for convenience and so the convention is
stated in one place, not because it is otherwise unobtainable.  The same is true
of ``deployed_logdet`` and ``deployed_sigma_gm``.

What genuinely cannot be recovered afterwards, and is the actual reason this
module exists:

* the **candidate** metric in a window where it was withheld -- it is a local of
  ``final()`` and is discarded when the gates do not fire;
* the **support** the controller consumed, once the buffer index is reset (and,
  when the window exceeded the buffer, at all);
* the **per-gate** outcomes, of which only the conjunction survives as
  ``has_escalated``;
* the **step size in force** for the transitions the window just completed,
  which the adaptation info stream never sees;
* on the multi-chain path, **branch history** -- deferred to a later schema.

Stability
---------
:data:`SCHEMA_VERSION` is **1** and the layout is **provisional**.  It is
expected to change when the multi-chain semantics are settled.  Consumers must
read :data:`SCHEMA_VERSION` and refuse a version they do not know rather than
inferring meaning from raw mask bits.

Gate masks
----------
Two masks are emitted, never one:

``gate_passed``
    bit set iff the predicate was evaluated *and* held.
``gate_evaluated``
    bit set iff the controller actually consulted the predicate this window.

A clear bit in ``gate_passed`` therefore means "failed" only when the same bit
is set in ``gate_evaluated``.  This distinction is load-bearing: the S_gap
stability predicate cannot be evaluated in the first window (it needs the
previous window's S_gap), and once the controller has escalated it stops
consulting the escalation gates altogether.  Reporting either case as a
failure would be wrong.
"""
from __future__ import annotations

from typing import Any, NamedTuple

import jax.numpy as jnp

from blackjax.mcmc.metrics import LowRankInverseMassMatrix
from blackjax.types import Array

__all__ = [
    "SCHEMA_VERSION",
    "GATE_BITS",
    "MetricPublicationRecord",
    "decode_gates",
    "encode_gates",
    "extract_publication_chronology",
    "publication_adapt_info_fn",
    "record_nbytes",
]

#: Provisional.  Bump on any layout or meaning change; consumers must check it.
SCHEMA_VERSION: int = 1

#: Bit positions for the **single-chain** escalation predicates.  Positions are
#: never reused across schema versions.  4-15 are reserved for the multi-chain
#: predicates and are NOT yet assigned -- the multi-chain gate set is unsettled.
GATE_BITS: dict[str, int] = {
    "r2": 0,
    "s_gap_magnitude": 1,
    "s_gap_stability": 2,
    "deadline": 3,
}


class MetricPublicationRecord(NamedTuple):
    """One slow-window metric-publication event (single-chain controller).

    Written inside ``final()`` from live values **before** the draw/gradient
    buffers are reset, so nothing here is reconstructed from cleared buffers.
    The four epsilon fields are stamped afterwards by the staged-adaptation
    host, which is the only layer that sees the step-size state.

    Between window boundaries the carry holds the previous window's record
    unchanged; deduplicate on ``window_index``.  It is ``-1`` before the first
    publication, which distinguishes "nothing published yet" from window 0.

    Numeric fields keep the dtype of the quantity they observe.  The epsilons
    all carry the dual-averaging scheme's own working dtype (float64 under
    ``jax_enable_x64``), never a forced float32.

    Attributes
    ----------
    window_index
        0-based publication counter; ``-1`` before the first publication.
    step_at_publication
        ``budget_used`` on entry to ``final()``.  Warmup **steps** for the
        single-chain controller (not gradient evaluations, and not a per-chain
        count -- those differ on the multi-chain path).
    support_n
        Draws the controller actually consumed: ``min(buffer_idx, buffer_capacity)``
        where ``buffer_idx`` is read *after* the window's last ``update()``.
    support_saturated
        ``True`` when ``buffer_idx > buffer_capacity``, i.e. the window was
        longer than the buffer and the oldest draws were overwritten.  When
        this is set, ``support_n`` is the capacity and the true window length
        is not recoverable from the record.
    buffer_capacity
        Buffer length ``B``; needed to interpret ``support_saturated``.
    gate_passed, gate_evaluated
        Escalation predicate masks; see the module docstring.  Decode with
        :func:`decode_gates`.
    escalated_now
        The controller escalated *in this window*.
    has_escalated_before
        The controller had already escalated on entry, in which case the
        escalation gates were not consulted (``gate_evaluated`` is 0).
    has_escalated
        Post-update monotone escalation flag.
    detection_rank
        ``k_new``: the rank chosen from the whitened-residual spectrum.  This
        is the *detection* quantity and need not equal the effective rank of
        either metric below.
    candidate_effective_rank
        Non-trivial eigenvalue count of the candidate metric.
    deployed_effective_rank
        Non-trivial eigenvalue count of the metric published for the next window,
        under the declared convention ``|lam - 1| > 1e-6``.  Recomputable from
        the returned metric; see the module docstring.
    escalation_rank
        The carried nominal rank recorded at escalation (0 before escalation).
    candidate_logdet, deployed_logdet, in_force_logdet
        ``log det M^-1 = 2*sum(log sigma) + sum(log lam)`` (exact for this
        parameterisation, whose ``U`` has orthonormal columns) for, respectively,
        the candidate, the metric published for the next window, and the metric
        that drove the window just completed.
    candidate_lam_max, candidate_lam_min
        Extremes of the candidate's eigenvalue vector.
    candidate_sigma_gm, deployed_sigma_gm
        Geometric means of the diagonal scalings.
    sigma_log_ratio_rms
        ``rms(log(candidate sigma) - log(deployed sigma))``; zero when the
        candidate is what was deployed.
    r2, r2_mode, s_gap, s_gap_prev, s_gap_relative_change
        Signals as measured this window.  ``r2`` is NaN on the deferred path.
    is_slow_mixing
        Transient-mixing class reported by the controller.
    epsilon_in_force
        Step size that drove the window's **last completed transition** -- the
        value handed to the MCMC kernel, captured before that step's
        dual-averaging update.
    epsilon_after_window_da
        Step size the dual averaging had reached after that final update, i.e.
        ``exp(log_step_size)`` immediately before the window-boundary reset.
    epsilon_window_average
        ``da_final(ss_state)`` -- the window's averaged step size, which seeds
        the next window's dual averaging.
    epsilon_next_window
        ``exp(log_step_size)`` after the boundary reset: the step size the first
        transition of the next window will use, under the newly published metric.

        These four are deliberately distinct and are never collapsed:
        ``epsilon_window_average`` and ``epsilon_next_window`` are related by
        ``exp(log(.))``, which is not guaranteed bitwise-identical, and
        ``epsilon_in_force`` is generally none of the others.
    candidate, deployed
        Full :class:`~blackjax.mcmc.metrics.LowRankInverseMassMatrix` factors,
        or ``None`` unless the core was built with ``full_matrices=True``.
        These are O(d*k) per record; see :func:`record_nbytes`.
    """

    window_index: Array
    step_at_publication: Array
    support_n: Array
    support_saturated: Array
    buffer_capacity: Array
    gate_passed: Array
    gate_evaluated: Array
    escalated_now: Array
    has_escalated_before: Array
    has_escalated: Array
    detection_rank: Array
    candidate_effective_rank: Array
    deployed_effective_rank: Array
    escalation_rank: Array
    candidate_logdet: Array
    deployed_logdet: Array
    in_force_logdet: Array
    candidate_lam_max: Array
    candidate_lam_min: Array
    candidate_sigma_gm: Array
    deployed_sigma_gm: Array
    sigma_log_ratio_rms: Array
    r2: Array
    r2_mode: Array
    s_gap: Array
    s_gap_prev: Array
    s_gap_relative_change: Array
    is_slow_mixing: Array
    epsilon_in_force: Array
    epsilon_after_window_da: Array
    epsilon_window_average: Array
    epsilon_next_window: Array
    candidate: LowRankInverseMassMatrix | None = None
    deployed: LowRankInverseMassMatrix | None = None


#: Fields the staged-adaptation host stamps after ``final()`` returns.
EPSILON_FIELDS: tuple[str, ...] = (
    "epsilon_in_force",
    "epsilon_after_window_da",
    "epsilon_window_average",
    "epsilon_next_window",
)


def encode_gates(**bits: Any) -> Array:
    """Pack named gate predicates into an int32 mask using :data:`GATE_BITS`.

    Unnamed gates are left clear.  Raises on an unknown name so a typo cannot
    silently produce an all-clear mask.
    """
    unknown = set(bits) - set(GATE_BITS)
    if unknown:
        raise KeyError(
            f"encode_gates: unknown gate name(s) {sorted(unknown)}. "
            f"known gates are {sorted(GATE_BITS)}"
        )
    mask = jnp.zeros((), dtype=jnp.int32)
    for name, value in bits.items():
        mask = mask | (
            jnp.asarray(value, dtype=jnp.int32) << jnp.int32(GATE_BITS[name])
        )
    return mask


def decode_gates(mask) -> dict[str, bool]:
    """Unpack an int32 gate mask into ``{gate_name: bool}``.

    Works on a concrete (post-``run``) mask, not inside a trace.
    """
    import numpy as np

    m = int(np.asarray(mask))
    return {name: bool((m >> bit) & 1) for name, bit in GATE_BITS.items()}


def publication_adapt_info_fn():
    """Build an ``adaptation_info_fn`` returning **only** the publication record.

    The generic :func:`~blackjax.adaptation.base.get_filter_adapt_info_fn`
    filters top-level fields only, so it can keep or drop ``imm_state`` whole --
    including the ``(buffer_size, d)`` draw and gradient buffers, which
    ``lax.scan`` would then stack once per step.  ``low_rank_adaptation``'s own
    default info fn documents a real 41 GB out-of-memory caused by exactly that.
    This helper stacks the record and nothing else.

    Returns
    -------
    Callable
        ``(state, info, adaptation_state) -> MetricPublicationRecord``, suitable
        for ``staged_adaptation(..., adaptation_info_fn=...)``.
    """

    def _publication_only(state, info, adaptation_state):
        del state, info
        return adaptation_state.imm_state.publication

    return _publication_only


def extract_publication_chronology(records: MetricPublicationRecord) -> list[dict]:
    """Reduce a per-step stacked record to one entry per publication.

    Drops the ``window_index == -1`` prefix (before the first publication) and
    keeps the first occurrence of each ``window_index``, since the carry holds
    a record unchanged between boundaries.

    Parameters
    ----------
    records
        A :class:`MetricPublicationRecord` whose leaves carry a leading
        ``num_steps`` axis -- the second return of ``warmup.run()`` when
        :func:`publication_adapt_info_fn` was used.

    Returns
    -------
    list[dict]
        One dict per publication, in window order, plus ``schema_version``.
        Full-matrix factors are omitted; read them off the stacked record
        directly when they are needed.
    """
    import numpy as np

    window_index = np.asarray(records.window_index)
    if window_index.ndim != 1:
        raise ValueError(
            "extract_publication_chronology: expected records stacked over a "
            f"leading step axis, got window_index with shape {window_index.shape}. "
            "Pass the adaptation info from a run that used publication_adapt_info_fn()."
        )

    scalar_fields = [
        f
        for f in records._fields
        if f not in ("candidate", "deployed") and getattr(records, f) is not None
    ]

    out: list[dict] = []
    seen: set[int] = set()
    for step in range(window_index.shape[0]):
        w = int(window_index[step])
        if w < 0 or w in seen:
            continue
        seen.add(w)
        entry: dict[str, Any] = {"schema_version": SCHEMA_VERSION}
        for name in scalar_fields:
            value = np.asarray(getattr(records, name))[step]
            if name in ("gate_passed", "gate_evaluated"):
                entry[name] = int(value)
            elif value.dtype == np.bool_:
                entry[name] = bool(value)
            elif np.issubdtype(value.dtype, np.integer):
                entry[name] = int(value)
            else:
                entry[name] = float(value)
        entry["gates_passed"] = decode_gates(entry["gate_passed"])
        entry["gates_evaluated"] = decode_gates(entry["gate_evaluated"])
        out.append(entry)

    out.sort(key=lambda e: e["window_index"])
    return out


def record_nbytes(record: MetricPublicationRecord) -> int:
    """Actual byte size of a record, summed over its real leaves.

    Reports what the arrays occupy rather than a nominal
    ``n_fields * 4`` figure: dtypes differ (``float64`` under x64) and the
    optional full-matrix factors dominate when they are present.
    """
    import jax

    return sum(int(jnp.asarray(leaf).nbytes) for leaf in jax.tree.leaves(record))
