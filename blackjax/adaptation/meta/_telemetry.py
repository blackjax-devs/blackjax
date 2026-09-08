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

The controller's ``final()`` computes candidates, predicates, and draw counts,
then resets its buffers; the adaptation info stream otherwise exposes only the
deployed metric.  This module records those inputs read-only.  It changes no
threshold, predicate, ordering, or published metric, and telemetry-disabled
runs construct or trace nothing here.

Two masks, and what they do *not* mean
--------------------------------------
The escalation expression (for example,
``~has_escalated & r2_gate & s_gap_gate & deadline_ok``) uses eager ``&``:
every predicate is computed every window, including after escalation and when
siblings are false.  Some also govern the multi-chain deferral latch, which is
not gated on ``has_escalated``.

Thus the masks report predicate truth and applicability, not computation:

``gate_predicate_true``
    Raw truth of each named predicate this window, always populated so branch
    decisions and the deferral latch remain reconstructable.

``escalation_gate_applicable``
    Whether the predicate bears on a **new** escalation decision.  Clear after
    escalation and when inputs are unavailable (the first S_gap stability test
    lacks a previous-window S_gap).

**The two are not nested.** A raw predicate can be true while its applicability
is zero; that is the normal post-escalation state and is not a contradiction.
Never read a clear ``gate_predicate_true`` bit as "failed" without checking
whether it was applicable, and never read a clear applicability bit as "the
controller did not compute this".

The ``sc_`` / ``w_`` / ``t_`` name prefixes are a **routing key**, not
decoration: :func:`extract_publication_chronology` selects which gates to decode
from them, so a new predicate must carry the prefix of the controller it belongs
to.  ``deadline`` is the sole unprefixed name, shared because both controllers
apply the identical budget test.

Stability
---------
Read :data:`SCHEMA_VERSION` from the module; do not hard-code it here or
anywhere else.  Consumers must check it and refuse a version they do not know
rather than inferring meaning from raw mask bits.  Bit positions are never
reused across versions.

Units
-----
Counts here are **not** warmup scan steps.  ``core_updates_per_chain``
counts calls to the metric core's ``update()``, which the host makes only on
slow-window steps: under Stan's :func:`~blackjax.adaptation.staged_adaptation.build_schedule`
it lags the scan index by the whole initial fast buffer.  Do not assume 75:
that is the default ``initial_buffer_size``, but ``build_schedule`` rescales the
buffers when they do not fit ``num_steps``, and the growing-window schedule the
auto path uses by default has no fast prefix at all — the lag is then zero.
Derive it from the schedule, never from the constant.
``warmup_step_index`` is the scan index and is stamped by the host.
``support_pooled_rows`` is a count of retained buffer rows; it is neither an
effective sample size nor a count of independent observations.  The rows
within a chain are autocorrelated draws, and the chains are not independent of
each other either: they share one adapted metric and one step size, so each
window's adaptation couples them.
``support_per_chain``, ``buffer_capacity`` and ``dropped_draws`` are per-chain;
``support_pooled_rows`` and ``core_chain_updates_total`` are the only
summed ones.
"""
from __future__ import annotations

from typing import Any, NamedTuple

import jax.numpy as jnp

from blackjax.adaptation.meta._calibration import (
    _DETECTION_BRANCH_BETWEEN_MEANS,
    _DETECTION_BRANCH_BOTH,
    _DETECTION_BRANCH_NONE,
    _DETECTION_BRANCH_POOLED_WITHIN,
    _LAM_NONTRIVIAL_TOL,
)
from blackjax.mcmc.metrics import LowRankInverseMassMatrix
from blackjax.types import Array

__all__ = [
    "SCHEMA_VERSION",
    "GATE_BITS",
    "ROUTE_DIAGONAL",
    "ROUTE_W",
    "ROUTE_T",
    "BRANCH_NONE",
    "BRANCH_W",
    "BRANCH_T",
    "BRANCH_BOTH",
    "CandidateSummary",
    "MetricPublicationRecord",
    "MultiChainDetail",
    "SingleChainDetail",
    "decode_gates",
    "encode_gates",
    "extract_publication_chronology",
    "publication_adapt_info_fn",
    "record_nbytes",
    "candidate_summary",
    "empty_candidate_summary",
    "empty_record",
]

SCHEMA_VERSION: int = 3

#: Bit positions, branch-scoped and disjoint.  ``deadline`` is shared because
#: the two controllers apply the identical budget test.  Positions are frozen
#: within a schema version and never reused across versions.
GATE_BITS: dict[str, int] = {
    # shared
    "deadline": 0,
    # single-chain controller
    "sc_r2": 1,
    "sc_s_gap_magnitude": 2,
    "sc_s_gap_stability": 3,
    # multi-chain, W branch (pooled within-chain spread)
    "w_magnitude": 4,
    "w_psi": 5,
    "w_r1": 6,
    "w_r2_raw": 7,
    # multi-chain, T branch (between-chain means)
    "t_magnitude": 8,
    "t_collinearity": 9,
    "t_loo": 10,
    "t_support": 11,
    "t_unimodality": 12,
    "t_r2_routed": 13,
}

#: Which metric the kernel will actually use next.  ``BOTH`` branches firing is
#: a *detection* outcome, not a route: the controller deploys the W metric.
ROUTE_DIAGONAL: int = 0
ROUTE_W: int = 1
ROUTE_T: int = 2

# Branch codes are the controller's own (_DETECTION_BRANCH_*), reused rather
# than re-encoded so a consumer never has to map between two vocabularies.
BRANCH_NONE: int = _DETECTION_BRANCH_NONE
BRANCH_W: int = _DETECTION_BRANCH_POOLED_WITHIN
BRANCH_T: int = _DETECTION_BRANCH_BETWEEN_MEANS
BRANCH_BOTH: int = _DETECTION_BRANCH_BOTH


class CandidateSummary(NamedTuple):
    """Compact description of one candidate inverse mass matrix.

    Emitted for every candidate, including candidates computed but not deployed
    before escalation.  On the multi-chain path, both freshly computed
    candidates remain recorded; history selects one route for deployment.

    Attributes
    ----------
    effective_rank
        Count of ``|lam - 1| > 1e-6``, matching the verdict convention.
    logdet
        ``2*sum(log sigma) + sum(log lam)`` under the documented
        :func:`_logdet` invariant: active ``lam != 1`` columns of ``U`` are
        orthonormal (including T's one-active-column case).
    lam_max, lam_min, sigma_gm
        Eigenvalue extremes and the geometric mean of the diagonal scaling.
        Only ``sigma_gm`` is shared by W and T on the multi-chain path; their
        eigenvalue summaries can differ.
    sigma_log_ratio_rms_vs_deployed
        RMS log-scale difference from the deployed metric.  This compares only
        diagonal scaling; W and T share the same value, so candidate identity
        requires full factors under a declared comparison convention.
    full
        The full factors, or ``None`` unless ``full_matrices=True``.
    """

    effective_rank: Array
    logdet: Array
    lam_max: Array
    lam_min: Array
    sigma_gm: Array
    sigma_log_ratio_rms_vs_deployed: Array
    full: Any = None


class SingleChainDetail(NamedTuple):
    """Fields specific to the single-chain meta-adaptation core.

    Attributes
    ----------
    detection_rank
        ``k_new`` from the whitened-residual spectrum; distinct from candidate,
        deployed, and stored nominal escalation ranks.
    s_gap, s_gap_prev, s_gap_relative_change
        Stability inputs; ``s_gap_prev`` is NaN in the first window, making
        ``sc_s_gap_stability`` inapplicable there.
    candidate
        The single low-rank candidate this controller builds.
    """

    detection_rank: Array
    s_gap: Array
    s_gap_prev: Array
    s_gap_relative_change: Array
    candidate: CandidateSummary


class MultiChainDetail(NamedTuple):
    """Fields specific to the multi-chain meta-adaptation core.

    Branch fields intentionally answer different questions:

    ``branch_fired_this_window``
        What escalated *now*: ``NONE`` / W / T / ``BOTH``.
    ``detection_branch_history``
        Carried last firing branch, unchanged when nothing fires; this selects
        the escalated metric and can diverge from the current branch.
    ``deployed_metric_route``
        Metric used by the kernel: diagonal before escalation, then W or T;
        BOTH routes to W.  It is carried to avoid reimplementing this rule.

    Attributes
    ----------
    branch_first_set_at_window
        First window where ``detection_branch`` left ``NONE``; ``-1`` before.
    t_detection_rank
        Between-chain ``k_new``.  The controller stores it as
        ``escalation_rank`` even when W fires, so the stored rank can describe a
        different branch; the record reports it unchanged.
    r2_routed
        GAIN-overridden R² used by T and the verdict, distinct from raw
        ``r2_raw`` used by W; they can disagree, especially when ``k << d``.
    is_converging, is_unimodal, any_mode_flag
        Separate observations behind the three-way unimodality rule.
    t_unimodality_resolved
        Resolved outcome ``is_converging | (is_unimodal & ~any_mode_flag)``;
        convergence is an override, not a fourth conjunct.  This policy can
        change and equals the ``t_unimodality`` gate bit by construction.
    t_contraction_stat, unimodality_gap_ratio
        The numerics behind those observations.
    unimodality_flag_count
        Consecutive flagged windows; deferral waits for the confirmation limit.
    deferred_to_ensemble
        Deferral latch; non-monotone and not gated on ``has_escalated``.
    chain_collinearity_f1, within_lam1, chain_consistency_psi, r1_top
        Detector signals measured this window.
    candidate_w, candidate_t
        Both candidates, built before routing: W is full Fisher-LR on pooled
        per-chain-centred buffers; T is a rank-1 geometric-mean correction.
    """

    branch_fired_this_window: Array
    detection_branch_history: Array
    deployed_metric_route: Array
    branch_first_set_at_window: Array
    t_detection_rank: Array
    r2_routed: Array
    is_converging: Array
    is_unimodal: Array
    any_mode_flag: Array
    t_unimodality_resolved: Array
    t_contraction_stat: Array
    unimodality_gap_ratio: Array
    unimodality_flag_count: Array
    deferred_to_ensemble: Array
    chain_collinearity_f1: Array
    within_lam1: Array
    chain_consistency_psi: Array
    r1_top: Array
    candidate_w: CandidateSummary
    candidate_t: CandidateSummary


class MetricPublicationRecord(NamedTuple):
    """One slow-window metric-publication event.

    Written inside ``final()`` from live values **before** the buffers are
    reset, so nothing is reconstructed from cleared buffers.  The epsilons and
    ``warmup_step_index`` are stamped afterwards by the staged-adaptation host,
    which is the only layer that sees the step-size state and the scan index.

    Exactly one of ``single_chain`` / ``multi_chain`` is populated; the other is
    ``None`` and contributes no pytree leaves.

    **A single-chain record and a multi-chain record therefore have different
    treedefs**, because the ``None`` sits in a different slot.  They cannot be
    stacked, ``tree_map``-ed or concatenated together as pytrees; combine them
    at the dict level, after :func:`extract_publication_chronology`.

    Between window boundaries the carry holds the previous record unchanged;
    deduplicate on ``window_index``, which is ``-1`` before the first
    publication.

    Attributes
    ----------
    window_index
        0-based publication counter; ``-1`` before the first publication.
    warmup_step_index
        The host's scan index at this boundary, or ``-1`` if the host did not
        stamp one.  This is the *only* field measured in warmup steps.
    core_updates_per_chain, core_chain_updates_total
        Metric-core updates, per chain and summed over chains.  One core
        ``update()`` consumes all ``n_chains`` chains at once, so a single
        update adds 1 to the first and ``n_chains`` to the second — the total
        counts chain-contributions, not invocations, which is why it is not
        named for calls.  ``total == per_chain * n_chains`` always: one quantity
        in two units, not two measurements.

        **Neither is a warmup step count.** The host calls ``update()`` only on
        slow stages, so under Stan's schedule both lag the scan index by the
        whole initial fast buffer.  ``warmup_step_index`` alone carries the host
        warmup clock, under the indexing convention stated on that field.

        **Cumulative across windows**, not per-window: the controller
        deliberately does not reset ``budget_used`` at a boundary, so these grow
        monotonically over the warmup.  Difference successive records for a
        per-window count.  The per-chain figure divides ``budget_used`` by
        ``n_chains``, which is exact because the counter advances by
        ``n_chains`` per update — unless a caller supplies an
        ``initial_metric_state`` carrying an off-multiple value.
    n_chains, dim
        Chain count and dimension.  Carried so a consumer can recompute the
        calibration thresholds (which are pure functions of ``M``, ``n``, ``d``)
        without duplicating them.
    support_per_chain
        Draws per chain the controller consumed: ``min(buffer_idx, buffer_capacity)``.
    support_pooled_rows
        ``support_per_chain * n_chains``, precomputed.  Read it as retained
        buffer rows; despite the name it is emphatically not an effective sample
        size and not a count of independent observations — multiplying a
        per-chain draw count by the chain count does not make observations
        independent.
    buffer_capacity
        Buffer length ``B``.
    buffer_capacity_reached
        ``buffer_idx >= B``: the buffer filled.  On its own this is not a loss.
    dropped_draws
        ``max(buffer_idx - B, 0)``: draws actually overwritten because the
        window was longer than the buffer.  Zero when the buffer merely filled
        exactly.  These two are separate because "full" and "lossy" are
        different facts.

        **Per chain**, like ``buffer_capacity`` and unlike ``support_pooled_rows``
        — a 7-step overflow at ``n_chains=8`` reports 7 here while 56 buffer rows
        were actually overwritten.  Multiply by ``n_chains`` for the pooled count.

        It is also the only surviving signal that the retained rows are
        *rotated* rather than chronological.  That is irrelevant to the
        covariance-style estimators, which are order-blind, but not to the
        transient-mixing and lag-1 autocorrelation signals, which read time
        order — a consumer diagnosing those can use this to spot the condition
        after the fact.
    gate_predicate_true, escalation_gate_applicable
        See the module docstring.  Not nested; decode with :func:`decode_gates`.
    escalated_now, has_escalated_before, has_escalated
        Escalation state around this window.

        ``escalation_gate_applicable`` is computed against
        ``has_escalated_before`` — the state on entry — not against
        ``has_escalated``.  Validating the mask contract with the post-state
        instead shows a spurious contradiction in exactly the escalation
        window, where applicability bits are set and ``has_escalated`` is
        already True.
    escalation_rank_stored
        The controller's carried nominal rank.  Distinct from the detection
        rank and from ``deployed_effective_rank``; on the multi-chain path it
        can even describe a different branch (see :class:`MultiChainDetail`).
    deployed_effective_rank, deployed_logdet, deployed_sigma_gm
        The metric published for the next window, under the declared
        ``|lam - 1| > 1e-6`` convention.

        Each is recomputable from a low-rank payload in principle, but the
        per-window *series* is not: ``run()`` returns only the final
        ``step_size`` and ``inverse_mass_matrix``, and under
        :func:`publication_adapt_info_fn` nothing else is stacked.  Recovering
        these three scalars per window otherwise means stacking ``O(d*k)``
        factors at every step.  They are the compact form of a series that does
        not otherwise exist, not a convenience copy of something you already
        have.
    in_force_logdet
        The metric that drove the window just completed.

        **A log-determinant is a volume summary, and differencing two of them
        does not answer "did the metric move".**  Equal log-determinants never
        imply equal metrics, even in exact arithmetic: a metric can change
        orientation or shape at unchanged volume, and this field would not
        move.  Read ``deployed_logdet - in_force_logdet`` as a change in volume
        and nothing more.  The stronger question needs the full factors, or the
        metric's action on vectors, under a declared comparison convention.

        Separately, the two are computed at different call sites from what is,
        in principle, the same matrix — the host carry was measured
        bit-identical across the window — and they can still disagree in the
        last few ulp.  One measured instance differed by ~8 ulp at float32 in
        one of six window pairs.  The mechanism was not isolated; differing
        floating-point summation or fusion order is the hypothesis, not an
        established cause, and that single measurement is not a universal noise
        floor.  So do not compare these for exact equality, and do not treat
        any particular tolerance as calibrated.
    r2_raw, r2_mode
        Score-linearity R² as measured, and which fit mode produced it.  On the
        multi-chain path this is the raw value the W branch uses; the routed
        value the T branch uses is in :class:`MultiChainDetail`.
    is_slow_mixing
        Transient-mixing class reported by the controller.  Meaningful on the
        single-chain path only: the multi-chain controller does not compute a
        transient-mixing signal and carries a constant ``False``, which this
        field faithfully reports.  Do not read a ``False`` there as "mixing is
        fast".
    epsilon_in_force
        Step size that drove the window's **last completed transition** — the
        value handed to the kernel, captured before that step's dual-averaging
        update.
    epsilon_after_window_da
        ``exp(log_step_size)`` after that final update, immediately before the
        boundary reset.
    epsilon_window_average
        ``da_final(ss_state)``, which seeds the next window's dual averaging.
    epsilon_next_window
        ``exp(log_step_size)`` after the reset: what the next window opens with.

        Four distinct quantities, never collapsed.  ``epsilon_window_average``
        and ``epsilon_next_window`` are related by ``exp(log(.))``, which is not
        guaranteed bitwise-identical, so no equality between them is asserted.
    deployed_full
        Full factors of the deployed metric, or ``None`` unless
        ``full_matrices=True``.
    single_chain, multi_chain
        Exactly one is populated.
    """

    window_index: Array
    warmup_step_index: Array
    core_updates_per_chain: Array
    core_chain_updates_total: Array
    n_chains: Array
    dim: Array
    support_per_chain: Array
    support_pooled_rows: Array
    buffer_capacity: Array
    buffer_capacity_reached: Array
    dropped_draws: Array
    gate_predicate_true: Array
    escalation_gate_applicable: Array
    escalated_now: Array
    has_escalated_before: Array
    has_escalated: Array
    escalation_rank_stored: Array
    deployed_effective_rank: Array
    deployed_logdet: Array
    deployed_sigma_gm: Array
    in_force_logdet: Array
    r2_raw: Array
    r2_mode: Array
    is_slow_mixing: Array
    epsilon_in_force: Array
    epsilon_after_window_da: Array
    epsilon_window_average: Array
    epsilon_next_window: Array
    deployed_full: Any = None
    single_chain: Any = None
    multi_chain: Any = None


def encode_gates(**bits: Any) -> Array:
    """Pack named predicates into an int32 mask using :data:`GATE_BITS`.

    Unnamed predicates are left clear.  Raises on an unknown name so a typo
    cannot silently produce an all-clear mask.
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


def decode_gates(mask, *, names: tuple[str, ...] | None = None) -> dict[str, bool]:
    """Unpack an int32 mask into ``{gate_name: bool}``.

    Pass ``names`` to restrict the result to one controller's predicates (the
    other controller's bits are always clear, and reporting them would suggest
    a decision that was never taken).  Works on a concrete mask, not in a trace.
    """
    import numpy as np

    m = int(np.asarray(mask))
    selected = GATE_BITS if names is None else {n: GATE_BITS[n] for n in names}
    return {name: bool((m >> bit) & 1) for name, bit in selected.items()}


def publication_adapt_info_fn():
    """Build an ``adaptation_info_fn`` returning **only** the publication record.

    The generic :func:`~blackjax.adaptation.base.get_filter_adapt_info_fn`
    filters top-level fields only, so it keeps or drops ``imm_state`` whole —
    including the ``(buffer_size, d)`` draw and gradient buffers, which
    ``lax.scan`` would then stack once per step.  ``low_rank_adaptation``'s own
    default info fn documents a real 41 GB out-of-memory caused by exactly that.
    This helper stacks the record and nothing else.
    """

    def _publication_only(state, info, adaptation_state):
        del state, info
        return adaptation_state.imm_state.publication

    return _publication_only


def _flatten_record(record, step: int) -> dict:
    """One publication's leaves as plain Python scalars, with nested prefixes."""
    import numpy as np

    out: dict[str, Any] = {}

    def walk(node, prefix: str):
        for name in node._fields:
            value = getattr(node, name)
            if value is None:
                continue
            key = f"{prefix}{name}"
            if isinstance(value, tuple) and hasattr(value, "_fields"):
                walk(value, f"{key}.")
                continue
            arr = np.asarray(value)[step]
            if arr.ndim > 0:
                continue  # full matrices: read them off the stacked record
            if arr.dtype == np.bool_:
                out[key] = bool(arr)
            elif np.issubdtype(arr.dtype, np.integer):
                out[key] = int(arr)
            else:
                out[key] = float(arr)

    walk(record, "")
    return out


def extract_publication_chronology(records) -> list[dict]:
    """Reduce a per-step stacked record to one entry per publication.

    Drops the ``window_index == -1`` prefix and keeps the first occurrence of
    each ``window_index``, since the carry holds a record unchanged between
    boundaries.

    Parameters
    ----------
    records
        A :class:`MetricPublicationRecord` whose leaves carry a leading
        ``num_steps`` axis — the second return of ``warmup.run()`` when
        :func:`publication_adapt_info_fn` was used.

    Returns
    -------
    list[dict]
        One dict per publication, in window order.  Nested detail is flattened
        with a dotted prefix (``single_chain.detection_rank``).  Each entry
        carries ``schema_version`` and decoded ``gates_true`` /
        ``gates_applicable`` restricted to the controller that produced it.
        Full matrices are omitted; read them off the stacked record.
    """
    import numpy as np

    window_index = np.asarray(records.window_index)
    if window_index.ndim != 1:
        raise ValueError(
            "extract_publication_chronology: expected records stacked over a "
            f"leading step axis, got window_index with shape {window_index.shape}. "
            "Pass the adaptation info from a run that used publication_adapt_info_fn()."
        )

    branch_prefixes = ("w_", "t_") if records.multi_chain is not None else ("sc_",)
    names = tuple(
        n for n in GATE_BITS if n == "deadline" or n.startswith(branch_prefixes)
    )

    out: list[dict] = []
    seen: set[int] = set()
    for step in range(window_index.shape[0]):
        w = int(window_index[step])
        if w < 0 or w in seen:
            continue
        seen.add(w)
        entry = _flatten_record(records, step)
        entry["schema_version"] = SCHEMA_VERSION
        entry["gates_true"] = decode_gates(entry["gate_predicate_true"], names=names)
        entry["gates_applicable"] = decode_gates(
            entry["escalation_gate_applicable"], names=names
        )
        out.append(entry)

    out.sort(key=lambda e: e["window_index"])
    return out


def record_nbytes(record) -> int:
    """Actual byte size of a record, summed over its real leaves.

    Reports what the arrays occupy rather than a nominal ``n_fields * 4``
    figure: dtypes differ (``float64`` under x64) and the optional full-matrix
    factors dominate when present.
    """
    import jax

    return sum(int(jnp.asarray(leaf).nbytes) for leaf in jax.tree.leaves(record))


# ---------------------------------------------------------------------------
# Record construction.  Kept here rather than in builders.py so the controller
# module stays about the controller; the shapes and dtypes these produce are
# the schema.
# ---------------------------------------------------------------------------


def _logdet(imm: LowRankInverseMassMatrix) -> Array:
    """``log det M^-1`` for ``M^-1 = diag(s)(I + U(L-I)U')diag(s)``.

    Returns ``2*sum(log sigma) + sum(log lam)``; no dense ``d x d`` matrix is
    formed.

    **Precondition: the *active* columns of ``U`` — those whose ``lam`` is not
    1 — must be orthonormal.**  Columns carrying ``lam == 1`` are annihilated by
    ``(L - I)`` and their orientation is irrelevant, so they need not be
    orthogonal to anything.  Under that condition the non-unit eigenvalues of
    the middle factor are exactly the non-unit ``lam`` and the determinant
    factorises.

    This is weaker than requiring ``U`` orthonormal, and deliberately so: not
    every metric this controller publishes has orthonormal ``U``.  A metric with
    two non-unit ``lam`` on columns that are not mutually orthogonal violates
    the precondition, and this function would be silently wrong for it.
    """
    return 2.0 * jnp.sum(jnp.log(imm.sigma)) + jnp.sum(jnp.log(imm.lam))


def _sigma_gm(imm: LowRankInverseMassMatrix) -> Array:
    return jnp.exp(jnp.mean(jnp.log(imm.sigma)))


def _effective_rank(imm: LowRankInverseMassMatrix) -> Array:
    """Eigenvalues the kernel actually sees as non-trivial.

    Distinct from any detection rank: the Fisher estimator can return
    sub-threshold directions that are numerically ``lam = 1`` and so contribute
    no structure to the deployed metric.
    """
    return jnp.sum(
        jnp.abs(imm.lam - 1.0) > jnp.asarray(_LAM_NONTRIVIAL_TOL, imm.lam.dtype)
    ).astype(jnp.int32)


def candidate_summary(
    candidate: LowRankInverseMassMatrix,
    deployed: LowRankInverseMassMatrix,
    *,
    full_matrices: bool,
) -> CandidateSummary:
    """Summarise one candidate metric against what was actually deployed."""
    ratio = jnp.log(candidate.sigma) - jnp.log(deployed.sigma)
    return CandidateSummary(
        effective_rank=_effective_rank(candidate),
        logdet=_logdet(candidate),
        lam_max=jnp.max(candidate.lam),
        lam_min=jnp.min(candidate.lam),
        sigma_gm=_sigma_gm(candidate),
        sigma_log_ratio_rms_vs_deployed=jnp.sqrt(jnp.mean(ratio**2)),
        full=candidate if full_matrices else None,
    )


def empty_candidate_summary(
    n_dims: int, rank: int, dtype, *, full_matrices: bool
) -> CandidateSummary:
    """Pre-first-publication candidate: dtypes matching what ``final()`` emits."""
    nan = jnp.array(float("nan"), dtype=dtype)
    return CandidateSummary(
        effective_rank=jnp.zeros((), dtype=jnp.int32),
        logdet=nan,
        lam_max=nan,
        lam_min=nan,
        sigma_gm=nan,
        sigma_log_ratio_rms_vs_deployed=nan,
        full=LowRankInverseMassMatrix(
            sigma=jnp.zeros(n_dims, dtype=dtype),
            U=jnp.zeros((n_dims, rank), dtype=dtype),
            lam=jnp.zeros(rank, dtype=dtype),
        )
        if full_matrices
        else None,
    )


def empty_record(
    n_dims: int,
    rank: int,
    dtype,
    *,
    n_chains: int,
    multi_chain: bool,
    full_matrices: bool,
) -> MetricPublicationRecord:
    """Record for a state that has not published yet.

    ``window_index`` is ``-1``.  Every dtype mirrors exactly what ``final()``
    will produce, so the two branches of the host's window-boundary
    ``lax.cond`` agree.  The host-stamped fields are placeholders here; the
    host overwrites them unconditionally at init and at every boundary, so
    their dtype never escapes.
    """
    nan = jnp.array(float("nan"), dtype=dtype)
    nan32 = jnp.array(float("nan"), dtype=jnp.float32)
    zero_i = jnp.zeros((), dtype=jnp.int32)
    minus_one = jnp.array(-1, dtype=jnp.int32)
    false_ = jnp.zeros((), dtype=jnp.bool_)
    blank = empty_candidate_summary(n_dims, rank, dtype, full_matrices=full_matrices)

    detail_sc = (
        None
        if multi_chain
        else SingleChainDetail(
            detection_rank=zero_i,
            s_gap=nan32,
            s_gap_prev=nan32,
            s_gap_relative_change=nan32,
            candidate=blank,
        )
    )
    detail_mc = (
        MultiChainDetail(
            branch_fired_this_window=jnp.array(BRANCH_NONE, dtype=jnp.int32),
            detection_branch_history=jnp.array(BRANCH_NONE, dtype=jnp.int32),
            deployed_metric_route=jnp.array(ROUTE_DIAGONAL, dtype=jnp.int32),
            branch_first_set_at_window=minus_one,
            t_detection_rank=zero_i,
            r2_routed=nan32,
            is_converging=false_,
            is_unimodal=false_,
            any_mode_flag=false_,
            t_unimodality_resolved=false_,
            t_contraction_stat=nan32,
            unimodality_gap_ratio=nan32,
            unimodality_flag_count=zero_i,
            deferred_to_ensemble=false_,
            chain_collinearity_f1=nan32,
            within_lam1=nan32,
            chain_consistency_psi=nan32,
            r1_top=nan32,
            candidate_w=blank,
            candidate_t=blank,
        )
        if multi_chain
        else None
    )

    return MetricPublicationRecord(
        window_index=minus_one,
        warmup_step_index=minus_one,
        core_updates_per_chain=zero_i,
        core_chain_updates_total=zero_i,
        n_chains=jnp.array(n_chains, dtype=jnp.int32),
        dim=jnp.array(n_dims, dtype=jnp.int32),
        support_per_chain=zero_i,
        support_pooled_rows=zero_i,
        buffer_capacity=zero_i,
        buffer_capacity_reached=false_,
        dropped_draws=zero_i,
        gate_predicate_true=zero_i,
        escalation_gate_applicable=zero_i,
        escalated_now=false_,
        has_escalated_before=false_,
        has_escalated=false_,
        escalation_rank_stored=zero_i,
        deployed_effective_rank=zero_i,
        deployed_logdet=nan,
        deployed_sigma_gm=nan,
        in_force_logdet=nan,
        r2_raw=nan32,
        r2_mode=zero_i,
        is_slow_mixing=false_,
        epsilon_in_force=nan,
        epsilon_after_window_da=nan,
        epsilon_window_average=nan,
        epsilon_next_window=nan,
        deployed_full=LowRankInverseMassMatrix(
            sigma=jnp.zeros(n_dims, dtype=dtype),
            U=jnp.zeros((n_dims, rank), dtype=dtype),
            lam=jnp.zeros(rank, dtype=dtype),
        )
        if full_matrices
        else None,
        single_chain=detail_sc,
        multi_chain=detail_mc,
    )
