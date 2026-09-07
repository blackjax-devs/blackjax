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

The meta-adaptation controller's ``final()`` computes candidate metrics, the
escalation predicates and the number of draws it consumed, then resets its
buffers and returns only the new state.  Everything except the deployed metric
is discarded, so a caller watching the adaptation info stream sees the
*outcome* of a publication decision but none of its *inputs*.

This module adds a read-only record of those inputs.  It changes no threshold,
no predicate, no ordering and no published metric: with telemetry disabled
nothing here is constructed or traced.

Two masks, and what they do *not* mean
--------------------------------------
The controller's escalation decision is a JAX expression such as
``~has_escalated & r2_gate & s_gap_gate & deadline_ok``.  ``&`` is eager:
**every** predicate is computed on **every** window, including after escalation
and including when a sibling conjunct is false.  There is no short-circuit, and
a failed conjunct does not prevent its siblings from being evaluated.  Some of
those same predicates additionally govern the multi-chain deferral latch, which
is *not* gated on ``has_escalated`` and can therefore fire post-escalation.

Reporting "was this computed?" would thus be uninformative — the answer is
always yes.  The two masks are defined instead as:

``gate_predicate_true``
    Raw truth of each named predicate this window, exactly as the controller
    computed it.  Always populated, including after escalation, so that the
    deferral latch and the branch decisions remain reconstructable.

``escalation_gate_applicable``
    Whether the predicate bears on a **new** escalation decision this window.
    Clear once the controller has escalated (the ``~has_escalated`` conjunct
    makes a new escalation impossible regardless of the others), and clear for
    a predicate whose inputs are not yet available — the S_gap stability test
    needs the previous window's S_gap, which the first window does not have.

**The two are not nested.** A raw predicate can be true while its applicability
is zero; that is the normal post-escalation state and is not a contradiction.
Never read a clear ``gate_predicate_true`` bit as "failed" without checking
whether it was applicable, and never read a clear applicability bit as "the
controller did not compute this".

Stability
---------
:data:`SCHEMA_VERSION` is **2**.  Consumers must read it and refuse a version
they do not know rather than inferring meaning from raw mask bits.  Bit
positions are never reused across versions.

Units
-----
Counts here are **not** warmup scan steps.  ``core_update_steps_per_chain``
counts calls to the metric core's ``update()``, which the host makes only on
slow-window steps: under Stan's :func:`~blackjax.adaptation.staged_adaptation.build_schedule`
it lags the scan index by the whole initial fast buffer (75 by default).
``warmup_step_index`` is the scan index and is stamped by the host.
``support_pooled_rows`` is a count of retained buffer rows; it is neither an
effective sample size nor a count of independent observations.
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

SCHEMA_VERSION: int = 2

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

    Emitted for every candidate the controller builds, including the ones it
    does not deploy — before escalation those are computed and discarded, which
    is the single most important thing this module preserves.

    After escalation the picture is different and the record reflects it: the
    historical route deploys a *freshly computed* candidate each window, so a
    candidate summary post-escalation generally describes the metric that was
    just published rather than one that was thrown away.

    Attributes
    ----------
    effective_rank
        Count of ``|lam - 1| > 1e-6``, the same convention
        :func:`~blackjax.adaptation.meta.verdict.extract_meta_verdict` uses.
    logdet
        ``2*sum(log sigma) + sum(log lam)``, exact for this parameterisation
        (``U`` has orthonormal columns, so the non-unit eigenvalues are ``lam``).
    lam_max, lam_min, sigma_gm
        Eigenvalue extremes and the geometric mean of the diagonal scaling.
    sigma_log_ratio_rms_vs_deployed
        ``rms(log(this sigma) - log(deployed sigma))``; zero when this candidate
        *is* what was deployed.
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
    """Fields specific to :func:`~blackjax.adaptation.meta.builders.build_meta_adaptation_core`.

    Attributes
    ----------
    detection_rank
        ``k_new``, chosen from the whitened-residual spectrum.  A detection
        quantity: distinct from the candidate's and the deployed metric's
        effective ranks, and distinct again from the stored nominal
        ``escalation_rank_stored`` on the parent record.
    s_gap, s_gap_prev, s_gap_relative_change
        The stability test's inputs.  ``s_gap_prev`` is NaN in the first window,
        which is why ``sc_s_gap_stability`` is inapplicable there.
    candidate
        The single low-rank candidate this controller builds.
    """

    detection_rank: Array
    s_gap: Array
    s_gap_prev: Array
    s_gap_relative_change: Array
    candidate: CandidateSummary


class MultiChainDetail(NamedTuple):
    """Fields specific to :func:`~blackjax.adaptation.meta.builders.build_multi_chain_meta_core`.

    Three branch fields, because three different questions have three different
    answers and the controller keeps them apart:

    ``branch_fired_this_window``
        What escalated *now*: ``NONE`` / W / T / ``BOTH``.
    ``detection_branch_history``
        The controller's carried ``detection_branch``, which holds the last
        *firing* window's branch and is unchanged in windows where nothing
        fires.  This carried value — not the current one — selects which
        escalated metric is deployed.
    ``deployed_metric_route``
        What the kernel will actually use: ``ROUTE_DIAGONAL`` before escalation,
        else ``ROUTE_W``/``ROUTE_T``.  ``BOTH`` firing deploys the W metric.

        This *is* derivable, from ``has_escalated`` and
        ``detection_branch_history``, by reapplying the controller's own routing
        rule (``BOTH`` and ``W`` both route to W).  It is carried anyway so that
        reading the record never requires reimplementing that rule — the same
        trade the unit fields make.  Do not read it as information held nowhere
        else; see the module docstring's list of what genuinely is.

    Attributes
    ----------
    branch_first_set_at_window
        ``window_index`` at which ``detection_branch`` first left ``NONE``,
        stamped on that transition only and unchanged afterwards; ``-1`` until
        it happens.
    t_detection_rank
        ``k_new``, the **between-chain** detection rank.  Note that the
        controller stores this into ``escalation_rank`` even when the W branch
        is what fired, so ``escalation_rank_stored`` can describe a different
        branch than the one that escalated.  Reported, not corrected.
    r2_routed
        The GAIN-overridden R² the T branch and the verdict use.  Distinct from
        the parent record's raw ``r2_raw``, which is what the W branch uses;
        the two can disagree, and the override exists precisely because the raw
        value is meaningless at ``k << d``.
    is_converging, is_unimodal, any_mode_flag
        The three separate observations behind the three-way unimodality rule.
    t_unimodality_resolved
        Its resolved outcome: ``is_converging | (is_unimodal & ~any_mode_flag)``.
    t_contraction_stat, unimodality_gap_ratio
        The numerics behind those observations.
    unimodality_flag_count
        Consecutive flagged windows; deferral needs this to reach
        ``_MC_UNIMODALITY_CONFIRM_WINDOWS``.
    deferred_to_ensemble
        The deferral latch.  Non-monotone, and *not* gated on ``has_escalated``,
        so it can fire after escalation.
    chain_collinearity_f1, within_lam1, chain_consistency_psi, r1_top
        Detector signals as measured this window.
    candidate_w, candidate_t
        **Both** candidates.  They are built before routing and are not
        interchangeable: W is the full Fisher-LR on per-chain-centred pooled
        buffers, T is a rank-1 geometric-mean slow-direction correction.
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
    core_update_steps_per_chain, core_update_chain_steps_total
        Calls to the metric core's ``update()``, per chain and summed over
        chains.  **Not** warmup steps: the host calls ``update()`` only on slow
        stages, so under Stan's schedule these lag the scan index by the whole
        initial fast buffer.  Equal to each other when ``n_chains == 1``.
    n_chains, dim
        Chain count and dimension.  Carried so a consumer can recompute the
        calibration thresholds (which are pure functions of ``M``, ``n``, ``d``)
        without duplicating them.
    support_per_chain
        Draws per chain the controller consumed: ``min(buffer_idx, buffer_capacity)``.
    support_pooled_rows
        ``support_per_chain * n_chains`` — retained buffer rows fed to the
        estimator.  Not an effective sample size and not independent observations.
    buffer_capacity
        Buffer length ``B``.
    buffer_capacity_reached
        ``buffer_idx >= B``: the buffer filled.  On its own this is not a loss.
    dropped_draws
        ``max(buffer_idx - B, 0)``: draws actually overwritten because the
        window was longer than the buffer.  Zero when the buffer merely filled
        exactly.  These two are separate because "full" and "lossy" are
        different facts.
    gate_predicate_true, escalation_gate_applicable
        See the module docstring.  Not nested; decode with :func:`decode_gates`.
    escalated_now, has_escalated_before, has_escalated
        Escalation state around this window.
    first_escalation_window_index
        ``window_index`` of the genuine ``False -> True`` transition of
        ``has_escalated``; ``-1`` until it happens, unchanged afterwards.
    escalation_rank_stored
        The controller's carried nominal rank.  Distinct from the detection
        rank and from ``deployed_effective_rank``; on the multi-chain path it
        can even describe a different branch (see :class:`MultiChainDetail`).
    deployed_effective_rank, deployed_logdet, deployed_sigma_gm
        The metric published for the next window.  Recomputable from the
        low-rank payload the warmup already returns, under the declared
        ``|lam - 1| > 1e-6`` convention; carried so the convention is stated
        once and so a per-window series exists.
    in_force_logdet
        The metric that drove the window just completed.
    r2_raw, r2_mode
        Score-linearity R² as measured, and which fit mode produced it.  On the
        multi-chain path this is the raw value the W branch uses; the routed
        value the T branch uses is in :class:`MultiChainDetail`.
    is_slow_mixing
        Transient-mixing class reported by the controller.
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
    core_update_steps_per_chain: Array
    core_update_chain_steps_total: Array
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
    first_escalation_window_index: Array
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

    ``U`` has orthonormal columns, so the non-unit eigenvalues of the middle
    factor are exactly ``lam`` and the determinant factorises; no dense ``d x d``
    matrix is formed.
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
        core_update_steps_per_chain=zero_i,
        core_update_chain_steps_total=zero_i,
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
        first_escalation_window_index=minus_one,
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
