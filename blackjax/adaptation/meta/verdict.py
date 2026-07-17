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
"""Post-run verdict extractors for the meta-adaptation controller.

Functions
---------
:func:`extract_meta_verdict` — single-chain verdict from
    :class:`~blackjax.adaptation.meta._state.MetaAdaptationCoreState`.
:func:`extract_multi_chain_verdict` — multi-chain verdict from
    :class:`~blackjax.adaptation.meta._state.MultiChainMetaAdaptationCoreState`.
"""
from __future__ import annotations

from typing import Any

from blackjax.adaptation.meta._calibration import (
    _AIRM_VELOCITY_TOL,
    _DETECTION_BRANCH_BETWEEN_MEANS,
    _DETECTION_BRANCH_BOTH,
    _DETECTION_BRANCH_NONE,
    _DETECTION_BRANCH_POOLED_WITHIN,
    _LAM_NONTRIVIAL_TOL,
    _MC_COLLINEARITY_TOL,
    _R2_DEFERRED,
    _R2_FULL_AFFINE,
    _R2_PROJECTED,
    _R_MIN,
    _S_MIN,
)
from blackjax.adaptation.meta._state import (
    MetaAdaptationCoreState,
    MetaAdaptationVerdict,
    MultiChainMetaAdaptationCoreState,
)


def extract_meta_verdict(
    final_state: MetaAdaptationCoreState,
    max_grad_budget: int,
    num_warmup_steps: int,
    adaptation_info: Any = None,
) -> MetaAdaptationVerdict:
    """Build a :class:`~blackjax.adaptation.meta._state.MetaAdaptationVerdict` from the final core state.

    Call after ``warmup.run()`` completes.  Pass ``adaptation_info`` (the
    second return of ``warmup.run()``) for true gradient counts.

    ``budget_returned_steps`` is ADVISORY: the scan runs its full length in v1;
    this field shows where a stopping host would have saved steps.
    """
    import numpy as np

    has_esc = bool(np.asarray(final_state.has_escalated))
    nominal_rank = int(np.asarray(final_state.escalation_rank))
    k = nominal_rank  # kept for clarity; not used in escalation decisions
    budget_used = int(np.asarray(final_state.budget_used))
    s_gap = float(np.asarray(final_state.s_gap_curr))
    r2 = float(np.asarray(final_state.r2_latest))
    mode_int = int(np.asarray(final_state.r2_mode))
    airm_v_prev = float(np.asarray(final_state.airm_vel_prev))
    airm_v_curr = float(np.asarray(final_state.airm_vel_curr))
    converged_at = int(np.asarray(final_state.converged_at_step))
    is_slow = bool(np.asarray(final_state.is_slow_mixing))

    # Deployed rank: count of non-trivial eigenvalue corrections in the
    # deployed metric (|lam_i - 1| > threshold).  This is the rank of the
    # structure the kernel actually uses, as opposed to the pre-mask count
    # stored in escalation_rank (nominal_rank), which can over-count when
    # the Fisher estimator zeroes sub-threshold directions back to lam=1.
    lam_np = np.asarray(final_state.inverse_mass_matrix.lam)
    effective_rank = int(np.sum(np.abs(lam_np - 1.0) > _LAM_NONTRIVIAL_TOL))

    r2_nan = np.isnan(r2)

    # Route
    r2_blocked = (not r2_nan) and (r2 < _R_MIN)
    if not has_esc and r2_blocked:
        route = "reparam_suggested"
    elif has_esc:
        route = "low_rank"
    else:
        route = "diagonal"

    # Confidence
    s_gap_valid = not np.isnan(s_gap)
    confidence = (
        "high"
        if (has_esc and not r2_nan and r2 >= _R_MIN and s_gap_valid and s_gap >= _S_MIN)
        else "low"
    )

    # Exit reason and advisory budget return
    airm_converged = (airm_v_prev < _AIRM_VELOCITY_TOL) and (
        airm_v_curr < _AIRM_VELOCITY_TOL
    )
    if airm_converged and has_esc:
        exit_reason = "airm_velocity_converged"
    elif budget_used >= num_warmup_steps:
        exit_reason = "warmup_budget_exhausted"
    else:
        exit_reason = "warmup_complete"

    # Advisory budget return: steps from first AIRM convergence to end of warmup.
    # converged_at_step = -1 means never converged (scan ran full length).
    budget_returned = (
        max(num_warmup_steps - converged_at, 0) if converged_at >= 0 else 0
    )

    # True gradient count from info stream
    budget_used_grads = -1
    if adaptation_info is not None:
        try:
            budget_used_grads = int(
                np.asarray(adaptation_info.num_integration_steps).sum()
            )
        except AttributeError:
            pass

    # Flags — mode_int is observed from the carry, not inferred.
    high_d_r2_mode = {
        _R2_DEFERRED: "deferred",
        _R2_PROJECTED: "projected",
        _R2_FULL_AFFINE: "full_affine",
    }.get(mode_int, "deferred")
    marginal_s_gap = (not has_esc) and s_gap_valid and (_S_MIN <= s_gap < 2.0 * _S_MIN)
    flags = {
        "reparam_hint": route == "reparam_suggested",
        "marginal_s_gap": marginal_s_gap,
        "wall_cost_discount": k > 0,
        "high_d_r2_mode": high_d_r2_mode,
        "mode_coverage": "single_chain_uncertified",
        # nominal_rank is the pre-filter count from the spectrum rank selector;
        # effective_rank (the top-level field) is the deployed count from
        # the Fisher metric's lam array.  Both are provided for diagnostics.
        "nominal_rank": nominal_rank,
    }

    return MetaAdaptationVerdict(
        route=route,
        metric=final_state.inverse_mass_matrix,
        effective_rank=effective_rank,
        confidence=confidence,
        exit_reason=exit_reason,
        budget_used_steps=budget_used,
        budget_returned_steps=budget_returned,
        budget_used_grads=budget_used_grads,
        r2_final=r2,
        s_gap_final=s_gap,
        transient_mixing_class="slow" if is_slow else "fast",
        buffer_policy="reset",
        flags=flags,
    )


def extract_multi_chain_verdict(
    final_state: MultiChainMetaAdaptationCoreState,
    max_grad_budget: int,
    num_warmup_steps: int,
    adaptation_info: Any = None,
    *,
    pooled_draws_by_window: Any = None,
) -> MetaAdaptationVerdict:
    """Build a :class:`~blackjax.adaptation.meta._state.MetaAdaptationVerdict` from a multi-chain final core state.

    Drop-in counterpart of :func:`extract_meta_verdict` for the
    :class:`~blackjax.adaptation.meta._state.MultiChainMetaAdaptationCoreState` produced by
    :func:`~blackjax.adaptation.meta.builders.build_multi_chain_meta_core`.

    Parameters
    ----------
    final_state
        Final :class:`~blackjax.adaptation.meta._state.MultiChainMetaAdaptationCoreState` after ``warmup.run()``.
    max_grad_budget, num_warmup_steps, adaptation_info
        Same semantics as :func:`extract_meta_verdict`.
    pooled_draws_by_window
        Optional per-window pooled draw array exposed for nested-R̂ diagnostics
        (shape ``(n_chains, n_per_window, d)`` per window).  Not used internally
        — passed through as ``flags["pooled_draws_by_window"]`` for the
        evaluation layer.

    Returns
    -------
    MetaAdaptationVerdict
        Verdict with multi-chain–specific flags: ``n_chains``,
        ``chain_collinearity``, ``unimodality_gate``, ``deferred_to_ensemble``,
        and ``mode_coverage="multi_chain_certified"`` when all gates passed.
    """
    import numpy as np

    has_esc = bool(np.asarray(final_state.has_escalated))
    nominal_rank = int(np.asarray(final_state.escalation_rank))
    k = nominal_rank
    budget_used = int(np.asarray(final_state.budget_used))
    # s_gap fields are NaN in multi-chain path (not the primary signal)
    s_gap = float(np.asarray(final_state.s_gap_curr))
    r2 = float(np.asarray(final_state.r2_latest))
    mode_int = int(np.asarray(final_state.r2_mode))
    airm_v_prev = float(np.asarray(final_state.airm_vel_prev))
    airm_v_curr = float(np.asarray(final_state.airm_vel_curr))
    converged_at = int(np.asarray(final_state.converged_at_step))
    is_slow = bool(np.asarray(final_state.is_slow_mixing))
    chain_collinearity_raw = float(np.asarray(final_state.chain_collinearity))
    unimodality_passed_raw = bool(np.asarray(final_state.unimodality_passed))
    deferred_raw = bool(np.asarray(final_state.deferred_to_ensemble))
    n_chains_actual: int = final_state.draws_buffer.shape[0]  # static

    lam_np = np.asarray(final_state.inverse_mass_matrix.lam)
    effective_rank = int(np.sum(np.abs(lam_np - 1.0) > _LAM_NONTRIVIAL_TOL))

    r2_nan = np.isnan(r2)

    # Route
    r2_blocked = (not r2_nan) and (r2 < _R_MIN)
    if not has_esc and r2_blocked:
        route = "reparam_suggested"
    elif has_esc:
        route = "low_rank"
    else:
        route = "diagonal"

    # Confidence — collinearity gate replaces s_gap stability for multi-chain path
    collinearity_valid = not np.isnan(chain_collinearity_raw)
    collinearity_passed = (
        collinearity_valid and chain_collinearity_raw >= _MC_COLLINEARITY_TOL
    )
    confidence = (
        "high"
        if (has_esc and not r2_nan and r2 >= _R_MIN and collinearity_passed)
        else "low"
    )

    # Exit reason and advisory budget return
    airm_converged = (airm_v_prev < _AIRM_VELOCITY_TOL) and (
        airm_v_curr < _AIRM_VELOCITY_TOL
    )
    if airm_converged and has_esc:
        exit_reason = "airm_velocity_converged"
    elif budget_used >= num_warmup_steps:
        exit_reason = "warmup_budget_exhausted"
    else:
        exit_reason = "warmup_complete"

    budget_returned = (
        max(num_warmup_steps - converged_at, 0) if converged_at >= 0 else 0
    )

    budget_used_grads = -1
    if adaptation_info is not None:
        try:
            budget_used_grads = int(
                np.asarray(adaptation_info.num_integration_steps).sum()
            )
        except AttributeError:
            pass

    high_d_r2_mode = {
        _R2_DEFERRED: "deferred",
        _R2_PROJECTED: "projected",
        _R2_FULL_AFFINE: "full_affine",
    }.get(mode_int, "deferred")

    # "need_more_chains": non-escalation with collinearity below threshold
    # (magnitude may have been present but gate didn't pass; more chains or
    # better-dispersed starts would help).
    spike_marginal = (not has_esc) and (nominal_rank > 0) and (not collinearity_passed)

    # mode_coverage: multi_chain_certified iff collinearity gate passed
    mode_coverage = (
        "multi_chain_certified"
        if (has_esc and collinearity_passed)
        else "multi_chain_uncertified"
        if n_chains_actual > 1
        else "single_chain_uncertified"
    )

    # start_dispersion_adequacy: non-escalation is one-sided-safe (conservative);
    # under-dispersed starts can miss slow directions but never over-escalate.
    start_dispersion_adequacy = (
        "adequate_if_overdispersed" if not has_esc else "not_applicable"
    )

    # v2.1 new diagnostic fields
    within_lam1_raw = float(np.asarray(final_state.within_lam1))
    chain_consistency_psi_raw = float(np.asarray(final_state.chain_consistency_psi))
    r1_top_raw = float(np.asarray(final_state.r1_top))
    detection_branch_raw = int(np.asarray(final_state.detection_branch))
    _BRANCH_NAMES = {
        _DETECTION_BRANCH_NONE: "none",
        _DETECTION_BRANCH_POOLED_WITHIN: "pooled_within",
        _DETECTION_BRANCH_BETWEEN_MEANS: "between_means",
        _DETECTION_BRANCH_BOTH: "both",
    }
    detection_branch_name = _BRANCH_NAMES.get(detection_branch_raw, "unknown")

    flags = {
        "reparam_hint": route == "reparam_suggested",
        "marginal_s_gap": False,  # s_gap not primary signal in multi-chain path
        "wall_cost_discount": k > 0,
        "high_d_r2_mode": high_d_r2_mode,
        "mode_coverage": mode_coverage,
        "nominal_rank": nominal_rank,
        # Multi-chain specific
        "n_chains": n_chains_actual,
        "chain_collinearity": chain_collinearity_raw,
        "need_more_chains": spike_marginal,
        "start_dispersion_adequacy": start_dispersion_adequacy,
        "unimodality_gate": "pass" if unimodality_passed_raw else "flag",
        "deferred_to_ensemble": deferred_raw,
        "pooled_draws_by_window": pooled_draws_by_window,
        # v2.1 branch diagnostics
        "within_lam1": within_lam1_raw,
        "chain_consistency_psi": chain_consistency_psi_raw,
        "r1_top": r1_top_raw,
        "detection_branch": detection_branch_name,
    }

    return MetaAdaptationVerdict(
        route=route,
        metric=final_state.inverse_mass_matrix,
        effective_rank=effective_rank,
        confidence=confidence,
        exit_reason=exit_reason,
        budget_used_steps=budget_used,
        budget_returned_steps=budget_returned,
        budget_used_grads=budget_used_grads,
        r2_final=r2,
        s_gap_final=s_gap,
        transient_mixing_class="slow" if is_slow else "fast",
        buffer_policy="reset",
        flags=flags,
    )
