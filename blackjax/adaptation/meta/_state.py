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
"""State NamedTuples for the meta-adaptation controller.

Three types:
- :class:`MetaAdaptationCoreState` — single-chain scan-carry state.
- :class:`MultiChainMetaAdaptationCoreState` — multi-chain (M-chain) scan-carry state.
- :class:`MetaAdaptationVerdict` — Python-side verdict extracted after the warmup scan.
"""
from __future__ import annotations

from typing import NamedTuple

from blackjax.mcmc.metrics import LowRankInverseMassMatrix
from blackjax.types import Array, ArrayLikeTree  # noqa: F401  (re-used by callers)


class MetaAdaptationCoreState(NamedTuple):
    """Scan-carry state for the meta-adaptation MetricCore.

    Buffer fields mirror ``LowRankMetricCoreState`` so the state is
    interchangeable in the staged_adaptation engine.  The ``inverse_mass_matrix``
    is always a :class:`~blackjax.mcmc.metrics.LowRankInverseMassMatrix`; before
    escalation, U=0 and lam=1 (bit-equivalent to the diagonal metric).
    """

    # Buffer (mirrors LowRankMetricCoreState layout)
    inverse_mass_matrix: LowRankInverseMassMatrix
    mu_star: Array  # optimal translation, (d,); zero before escalation
    draws_buffer: Array  # (buffer_size, d)
    grads_buffer: Array  # (buffer_size, d)
    buffer_idx: Array  # int32 scalar; reset to 0 after each window (v1 reset policy)
    background_split: Array  # always 0 in v1
    recompute_counter: Array  # always 0 in v1
    # Controller carry
    has_escalated: Array  # bool scalar; monotone True-once
    escalation_rank: Array  # int32; the rank k chosen at escalation (0 before)
    s_gap_prev: Array  # float32; S_gap from window before last (NaN initially)
    s_gap_curr: Array  # float32; S_gap from most recent window (NaN initially)
    r2_latest: Array  # float32; most recent R² (NaN if deferred or not yet computed)
    r2_mode: Array  # int32; 0=deferred, 1=projected, 2=full_affine (last window)
    budget_used: Array  # int32; warmup steps elapsed (step-clock proxy)
    converged_at_step: Array  # int32; step when AIRM velocity first fired (<0 = not yet)
    prev_lam: Array  # (max_rank,); lam from previous window for AIRM velocity
    airm_vel_prev: Array  # float32; AIRM velocity proxy from window before last
    airm_vel_curr: Array  # float32; AIRM velocity proxy from most recent window
    is_slow_mixing: Array  # bool; True = slow-mixing (RESET preferred by v1.1 switch)


class MetaAdaptationVerdict(NamedTuple):
    """Verdict emitted by :func:`~blackjax.adaptation.meta.verdict.extract_meta_verdict`
    after the warmup scan.

    All budget numbers are in warmup steps (step-clock proxy) unless the
    info stream is provided for true gradient counts.

    ``budget_returned_steps`` is ADVISORY in v1: the scan runs its full length;
    a stopping host (lax.while) is the named v1.1 upgrade.
    """

    route: str  # "diagonal" | "low_rank" | "reparam_suggested"
    metric: LowRankInverseMassMatrix
    effective_rank: int  # deployed rank: count of |lam_i − 1| > _LAM_NONTRIVIAL_TOL (0 for diagonal route)
    confidence: str  # "high" | "low"
    exit_reason: str  # "warmup_complete" | "airm_velocity_converged" | "warmup_budget_exhausted"
    budget_used_steps: int
    budget_returned_steps: int  # advisory (see docstring)
    budget_used_grads: int  # -1 if info stream not provided
    r2_final: float
    s_gap_final: float
    transient_mixing_class: str  # "slow" | "fast"
    buffer_policy: str  # always "reset" in v1
    flags: dict  # reparam_hint, marginal_s_gap, wall_cost_discount, high_d_r2_mode, mode_coverage, nominal_rank


class MultiChainMetaAdaptationCoreState(NamedTuple):
    """Scan-carry state for the multi-chain meta-adaptation MetricCore.

    Extends :class:`MetaAdaptationCoreState` with per-chain draw/grad buffers
    of shape ``(n_chains, buf_size, d)`` so that cross-chain projector agreement
    can be computed at each window boundary.  The ``inverse_mass_matrix`` is
    always shared across all chains (one adapted metric for all M chains).

    All controller carry fields (``has_escalated``, ``s_gap_*``, ``r2_*``,
    ``airm_vel_*``, …) are identical in semantics to the single-chain state;
    ``chain_collinearity`` carries the collinearity score f₁ from the most
    recent window boundary (NaN until the first window is complete).

    When ``n_chains=1`` the single-chain path is used instead (see
    :func:`~blackjax.adaptation.meta.builders.build_meta_adaptation_core`);
    this state is never constructed for ``n_chains=1``.
    """

    # Shared metric (all M chains adopt the same inverse mass matrix)
    inverse_mass_matrix: LowRankInverseMassMatrix
    mu_star: Array  # optimal translation, (d,)
    # Per-chain buffers: (n_chains, buf_size, d)
    draws_buffer: Array
    grads_buffer: Array
    buffer_idx: Array  # int32 scalar; steps elapsed in the current window
    background_split: Array  # always 0 (protocol compat)
    recompute_counter: Array  # always 0 (protocol compat)
    # Controller carry — same semantics as MetaAdaptationCoreState
    has_escalated: Array  # bool scalar; monotone True-once
    escalation_rank: Array  # int32; rank k chosen at escalation (0 before)
    s_gap_prev: Array  # float32; retained for diagnostic compatibility (NaN in v2 path)
    s_gap_curr: Array  # float32; retained for diagnostic compatibility (NaN in v2 path)
    r2_latest: Array  # float32; most recent R² from pooled draws
    r2_mode: Array  # int32; _R2_DEFERRED / _R2_PROJECTED / _R2_FULL_AFFINE
    budget_used: Array  # int32; warmup step evaluations elapsed
    converged_at_step: Array  # int32; step of first AIRM convergence (<0 = not yet)
    prev_lam: Array  # (max_rank,); lam from previous window for AIRM velocity
    airm_vel_prev: Array  # float32
    airm_vel_curr: Array  # float32
    is_slow_mixing: Array  # bool; always False in the multi-chain path (pooled diagnostic)
    # Multi-chain-specific carry (v2 fields)
    chain_collinearity: Array  # float32; collinearity score f₁ from most recent window (NaN initially)
    unimodality_passed: Array  # bool; True = gap-stat found unimodal distribution (False = mode-split flag)
    deferred_to_ensemble: Array  # bool; True = other gates passed but unimodality blocked (P1→P3 handoff)
    # v2.1 additions — W-branch diagnostics + T-branch guard state
    within_lam1: Array  # float32; top eigenvalue of pooled within-chain residual (NaN until first window)
    chain_consistency_psi: Array  # float32; Ψ cross-chain consistency cosine (NaN until first window)
    r1_top: Array  # float32; lag-1 autocorr in top W-branch direction (NaN until first window)
    detection_branch: Array  # int32; _DETECTION_BRANCH_* code from the most recent firing window
    unimodality_flag_count: Array  # int32; consecutive windows gap-stat flagged (for 2-window confirmation)
