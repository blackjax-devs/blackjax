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
"""Meta-adaptation controller for the HMC-family warmup.

At each window boundary the controller computes two signals: (1) held-out
score-linearity R² — the curvature gate (funnel R²≈0.007 vs ≥0.54 for all
metric-fixable classes); (2) S_gap(k) = λ₁/λ_{k+1} of the Welford-whitened
residual — the magnitude predictor (Spearman 1.0 with measured rank-k payoff).
Escalate diagonal → rank-k iff R² ≥ _R_MIN AND S_gap ≥ _S_MIN AND stable
over two consecutive windows AND budget deadline clear. Growing-window schedule
(nutpie-style) is the default; AIRM-velocity early exit is advisory in v1
(the scan runs its full length; ``converged_at_step`` records where stopping
would have helped — the actual early-stop host is the named v1.1 upgrade).

.. warning::

   ``metric="auto"`` is **experimental (v1)**.  The low-rank escalation is not
   robustly calibrated at high dimension: when the residual spectrum's dominant
   structure sits near the detection boundary, whether the controller escalates
   can depend on the random seed used for sampling.  Use ``metric="auto"`` for
   exploration and algorithm development, not for production efficiency claims.
   A multi-chain escalation trigger (planned for v2) is expected to make the
   escalation decision robust across seeds.

**Dtype note**: the composed estimator ``_compute_low_rank_metric`` produces
numerically indefinite metrics under float32 (~98% of runs). Enable x64 via
``jax.config.update("jax_enable_x64", True)`` for production use and for the
production use and for numerical-precision-sensitive acceptance runs; all optpath harnesses ran with x64 enabled.

See :mod:`blackjax.adaptation.metric_recipes` for the MetricCore protocol and
:mod:`blackjax.adaptation.staged_adaptation` for the host engine.
"""
from __future__ import annotations

from typing import Any, NamedTuple

import jax
import jax.flatten_util as fu
import jax.numpy as jnp

from blackjax.adaptation.metric_estimators import _compute_low_rank_metric
from blackjax.adaptation.metric_recipes import MetricCore
from blackjax.mcmc.metrics import LowRankInverseMassMatrix
from blackjax.types import Array, ArrayLikeTree

__all__ = [
    "MetaAdaptationCoreState",
    "MetaAdaptationVerdict",
    "MultiChainMetaAdaptationCoreState",
    "build_meta_adaptation_core",
    "build_multi_chain_meta_core",
    "extract_meta_verdict",
    "extract_multi_chain_verdict",
    "_between_chain_detection",
    "_compute_within_chain_stats",
    "_mc_detection_edge",
    "_mc_unimodality_threshold",
]

# ---------------------------------------------------------------------------
# Module constants — empirical calibration anchors (not user knobs).
# ---------------------------------------------------------------------------

_R_MIN: float = 0.5
"""R² gate: curvature targets score ≈0.007–0.09; metric-fixable ≥0.54 (optpath-B)."""

_S_MIN: float = 2.0
"""S_gap magnitude gate: stoch_vol (marginal, S_gap≈1.5) must not escalate."""

_S_GAP_STABILITY_TOL: float = 0.3
"""Max relative S_gap change between consecutive windows before acting.
Warmup-phase S_gap is transient-inflated; two stable reads prevent acting early."""

_MIN_TRAIN_D_RATIO: int = 8
"""Full-affine fit: n_half ≥ 8d required; stoch_vol d=503: R²=-109 at n/d≈2."""

_MIN_TRAIN_K_RATIO: int = 4
"""Projected fit: n_half ≥ 4·(actual_rank+1) required (total n ≳ 8·(k+1)).
d-independent threshold — radon (d=390, k≈2, max_rank=50, n≳408) fits within
the growing-window schedule at 50k-grad budget (max window ≈725 ≥ 408)."""

_AIRM_VELOCITY_TOL: float = 0.05
"""Frobenius-norm lam-change threshold for AIRM early-exit (advisory, v1)."""

_STEP_SIZE_READAPT_BUFFER: int = 50
"""Steps reserved for step-size re-adaptation after escalation."""

_ASSUMED_AVG_LEAPFROGS_PER_STEP: int = 20
"""Conservative grad-per-step divisor: max_grad_budget → num_steps.
Most NUTS runs average 4–15 leapfrogs per step; 20 is conservative."""

_TRANSIENT_MIXING_THRESHOLD: float = 1.0
"""Split-half normalised mean-diff threshold for slow-mixing class detection."""

_MAX_RANK_CAP: int = 50
"""Static rank cap for buffer allocation; per-window rank ≤ min(cap, n//2)."""

_LAM_NONTRIVIAL_TOL: float = 1e-6
"""Threshold for a deployed-metric eigenvalue correction to be considered
non-trivial: ``|lam_i - 1| > _LAM_NONTRIVIAL_TOL``.  Directions with
``lam_i = 1.0`` exactly (set by the Fisher estimator for sub-threshold
directions) contribute zero to the deployed rank.
"""

# Multi-chain constants — escalation trigger v2.
_MULTI_CHAIN_DEFAULT_N_CHAINS: int = 8
"""Default number of independent overdispersed chains for the multi-chain
escalation trigger.  Grounded in the sharp-transition point where the
between-chain look-count makes escalation robust for near-edge cases.
"""

_MC_MIN_CHAINS: int = 6
"""Minimum safe chain count for the multi-chain detection gate.

Below M=6 the collinearity null-margin is unsafe (iid null f₁ can reach 0.79
at M=4, which is above the 0.70 threshold), and the unimodality gap-ratio for
a perfect 2-cluster split (≈ M−1 = 3) falls below the threshold for M<4.
At M≥6 both regimes have sufficient separation.  :func:`build_multi_chain_meta_core`
warns when M is below this value.
"""

_MC_COLLINEARITY_TOL: float = 0.7
"""Minimum collinearity score f₁ to accept a between-chain spike.

f₁ = fraction of total between-chain scatter variance in the top singular
direction.  Genuine slow directions produce near-rank-1 concentration (f₁→1);
within-chain autocorrelation artifacts scatter nearly isotropically across
independent chains (f₁ ≈ 1/(M−1)).  Threshold is between those regimes.
"""

_MC_UNIMODALITY_GAP_FRACTION: float = 0.5
"""Gap-statistic threshold fraction for the unimodality guard.

Threshold = max(0.5 * (M−1), 3.0).  A perfect 2-cluster bimodal split with
M chains gives max_gap/mean_gap ≈ M−1; the genuine-unimodal null gives
≈ 1.0–2.3 empirically.  The 0.5 fraction places the threshold halfway between
those regimes for any M≥6 (the fenced minimum).  The floor of 3.0 adds
headroom against noise for small M.  This is a FLAG, not a proof: noisy at
small M or uneven splits.  Computed dynamically by :func:`_mc_unimodality_threshold`.
"""

# v2.1 W-branch and T-branch guard constants.
_W_BRANCH_PSI_FLOOR: float = 0.15
"""Minimum Ψ (cross-chain consistency) threshold.

All measured null q999 values sit ≤ 0.095 across every null tried (iid, AR
ρ∈{0.5,0.8,0.95}, oscillatory, shared slow-mixing direction).  The threshold
0.15 sits ≥ 2× above the largest observed null q999 and ~6× below genuine
deep-spread values (0.91–0.98).  Used as the floor: actual gate is
max(3·q99_null(M,n,d), 0.15); for d≥50 this equals ~0.19 vs genuine ~0.96.
"""

_W_BRANCH_R1_TOL: float = -0.2
"""Lag-1 autocorrelation lower bound for the oscillation screen.

A genuine under-resolved slow direction is diffusive (r₁ > 0); integrator
resonance alternates (r₁ < 0).  The −0.2 lower bound admits all positive and
mildly-negative lag-1 values while blocking the strongly-negative oscillatory
direction.
"""

_W_BRANCH_NULL_EDGE_TW_FACTOR: float = 1.02
"""Tracy-Widom finite-N inflation factor applied to the iid-null bulk upper edge.

At the regime-relevant pool size N = M(n-1) approx 300 (8 chains, 40-draw window),
the iid-null q99 of the top eigenvalue exceeds the asymptotic analytic edge by
~2%, as measured on 1000 iid null replicates at M=8, n=40, d in {10,20,50,100}.
This constant is isolated in one place so the next calibration pass can replace
the analytic formula with a per-(M,n,d) lookup without touching any other constant.
See :func:`_w_branch_null_edge`.
"""

_MC_UNIMODALITY_Q99_TABLE: dict[int, float] = {
    6: 3.8,  # conservative estimate; dedicated calibration needed for M<8
    7: 4.2,  # conservative estimate
    8: 4.54,  # MC-calibrated from 1000 reps iid null at M=8
}
"""Gap-stat null q99 per M for the unimodality guard (v2.1 recalibration).

v2 used 3.5 = the measured null q90 → 10%/window FPR, causing 5/5 radon
false-positives and 11-window cumulative rate ≈65%.  v2.1 targets 1%/window
(q99 = 4.54 at M=8) and requires 2-consecutive-window confirmation.
"""

_MC_UNIMODALITY_CONFIRM_WINDOWS: int = 2
"""Consecutive windows the gap-stat must flag before deferred_to_ensemble latches.

Early transient windows can produce accidental splits (chains haven't yet
mixed across the target); requiring 2 consecutive flags eliminates most
transient false-defers.  Two is the minimum distinguishing confirmed from
transient; three would be more conservative but too slow for the schedule.
"""

# Detection branch codes (int32 stored in state.detection_branch).
_DETECTION_BRANCH_NONE: int = 0
_DETECTION_BRANCH_POOLED_WITHIN: int = 1  # W-branch fired
_DETECTION_BRANCH_BETWEEN_MEANS: int = 2  # T-branch fired
_DETECTION_BRANCH_BOTH: int = 3  # both branches fired this window

# R² mode integers (stored in carry as int8).
_R2_DEFERRED: int = 0
_R2_PROJECTED: int = 1
_R2_FULL_AFFINE: int = 2


# ---------------------------------------------------------------------------
# State types
# ---------------------------------------------------------------------------


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
    """Verdict emitted by :func:`extract_meta_verdict` after the warmup scan.

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
    :func:`build_meta_adaptation_core`); this state is never constructed for
    ``n_chains=1``.
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


# ---------------------------------------------------------------------------
# Signal computation — pure JAX functions, module-private
# ---------------------------------------------------------------------------


def _compute_whitened_spectrum(
    draws_buffer: Array,
    sigma: Array,
    n: Array,
    max_rank: int,
) -> tuple[Array, Array]:
    """Top ``max_rank`` eigenvalues and eigenvectors of the diagonal-whitened
    sample covariance R = D^{-1/2}ΣD^{-1/2} (D = diag(σ²)), computed via the
    thin SVD of the centred whitened draw matrix.

    Returns ``(eigenvalues, U_k)`` with shapes ``(max_rank,)`` and
    ``(d, max_rank)``; zero-padded when fewer than max_rank components are
    available.  Invalid rows (beyond index n) are zeroed via a mask.
    """
    B, d = draws_buffer.shape
    n_safe = jnp.maximum(n.astype(draws_buffer.dtype), 1.0)
    mask = (jnp.arange(B) < n).astype(draws_buffer.dtype)
    sigma_safe = jnp.maximum(sigma, 1e-20)
    mean_x = (mask[:, None] * draws_buffer).sum(0) / n_safe
    w = mask[:, None] * (draws_buffer - mean_x[None, :]) / sigma_safe[None, :]
    _, s, Vt = jnp.linalg.svd(w, full_matrices=False)
    eigs_all = (s**2) / n_safe
    actual = min(max_rank, min(B, d))  # static
    if actual < max_rank:
        pad = max_rank - actual
        eigenvalues = jnp.concatenate(
            [eigs_all[:actual], jnp.zeros(pad, dtype=eigs_all.dtype)]
        )
        U_k = jnp.concatenate(
            [Vt[:actual].T, jnp.zeros((d, pad), dtype=Vt.dtype)], axis=1
        )
    else:
        eigenvalues = eigs_all[:max_rank]
        U_k = Vt[:max_rank].T
    return eigenvalues, U_k


def _choose_rank(
    eigenvalues: Array,
    n: Array,
    max_rank: int,
    cutoff: float = 2.0,
) -> Array:
    """Count informative eigenvalues (outside [1/cutoff, cutoff]), capped by
    n//2 (estimation-support limit) and max_rank.
    """
    is_informative = (eigenvalues > cutoff) | (eigenvalues < 1.0 / cutoff)
    count = is_informative.sum().astype(jnp.int32)
    support_cap = (n // 2).astype(jnp.int32)
    return jnp.minimum(count, jnp.minimum(support_cap, jnp.int32(max_rank)))


def _compute_s_gap(eigenvalues: Array, k: Array) -> Array:
    """S_gap at rank cut k: λ₁/λ_{k+1}.  Returns 1.0 when k=0."""
    max_rank = eigenvalues.shape[0]
    k_clipped = jnp.clip(k.astype(jnp.int32), 0, max_rank - 1)
    lambda_1 = jnp.maximum(eigenvalues[0], 1e-10)
    lambda_k1 = jax.lax.dynamic_index_in_dim(eigenvalues, k_clipped, keepdims=False)
    lambda_k1_safe = jnp.maximum(lambda_k1, 1e-10)
    return jnp.where(
        k.astype(jnp.int32) == 0, jnp.ones_like(lambda_1), lambda_1 / lambda_k1_safe
    )


def _compute_r2_score_linearity(
    draws_buffer: Array,
    grads_buffer: Array,
    sigma: Array,
    n: Array,
    U_k: Array,
    max_rank: int,
) -> tuple[Array, Array]:
    """Held-out score-linearity R² with three-tier fallback.

    Fit s_w ≈ A·w + b train/test (first/last n//2 valid rows).  Three modes:
    full-affine (n ≥ 2·8·d), projected on max_rank+1 candidate directions
    (n ≥ 2·8·(max_rank+1)), deferred/NaN otherwise.  Returns (r2, mode_int)
    where mode_int is a traced JAX int32: _R2_DEFERRED=0, _R2_PROJECTED=1,
    _R2_FULL_AFFINE=2.  Mode is observed from the actual branch taken — not
    inferred post-hoc.
    """
    B, d = draws_buffer.shape
    n_f = n.astype(jnp.float32)
    n_safe = jnp.maximum(n_f, 2.0)
    mask = (jnp.arange(B) < n).astype(draws_buffer.dtype)
    sigma_safe = jnp.maximum(sigma, 1e-20)
    mean_x = (mask[:, None] * draws_buffer).sum(0) / n_safe
    mean_g = (mask[:, None] * grads_buffer).sum(0) / n_safe
    w = mask[:, None] * (draws_buffer - mean_x[None, :]) / sigma_safe[None, :]
    s_w = mask[:, None] * (grads_buffer - mean_g[None, :]) * sigma_safe[None, :]
    n_train_int = n // 2
    train_mask = mask * (jnp.arange(B) < n_train_int).astype(mask.dtype)
    test_mask = mask * (jnp.arange(B) >= n_train_int).astype(mask.dtype)

    def _r2_from_features(feats: Array, resp: Array) -> Array:
        """Held-out R² of ``resp`` predicted from ``feats`` (train/test split)."""
        n_feats = feats.shape[1]
        FtF = (train_mask[:, None] * feats).T @ (train_mask[:, None] * feats)
        FtS = (train_mask[:, None] * feats).T @ (train_mask[:, None] * resp)
        A = jnp.linalg.lstsq(
            FtF + 1e-8 * jnp.eye(n_feats, dtype=FtF.dtype), FtS, rcond=None
        )[0]
        s_pred = (test_mask[:, None] * feats) @ A
        s_test = test_mask[:, None] * resp
        n_test_safe = jnp.maximum(test_mask.sum().astype(jnp.float32), 2.0)
        s_mean = s_test.sum(0) / n_test_safe
        tss = ((s_test - test_mask[:, None] * s_mean[None, :]) ** 2).sum(0)
        rss = ((s_test - s_pred) ** 2).sum(0)
        return jnp.median(1.0 - rss / jnp.maximum(tss, 1e-10))

    def _full_affine() -> tuple[Array, Array]:
        feats = jnp.concatenate([w, jnp.ones((B, 1), dtype=w.dtype)], axis=1)
        return _r2_from_features(feats, s_w), jnp.int32(_R2_FULL_AFFINE)

    def _projected() -> tuple[Array, Array]:
        # Project BOTH position and score onto U_k.  Rank-k acts only in
        # span(U_k), so linearity of the score RESTRICTED to that subspace is
        # the escalation-safety question; nonlinearity orthogonal to U_k is
        # invisible to rank-k and should not gate escalation.
        # Bug fixed: old code regressed d-dim score on k+1 features → median
        # R² ≈ 0 at d≫k (radon d=390 always emitted reparam_suggested).
        w_proj = w @ U_k  # (B, actual_rank): position projected onto U_k
        s_w_proj = s_w @ U_k  # (B, actual_rank): score projected onto U_k
        feats = jnp.concatenate([w_proj, jnp.ones((B, 1), dtype=w.dtype)], axis=1)
        return _r2_from_features(feats, s_w_proj), jnp.int32(_R2_PROJECTED)

    def _deferred() -> tuple[Array, Array]:
        # dtype must match sibling branches (s_w.dtype); hardcoding float32 crashes
        # under x64 where sibling branches return float64.
        return jnp.asarray(float("nan"), dtype=s_w.dtype), jnp.int32(_R2_DEFERRED)

    min_n_full = float(2 * _MIN_TRAIN_D_RATIO * d)  # each half needs ≥ 8d samples
    min_n_proj = float(2 * _MIN_TRAIN_K_RATIO * (max_rank + 1))  # each half ≥ 8(k+1)

    return jax.lax.cond(
        n_f >= min_n_full,
        _full_affine,
        lambda: jax.lax.cond(n_f >= min_n_proj, _projected, _deferred),
    )


def _compute_transient_mixing_signal(
    draws_buffer: Array,
    sigma: Array,
    n: Array,
) -> Array:
    """Split-half mean-difference proxy for the warmup transient-mixing class.

    Returns traced bool: True = slow-mixing (RESET preferred by v1.1 switch).
    V1 always uses RESET; this signal is reported for the v1.1 anchor.
    """
    B, _ = draws_buffer.shape
    n_f = n.astype(draws_buffer.dtype)
    n_safe = jnp.maximum(n_f, 2.0)
    mask = (jnp.arange(B) < n).astype(draws_buffer.dtype)
    sigma_safe = jnp.maximum(sigma, 1e-20)
    mean_x = (mask[:, None] * draws_buffer).sum(0) / n_safe
    w = mask[:, None] * (draws_buffer - mean_x[None, :]) / sigma_safe[None, :]
    n_train_int = n // 2
    m1 = mask * (jnp.arange(B) < n_train_int).astype(mask.dtype)
    m2 = mask * (jnp.arange(B) >= n_train_int).astype(mask.dtype)
    n1 = jnp.maximum(m1.sum().astype(jnp.float32), 1.0)
    n2 = jnp.maximum(m2.sum().astype(jnp.float32), 1.0)
    mu1 = (m1[:, None] * w).sum(0) / n1
    mu2 = (m2[:, None] * w).sum(0) / n2
    std = jnp.maximum(((mask[:, None] * w**2).sum(0) / n_safe) ** 0.5, 1e-10)
    return jnp.max(jnp.abs(mu1 - mu2) / std) > _TRANSIENT_MIXING_THRESHOLD


def _compute_within_chain_stats(
    draws_buffer_mc: Array,
    n: Array,
) -> tuple[Array, Array]:
    """Per-chain means and pooled within-chain diagonal variance.

    Parameters
    ----------
    draws_buffer_mc
        Per-chain draw buffers, shape ``(M, buf_size, d)``.
    n
        Dynamic fill count (same for all chains), int32 scalar.

    Returns
    -------
    chain_means
        Shape ``(M, d)`` — per-chain mean over valid draws.
    W_diag
        Shape ``(d,)`` — average within-chain diagonal variance across M chains.
    """
    M_s, B, d = draws_buffer_mc.shape  # all static
    n_f = n.astype(draws_buffer_mc.dtype)
    n_safe = jnp.maximum(n_f, 1.0)
    step_mask = (jnp.arange(B) < n).astype(draws_buffer_mc.dtype)  # (B,)

    # Per-chain means: (M, d)
    chain_means = (step_mask[None, :, None] * draws_buffer_mc).sum(
        1
    ) / n_safe  # broadcast (M, B, d) × (B,)

    # Per-chain within-chain variance, vmapped over M: (M, d)
    def _chain_var(draws_m: Array, mean_m: Array) -> Array:
        centered = step_mask[:, None] * (draws_m - mean_m[None, :])
        return (centered**2).sum(0) / jnp.maximum(n_safe - 1.0, 1.0)

    per_chain_vars = jax.vmap(_chain_var)(draws_buffer_mc, chain_means)  # (M, d)
    W_diag = per_chain_vars.mean(0)  # (d,) pooled within-chain variance
    return chain_means, W_diag


def _between_chain_detection(
    chain_means: Array,
    W_diag: Array,
    n: Array,
    M: int,
    d: int,
) -> tuple[Array, Array, Array]:
    """Between-chain detection statistic via the M×M Gram.

    Computes T = (n/(M−1))·ZᵀZ (rank ≤ M−1) via the M×M Gram ZZᵀ.
    T's eigenvalues are the per-direction Gelman–Rubin B/W ratios (R̂²).
    Null distribution has top eigenvalue → (1+√(d/(M−1)))² (the detection edge).

    Parameters
    ----------
    chain_means
        Shape ``(M, d)`` — per-chain mean.
    W_diag
        Shape ``(d,)`` — within-chain diagonal variance (whitening basis).
    n
        Dynamic fill count, int32 scalar (within-chain number of draws).
    M, d
        Static integers.

    Returns
    -------
    T_eigenvalues
        Shape ``(M,)`` — eigenvalues of T in descending order (last M−rank ≈ 0).
    V_top
        Shape ``(d, M−1)`` — top M−1 directions of T in whitened space (columns).
    f1
        float32 scalar — collinearity score: fraction of total between-chain
        scatter variance in the leading direction (→1 for genuine slow dir,
        ≈1/(M−1) for isotropic scatter).
    """
    n_f = n.astype(chain_means.dtype)
    grand_mean = chain_means.mean(0)  # (d,)
    W_safe = jnp.maximum(W_diag, jnp.float32(1e-20))
    sigma_w = jnp.sqrt(W_safe)  # (d,)

    # Whitened chain-mean deviations: z_m = (mu_m − mu_bar) / sigma_w, (M, d)
    Z = (chain_means - grand_mean[None, :]) / sigma_w[None, :]

    # M×M Gram: cheaper than d×d when M ≪ d
    gram = Z @ Z.T  # (M, M)

    # Symmetric eigendecomp; eigh returns ascending order
    eigvals_gram, eigvecs_gram = jnp.linalg.eigh(gram)  # (M,), (M, M)
    # Descending order
    eigvals_gram = jnp.flip(eigvals_gram)
    eigvecs_gram = jnp.flip(eigvecs_gram, axis=1)

    # T eigenvalues: scale by c = n/(M−1)
    c = n_f / jnp.float32(M - 1)
    T_eigenvalues = eigvals_gram * c  # (M,)

    # Collinearity score f₁ = λ_max(gram) / trace(gram)
    trace_gram = jnp.maximum(jnp.trace(gram), jnp.float32(1e-20))
    f1 = (eigvals_gram[0] / trace_gram).astype(jnp.float32)

    # Top M−1 directions in d-space via Z^T U / sqrt(gram_eigval)
    # (eigenvectors of Z^T Z = right singular vectors of Z)
    eps = jnp.float32(1e-10)
    top_m1 = min(M - 1, d)  # static
    s_safe = jnp.sqrt(jnp.maximum(eigvals_gram[:top_m1], eps))  # (M−1,)
    V_top = Z.T @ eigvecs_gram[:, :top_m1] / s_safe[None, :]  # (d, M−1)

    return T_eigenvalues, V_top, f1


def _mc_detection_edge(d: int, dof: int) -> float:
    """Between-chain bulk-separation edge: ``(1 + √(d/dof))²``.

    Calibrated for the M×M Gram whose null Wishart has ``dof = M−1`` degrees
    of freedom (grand-mean constraint removes one dof from M chains).  Using
    M−1 (not M) is rigorous: rank(T) = M−1 exactly.

    Both ``d`` and ``dof`` are Python ints (static at construction time).
    """
    return (1.0 + (d / dof) ** 0.5) ** 2


def _loo_detection_passes(
    chain_means: Array,
    W_diag: Array,
    n: Array,
    M: int,
    d: int,
    edge_loo: float,
) -> Array:
    """Leave-one-out check: detection must survive dropping any single chain.

    For each m' = 0..M−1, re-centres the remaining M−1 chains, computes the
    top T eigenvalue with (M−2) dof, and checks it clears ``edge_loo``.
    All M checks must pass (conjunction).

    Parameters
    ----------
    chain_means
        Shape ``(M, d)``.
    W_diag
        Shape ``(d,)`` — within-chain diagonal variance.
    n
        Dynamic fill count, int32.
    M, d
        Static.
    edge_loo
        Detection edge for M−2 dof: ``(1+√(d/(M−2)))²`` (Python float).

    Returns
    -------
    bool scalar (JAX) — True if all M leave-one-out checks pass.
    """
    n_f = n.astype(chain_means.dtype)
    c_loo = n_f / jnp.float32(M - 2)
    W_safe = jnp.maximum(W_diag, jnp.float32(1e-20))
    sigma_w = jnp.sqrt(W_safe)
    edge_f32 = jnp.float32(edge_loo)

    all_pass = jnp.ones((), dtype=jnp.bool_)
    for m_drop in range(M):
        # Remaining M−1 chain means (static Python-time indexing)
        rows = [chain_means[m] for m in range(M) if m != m_drop]
        Z_loo = jnp.stack(rows)  # (M−1, d)
        mu_loo = Z_loo.mean(0)  # (d,)
        Z_loo_c = (Z_loo - mu_loo[None, :]) / sigma_w[None, :]  # whitened + centred
        gram_loo = Z_loo_c @ Z_loo_c.T  # (M−1, M−1)
        max_eigval_loo = jnp.linalg.eigvalsh(gram_loo)[-1]  # largest (ascending)
        T_max_loo = max_eigval_loo * c_loo
        all_pass = all_pass & (T_max_loo > edge_f32)

    return all_pass


def _mc_unimodality_threshold(M: int) -> float:
    """Gap-stat threshold for the unimodality guard, calibrated at null q99.

    v2.1 recalibration: threshold = MC null q99 per M (table-based where
    calibrated, conservative fallback otherwise).  v2 used q90 = 3.5 at M=8,
    causing 10%/window FPR and 5/5 radon false-defers across 11 windows.
    The table ``_MC_UNIMODALITY_Q99_TABLE`` holds measured values; M values
    not in the table fall back to ``max(0.5*(M−1), 3.0)`` (the v2 formula,
    now treated as a conservative estimate rather than a design threshold).

    Requires 2-consecutive-window confirmation (``_MC_UNIMODALITY_CONFIRM_WINDOWS``)
    before ``deferred_to_ensemble`` latches — this is a FLAG statistic, not a proof.

    Parameters
    ----------
    M
        Number of chains (static Python int).

    Returns
    -------
    float — threshold above which the projected chain-means are flagged as
    mode-split (one window; requires 2 consecutive to latch).
    """
    return _MC_UNIMODALITY_Q99_TABLE.get(
        M, max(_MC_UNIMODALITY_GAP_FRACTION * (M - 1), 3.0)
    )


def _unimodality_gap_stat(
    chain_means: Array,
    top_direction: Array,
    M: int,
) -> tuple[Array, Array]:
    """Gap-statistic check on projected chain-means.

    Projects the M chain-means onto ``top_direction`` and measures whether
    the projections cluster into one group (unimodal) or two or more groups
    (mode-split).  Uses max_gap / mean_gap as the discreteness signal.

    Parameters
    ----------
    chain_means
        Shape ``(M, d)``.
    top_direction
        Shape ``(d,)`` — unit-vector slow direction (unwhitened).
    M
        Static int — number of chains.

    Returns
    -------
    is_unimodal
        bool scalar — True iff gap ratio < :func:`_mc_unimodality_threshold`.
    gap_ratio
        float32 scalar — max_gap / mean_gap (diagnostic; large ⇒ mode-split).
    """
    threshold = _mc_unimodality_threshold(M)  # Python float, static
    projections = chain_means @ top_direction  # (M,)
    sorted_proj = jnp.sort(projections)  # ascending
    gaps = sorted_proj[1:] - sorted_proj[:-1]  # (M−1,) inter-point gaps
    mean_gap = gaps.mean()
    max_gap = gaps.max()
    gap_ratio = max_gap / jnp.maximum(mean_gap, jnp.float32(1e-10))
    is_unimodal = gap_ratio < jnp.float32(threshold)
    return is_unimodal, gap_ratio.astype(jnp.float32)


def _geometric_mean_deploy_scale(
    chain_means: Array,
    pooled_grads: Array,
    step_mask_all: Array,
    grand_mean: Array,
    e: Array,
    n_pool: Array,
    M: int,
) -> Array:
    """Geometric-mean metric variance for the slow direction.

    σ̂²_deploy = √( (B/n) · 1/(eᵀF̂e) )

    where B/n is the between-chain variance of chain-means in direction e and
    1/(eᵀF̂e) is the pooled-Fisher curvature scale in direction e.  The
    geometric mean cancels the init-dispersion factor f_disp that makes each
    term individually biased: B/n over-estimates by f_disp, Fisher curvature
    under-estimates by f_disp; the product is f_disp-independent.

    Parameters
    ----------
    chain_means
        Shape ``(M, d)`` — per-chain means.
    pooled_grads
        Shape ``(M*B, d)`` — all chains' gradient buffers flattened.
    step_mask_all
        Shape ``(M*B,)`` — 1 for valid rows, 0 otherwise.
    grand_mean
        Shape ``(d,)`` — grand mean of chain means.
    e
        Shape ``(d,)`` — unit-vector slow direction in original (unwhitened) space.
    n_pool
        Dynamic int32 — total valid gradient count (M * n_per_chain).
    M
        Static int — number of chains.

    Returns
    -------
    float32 scalar — σ̂²_deploy (positive).
    """
    # B/n: between-chain variance of chain-means projected onto e
    mu_proj = (chain_means - grand_mean[None, :]) @ e  # (M,)
    B_over_n = (mu_proj**2).sum() / jnp.float32(M - 1)

    # Fisher curvature: eᵀF̂_pooled e
    n_pool_f = n_pool.astype(pooled_grads.dtype)
    n_pool_safe = jnp.maximum(n_pool_f, jnp.float32(1.0))
    g_proj = pooled_grads @ e  # (M*B,)
    fisher_curv_e = (step_mask_all * g_proj**2).sum() / n_pool_safe

    # Geometric mean: sqrt(B/n * 1/fisher_curv_e)
    sigma_sq_deploy = jnp.sqrt(
        jnp.maximum(B_over_n, jnp.float32(1e-20))
        / jnp.maximum(fisher_curv_e, jnp.float32(1e-20))
    )
    return sigma_sq_deploy.astype(jnp.float32)


# ---------------------------------------------------------------------------
# v2.1 W-branch signal functions (pooled within-chain residual detector)
# ---------------------------------------------------------------------------


def _build_mc_window_schedule(num_steps: int, M: int, actual_rank: int) -> Array:
    """Pooled-aware growing-window schedule for the M-chain meta-adaptation path.

    The single-chain schedule's 30%-early phase and 80-step starting window are
    sized for a single chain.  With M chains the detection-relevant count is the
    POOLED ``M * n``, so windows should be sized on pooled samples: the first
    main window is ``n1 = ceil(min_n_proj / M)`` per chain, ensuring
    ``n_pool = n1 * M >= min_n_proj = 8 * (actual_rank + 1)`` (projected-tier
    floor).  This restores early-escalation capability that the single-chain
    schedule loses at M >= 4.

    Example (M=8, actual_rank=25, num_steps=312):
        ``n1 = ceil(208 / 8) = 26``.
        Windows end at steps 1, 27, 66, 124, 265.
        Steps 27, 66, 124 have n_pool ≥ 208 AND budget_remaining ≥ 50 —
        all three are escalation-eligible.

    The ``early_window=0.0`` argument to :func:`build_growing_window_schedule`
    gives a harmless 1-step early window (due to the ``max(..., 1)`` guard in
    that function) followed by the main pooled-aware phase.

    Parameters
    ----------
    num_steps
        Per-chain warmup step count (= total_budget // (LEAPS * M)).
    M
        Number of chains (static Python int).
    actual_rank
        Estimated rank capacity (static Python int; = min(d//2, _MAX_RANK_CAP)).

    Returns
    -------
    Array
        ``(num_steps, 2)`` schedule array in the same ``(stage, is_window_end)``
        format as :func:`~blackjax.adaptation.low_rank_adaptation.build_growing_window_schedule`.
    """
    from blackjax.adaptation.low_rank_adaptation import build_growing_window_schedule

    min_n_proj = 2 * _MIN_TRAIN_K_RATIO * (actual_rank + 1)  # 8*(actual_rank+1)
    n1 = max(-(-min_n_proj // M), 1)  # ceil division, ensure ≥ 1
    return build_growing_window_schedule(
        num_steps,
        early_window=0.0,  # suppress early phase; harmless 1-step leftover is fine
        window_size=n1,
        window_growth=1.5,
    )


def _w_branch_null_edge(M: int, n: Array, d: int) -> Array:
    """Null bulk upper edge for the pooled within-chain residual spectrum.

    Formula: ``_W_BRANCH_NULL_EDGE_TW_FACTOR * (1 + sqrt(d / (M*(n-1))))^2``.

    **Conservative for positively-autocorrelated series.** The base analytic
    form ``(1 + sqrt(d/N))^2`` is the iid-null upper bulk edge for N = M*(n-1)
    degrees of freedom (within-chain residuals pooled across M chains, each
    chain centered on its own mean).  For positively-autocorrelated chains
    (AR rho > 0), within-chain variance is inflated above the iid level, which
    reduces effective N below M*(n-1) — meaning genuine structure clears this
    edge more easily than the iid calibration predicts.  The cross-chain Psi
    consistency gate is the primary FPR control; this edge is secondary.

    **TW-inflation factor.** :data:`_W_BRANCH_NULL_EDGE_TW_FACTOR` = 1.02
    accounts for Tracy-Widom finite-N fluctuations: at the regime-relevant
    pool size N approx 300 (M=8, n=40-draw window), the iid-null q99 exceeds
    the asymptotic edge by ~2% (measured on 1000 iid null replicates at M=8,
    n=40, d in {10,20,50,100}).

    **Upgrade path.** MC-per-(M,n,d) calibration replaces this analytic formula
    with a lookup table of measured null quantiles, eliminating the iid
    assumption and the TW approximation.  This function is designed as one
    swappable block for that transition: replace the body, keep the signature.

    Parameters
    ----------
    M : int
        Number of chains (static Python int; baked into N = M*(n-1)).
    n : Array
        Valid draws per chain in this window (dynamic JAX int32 or int).
    d : int
        Dimension (static Python int; baked into the aspect-ratio term d/N).

    Returns
    -------
    Array
        float32 null edge; compare directly to ``lam1`` from
        :func:`_compute_pooled_within_spectrum`.
    """
    N_safe = jnp.maximum(
        jnp.float32(M) * (jnp.asarray(n, dtype=jnp.float32) - jnp.float32(1.0)),
        jnp.float32(1.0),
    )
    base_edge = (jnp.float32(1.0) + jnp.sqrt(jnp.float32(d) / N_safe)) ** 2
    return jnp.float32(_W_BRANCH_NULL_EDGE_TW_FACTOR) * base_edge


# ---------------------------------------------------------------------------
# BLOCKER-2 GAIN+abstain: projected-tier router for M-chain path
# ---------------------------------------------------------------------------

_GAIN_THRESHOLD: float = 0.3
"""Slope-heterogeneity GAIN threshold for projected-tier reparam signal.

Reparam requires per-chain fits to beat the shared pooled fit by this margin
on held-out data.  A Gaussian (no structure) gives GAIN <= 0 (per-chain fits
overfit relative to shared); a funnel-at-levels gives GAIN ~ +0.83.
"""

_GAIN_READABILITY_FLOOR: float = 0.5
"""Minimum per-chain R² for the projected-tier fits to be considered readable.

Below this threshold the fits are garbage (starved / transient chains) and
the result is abstain (route = diagonal, confidence = low) rather than
reparam.  Equals _R_MIN by construction (same curvature semantics).
"""


def _compute_projected_gain_r2_mc(
    pc_draws_tm: Array,
    pc_grads_tm: Array,
    sigma_w_diag: Array,
    n: Array,
    M: int,
    U_k: Array,
) -> tuple[Array, Array]:
    """Slope-heterogeneity GAIN for the projected router tier (M-chain path).

    Both the per-chain fit and the pooled-shared fit are evaluated on held-out
    data (train = first half of each chain, test = second half).  The GAIN

    .. code-block:: text

        GAIN = R2_perchain - R2_shared

    is positive only when genuine curvature heterogeneity exists across chain
    regions.  A Gaussian null produces GAIN <= 0 (per-chain fits overfit),
    so the threshold 0.3 gives near-zero false-reparam rate.

    **Abstain rule**: if R2_perchain < :data:`_GAIN_READABILITY_FLOOR` (garbage
    fits due to starvation / transience), return (NaN, NaN).  The caller maps
    NaN to diagonal + confidence=low (not reparam).

    Parameters
    ----------
    pc_draws_tm
        Per-chain-centered time-major pooled draws, shape ``(B*M, d)``.
        Row ``t*M + m`` = (step t, chain m).  Valid rows: first ``n*M``.
    pc_grads_tm
        Same layout as ``pc_draws_tm`` for score vectors.
    sigma_w_diag
        Within-chain whitening std, shape ``(d,)``; ``sqrt(W_diag)``.
    n
        Per-chain valid draw count (dynamic JAX int32).
    M
        Chain count (static Python int; baked into reshape).
    U_k
        Top-k whitened eigenvectors, shape ``(d, k)`` where k = actual_rank.

    Returns
    -------
    tuple[Array, Array]
        ``(gain, r2_perchain)`` both float32.  ``NaN`` on abstain (unreadable
        fits).
    """
    B_M = pc_draws_tm.shape[0]  # static: B*M
    B = B_M // M  # per-chain buffer size (static)
    k = U_k.shape[1]  # projected rank (static)

    sigma_safe = jnp.maximum(sigma_w_diag, jnp.float32(1e-20))

    # Reshape to step-major-chain: (B, M, d); chain m at [:, m, :].
    draws_3d = pc_draws_tm.reshape(B, M, -1)  # (B, M, d)
    grads_3d = pc_grads_tm.reshape(B, M, -1)  # (B, M, d)

    # Whiten and project to k directions.
    w_proj_3d = (draws_3d / sigma_safe[None, None, :]) @ U_k  # (B, M, k)
    s_proj_3d = (grads_3d * sigma_safe[None, None, :]) @ U_k  # (B, M, k)

    # Per-chain train/test masks (same split boundary for all chains).
    n_half = n // 2
    step_idx = jnp.arange(B, dtype=jnp.int32)
    train_mask = (step_idx < n_half).astype(pc_draws_tm.dtype)  # (B,)
    test_mask = ((step_idx >= n_half) & (step_idx < n)).astype(
        pc_draws_tm.dtype
    )  # (B,)
    n_test_safe = jnp.maximum(test_mask.sum().astype(jnp.float32), jnp.float32(2.0))

    # ---- Shared fit: pool all chains' train data ----
    # Layout: (B, M, k) → chain-major (M, B, k) → reshape (M*B, k).
    # Shared train mask: broadcast step mask over M chains.
    w_pool = w_proj_3d.transpose(1, 0, 2).reshape(M * B, k)  # (M*B, k)
    s_pool = s_proj_3d.transpose(1, 0, 2).reshape(M * B, k)  # (M*B, k)
    # train_mask for pool: repeat each step's mask M times (chain-major order)
    train_pool = jnp.tile(train_mask, M)  # (M*B,) — step t repeated for M chains
    feats_sh = jnp.concatenate(
        [w_pool, jnp.ones((M * B, 1), dtype=w_pool.dtype)], axis=1
    )
    tm = train_pool[:, None]
    FtF_sh = (tm * feats_sh).T @ (tm * feats_sh)  # (k+1, k+1)
    FtS_sh = (tm * feats_sh).T @ (tm * s_pool)  # (k+1, k)
    A_sh = jnp.linalg.lstsq(
        FtF_sh + jnp.float32(1e-8) * jnp.eye(k + 1, dtype=FtF_sh.dtype),
        FtS_sh,
        rcond=None,
    )[
        0
    ]  # (k+1, k)

    # ---- Per-chain r2_shared and r2_perchain via vmap ----
    # Transpose to chain-major so vmap sees chain as batch dim.
    w_proj_MC = w_proj_3d.transpose(1, 0, 2)  # (M, B, k)
    s_proj_MC = s_proj_3d.transpose(1, 0, 2)  # (M, B, k)

    def _chain_r2s(w_m: Array, s_m: Array) -> tuple[Array, Array]:
        """Per-chain R2_shared and R2_perchain for one chain (vmapped)."""
        feats_m = jnp.concatenate(
            [w_m, jnp.ones((B, 1), dtype=w_m.dtype)], axis=1
        )  # (B, k+1)
        tm_col = test_mask[:, None]

        # Shared model predictions on this chain's test data
        pred_sh = (tm_col * feats_m) @ A_sh  # (B, k)
        s_test = tm_col * s_m
        s_mean = s_test.sum(0) / n_test_safe
        tss = ((s_test - tm_col * s_mean[None, :]) ** 2).sum(0)
        rss_sh = ((s_test - pred_sh) ** 2).sum(0)
        r2_sh = jnp.median(
            jnp.float32(1.0) - rss_sh / jnp.maximum(tss, jnp.float32(1e-10))
        )

        # Per-chain OLS on this chain's train, evaluated on its test
        tr_col = train_mask[:, None]
        FtF_m = (tr_col * feats_m).T @ (tr_col * feats_m)
        FtS_m = (tr_col * feats_m).T @ (tr_col * s_m)
        A_m = jnp.linalg.lstsq(
            FtF_m + jnp.float32(1e-8) * jnp.eye(k + 1, dtype=FtF_m.dtype),
            FtS_m,
            rcond=None,
        )[0]
        pred_pc = (tm_col * feats_m) @ A_m
        rss_pc = ((s_test - pred_pc) ** 2).sum(0)
        r2_pc = jnp.median(
            jnp.float32(1.0) - rss_pc / jnp.maximum(tss, jnp.float32(1e-10))
        )
        return r2_sh, r2_pc

    r2_sh_all, r2_pc_all = jax.vmap(_chain_r2s)(w_proj_MC, s_proj_MC)  # (M,) each
    r2_shared = jnp.median(r2_sh_all)
    r2_perchain = jnp.median(r2_pc_all)

    gain = r2_perchain - r2_shared

    # Abstain: unreadable fits → return NaN to signal "no evidence"
    abstain = (r2_perchain < jnp.float32(_GAIN_READABILITY_FLOOR)) | ~jnp.isfinite(
        r2_perchain
    )
    nan32 = jnp.array(float("nan"), dtype=jnp.float32)
    gain_out = jnp.where(abstain, nan32, gain.astype(jnp.float32))
    r2_pc_out = jnp.where(abstain, nan32, r2_perchain.astype(jnp.float32))
    return gain_out, r2_pc_out


def _compute_pooled_within_spectrum(
    draws_buffer_mc: Array,
    chain_means: Array,
    W_diag: Array,
    n: Array,
    M: int,
    max_rank: int,
) -> tuple[Array, Array]:
    """Top eigenvalue and top eigenvector of the pooled within-chain residual
    correlation matrix R_W.

    Pools M chains' per-chain-centered, diag-whitened residuals into an
    M(n-1)-dof estimate of the within-chain correlation.  Computed via thin SVD
    of the stacked (M*B, d) masked residual matrix — never a d×d eigendecomp.

    Structural facts (exact, not asymptotic):
    - diag(R_W) = 1 exactly: per-chain centering removes the between-chain
      component so W_diag is exactly R_W's diagonal.
    - R_W is f_disp-free: chain means are removed before pooling so init
      overdispersion never enters the estimate.
    - R_W is mode-blind: per-chain centering within each mode gives white
      residuals if the modes have similar within-mode geometry; multi-modal
      chains escalate on the within-mode correlation structure only.

    Parameters
    ----------
    draws_buffer_mc
        Shape ``(M, B, d)``.
    chain_means
        Shape ``(M, d)``; per-chain means over valid draws.
    W_diag
        Shape ``(d,)``; pooled within-chain diagonal variance.
    n
        Dynamic fill count, int32 scalar.
    M
        Static chain count.
    max_rank
        Maximum rank for the SVD (determines how many right singular vectors
        are returned; only the top one is used for the W-branch gate).

    Returns
    -------
    lam1
        float32 scalar — top eigenvalue of R_W.
    top_eigvec
        Shape ``(d,)`` float32 — top right singular vector (principal direction
        in the whitened space).
    """
    _M, B, d = draws_buffer_mc.shape
    W_safe = jnp.maximum(W_diag, jnp.float32(1e-20))
    sigma_w = jnp.sqrt(W_safe)

    # Per-chain centering + whitening: Y[m,t] = (X[m,t] - mu_m) / sigma_w
    centered = draws_buffer_mc - chain_means[:, None, :]  # (M, B, d)
    whitened = centered / sigma_w[None, None, :]  # (M, B, d)

    # Apply valid-row mask within each chain; flatten chain-major → (M*B, d)
    step_mask = (jnp.arange(B) < n).astype(whitened.dtype)  # (B,)
    Y_pool = (step_mask[None, :, None] * whitened).reshape(M * B, d)  # (M*B, d)

    # Dof: N = M*(n-1).  Scale so S = Y_scaled^T Y_scaled ≈ R_W = (1/N) Y_pool^T Y_pool
    N_dof = jnp.maximum(
        n.astype(jnp.int32) * jnp.int32(M) - jnp.int32(M), jnp.int32(1)
    )  # M*(n-1)
    N_f = N_dof.astype(Y_pool.dtype)
    Y_scaled = Y_pool / jnp.sqrt(jnp.maximum(N_f, jnp.float32(1.0)))

    # Finite guard before SVD (NaN/Inf rows from near-zero W_diag)
    Y_safe = jnp.where(jnp.isfinite(Y_scaled), Y_scaled, jnp.zeros_like(Y_scaled))

    # Thin SVD: singular values s satisfy lam_i(R_W) = s_i^2
    _, s, Vt = jnp.linalg.svd(Y_safe, full_matrices=False)

    lam1 = (s[0] ** 2).astype(jnp.float32)
    top_eigvec = Vt[0].astype(jnp.float32)  # (d,) — top direction in whitened space

    return lam1, top_eigvec


def _compute_mode_consistency_flag(
    pc_draws_tm: Array,
    pc_grads_tm: Array,
    grads_buffer_mc: Array,
    chain_means: Array,
    grand_mean: Array,
    V_top: Array,
    sigma_w_diag: Array,
    T_eigenvalues: Array,
    edge_full: Array,
    n: Array,
    M: int,
) -> Array:
    """Per-direction mode-consistency flag for T-branch multimodality detection.

    BLOCKER-3 fix.  For each admitted T-spike direction ``e_j`` (where
    ``T_eigenvalues[j] > edge_full``), computes

    .. code-block:: text

        mode_flag(j) = (R²_local(e_j) − R²_global(e_j) > 0.3) AND (R²_local(e_j) ≥ 0.5)

    **R²_local**: per-chain OLS of (per-chain-centered score, per-chain-centered
    position) projected to ``e_j``, then median over chains.  Captures local
    score linearity within each chain's region.

    **R²_global**: pooled OLS of (grand-centered score, grand-centered position)
    projected to ``e_j``.  For a unimodal target the global score is globally
    linear → R²_global ≈ R²_local.  For a mode-split the scores have different
    intercepts per mode → the global linear fit collapses → R²_global << R²_local.

    **Critical distinction**: R²_global must use the RAW (not per-chain-centered)
    gradients, grand-mean-centered.  If per-chain-centered gradients were used for
    R²_global, a unimodal overdispersed target (chains at different positions on
    the same Gaussian) would falsely flag: the per-chain-centered score doesn't
    correlate with the grand-centered position (the offset contributes to x but
    not to pc_s), mimicking a mode-split.  Using grand-centered raw gradients
    restores the global linear relationship for unimodal targets.

    Returns True if ANY admitted direction flags.  W-branch is unaffected (never
    gated on this — the W-branch is structurally mode-blind via per-chain centering).

    Parameters
    ----------
    pc_draws_tm
        Per-chain-centered time-major draws, shape ``(B*M, d)``.  Row ``t*M+m``.
    pc_grads_tm
        Per-chain-centered time-major grads, shape ``(B*M, d)``.  Used for R²_local.
    grads_buffer_mc
        Raw (non-centered) grads, shape ``(M, B, d)``.  Used for R²_global (grand-
        centered).  For a unimodal Gaussian the raw grads ARE globally linear in
        position; per-chain-centered grads lose this between-chain signal.
    chain_means
        Shape ``(M, d)`` — per-chain draw means.
    grand_mean
        Shape ``(d,)`` — mean of chain means.
    V_top
        Shape ``(d, k)`` — top T-eigenvectors in W-whitened space.
    sigma_w_diag
        Shape ``(d,)`` — within-chain whitening std.
    T_eigenvalues
        Shape ``(k,)`` — T-branch eigenvalues (to mask admitted dirs).
    edge_full
        Scalar float32 — T magnitude threshold.
    n
        Per-chain valid draw count (dynamic int32).
    M
        Static chain count.

    Returns
    -------
    bool scalar — True if any T-spike direction exhibits mode-consistency signature.
    """
    BM, d = pc_draws_tm.shape
    B = BM // M  # static

    # Grand-mean-centered raw grads for R²_global (shape M, B, d → B*M, d time-major)
    step_mask_col = (jnp.arange(B) < n).astype(grads_buffer_mc.dtype)  # (B,)
    n_f = jnp.maximum(n.astype(grads_buffer_mc.dtype), jnp.float32(1.0))
    # grand mean of raw grads (over all valid draws across all chains)
    total_valid = n_f * jnp.float32(M)
    grad_sum = (step_mask_col[None, :, None] * grads_buffer_mc).sum(axis=(0, 1))  # (d,)
    grand_mean_grad = grad_sum / jnp.maximum(total_valid, jnp.float32(1.0))  # (d,)
    # Grand-mean-centered raw grads, time-major layout
    gc_grads_mc = grads_buffer_mc - grand_mean_grad[None, None, :]  # (M, B, d)
    # swapaxes(0,1) → (B, M, d) → reshape → (B*M, d)
    gc_grads_tm = gc_grads_mc.swapaxes(0, 1).reshape(BM, d)  # (B*M, d)

    # Grand-centered position offset in time-major layout
    gc_offset = chain_means - grand_mean[None, :]  # (M, d)
    gc_offset_tm = jnp.tile(gc_offset, (B, 1))  # (B*M, d)

    # Per-row valid mask: row t*M+m is valid when step t < n
    row_idx = jnp.arange(BM, dtype=jnp.int32)
    t_idx = row_idx // M  # (B*M,) time step per row
    valid_mask = (t_idx < n).astype(pc_draws_tm.dtype)  # (B*M,)

    # Step-wise valid mask for per-chain reshape
    step_valid = step_mask_col.astype(pc_draws_tm.dtype)  # (B,)

    def _r2_for_direction(j: int) -> Array:
        """Mode-consistency flag (bool Array) for direction j."""
        # Unwhiten V_top[:, j] → original space unit vector
        e_unnorm = sigma_w_diag * V_top[:, j]
        e_j = e_unnorm / jnp.maximum(jnp.linalg.norm(e_unnorm), jnp.float32(1e-10))

        # ---- R²_global: OLS on grand-centered raw grads vs grand-centered positions ----
        x_gc_proj = (pc_draws_tm + gc_offset_tm) @ e_j  # (B*M,) grand-centered position
        s_gc_proj = gc_grads_tm @ e_j  # (B*M,) grand-centered raw score

        n_total = jnp.maximum(valid_mask.sum(), jnp.float32(2.0))
        x_gc_mean = (valid_mask * x_gc_proj).sum() / n_total
        s_gc_mean = (valid_mask * s_gc_proj).sum() / n_total
        x_gc_c = x_gc_proj - x_gc_mean
        s_gc_c = s_gc_proj - s_gc_mean
        Sxx_g = (valid_mask * x_gc_c**2).sum()
        Sxs_g = (valid_mask * x_gc_c * s_gc_c).sum()
        beta_g = Sxs_g / jnp.maximum(Sxx_g, jnp.float32(1e-20))
        rss_g = (valid_mask * (s_gc_c - beta_g * x_gc_c) ** 2).sum()
        ss_tot_g = (valid_mask * s_gc_c**2).sum()
        r2_global = jnp.clip(
            jnp.float32(1.0) - rss_g / jnp.maximum(ss_tot_g, jnp.float32(1e-20)),
            jnp.float32(-10.0),
            jnp.float32(1.0),
        )

        # ---- R²_local: per-chain OLS on per-chain-centered (x, s), median ----
        x_pc_proj = pc_draws_tm @ e_j  # (B*M,) per-chain-centered position
        s_pc_proj = pc_grads_tm @ e_j  # (B*M,) per-chain-centered score

        # Reshape time-major (B*M,) → (B, M): column m = chain m's values
        x_pc_2d = x_pc_proj.reshape(B, M)  # (B, M)
        s_2d = s_pc_proj.reshape(B, M)  # (B, M)

        def _chain_r2_1d(x_col: Array, s_col: Array) -> Array:
            n_c = jnp.maximum(step_valid.sum(), jnp.float32(2.0))
            x_m = (step_valid * x_col).sum() / n_c
            s_m = (step_valid * s_col).sum() / n_c
            x_c = x_col - x_m
            s_c = s_col - s_m
            Sxx = (step_valid * x_c**2).sum()
            Sxs = (step_valid * x_c * s_c).sum()
            beta = Sxs / jnp.maximum(Sxx, jnp.float32(1e-20))
            rss = (step_valid * (s_c - beta * x_c) ** 2).sum()
            ss_tot = (step_valid * s_c**2).sum()
            r2 = jnp.float32(1.0) - rss / jnp.maximum(ss_tot, jnp.float32(1e-20))
            return jnp.clip(r2, jnp.float32(-10.0), jnp.float32(1.0))

        r2_per_chain = jax.vmap(_chain_r2_1d, in_axes=(1, 1))(x_pc_2d, s_2d)  # (M,)
        r2_local = jnp.median(r2_per_chain)

        # Admitted direction only (mask out by eigenvalue threshold)
        admitted = T_eigenvalues[j] > edge_full
        return (
            admitted
            & (r2_local - r2_global > jnp.float32(0.3))
            & (r2_local >= jnp.float32(0.5))
        )

    # Compute flag for each static direction index; OR together
    k = V_top.shape[1]  # static Python int
    flags = jnp.stack([_r2_for_direction(j) for j in range(k)])  # (k,) bool
    return flags.any()


def _compute_contraction_stat(
    draws_buffer: Array,
    chain_means: Array,
    grand_mean: Array,
    n: Array,
    M: int,
) -> Array:
    """Per-chain split-half drift t-statistic for the T-branch three-way rule.

    BLOCKER-3 fix.  For each chain m, computes the whitened drift along the
    chain's own offset direction

    .. code-block:: text

        c_m = ⟨mean(late half) − mean(early half), ô_m⟩ / SE_m

    where ``ô_m = (chain_mean_m − grand_mean) / ‖·‖`` and ``SE_m`` is the
    pooled SE of the two half-means along ``ô_m``.  A one-sided t across the
    M chains is returned.

    Interpretation (at M=8, critical value -2.365 at α=2.5%):

    - ``t_contr < -2.365``: chains are converging toward the grand mean →
      unimodal-safe, escalate if the other T gates pass.
    - ``t_contr ≈ 0``: chains are frozen / not drifting (mode-split or
      genuinely slow direction); combine with mode-consistency flag.
    - ``t_contr >> 0``: chains are diverging (unusual, flag as inconclusive).

    Parameters
    ----------
    draws_buffer
        Shape ``(M, B, d)`` — raw (not centered) per-chain draws.
    chain_means
        Shape ``(M, d)`` — per-chain draw means.
    grand_mean
        Shape ``(d,)`` — mean of chain means.
    n
        Per-chain valid draw count (dynamic int32).
    M
        Static chain count.

    Returns
    -------
    float32 scalar — one-sided t statistic across M chains.
    """
    _M, B, d = draws_buffer.shape  # all static

    # Per-chain offset directions ô_m = (chain_means_m - grand_mean) / ||·||
    offsets = chain_means - grand_mean[None, :]  # (M, d)
    norms = jnp.linalg.norm(offsets, axis=1, keepdims=True)  # (M, 1)
    o_hat = offsets / jnp.maximum(norms, jnp.float32(1e-10))  # (M, d)

    # Project draws onto ô_m: (M, B, d) * (M, 1, d) → sum over d → (M, B)
    proj = (draws_buffer * o_hat[:, None, :]).sum(axis=-1)  # (M, B)

    # Build valid/early/late masks
    step_idx = jnp.arange(B, dtype=jnp.int32)
    n_half = n // 2
    early_mask = (step_idx < n_half).astype(proj.dtype)  # (B,)
    late_mask = ((step_idx >= n_half) & (step_idx < n)).astype(proj.dtype)
    valid_mask = (step_idx < n).astype(proj.dtype)

    n_half_f = jnp.maximum(n_half.astype(jnp.float32), jnp.float32(1.0))
    n_late_f = jnp.maximum((n - n_half).astype(jnp.float32), jnp.float32(1.0))
    n_total_f = jnp.maximum(n.astype(jnp.float32), jnp.float32(1.0))

    # Per-chain half-means: broadcast mask (B,) over chains (M, B)
    early_proj = (proj * early_mask[None, :]).sum(axis=1) / n_half_f  # (M,)
    late_proj = (proj * late_mask[None, :]).sum(axis=1) / n_late_f  # (M,)

    # Per-chain within-window std for SE estimation
    proj_mean = (proj * valid_mask[None, :]).sum(axis=1) / n_total_f  # (M,)
    proj_var = (valid_mask[None, :] * (proj - proj_mean[:, None]) ** 2).sum(
        axis=1
    ) / jnp.maximum(n_total_f - jnp.float32(1.0), jnp.float32(1.0))
    se = jnp.sqrt(
        jnp.maximum(proj_var, jnp.float32(1e-10)) * jnp.float32(2.0) / n_half_f
    )  # (M,) — SE of mean difference between two half-means

    # Standardized drift per chain
    c_std = (late_proj - early_proj) / jnp.maximum(se, jnp.float32(1e-10))  # (M,)

    # One-sided t across M chains
    c_mean = c_std.mean()
    c_se = jnp.std(c_std, ddof=1) / jnp.sqrt(jnp.float32(M))
    t_contr = c_mean / jnp.maximum(c_se, jnp.float32(1e-10))

    return t_contr.astype(jnp.float32)


def _w_branch_psi_threshold(M: int, n: Array, d: int) -> Array:
    """Adaptive Ψ threshold: max(3 * q99_null(M, n, d), 0.15).

    FOLD-IN 4: the flat 0.15 floor causes 16–17% null leak at d=10.  The
    adaptive threshold is calibrated as 3 * q99_null, where q99_null is the
    q99 of Ψ under iid chains at (M, n, d) = (8, n, d).  Values are measured
    at N_base = M * (n_base - 1) = 1360 (M=8, n_base=171); other N are scaled
    by sqrt(N_base / N).

    Calibration table (q99_null at N=1360, M=8):
        d=10: q99=0.129 → threshold = 3*0.129 = 0.387
        d=26: q99=0.040 → threshold = 3*0.040 = 0.120
        d=50: q99=0.023 → threshold = 3*0.023 = 0.068

    The floor 0.15 applies at all dimensions (avoids sub-0.15 thresholds for
    large d at sparse N) and is the spec-provided minimum.

    Parameters
    ----------
    M
        Chain count (static Python int; baked into the N computation).
    n
        Per-chain valid draw count (dynamic JAX int32 scalar).
    d
        Observation dimension (static Python int).

    Returns
    -------
    float32 scalar — adaptive Ψ gate threshold.
    """
    # Calibration table at N_base = 1360 (M=8, n=171):
    # format: (d_anchor, q99_at_N_base)
    _PSI_CAL_D = jnp.array([10.0, 26.0, 50.0], dtype=jnp.float32)
    _PSI_CAL_Q99 = jnp.array([0.129, 0.040, 0.023], dtype=jnp.float32)
    _N_BASE = jnp.float32(1360.0)

    # Pool size for this (M, n): N = M * max(n-1, 1)
    N = jnp.float32(M) * jnp.maximum(
        n.astype(jnp.float32) - jnp.float32(1.0), jnp.float32(1.0)
    )

    # Log-interpolate q99 on d (linear interp in log-log space)
    d_f = jnp.float32(d)
    log_d = jnp.log(jnp.maximum(d_f, jnp.float32(1.0)))
    log_d0 = jnp.log(_PSI_CAL_D[0])
    log_d1 = jnp.log(_PSI_CAL_D[1])
    log_d2 = jnp.log(_PSI_CAL_D[2])

    log_q0 = jnp.log(jnp.maximum(_PSI_CAL_Q99[0], jnp.float32(1e-6)))
    log_q1 = jnp.log(jnp.maximum(_PSI_CAL_Q99[1], jnp.float32(1e-6)))
    log_q2 = jnp.log(jnp.maximum(_PSI_CAL_Q99[2], jnp.float32(1e-6)))

    # Piecewise linear in log-log: clamp to calibrated range
    t01 = jnp.clip((log_d - log_d0) / (log_d1 - log_d0), 0.0, 1.0)
    t12 = jnp.clip((log_d - log_d1) / (log_d2 - log_d1), 0.0, 1.0)
    log_q_01 = log_q0 + t01 * (log_q1 - log_q0)
    log_q_12 = log_q1 + t12 * (log_q2 - log_q1)
    log_q_interp = jnp.where(d_f <= _PSI_CAL_D[1], log_q_01, log_q_12)
    q99_base = jnp.exp(log_q_interp)

    # Scale by sqrt(N_base / N): tighter calibration at larger N
    scale = jnp.sqrt(
        jnp.maximum(_N_BASE / jnp.maximum(N, jnp.float32(1.0)), jnp.float32(0.01))
    )
    q99_at_N = q99_base * scale

    adaptive_thresh = jnp.float32(3.0) * q99_at_N
    return jnp.maximum(adaptive_thresh, jnp.float32(_W_BRANCH_PSI_FLOOR))


def _compute_chain_consistency_psi(
    draws_buffer_mc: Array,
    chain_means: Array,
    W_diag: Array,
    n: Array,
    M: int,
) -> Array:
    """Cross-chain consistency Ψ of the off-diagonal residual correlation.

    Splits the M chains into fixed halves A (first M//2) and B (remaining),
    computes the pooled within-chain residual correlation for each half using
    the shared W_diag whitener, then computes the cosine similarity of their
    off-diagonal parts in Frobenius matrix space.

    Null property: chains are independent, so E[<C_A, C_B>_F] = ||E C||²_F
    is the off-diagonal signal energy; the cross-noise term is exactly zero
    regardless of within-chain autocorrelation law.  This makes Ψ τ-blind:
    it carries genuine target structure but is insensitive to per-chain
    mixing rate.

    Parameters
    ----------
    draws_buffer_mc
        Shape ``(M, B, d)``.
    chain_means
        Shape ``(M, d)``.
    W_diag
        Shape ``(d,)`` — pooled within-chain diagonal variance.
    n
        Dynamic fill count, int32 scalar.
    M
        Static chain count.

    Returns
    -------
    float32 scalar — Ψ (ranges nominally in [−1, 1]; genuine deep-spread
    ≈ 0.91–0.98; iid-null q999 ≤ 0.095 across all null types tested;
    threshold floor = :data:`_W_BRANCH_PSI_FLOOR` = 0.15).
    """
    _M, B, d = draws_buffer_mc.shape
    M_A = M // 2
    M_B = M - M_A

    W_safe = jnp.maximum(W_diag, jnp.float32(1e-20))
    sigma_w = jnp.sqrt(W_safe)

    # Per-chain centering + whitening
    centered = draws_buffer_mc - chain_means[:, None, :]  # (M, B, d)
    whitened = centered / sigma_w[None, None, :]  # (M, B, d)

    step_mask = (jnp.arange(B) < n).astype(whitened.dtype)  # (B,)

    # Half-A chains: flatten chain-major (M_A*B, d) with valid-row mask
    Y_A = (step_mask[None, :, None] * whitened[:M_A]).reshape(M_A * B, d)
    # Half-B chains: flatten chain-major (M_B*B, d)
    Y_B = (step_mask[None, :, None] * whitened[M_A:]).reshape(M_B * B, d)

    # Finite guards
    Y_A = jnp.where(jnp.isfinite(Y_A), Y_A, jnp.zeros_like(Y_A))
    Y_B = jnp.where(jnp.isfinite(Y_B), Y_B, jnp.zeros_like(Y_B))

    # <R_A, R_B>_F = (1/(N_A*N_B)) * ||Y_A Y_B^T||_F^2 = (1/(N_A*N_B)) * ||cross_gram||_F^2
    # (uses the identity trace(A^T A B^T B) = ||A B^T||_F^2 for A=Y_A, B=Y_B)
    cross_gram = Y_A @ Y_B.T  # (M_A*B, M_B*B)
    inner_R_AB = jnp.sum(cross_gram**2)  # = N_A*N_B * <R_A,R_B>_F

    # Diagonal correction: sum_i (R_A)_{ii}(R_B)_{ii} = (1/(N_A*N_B)) * dot(d_A, d_B)
    # where d_A[i] = ||Y_A[:,i]||^2 = N_A * (R_A)_{ii}
    d_A = jnp.sum(Y_A**2, axis=0)  # (d,) = N_A * diag(R_A)
    d_B = jnp.sum(Y_B**2, axis=0)  # (d,) = N_B * diag(R_B)
    diag_corr = jnp.dot(d_A, d_B)  # = N_A*N_B * sum_i (R_A)_{ii}(R_B)_{ii}

    # Frobenius inner product of off-diagonal parts
    inner_C_AB = inner_R_AB - diag_corr  # = N_A*N_B * <C_A, C_B>_F

    # Norms: ||C_X||_F^2 = ||R_X||_F^2 - sum_i (R_X)_{ii}^2
    # ||R_A||_F^2 = (1/N_A^2) * trace((Y_A^T Y_A)^2) = (1/N_A^2) * ||Y_A Y_A^T||_F^2
    auto_gram_A = Y_A @ Y_A.T  # (M_A*B, M_A*B)
    auto_gram_B = Y_B @ Y_B.T  # (M_B*B, M_B*B)
    inner_R_AA = jnp.sum(auto_gram_A**2)  # = N_A^2 * ||R_A||_F^2
    inner_R_BB = jnp.sum(auto_gram_B**2)  # = N_B^2 * ||R_B||_F^2
    diag_AA = jnp.dot(d_A, d_A)  # = N_A^2 * sum_i (R_A)_{ii}^2
    diag_BB = jnp.dot(d_B, d_B)  # = N_B^2 * sum_i (R_B)_{ii}^2
    inner_C_AA = inner_R_AA - diag_AA  # = N_A^2 * ||C_A||_F^2
    inner_C_BB = inner_R_BB - diag_BB  # = N_B^2 * ||C_B||_F^2

    # Ψ = <C_A,C_B>_F / (||C_A||_F * ||C_B||_F)
    # Numerator: inner_C_AB / (N_A * N_B)
    # Denominator: sqrt(inner_C_AA / N_A^2) * sqrt(inner_C_BB / N_B^2)
    #            = sqrt(inner_C_AA * inner_C_BB) / (N_A * N_B)
    # Simplifies to: inner_C_AB / sqrt(max(inner_C_AA * inner_C_BB, eps))
    psi = inner_C_AB / jnp.maximum(
        jnp.sqrt(jnp.maximum(inner_C_AA * inner_C_BB, jnp.float32(1e-30))),
        jnp.float32(1e-20),
    )
    return psi.astype(jnp.float32)


def _compute_lag1_autocorr_top_dir(
    draws_buffer_mc: Array,
    chain_means: Array,
    W_diag: Array,
    top_eigvec: Array,
    n: Array,
    M: int,
) -> Array:
    """Pooled lag-1 autocorrelation of projections onto the top W-branch direction.

    The oscillation screen: a genuine under-resolved slow direction is
    diffusive (r₁ > 0), while integrator resonance alternates (r₁ < 0).
    Pooling M chains' lag-1 estimates reduces variance relative to single-chain.

    Gate: ``r1_top > _W_BRANCH_R1_TOL`` (= −0.2).  The gate is a LOWER bound
    only — no requirement of large positive r₁, or every isotropic-AC null
    direction would pass (at ρ=0.8, AR direction r₁ ≈ +0.8 — the screen
    must not penalize high-AC genuine slow directions).

    Parameters
    ----------
    draws_buffer_mc
        Shape ``(M, B, d)``.
    chain_means
        Shape ``(M, d)``.
    W_diag
        Shape ``(d,)`` — pooled within-chain diagonal variance.
    top_eigvec
        Shape ``(d,)`` — top right singular vector in whitened space (from
        :func:`_compute_pooled_within_spectrum`).
    n
        Dynamic fill count, int32 scalar.
    M
        Static chain count.

    Returns
    -------
    float32 scalar — pooled lag-1 autocorrelation (mean across M chains).
    """
    _M, B, d = draws_buffer_mc.shape
    W_safe = jnp.maximum(W_diag, jnp.float32(1e-20))
    sigma_w = jnp.sqrt(W_safe)

    # Per-chain centering + whitening, then project onto top_eigvec (whitened space)
    centered = draws_buffer_mc - chain_means[:, None, :]  # (M, B, d)
    whitened = centered / sigma_w[None, None, :]  # (M, B, d)
    proj = whitened @ top_eigvec  # (M, B)

    step_mask = (jnp.arange(B) < n).astype(proj.dtype)  # (B,)
    n_f = jnp.maximum(n.astype(proj.dtype), jnp.float32(2.0))

    def _chain_lag1(proj_m: Array) -> Array:
        """Lag-1 autocorrelation for one chain's projection time series."""
        mu = (step_mask * proj_m).sum() / n_f
        p_c = step_mask * (proj_m - mu)  # centered
        var = (p_c**2).sum() / jnp.maximum(n_f - jnp.float32(1.0), jnp.float32(1.0))
        lag1_cov = (
            p_c[:-1] * p_c[1:] * step_mask[:-1] * step_mask[1:]
        ).sum() / jnp.maximum(n_f - jnp.float32(2.0), jnp.float32(1.0))
        return lag1_cov / jnp.maximum(var, jnp.float32(1e-20))

    per_chain_r1 = jax.vmap(_chain_lag1)(proj)  # (M,)
    return per_chain_r1.mean().astype(jnp.float32)


def _build_pc_centered_time_major_pool(
    draws_buffer_mc: Array,
    grads_buffer_mc: Array,
    chain_means: Array,
    n: Array,
    M: int,
) -> tuple[Array, Array, Array]:
    """Per-chain-centered draws/grads in time-major layout with the valid-row mask.

    Subtracts each chain's own mean from positions and gradients, then
    reshapes from chain-major ``(M, B, d)`` to time-major ``(B*M, d)`` where
    the first ``n*M`` rows are valid (one real draw per chain per time step).

    This simultaneously fixes two v2 defects in the pooled R² path:
    1. **Padding bug**: chain-major layout puts zero-padded rows in the
       middle of the buffer (rows ``m*B + n .. m*B + B-1``), so the valid-row
       mask ``arange < n_pool`` incorrectly includes padding from early chains
       and drops later chains.  Time-major layout makes all valid rows
       contiguous at the start.
    2. **Between-chain inflation**: grand-centered pooled data conflates local
       curvature with transient chain-offset variance, producing wildly wrong
       R² on real multi-chain draws.  Per-chain centering removes the offset.

    Parameters
    ----------
    draws_buffer_mc
        Shape ``(M, B, d)``.
    grads_buffer_mc
        Shape ``(M, B, d)``.
    chain_means
        Shape ``(M, d)`` — per-chain position means (computed from valid rows).
    n
        Dynamic valid-row fill count, int32.
    M
        Static chain count.

    Returns
    -------
    pc_draws_tm : shape ``(B*M, d)`` — per-chain-centered draws, time-major.
    pc_grads_tm : shape ``(B*M, d)`` — per-chain-centered grads, time-major.
    step_mask_tm : shape ``(B*M,)`` — 1 for valid rows (first n*M), 0 otherwise.
    """
    _M, B, d = draws_buffer_mc.shape

    # Gradient means per chain (computed from valid rows)
    step_mask = (jnp.arange(B) < n).astype(draws_buffer_mc.dtype)  # (B,)
    n_f = jnp.maximum(n.astype(draws_buffer_mc.dtype), jnp.float32(1.0))

    def _chain_grad_mean(g_m: Array) -> Array:
        return (step_mask[:, None] * g_m).sum(0) / n_f

    grad_means = jax.vmap(_chain_grad_mean)(grads_buffer_mc)  # (M, d)

    # Per-chain centering
    pc_draws = draws_buffer_mc - chain_means[:, None, :]  # (M, B, d)
    pc_grads = grads_buffer_mc - grad_means[:, None, :]  # (M, B, d)

    # Time-major reshape: (M, B, d) → swap → (B, M, d) → reshape → (B*M, d)
    # Valid rows: first n*M (all M chains at time step t, for t = 0..n-1)
    pc_draws_tm = pc_draws.swapaxes(0, 1).reshape(B * M, d)
    pc_grads_tm = pc_grads.swapaxes(0, 1).reshape(B * M, d)
    # step_mask[t] = 1 for t < n; repeat M times → M*B mask, valid = first n*M
    step_mask_tm = jnp.repeat(step_mask, M)  # (B*M,)

    return pc_draws_tm, pc_grads_tm, step_mask_tm


# ---------------------------------------------------------------------------
# Core builder
# ---------------------------------------------------------------------------


def build_meta_adaptation_core(
    max_grad_budget: int,
    *,
    max_rank: int | None = None,
    gamma: float = 1e-5,
    cutoff: float = 2.0,
) -> MetricCore:
    """Build the meta-adaptation :class:`~blackjax.adaptation.metric_recipes.MetricCore`.

    Parameters
    ----------
    max_grad_budget
        Maximum total gradient budget (leapfrog evaluations).  Converted to
        warmup steps via ``_ASSUMED_AVG_LEAPFROGS_PER_STEP`` at Python time.
    max_rank
        Maximum low-rank rank; ``None`` uses :data:`_MAX_RANK_CAP`.
    gamma, cutoff
        Fisher-estimator parameters; defaults match ``fisher_low_rank`` recipe.

    Returns
    -------
    MetricCore
        Embeddable init/update/final bundle.
    """
    _max_rank: int = _MAX_RANK_CAP if max_rank is None else max_rank
    max_budget_steps: int = max(max_grad_budget // _ASSUMED_AVG_LEAPFROGS_PER_STEP, 1)

    def init(n_dims: int) -> MetaAdaptationCoreState:
        # Buffer sized to half the budget so the growing-window schedule's largest
        # window (≈ max_budget_steps * 0.4–0.5) does not overflow the buffer.
        # With the cap n=min(buffer_idx, B) in final(), overflow is still safe
        # (RESET policy keeps the most-recent B draws = the more-stationary tail).
        buf = min(max(max_budget_steps // 2, 256), max_budget_steps)
        buf = max(buf, 2 * (_max_rank + 1) * _MIN_TRAIN_K_RATIO)
        buf = min(buf, max_budget_steps)
        actual_rank = min(_max_rank, max(n_dims // 2, 1), _MAX_RANK_CAP)
        return MetaAdaptationCoreState(
            inverse_mass_matrix=LowRankInverseMassMatrix(
                sigma=jnp.ones(n_dims),
                U=jnp.zeros((n_dims, actual_rank)),
                lam=jnp.ones(actual_rank),
            ),
            mu_star=jnp.zeros(n_dims),
            draws_buffer=jnp.zeros((buf, n_dims)),
            grads_buffer=jnp.zeros((buf, n_dims)),
            buffer_idx=jnp.zeros((), dtype=jnp.int32),
            background_split=jnp.zeros((), dtype=jnp.int32),
            recompute_counter=jnp.zeros((), dtype=jnp.int32),
            has_escalated=jnp.zeros((), dtype=jnp.bool_),
            escalation_rank=jnp.zeros((), dtype=jnp.int32),
            s_gap_prev=jnp.array(float("nan"), dtype=jnp.float32),
            s_gap_curr=jnp.array(float("nan"), dtype=jnp.float32),
            r2_latest=jnp.array(float("nan"), dtype=jnp.float32),
            r2_mode=jnp.array(_R2_DEFERRED, dtype=jnp.int32),
            budget_used=jnp.zeros((), dtype=jnp.int32),
            converged_at_step=jnp.array(-1, dtype=jnp.int32),  # -1 = not yet converged
            prev_lam=jnp.ones(actual_rank, dtype=jnp.float32),
            airm_vel_prev=jnp.array(float("inf"), dtype=jnp.float32),
            airm_vel_curr=jnp.array(float("inf"), dtype=jnp.float32),
            is_slow_mixing=jnp.zeros((), dtype=jnp.bool_),
        )

    def update(
        state: MetaAdaptationCoreState,
        position: ArrayLikeTree,
        grad: ArrayLikeTree | None = None,
    ) -> MetaAdaptationCoreState:
        pos_flat, _ = fu.ravel_pytree(position)
        grad_flat, _ = fu.ravel_pytree(grad)
        B = state.draws_buffer.shape[0]
        idx = state.buffer_idx % B
        # Cast the column offset to idx.dtype — under x64 the literal 0 is int64
        # but idx is int32, causing "index arguments must be integers of the same
        # type" at trace time.
        col0 = jnp.zeros((), dtype=idx.dtype)
        return state._replace(
            draws_buffer=jax.lax.dynamic_update_slice(
                state.draws_buffer, pos_flat[None, :], (idx, col0)
            ),
            grads_buffer=jax.lax.dynamic_update_slice(
                state.grads_buffer, grad_flat[None, :], (idx, col0)
            ),
            buffer_idx=state.buffer_idx + 1,
            budget_used=state.budget_used + 1,
        )

    def final(state: MetaAdaptationCoreState) -> MetaAdaptationCoreState:
        """Window-boundary controller: compute signals → escalation decision →
        choose IMM → hard-reset buffer (v1 reset policy always).
        """
        B, d = state.draws_buffer.shape
        n = jnp.minimum(state.buffer_idx, jnp.int32(B))  # cap at B (Fix 4)
        actual_rank = state.inverse_mass_matrix.U.shape[1]  # static

        # Welford diagonal sigma: sample std from the buffer.  Used for:
        # (a) stay-diagonal metric (measured ≥0.62× welford vs 0.11× fisher-diag
        #     on classes where escalation is correctly refused), (b) S_gap whitening
        #     basis (where _S_MIN=2.0 and the anchors radon≈8.5/stoch_vol≈1.6 are
        #     calibrated).
        n_f = n.astype(state.draws_buffer.dtype)
        n_safe = jnp.maximum(n_f, 1.0)
        mask_w = (jnp.arange(B) < n).astype(state.draws_buffer.dtype)
        mean_x = (mask_w[:, None] * state.draws_buffer).sum(0) / n_safe
        var_x = (mask_w[:, None] * (state.draws_buffer - mean_x[None, :]) ** 2).sum(
            0
        ) / jnp.maximum(n_safe - 1.0, 1.0)
        sigma_welford = jnp.sqrt(jnp.maximum(var_x, 1e-10))

        # Fisher-LR metric (sigma + low-rank U, lam) — used only for the escalated
        # path.  Its sigma is informative for the escalated metric but measured worse
        # on the stay-diagonal classes (funnel 0.11×, stoch_vol 0.54× vs welford).
        sigma_lr, mu_star_new, U_lr, lam_lr = _compute_low_rank_metric(
            state.draws_buffer, state.grads_buffer, n, actual_rank, gamma, cutoff
        )

        # Stay-diagonal metric: Welford sigma, U=0, lam=1.
        # Recovers-classical anchor: this IS the nutpie welford-diagonal scheme that
        # the controller converges to on isotropic/curvature targets.
        diag_imm = LowRankInverseMassMatrix(
            sigma=sigma_welford,
            U=jnp.zeros((d, actual_rank), dtype=sigma_welford.dtype),
            lam=jnp.ones(actual_rank, dtype=sigma_welford.dtype),
        )
        # Escalated metric: full Fisher-LR (measured payoffs are relative to welford
        # baseline, so the two-phase design diag(welford) → rank-k(fisher) is correct).
        lr_imm = LowRankInverseMassMatrix(sigma=sigma_lr, U=U_lr, lam=lam_lr)

        # Whitened-residual spectrum on the WELFORD basis (where _S_MIN and anchors
        # were calibrated; the Fisher whitening inflates S_gap vs the calibrated scale).
        eigenvalues, U_k = _compute_whitened_spectrum(
            state.draws_buffer, sigma_welford, n, actual_rank
        )
        k_new = _choose_rank(eigenvalues, n, actual_rank, cutoff)
        s_gap_new = _compute_s_gap(eigenvalues, k_new)

        # Score-linearity R² on the Welford-whitened space.
        # (mode is observed from the taken branch — not inferred post-hoc)
        r2_new, mode_new = _compute_r2_score_linearity(
            state.draws_buffer, state.grads_buffer, sigma_welford, n, U_k, actual_rank
        )
        # FOLD-IN 6: clip R² to [-10, 1]; values below -10 → NaN (deferred path).
        r2_new = jnp.where(
            r2_new < jnp.float32(-10.0),
            jnp.array(float("nan"), dtype=r2_new.dtype),
            jnp.clip(r2_new, max=jnp.float32(1.0)),
        )

        # Transient-mixing class (reported in verdict; v1 always uses RESET).
        is_slow = _compute_transient_mixing_signal(state.draws_buffer, sigma_welford, n)

        # ---- Escalation decision (all three gates must pass) ----
        r2_gate = (
            r2_new >= _R_MIN
        )  # False when r2_new is NaN (JAX comparison semantics)
        s_gap_prev_valid = ~jnp.isnan(state.s_gap_curr)
        relative_change = jnp.abs(s_gap_new - state.s_gap_curr) / jnp.maximum(
            s_gap_new, 1e-10
        )
        s_gap_gate = (
            (s_gap_new >= _S_MIN)
            & s_gap_prev_valid
            & (relative_change < _S_GAP_STABILITY_TOL)
        )
        budget_remaining = jnp.int32(max_budget_steps) - state.budget_used.astype(
            jnp.int32
        )
        deadline_ok = budget_remaining >= 2 * k_new.astype(jnp.int32) + jnp.int32(
            _STEP_SIZE_READAPT_BUFFER
        )

        escalate_now = ~state.has_escalated & r2_gate & s_gap_gate & deadline_ok
        new_has_escalated = state.has_escalated | escalate_now
        new_escalation_rank = jnp.where(escalate_now, k_new, state.escalation_rank)

        chosen_imm = jax.lax.cond(new_has_escalated, lambda: lr_imm, lambda: diag_imm)
        chosen_mu = jax.lax.cond(
            new_has_escalated, lambda: mu_star_new, lambda: jnp.zeros_like(mu_star_new)
        )

        # ---- AIRM velocity proxy + advisory convergence step ----
        # Cast prev_lam to lam_lr dtype for the norm so the subtraction is
        # numerically sound; then cast back to float32 before storing.
        lam_diff = jnp.linalg.norm(lam_lr - state.prev_lam.astype(lam_lr.dtype))
        # Controller-signal scalars are always stored as float32: they are gate
        # comparisons and AIRM velocities that don't benefit from float64 precision.
        # Explicit casts here are required so that both branches of the
        # jax.lax.cond in staged_adaptation produce identical types under x64
        # (the false branch returns the init-time float32 carry unchanged).
        lam_diff_f32 = lam_diff.astype(jnp.float32)
        new_airm_vel_prev = state.airm_vel_curr  # already float32
        new_airm_vel_curr = jnp.where(
            new_has_escalated, lam_diff_f32, state.airm_vel_curr
        )
        # Set converged_at_step once (monotone, like has_escalated) when AIRM criterion fires.
        airm_converged_now = (
            new_has_escalated
            & (new_airm_vel_curr < _AIRM_VELOCITY_TOL)
            & (new_airm_vel_prev < _AIRM_VELOCITY_TOL)
        )
        new_converged_at = jnp.where(
            (state.converged_at_step < 0) & airm_converged_now,
            state.budget_used,
            state.converged_at_step,
        )

        return MetaAdaptationCoreState(
            inverse_mass_matrix=chosen_imm,
            mu_star=chosen_mu,
            draws_buffer=jnp.zeros_like(state.draws_buffer),
            grads_buffer=jnp.zeros_like(state.grads_buffer),
            buffer_idx=jnp.zeros_like(state.buffer_idx),
            background_split=jnp.zeros_like(state.background_split),
            recompute_counter=jnp.zeros_like(state.recompute_counter),
            has_escalated=new_has_escalated,
            escalation_rank=new_escalation_rank,
            s_gap_prev=state.s_gap_curr,  # float32, propagated from prior window
            s_gap_curr=s_gap_new.astype(jnp.float32),
            r2_latest=r2_new.astype(jnp.float32),
            r2_mode=mode_new,
            budget_used=state.budget_used,
            converged_at_step=new_converged_at,
            prev_lam=lam_lr.astype(jnp.float32),
            airm_vel_prev=new_airm_vel_prev,
            airm_vel_curr=new_airm_vel_curr,
            is_slow_mixing=is_slow,
        )

    return MetricCore(init=init, update=update, final=final)


def build_multi_chain_meta_core(
    max_grad_budget: int,
    n_chains: int = _MULTI_CHAIN_DEFAULT_N_CHAINS,
    *,
    max_rank: int | None = None,
    gamma: float = 1e-5,
    cutoff: float = 2.0,
) -> MetricCore:
    """Build the multi-chain meta-adaptation :class:`~blackjax.adaptation.metric_recipes.MetricCore`.

    Runs M independent chains sharing one adapted metric; the escalation
    decision uses pooled M-chain information instead of a single-chain
    stability check.  The pooled between-chain signal makes the escalation
    decision robust to seed variation for near-edge posterior structures.

    The multi-chain gate (replaces the single-chain S_gap-stability check).
    Five conditions must all hold to escalate:

    1. **Magnitude.** Top eigenvalue of the between-chain T matrix exceeds the
       detection edge ``(1 + √(d/(M−1)))²`` (M−1 dof, grand-mean constraint).
    2. **Collinearity.** Fraction of total between-chain scatter in the top
       singular direction f₁ ≥ :data:`_MC_COLLINEARITY_TOL`.  Genuine slow
       directions produce near-rank-1 concentration (f₁→1); isotropic spurious
       scatter gives f₁ ≈ 1/(M−1).
    3. **Leave-one-out.** Detection must survive dropping any single chain,
       preventing a single outlier chain from driving the verdict.  Leave-two-out
       (dropping any pair) is subsumed by the collinearity + unimodality conjunction
       for the aligned-pair threat model and is deferred to v2.1.
    4. **Support floor.** At least one spike is admitted (k ≥ 1).
    5. **Unimodality guard.** Gap-statistic on the projected chain-means must
       not flag mode-split; mode-separated chains are deferred to the ensemble
       (Paper-3 scope) and reported via ``deferred_to_ensemble=True`` in the
       verdict.

    Plus R² curvature gate and budget deadline (same as single-chain).

    Budget re-allocation: ``max_grad_budget`` is the TOTAL gradient budget,
    shared across all M chains.  Providing ``n_chains=M`` overdispersed
    starting positions to ``run()`` causes each chain to run for
    ``total // M`` leapfrog evaluations — the total cost equals the
    single-chain budget, not M× it.

    For ``n_chains=1`` use :func:`build_meta_adaptation_core` directly to
    obtain exact single-chain (v1) behaviour; the ``staged_adaptation`` engine
    routes to it automatically when ``n_chains=1``.

    Parameters
    ----------
    max_grad_budget
        Maximum total gradient budget (leapfrog evaluations) across all M chains.
    n_chains
        Number of independent chains.  Must be ≥ 2.  Defaults to
        :data:`_MULTI_CHAIN_DEFAULT_N_CHAINS` (8).
    max_rank, gamma, cutoff
        Same as :func:`build_meta_adaptation_core`.

    Returns
    -------
    MetricCore
        Embeddable init/update/final bundle.  ``update`` expects ``position``
        of shape ``(n_chains, d)`` and ``grad`` of shape ``(n_chains, d)``.
    """
    if n_chains < 2:
        raise ValueError(
            f"build_multi_chain_meta_core: n_chains must be >= 2, got {n_chains}. "
            "For single-chain use, call build_meta_adaptation_core instead."
        )
    if n_chains < _MC_MIN_CHAINS:
        import warnings

        warnings.warn(
            f"build_multi_chain_meta_core: n_chains={n_chains} < {_MC_MIN_CHAINS} "
            f"(the recommended minimum).  At M<{_MC_MIN_CHAINS} the collinearity "
            "null-margin is unsafe (iid null f₁ can exceed the 0.70 threshold) and "
            "the unimodality gap-ratio for a perfect 2-cluster split falls below "
            "the detection threshold.  Use n_chains >= 6 (default 8) for reliable "
            "gate behaviour.",
            stacklevel=2,
        )
    _max_rank: int = _MAX_RANK_CAP if max_rank is None else max_rank
    # Per-chain step budget: total budget divided across M chains
    max_budget_steps_total: int = max(
        max_grad_budget // _ASSUMED_AVG_LEAPFROGS_PER_STEP, 1
    )
    max_budget_steps_per_chain: int = max(max_budget_steps_total // n_chains, 1)

    def init(n_dims: int) -> MultiChainMetaAdaptationCoreState:
        # buf_size: sized to hold the largest single-chain window.  With the
        # budget split across n_chains, each chain runs max_budget_steps_per_chain
        # steps, so buf_size is half that (growing-window largest window ≈ 40–50%).
        buf = min(max(max_budget_steps_per_chain // 2, 256), max_budget_steps_per_chain)
        buf = max(buf, 2 * (_max_rank + 1) * _MIN_TRAIN_K_RATIO)
        buf = min(buf, max_budget_steps_per_chain)
        actual_rank = min(_max_rank, max(n_dims // 2, 1), _MAX_RANK_CAP)
        return MultiChainMetaAdaptationCoreState(
            inverse_mass_matrix=LowRankInverseMassMatrix(
                sigma=jnp.ones(n_dims),
                U=jnp.zeros((n_dims, actual_rank)),
                lam=jnp.ones(actual_rank),
            ),
            mu_star=jnp.zeros(n_dims),
            draws_buffer=jnp.zeros((n_chains, buf, n_dims)),
            grads_buffer=jnp.zeros((n_chains, buf, n_dims)),
            buffer_idx=jnp.zeros((), dtype=jnp.int32),
            background_split=jnp.zeros((), dtype=jnp.int32),
            recompute_counter=jnp.zeros((), dtype=jnp.int32),
            has_escalated=jnp.zeros((), dtype=jnp.bool_),
            escalation_rank=jnp.zeros((), dtype=jnp.int32),
            s_gap_prev=jnp.array(float("nan"), dtype=jnp.float32),
            s_gap_curr=jnp.array(float("nan"), dtype=jnp.float32),
            r2_latest=jnp.array(float("nan"), dtype=jnp.float32),
            r2_mode=jnp.array(_R2_DEFERRED, dtype=jnp.int32),
            budget_used=jnp.zeros((), dtype=jnp.int32),
            converged_at_step=jnp.array(-1, dtype=jnp.int32),
            prev_lam=jnp.ones(actual_rank, dtype=jnp.float32),
            airm_vel_prev=jnp.array(float("inf"), dtype=jnp.float32),
            airm_vel_curr=jnp.array(float("inf"), dtype=jnp.float32),
            is_slow_mixing=jnp.zeros((), dtype=jnp.bool_),
            chain_collinearity=jnp.array(float("nan"), dtype=jnp.float32),
            unimodality_passed=jnp.ones((), dtype=jnp.bool_),  # True until first window
            deferred_to_ensemble=jnp.zeros((), dtype=jnp.bool_),
            # v2.1 new fields — NaN / 0 until first window
            within_lam1=jnp.array(float("nan"), dtype=jnp.float32),
            chain_consistency_psi=jnp.array(float("nan"), dtype=jnp.float32),
            r1_top=jnp.array(float("nan"), dtype=jnp.float32),
            detection_branch=jnp.array(_DETECTION_BRANCH_NONE, dtype=jnp.int32),
            unimodality_flag_count=jnp.zeros((), dtype=jnp.int32),
        )

    def update(
        state: MultiChainMetaAdaptationCoreState,
        positions: Array,
        grads: Array,
    ) -> MultiChainMetaAdaptationCoreState:
        """Accumulate one step of M chains into the per-chain buffers.

        Parameters
        ----------
        positions
            Shape ``(n_chains, d)`` — flat 2-D array, one row per chain.
        grads
            Shape ``(n_chains, d)`` — flat 2-D array, one row per chain.
        """
        # positions and grads are already flat (n_chains, d) arrays.
        # Ravel each chain's row in case a pytree is passed (no-op for 1-D rows).
        B = state.draws_buffer.shape[1]  # buf_size, static
        idx = state.buffer_idx % B
        col0 = jnp.zeros((), dtype=idx.dtype)

        def _write_chain(draws_m: Array, grads_m: Array, pos_m: Array, grad_m: Array):
            pos_flat, _ = fu.ravel_pytree(pos_m)
            grad_flat, _ = fu.ravel_pytree(grad_m)
            new_d = jax.lax.dynamic_update_slice(
                draws_m, pos_flat[None, :], (idx, col0)
            )
            new_g = jax.lax.dynamic_update_slice(
                grads_m, grad_flat[None, :], (idx, col0)
            )
            return new_d, new_g

        new_draws_buffer, new_grads_buffer = jax.vmap(_write_chain)(
            state.draws_buffer, state.grads_buffer, positions, grads
        )

        return state._replace(
            draws_buffer=new_draws_buffer,
            grads_buffer=new_grads_buffer,
            buffer_idx=state.buffer_idx + 1,
            budget_used=state.budget_used + n_chains,
        )

    def final(
        state: MultiChainMetaAdaptationCoreState,
    ) -> MultiChainMetaAdaptationCoreState:
        """Window-boundary controller: W-branch ∪ T-branch detection (v2.1).

        v2.1 two-branch union:
        - **W-branch (primary, deep spread):** pools per-chain-centered,
          within-chain-whitened residuals into a M(n−1)-dof spectrum; escalates
          when λ₁ clears the MP null edge AND the cross-chain consistency Ψ
          confirms AND the oscillation screen passes AND R² ≥ 0.5.
        - **T-branch (pure spike):** v2's between-means detector kept for the
          fully-unmixed regime.  Unimodality guard recalibrated to q99 (4.54
          at M=8) with 2-window confirmation; latch is non-monotone.

        Router fix: per-chain-centered draws/grads in time-major layout (first
        n*M rows valid), replacing the v2 chain-major reshape that caused
        r2 ∈ {−1.96×10¹⁷, +1.000} on real multi-chain draws.
        """
        M_stat, B, d = state.draws_buffer.shape  # all static
        n = jnp.minimum(state.buffer_idx, jnp.int32(B))
        actual_rank = state.inverse_mass_matrix.U.shape[1]  # static

        # ---- Within-chain statistics ----
        chain_means, W_diag = _compute_within_chain_stats(state.draws_buffer, n)
        grand_mean = chain_means.mean(0)  # (d,)
        W_safe = jnp.maximum(W_diag, jnp.float32(1e-20))
        sigma_w_diag = jnp.sqrt(W_safe)

        # ---- T-branch: between-chain detection via M×M Gram ----
        dof = M_stat - 1  # rank of T (grand-mean constraint)
        edge_full = _mc_detection_edge(d, dof)  # (1+√(d/(M−1)))²
        edge_loo = _mc_detection_edge(d, max(dof - 1, 1))

        T_eigenvalues, V_top, f1 = _between_chain_detection(
            chain_means, W_diag, n, M_stat, d
        )
        T_top = T_eigenvalues[0]

        k_raw = (T_eigenvalues > jnp.float32(edge_full)).sum().astype(jnp.int32)
        k_new = jnp.minimum(
            jnp.minimum(k_raw, jnp.int32(max(dof - 1, 1))), jnp.int32(actual_rank)
        )

        t_magnitude = T_top > jnp.float32(edge_full)
        t_collinearity = f1 >= jnp.float32(_MC_COLLINEARITY_TOL)
        t_loo = _loo_detection_passes(chain_means, W_diag, n, M_stat, d, edge_loo)
        t_support = k_new >= jnp.int32(1)

        # ---- T-branch Gate 5: unimodality (gap-stat, corroborator after BLOCKER-3) ----
        # Unwhiten top T direction: V_top[:, 0] is in whitened space → unwhiten by sigma_w
        e_unnorm = sigma_w_diag * V_top[:, 0]  # (d,) in original space
        e_norm = jnp.linalg.norm(e_unnorm)
        e_dir = e_unnorm / jnp.maximum(e_norm, jnp.float32(1e-10))
        is_unimodal, _gap_ratio = _unimodality_gap_stat(chain_means, e_dir, M_stat)
        # NOTE: new_flag_count is computed AFTER BLOCKER-3 (mode-consistency + contraction)
        # because the primary multimodality signal is any_mode_flag (mode-consistency),
        # not the gap-stat alone.  The flag counter uses multimodality_signal (either signal)
        # to count consecutive windows for the 2-window confirmation gate.

        # T pre-unimodality conjunction (four non-unimodality gates)
        t_pre_uni = t_magnitude & t_collinearity & t_loo & t_support

        # ---- Per-chain-centered pooled buffers in time-major layout ----
        # Fixes the v2 padding bug (chain-major zeros contaminated R² / Fisher paths)
        # and removes between-chain transient inflation from the curvature gate.
        pc_draws_tm, pc_grads_tm, step_mask_tm = _build_pc_centered_time_major_pool(
            state.draws_buffer, state.grads_buffer, chain_means, n, M_stat
        )
        n_pool = n.astype(jnp.int32) * jnp.int32(M_stat)
        # step_mask_all for geometric_mean_deploy_scale (time-major, repeat)
        step_mask = (jnp.arange(B) < n).astype(state.draws_buffer.dtype)  # (B,)
        step_mask_all = jnp.repeat(step_mask, M_stat)  # (B*M,)

        # Finite guard before SVD/eigh (sanitise NaN/Inf from near-zero W_diag dims)
        pc_draws_safe = jnp.where(
            jnp.isfinite(pc_draws_tm), pc_draws_tm, jnp.zeros_like(pc_draws_tm)
        )
        pc_grads_safe = jnp.where(
            jnp.isfinite(pc_grads_tm), pc_grads_tm, jnp.zeros_like(pc_grads_tm)
        )

        # ---- Fisher-LR metric on per-chain-centered pooled buffers ----
        # Used by the W-branch metric and by the T-branch sigma baseline.
        sigma_lr, mu_star_new, U_lr, lam_lr = _compute_low_rank_metric(
            pc_draws_safe, pc_grads_safe, n_pool, actual_rank, gamma, cutoff
        )

        # ---- R² curvature gate (per-chain-centered, time-major) ----
        # Per-chain centering removes between-chain variance, so the shared-slope
        # fit measures local metric-fixability (curvature), not global linearity.
        _, U_k_pooled = _compute_whitened_spectrum(
            pc_draws_safe, sigma_w_diag, n_pool, actual_rank
        )
        r2_new, mode_new = _compute_r2_score_linearity(
            pc_draws_safe, pc_grads_safe, sigma_w_diag, n_pool, U_k_pooled, actual_rank
        )
        # FOLD-IN 6: clip R² to [-10, 1] before any gate or carry.
        # Fit values below -10 are garbage (starved / degenerate data) and should
        # trigger the deferred path (NaN), not pollute the carry with extreme negatives.
        r2_new = jnp.where(
            r2_new < jnp.float32(-10.0),
            jnp.array(float("nan"), dtype=r2_new.dtype),
            jnp.clip(r2_new, max=jnp.float32(1.0)),
        )

        # W-branch r² gate uses the raw (pre-GAIN-override) R² so that a genuine
        # linear Gaussian with GAIN ≈ 0 in the projected tier is not blocked.
        # The W-branch question is "is the metric fixable?" — answered by the
        # original per-chain-centered projected fit, not by the slope-heterogeneity test.
        r2_gate_w = r2_new >= jnp.float32(_R_MIN)  # for W-branch only

        # BLOCKER-2: projected-tier GAIN+abstain router for T-branch routing.
        # When _R2_PROJECTED was taken (n_pool < 16d), the raw r2_new is
        # meaningless at k<<d (can be ≤ 0 for valid curvature targets → false
        # reparam_suggested in verdict; stoch_vol r2=-0.117, radon r2=-0.981).
        # Override: compute slope-heterogeneity GAIN = R²_perchain − R²_shared.
        #   abstain (NaN) → r2_routing = NaN → no T escalation / no reparam
        #   GAIN > 0.3 AND R²_pc ≥ 0.5 → r2_routing = R²_pc → T can escalate
        #   GAIN ≤ 0.3 (no evidence) → r2_routing = NaN → diagonal, no T action
        # W-branch is NOT gated on r2_routing (uses r2_gate_w from the raw R²).
        def _gain_r2_override() -> Array:
            gain_proj, r2_pc_proj = _compute_projected_gain_r2_mc(
                pc_draws_safe, pc_grads_safe, sigma_w_diag, n, M_stat, U_k_pooled
            )
            reparam_signal = (
                jnp.isfinite(gain_proj)
                & (gain_proj > jnp.float32(_GAIN_THRESHOLD))
                & (r2_pc_proj >= jnp.float32(_R_MIN))
            )
            return jnp.where(
                reparam_signal,
                r2_pc_proj,
                jnp.array(float("nan"), dtype=r2_new.dtype),
            )

        r2_routing = jax.lax.cond(
            mode_new == jnp.int32(_R2_PROJECTED),
            _gain_r2_override,
            lambda: r2_new,
        )
        r2_gate = r2_routing >= jnp.float32(
            _R_MIN
        )  # for T-branch + verdict; False when NaN

        # ---- W-branch: pooled within-chain whiteness detector ----
        lam1_w, top_eigvec_w = _compute_pooled_within_spectrum(
            state.draws_buffer, chain_means, W_diag, n, M_stat, actual_rank
        )
        w_edge = _w_branch_null_edge(M_stat, n, d)
        w_magnitude = lam1_w > w_edge

        psi_w = _compute_chain_consistency_psi(
            state.draws_buffer, chain_means, W_diag, n, M_stat
        )
        # FOLD-IN 4: adaptive Ψ threshold max(3*q99_null(M,n,d), 0.15).
        # The flat 0.15 causes 16-17% null leak at d=10; the calibrated table
        # corrects this without changing the floor for large d (sparse N).
        w_psi_thresh = _w_branch_psi_threshold(M_stat, n, d)
        w_psi_gate = psi_w > w_psi_thresh

        r1_w = _compute_lag1_autocorr_top_dir(
            state.draws_buffer, chain_means, W_diag, top_eigvec_w, n, M_stat
        )
        w_r1_gate = r1_w > jnp.float32(_W_BRANCH_R1_TOL)

        # ---- Budget deadline ----
        budget_remaining = jnp.int32(max_budget_steps_per_chain) - (
            state.budget_used.astype(jnp.int32) // jnp.int32(n_chains)
        )
        deadline_ok = budget_remaining >= 2 * k_new.astype(jnp.int32) + jnp.int32(
            _STEP_SIZE_READAPT_BUFFER
        )

        # ---- W-branch conjunction ----
        # No unimodality gate: per-chain centering makes W structurally mode-blind.
        # No support gate: W detects continuous spread, not discrete spike count.
        # Uses r2_gate_w (raw R², pre-GAIN-override) so that a genuine linear
        # Gaussian with low projected GAIN is not incorrectly blocked here.
        escalate_W = (
            ~state.has_escalated
            & w_magnitude
            & w_psi_gate
            & w_r1_gate
            & r2_gate_w
            & deadline_ok
        )

        # ---- BLOCKER-3: mode-consistency flag + contraction stat (T-branch guard) ----
        # Mode-consistency (BLOCKER-3): per direction e_j, flag iff
        # (R²_local(e_j) − R²_global(e_j) > 0.3) AND (R²_local(e_j) ≥ 0.5).
        # Replaces the gap-stat as the primary multimodality signal — k-mode-agnostic.
        any_mode_flag = _compute_mode_consistency_flag(
            pc_draws_safe,
            pc_grads_safe,
            state.grads_buffer,  # raw (M, B, d) for grand-centered R²_global
            chain_means,
            grand_mean,
            V_top,
            sigma_w_diag,
            T_eigenvalues,
            jnp.float32(edge_full),
            n,
            M_stat,
        )
        # Contraction stat (BLOCKER-3): per-chain split-half drift t.
        # t < -2.365 at M=8 → chains are converging → unimodal-safe, T can escalate.
        t_contr = _compute_contraction_stat(
            state.draws_buffer, chain_means, grand_mean, n, M_stat
        )
        _T_CONTR_CRIT = jnp.float32(-2.365)  # one-sided at α=2.5%, M=8 dof=7
        is_converging = t_contr < _T_CONTR_CRIT

        # ---- T-branch three-way (BLOCKER-3 fix, replaces binary gap-stat gate) ----
        # (i) Converging → chains drifting toward grand mean → unimodal-safe → escalate
        #     if pre-uni gates pass (override gap-stat).
        # (ii) Any mode_flag AND NOT converging → defer (genuine or uncertain multimodality).
        # (iii) Inconclusive (not converging, no mode flag) → existing gap-stat gate (is_unimodal).
        t_unimodality_override = (
            is_converging  # (i): converging overrides gap-stat block
        )
        t_unimodality_default = (
            is_unimodal & ~any_mode_flag
        )  # (iii): gap-stat + no flag
        t_unimodality = t_unimodality_override | t_unimodality_default

        mc_detection_T = t_pre_uni & t_unimodality
        escalate_T = ~state.has_escalated & r2_gate & mc_detection_T & deadline_ok

        # ---- Union escalation (computed before deferred to enable the gate) ----
        escalate_now = escalate_W | escalate_T
        new_has_escalated = state.has_escalated | escalate_now

        # ---- Scoped latch: deferred_to_ensemble (non-monotone, v2.1 rule) ----
        # Deferred is set when T-branch wants to escalate (t_pre_uni=True) but the
        # multimodality guard fires (any_mode_flag OR gap-stat flags), AND this has
        # been true for _MC_UNIMODALITY_CONFIRM_WINDOWS=2 consecutive windows.
        # The 2-window confirmation prevents transient flags from triggering P1→P3.
        # Mode-consistency (any_mode_flag) is the primary multimodality signal;
        # gap-stat (~is_unimodal) is a corroborating signal.  Either alone sets
        # the flag counter; both together confirm split faster.
        # Non-monotone: recomputed each window; if chains merge / flags clear, resets.
        # Post-escalation: deferred=False once escalated (moot after metric deployed). ✓
        # Impossible combo (route=low_rank ∧ deferred ∧ detection_branch=between_means):
        # T escalation requires t_unimodality=True → flag_count resets to 0 →
        # confirmed_split=False → new_deferred=False. ✓
        # W-escalation + deferred is LEGAL (cross-branch coexistence). ✓
        # Primary = mode-consistency (any_mode_flag); corroborator = gap-stat (~is_unimodal).
        # Either alone increments the flag counter for the 2-window confirmation gate.
        multimodality_signal = any_mode_flag | ~is_unimodal
        new_flag_count = jnp.where(
            multimodality_signal,
            state.unimodality_flag_count + jnp.int32(1),
            jnp.int32(0),
        )
        # Confirmed split requires _MC_UNIMODALITY_CONFIRM_WINDOWS (=2) consecutive windows
        unimodality_confirmed_split = new_flag_count >= jnp.int32(
            _MC_UNIMODALITY_CONFIRM_WINDOWS
        )

        new_deferred = (
            t_pre_uni
            & multimodality_signal
            & unimodality_confirmed_split
            & r2_gate
            & ~new_has_escalated
        )
        new_escalation_rank = jnp.where(escalate_now, k_new, state.escalation_rank)

        # Detection branch code (carry from last firing window if no fire this window)
        branch_when_fires = jnp.where(
            escalate_W & escalate_T,
            jnp.int32(_DETECTION_BRANCH_BOTH),
            jnp.where(
                escalate_W,
                jnp.int32(_DETECTION_BRANCH_POOLED_WITHIN),
                jnp.int32(_DETECTION_BRANCH_BETWEEN_MEANS),
            ),
        )
        new_detection_branch = jnp.where(
            escalate_now, branch_when_fires, state.detection_branch
        )

        # ---- Metric selection ----
        # W fires: full Fisher-LR on per-chain-centered pooled buffers.
        # T fires (alone): Fisher-LR sigma + rank-1 geometric-mean slow-dir correction.
        # Neither fires post-escalation: carry the branch-appropriate LR metric.

        # T-branch escalated metric (v2 rank-1 update, unchanged)
        sigma_sq_deploy = _geometric_mean_deploy_scale(
            chain_means, pc_grads_safe, step_mask_all, grand_mean, e_dir, n_pool, M_stat
        )
        sigma_lr_e_sq = jnp.maximum(
            ((sigma_lr**2) * (e_dir**2)).sum(), jnp.float32(1e-20)
        )
        lam_slow = sigma_sq_deploy / sigma_lr_e_sq
        U_slow = e_dir[:, None]  # (d, 1)
        lam_slow_vec = jnp.concatenate(
            [lam_slow[None], jnp.ones(actual_rank - 1, dtype=lam_slow.dtype)]
        )
        t_lr_imm = LowRankInverseMassMatrix(
            sigma=sigma_lr,
            U=jnp.concatenate([U_slow, U_lr[:, 1:]], axis=1),
            lam=lam_slow_vec,
        )

        # W-branch metric: full Fisher-LR (all directions, no rank-1 truncation)
        w_lr_imm = LowRankInverseMassMatrix(sigma=sigma_lr, U=U_lr, lam=lam_lr)

        # Stay-diagonal metric (pre-escalation)
        diag_imm = LowRankInverseMassMatrix(
            sigma=sigma_w_diag,
            U=jnp.zeros((d, actual_rank), dtype=sigma_w_diag.dtype),
            lam=jnp.ones(actual_rank, dtype=sigma_w_diag.dtype),
        )

        # Select escalated metric: W fires (now or previously) → W metric; else T metric.
        prev_was_w = (
            new_detection_branch == jnp.int32(_DETECTION_BRANCH_POOLED_WITHIN)
        ) | (new_detection_branch == jnp.int32(_DETECTION_BRANCH_BOTH))
        escalated_imm = jax.lax.cond(prev_was_w, lambda: w_lr_imm, lambda: t_lr_imm)
        chosen_imm = jax.lax.cond(
            new_has_escalated, lambda: escalated_imm, lambda: diag_imm
        )
        chosen_mu = jax.lax.cond(
            new_has_escalated, lambda: mu_star_new, lambda: jnp.zeros_like(mu_star_new)
        )

        # ---- AIRM velocity proxy (on per-chain-centered pooled lam) ----
        lam_diff = jnp.linalg.norm(lam_lr - state.prev_lam.astype(lam_lr.dtype))
        lam_diff_f32 = lam_diff.astype(jnp.float32)
        new_airm_vel_prev = state.airm_vel_curr
        new_airm_vel_curr = jnp.where(
            new_has_escalated, lam_diff_f32, state.airm_vel_curr
        )
        airm_converged_now = (
            new_has_escalated
            & (new_airm_vel_curr < _AIRM_VELOCITY_TOL)
            & (new_airm_vel_prev < _AIRM_VELOCITY_TOL)
        )
        new_converged_at = jnp.where(
            (state.converged_at_step < 0) & airm_converged_now,
            state.budget_used,
            state.converged_at_step,
        )

        return MultiChainMetaAdaptationCoreState(
            inverse_mass_matrix=chosen_imm,
            mu_star=chosen_mu,
            draws_buffer=jnp.zeros_like(state.draws_buffer),
            grads_buffer=jnp.zeros_like(state.grads_buffer),
            buffer_idx=jnp.zeros_like(state.buffer_idx),
            background_split=jnp.zeros_like(state.background_split),
            recompute_counter=jnp.zeros_like(state.recompute_counter),
            has_escalated=new_has_escalated,
            escalation_rank=new_escalation_rank,
            s_gap_prev=state.s_gap_curr,  # NaN chain; diagnostic compat
            s_gap_curr=jnp.array(float("nan"), dtype=jnp.float32),
            r2_latest=r2_routing.astype(jnp.float32),  # GAIN-corrected for verdict
            r2_mode=mode_new,
            budget_used=state.budget_used,
            converged_at_step=new_converged_at,
            prev_lam=lam_lr.astype(jnp.float32),
            airm_vel_prev=new_airm_vel_prev,
            airm_vel_curr=new_airm_vel_curr,
            is_slow_mixing=jnp.zeros((), dtype=jnp.bool_),
            chain_collinearity=f1,
            unimodality_passed=is_unimodal,
            deferred_to_ensemble=new_deferred,
            # v2.1 new fields
            within_lam1=lam1_w.astype(jnp.float32),
            chain_consistency_psi=psi_w.astype(jnp.float32),
            r1_top=r1_w.astype(jnp.float32),
            detection_branch=new_detection_branch,
            unimodality_flag_count=new_flag_count,
        )

    return MetricCore(init=init, update=update, final=final)


# ---------------------------------------------------------------------------
# Verdict extraction — Python-side, runs after the warmup scan
# ---------------------------------------------------------------------------


def extract_meta_verdict(
    final_state: MetaAdaptationCoreState,
    max_grad_budget: int,
    num_warmup_steps: int,
    adaptation_info: Any = None,
) -> MetaAdaptationVerdict:
    """Build a :class:`MetaAdaptationVerdict` from the final core state.

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
    """Build a :class:`MetaAdaptationVerdict` from a multi-chain final core state.

    Drop-in counterpart of :func:`extract_meta_verdict` for the
    :class:`MultiChainMetaAdaptationCoreState` produced by
    :func:`build_multi_chain_meta_core`.

    Parameters
    ----------
    final_state
        Final :class:`MultiChainMetaAdaptationCoreState` after ``warmup.run()``.
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
