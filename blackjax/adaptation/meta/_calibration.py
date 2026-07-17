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
"""Calibration surface for the meta-adaptation controller.

ALL module-level constants and the swappable calibration functions live here.
This file is the V-phase calibration surface: to recalibrate any gate
threshold, change a constant or swap one of the functions below — no other
file needs to change.

Constants
---------
Single-chain gates: _R_MIN, _S_MIN, _S_GAP_STABILITY_TOL, _MIN_TRAIN_D_RATIO,
    _MIN_TRAIN_K_RATIO, _AIRM_VELOCITY_TOL, _STEP_SIZE_READAPT_BUFFER,
    _ASSUMED_AVG_LEAPFROGS_PER_STEP, _TRANSIENT_MIXING_THRESHOLD, _MAX_RANK_CAP,
    _LAM_NONTRIVIAL_TOL.

Multi-chain gates: _MULTI_CHAIN_DEFAULT_N_CHAINS, _MC_MIN_CHAINS,
    _MC_COLLINEARITY_TOL, _MC_UNIMODALITY_GAP_FRACTION, _W_BRANCH_PSI_FLOOR,
    _W_BRANCH_R1_TOL, _W_BRANCH_NULL_EDGE_TW_FACTOR, _MC_UNIMODALITY_Q99_TABLE,
    _MC_UNIMODALITY_CONFIRM_WINDOWS.

Router: _GAIN_THRESHOLD, _GAIN_READABILITY_FLOOR.

Detection-branch codes: _DETECTION_BRANCH_NONE/POOLED_WITHIN/BETWEEN_MEANS/BOTH.
R²-mode codes: _R2_DEFERRED/PROJECTED/FULL_AFFINE.

Swappable calibration functions
--------------------------------
:func:`_mc_detection_edge`, :func:`_mc_unimodality_threshold`,
:func:`_w_branch_null_edge`, :func:`_w_branch_psi_threshold`.
"""
from __future__ import annotations

import jax.numpy as jnp

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

# Multi-chain escalation-trigger constants.
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
At M≥6 both regimes have sufficient separation.  :func:`~blackjax.adaptation.meta.builders.build_multi_chain_meta_core`
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

# W-branch and T-branch guard constants.
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
# Projected-tier GAIN+abstain: router constants for M-chain path
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


# ---------------------------------------------------------------------------
# Swappable calibration functions
# ---------------------------------------------------------------------------


def _mc_detection_edge(d: int, dof: int) -> float:
    """Between-chain bulk-separation edge: ``(1 + √(d/dof))²``.

    Calibrated for the M×M Gram whose null Wishart has ``dof = M−1`` degrees
    of freedom (grand-mean constraint removes one dof from M chains).  Using
    M−1 (not M) is rigorous: rank(T) = M−1 exactly.

    Both ``d`` and ``dof`` are Python ints (static at construction time).
    """
    return (1.0 + (d / dof) ** 0.5) ** 2


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


def _w_branch_null_edge(M: int, n, d: int):
    """Null bulk upper edge for the pooled within-chain residual spectrum.

    Formula: ``_W_BRANCH_NULL_EDGE_TW_FACTOR * (1 + sqrt(d / (M*(n-1))))^2``.

    **Role of this edge — magnitude gate only, not the FP control.**
    The edge is a NECESSARY condition for W-branch escalation, not a sufficient
    one.  The cross-chain Ψ consistency gate is the load-bearing false-positive
    control.  Measured on 2000 iid-null replicates at (M=8, n=40, d in
    {10,20,50,100}), the magnitude-alone FPR at this edge is 0.6–1.8% (~q98 at
    small d, tightening toward q99 at large d).  Under AR(0.9) chains,
    magnitude-alone FPR is ~100% (autocorrelation inflates lam1 well above the
    edge), but the joint (magnitude AND Ψ) FPR is 0.000% at every measured cell
    because the Ψ gate correctly identifies independent-chain autocorrelation as
    isotropic and refuses escalation.  Do NOT interpret the magnitude-alone rate
    as a standalone FPR guarantee.

    **Conservative for positively-autocorrelated series.** For AR rho > 0,
    within-chain variance is inflated above the iid level, reducing effective N
    below M*(n-1) — genuine structure clears the edge more easily than the iid
    calibration predicts.  The Ψ gate handles the resulting magnitude-alone
    inflation; the edge provides an efficient first-pass screen.

    **TW-inflation factor.** :data:`_W_BRANCH_NULL_EDGE_TW_FACTOR` = 1.02
    is a finite-N correction derived from the measured iid-null magnitude-alone
    rates above (factor chosen so the edge sits at approximately the iid null
    q98–q99 boundary across d in {10,20,50,100} at the regime-relevant N ~ 300).

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
        :func:`~blackjax.adaptation.meta._detection._compute_pooled_within_spectrum`.
    """
    N_safe = jnp.maximum(
        jnp.float32(M) * (jnp.asarray(n, dtype=jnp.float32) - jnp.float32(1.0)),
        jnp.float32(1.0),
    )
    base_edge = (jnp.float32(1.0) + jnp.sqrt(jnp.float32(d) / N_safe)) ** 2
    return jnp.float32(_W_BRANCH_NULL_EDGE_TW_FACTOR) * base_edge


def _w_branch_psi_threshold(M: int, n, d: int):
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
        jnp.asarray(n, dtype=jnp.float32) - jnp.float32(1.0), jnp.float32(1.0)
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
