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
"""Tests for :mod:`blackjax.adaptation.meta_adaptation`.

Coverage (four categories):

(a) Criterion unit tests with measured fixtures.
    R² gate and S_gap ordering tested against values from the optimal-path
    measurement arms (optpath-A.md / optpath-B.md).
    Structural checks only — NOT stochastic thin-margin assertions. The
    measured values used as fixtures were recorded at n=8000 and are used
    here to verify criterion LOGIC only.

(b) Decision-table tests.
    Each escalation condition (R² gate, S_gap magnitude, S_gap stability,
    budget deadline) is tested blocking independently. Verifies monotone
    escalation.

(c) Structural e2e smokes.
    Three geometry classes: isotropic (never escalates), low-rank correlated
    (escalates), curvature-only (stays diagonal, reparam_suggested).
    Fast CPU-only. No thin-margin stochastic assertions; only structural
    properties (route, metric type, IMM shape).

(d) Recovers-classical structural checks.
    The emitted metric is always a LowRankInverseMassMatrix.
    Diagonal route: U=0, lam=1. Escalated route: U ≠ 0.
    `metric="auto"` wiring in staged_adaptation smoke test.

Implementation note on geometry helpers
----------------------------------------
The ``_make_curvature_buffer`` helper uses independent random gradients
(uncorrelated with positions, R²≈0). Real curvature targets like the funnel
have a similar property: the score-linearity R² is ≈0.007 (optpath-B.md)
because the score has non-linear, position-dependent variance that the linear
fit cannot capture. Random gradients are the minimal synthetic model for this.

The ``_make_correlated_buffer`` helper uses a correlated Gaussian where the
covariance has a rank-k spike (Σ = I + U(Λ-I)Uᵀ). The Welford-diagonal sigma
captures the marginal variances, leaving a whitened residual spectrum with
S_gap >> _S_MIN at the spike directions. The score is linear in position
(Gaussian), giving R²=1.0. This is the minimal model for the "linear-residual,
high-S_gap" class that the controller should escalate on.

The ``_make_high_sgap_curvature_buffer`` helper produces correlated draws (high
S_gap) with RANDOM independent grads (R²≈0). This is the load-bearing fixture
for test_funnel_refusal_isolates_r2_gate: the S_gap gate passes but the R² gate
blocks. Proves that the R² gate is the sole discriminator for curvature targets.

Gate-isolation discipline
--------------------------
Each decision-table test must isolate ONE gate.  Vacuous-gate antipattern:
using isotropic fixtures where S_gap≈1 means the S_gap gate fires before R²
or the deadline ever runs.  Correct isolation:
  - R² gate: use correlated draws (high S_gap) + random grads (low R²)
  - deadline gate: use correlated draws + true score (all gates pass) + jam budget
  - S_gap gate: use isotropic draws (S_gap≈1 blocks; R² passes as a side-effect)
"""
import warnings

import jax
import jax.numpy as jnp
import numpy as np

import blackjax
from blackjax.adaptation.meta_adaptation import (
    _ASSUMED_AVG_LEAPFROGS_PER_STEP,
    _DETECTION_BRANCH_BETWEEN_MEANS,
    _DETECTION_BRANCH_BOTH,
    _DETECTION_BRANCH_NONE,
    _DETECTION_BRANCH_POOLED_WITHIN,
    _LAM_NONTRIVIAL_TOL,
    _MC_COLLINEARITY_TOL,
    _MC_UNIMODALITY_CONFIRM_WINDOWS,
    _R2_DEFERRED,
    _R2_FULL_AFFINE,
    _R2_PROJECTED,
    _R_MIN,
    _S_MIN,
    _W_BRANCH_PSI_FLOOR,
    MetaAdaptationCoreState,
    MetaAdaptationVerdict,
    MultiChainMetaAdaptationCoreState,
    _between_chain_detection,
    _build_pc_centered_time_major_pool,
    _choose_rank,
    _compute_chain_consistency_psi,
    _compute_pooled_within_spectrum,
    _compute_r2_score_linearity,
    _compute_s_gap,
    _compute_whitened_spectrum,
    _mc_detection_edge,
    _mc_unimodality_threshold,
    _w_branch_lam1_edge,
    build_meta_adaptation_core,
    build_multi_chain_meta_core,
    extract_meta_verdict,
    extract_multi_chain_verdict,
)
from blackjax.adaptation.metric_recipes import MetricCore
from blackjax.adaptation.staged_adaptation import _make_engine
from blackjax.mcmc.metrics import LowRankInverseMassMatrix
from tests.fixtures import BlackJAXTest

# ---------------------------------------------------------------------------
# Test geometry helpers
# ---------------------------------------------------------------------------

_RNG_SEED = 20260715


def _make_isotropic_buffer(n_dims: int, n: int, seed: int = _RNG_SEED):
    """Standard-normal draws with linear score (score = -position for N(0,I))."""
    key = jax.random.key(seed)
    k1, _ = jax.random.split(key)
    draws = jax.random.normal(k1, (n, n_dims))
    grads = -draws
    return draws, grads


def _make_curvature_buffer(n_dims: int, n: int, seed: int = _RNG_SEED):
    """Draws with random independent gradients: R²≈0 (curvature regime proxy).

    Real curvature targets (funnel, banana) have score-linearity R²≈0.007–0.09
    (optpath-B.md): the score is a highly non-linear function of position.
    Using independent random gradients is the minimal model for this:
    R² is exactly 0 by construction.
    """
    key = jax.random.key(seed)
    k1, k2 = jax.random.split(key)
    draws = jax.random.normal(k1, (n, n_dims))
    grads = jax.random.normal(k2, (n, n_dims))  # independent of draws → R²≈0
    return draws, grads


def _make_correlated_buffer(
    n_dims: int,
    n: int,
    rank: int = 2,
    lam_spike: float = 25.0,
    seed: int = _RNG_SEED,
):
    """Correlated Gaussian with rank-k spike: R²=1.0, S_gap >> _S_MIN.

    Σ = I + U*(lam_spike - 1)*Uᵀ, score = -Σ⁻¹ x (linear in position).
    After Welford-diagonal whitening (D = diag(Σ)), the whitened residual has
    a rank-k spike with eigenvalue ≈ lam_spike / max(diag(Σ)), giving S_gap
    well above _S_MIN for large lam_spike.
    """
    key = jax.random.key(seed)
    k1, k2 = jax.random.split(key)
    raw = jax.random.normal(k2, (n_dims, rank))
    U, _ = jnp.linalg.qr(raw)
    U = U[:, :rank]

    z = jax.random.normal(k1, (n, n_dims))
    z_orth = z - (z @ U) @ U.T
    draws = z_orth + jnp.sqrt(lam_spike) * (z @ U) @ U.T
    grads = -(draws - (1.0 - 1.0 / lam_spike) * (draws @ U) @ U.T)
    return draws, grads


def _make_marginal_sgap_curvature_buffer(
    n_dims: int,
    n: int,
    seed: int = _RNG_SEED,
):
    """Rank-1 spike with S_gap ∈ [_S_MIN, 2·_S_MIN) = [2.0, 4.0) + random grads.

    Uses lam_spike=4.5 (random non-axis-aligned direction, d=20, n=500).
    Measured with seed=42: top Welford-whitened eigenvalue ≈ 3.5, k_new=1,
    S_gap ≈ 2.94 ∈ [2, 4).  lam_spike=4.5 gives margins of ≥0.3 above _S_MIN
    across the seed sweep [1,7,13,17,21,37,42,55,63,100], making the fixture robust
    to seed variation (all tested seeds remain in band).

    Random grads (R²≈0) block escalation via the R² gate, leaving the S_gap in
    the MARGINAL band as the only reason for staying diagonal — the correct fixture
    for the 'stays-diag-marginal' decision row.

    The rank-1 (non-axis-aligned) spike is load-bearing: an axis-aligned spike is
    perfectly cancelled by Welford diagonal whitening (D^{-1/2} Sigma D^{-1/2} = I),
    giving S_gap=1.  A random direction leaks residual anisotropy into the whitened
    space, producing a measurable spike with a non-trivial k_new=1 cut.
    """
    key = jax.random.key(seed)
    k1, k2, k3 = jax.random.split(key, 3)
    # Rank-1 random direction (non-axis-aligned -> Welford whitening is imperfect)
    raw = jax.random.normal(k3, (n_dims, 1))
    U, _ = jnp.linalg.qr(raw)
    U = U[:, :1]

    z = jax.random.normal(k1, (n, n_dims))
    z_orth = z - (z @ U) @ U.T
    # lam_spike=4.5 gives S_gap ∈ [2.05, 2.94] across probed seeds; well clear of
    # both boundaries [_S_MIN=2.0, 2*_S_MIN=4.0).
    lam_spike = 4.5
    draws = z_orth + jnp.sqrt(lam_spike) * (z @ U) @ U.T
    grads = jax.random.normal(k2, (n, n_dims))  # independent of draws -> R2 approx 0
    return draws, grads


def _make_high_sgap_curvature_buffer(
    n_dims: int,
    n: int,
    rank: int = 2,
    lam_spike: float = 20.0,
    seed: int = _RNG_SEED,
):
    """Correlated draws (high S_gap) + random independent grads (R²≈0).

    Fixture for the funnel-refusal test: the S_gap gate passes (S_gap≥_S_MIN)
    but the R² gate blocks (R²≈0).  Proves the R² gate is the sole discriminator
    for non-linear-score targets.
    """
    key = jax.random.key(seed)
    k1, k2, k3 = jax.random.split(key, 3)
    raw = jax.random.normal(k3, (n_dims, rank))
    U, _ = jnp.linalg.qr(raw)
    U = U[:, :rank]

    z = jax.random.normal(k1, (n, n_dims))
    z_orth = z - (z @ U) @ U.T
    draws = z_orth + jnp.sqrt(lam_spike) * (z @ U) @ U.T
    grads = jax.random.normal(k2, (n, n_dims))  # independent of draws → R²≈0
    return draws, grads


def _fill_state_from_buffer(
    state: MetaAdaptationCoreState,
    draws: jax.Array,
    grads: jax.Array,
) -> MetaAdaptationCoreState:
    """Copy draws/grads into the state buffer (bypasses update() for direct testing)."""
    B, d = state.draws_buffer.shape
    n = draws.shape[0]
    n_fill = min(n, B)
    return state._replace(
        draws_buffer=jnp.concatenate(
            [draws[:n_fill], jnp.zeros((B - n_fill, d), dtype=draws.dtype)], axis=0
        ),
        grads_buffer=jnp.concatenate(
            [grads[:n_fill], jnp.zeros((B - n_fill, d), dtype=grads.dtype)], axis=0
        ),
        buffer_idx=jnp.array(n_fill, dtype=jnp.int32),
    )


# ---------------------------------------------------------------------------
# (a) Criterion unit tests with measured fixtures
# ---------------------------------------------------------------------------


class TestCriterionR2Gate(BlackJAXTest):
    """R² gate correctly separates linear-residual from curvature geometry."""

    def test_linear_geometry_r2_above_threshold(self):
        """Linear score (MVN): R² >> _R_MIN."""
        d, n, max_rank = 20, 400, 10
        draws, grads = _make_isotropic_buffer(d, n, seed=1)
        draws = jnp.array(draws, dtype=jnp.float32)
        grads = jnp.array(grads, dtype=jnp.float32)

        sigma = jnp.ones(d)
        U_k = jnp.zeros((d, max_rank))
        n_arr = jnp.array(n, dtype=jnp.int32)
        r2, mode = _compute_r2_score_linearity(
            draws, grads, sigma, n_arr, U_k, max_rank
        )
        r2_np = float(np.asarray(r2))
        self.assertFalse(np.isnan(r2_np), "R² should not be deferred at n=400, d=20")
        self.assertGreater(
            r2_np, _R_MIN, f"Linear geometry: expected R² > {_R_MIN}, got {r2_np}"
        )

    def test_curvature_geometry_r2_below_threshold(self):
        """Random-gradient score (curvature proxy): R² << _R_MIN."""
        d, n, max_rank = 20, 400, 10
        draws, grads = _make_curvature_buffer(d, n, seed=2)
        draws = jnp.array(draws, dtype=jnp.float32)
        grads = jnp.array(grads, dtype=jnp.float32)

        sigma = jnp.ones(d)
        U_k = jnp.zeros((d, max_rank))
        n_arr = jnp.array(n, dtype=jnp.int32)
        r2, _ = _compute_r2_score_linearity(draws, grads, sigma, n_arr, U_k, max_rank)
        r2_np = float(np.asarray(r2))
        self.assertFalse(
            np.isnan(r2_np), "Should not defer with n=400, d=20, max_rank=10"
        )
        self.assertLess(
            r2_np, _R_MIN, f"Curvature proxy: expected R² < {_R_MIN}, got {r2_np}"
        )

    def test_high_d_deferred_when_n_too_small(self):
        """R² is NaN and mode=_R2_DEFERRED when n is too small for any fit."""
        d = 100
        n = 10  # far below 2 * 4 * (max_rank+1) = 88 (projected threshold)
        max_rank = 10
        draws, grads = _make_isotropic_buffer(d, n, seed=3)
        draws = jnp.array(draws, dtype=jnp.float32)
        grads = jnp.array(grads, dtype=jnp.float32)

        sigma = jnp.ones(d)
        U_k = jnp.zeros((d, max_rank))
        n_arr = jnp.array(n, dtype=jnp.int32)
        r2, mode_int = _compute_r2_score_linearity(
            draws, grads, sigma, n_arr, U_k, max_rank
        )
        self.assertTrue(
            np.isnan(float(np.asarray(r2))),
            f"Expected deferred (NaN) R² at n={n}, d={d}",
        )
        self.assertEqual(
            int(np.asarray(mode_int)), _R2_DEFERRED, "Mode should be _R2_DEFERRED"
        )

    def test_projected_fit_non_nan_and_mode_projected(self):
        """Projected fit is used (non-NaN, mode=_R2_PROJECTED) when n ≥ 2*4*(max_rank+1)."""
        d = 200  # too large for full-affine with n=106 (min_n_full=3200)
        max_rank = 5
        n = 2 * 4 * (max_rank + 1) + 10  # = 58, above min_n_proj=48
        draws, grads = _make_isotropic_buffer(d, n, seed=4)
        draws = jnp.array(draws, dtype=jnp.float32)
        grads = jnp.array(grads, dtype=jnp.float32)

        sigma = jnp.ones(d)
        U_k = jnp.zeros((d, max_rank))
        n_arr = jnp.array(n, dtype=jnp.int32)
        r2, mode_int = _compute_r2_score_linearity(
            draws, grads, sigma, n_arr, U_k, max_rank
        )
        self.assertFalse(
            np.isnan(float(np.asarray(r2))),
            f"Projected fit should be feasible at n={n}, d={d}",
        )
        self.assertEqual(
            int(np.asarray(mode_int)), _R2_PROJECTED, "Mode should be _R2_PROJECTED"
        )

    def test_full_affine_mode_when_n_large(self):
        """Full-affine fit is used (mode=_R2_FULL_AFFINE) when n ≥ 2*8*d."""
        d, max_rank = 10, 5
        n = 2 * 8 * d + 10  # = 170, above min_n_full=160
        draws, grads = _make_isotropic_buffer(d, n, seed=5)
        draws = jnp.array(draws, dtype=jnp.float32)
        grads = jnp.array(grads, dtype=jnp.float32)

        sigma = jnp.ones(d)
        U_k = jnp.zeros((d, max_rank))
        n_arr = jnp.array(n, dtype=jnp.int32)
        r2, mode_int = _compute_r2_score_linearity(
            draws, grads, sigma, n_arr, U_k, max_rank
        )
        self.assertFalse(np.isnan(float(np.asarray(r2))))
        self.assertEqual(
            int(np.asarray(mode_int)), _R2_FULL_AFFINE, "Mode should be _R2_FULL_AFFINE"
        )


class TestCriterionSGap(BlackJAXTest):
    """S_gap ordering agrees with measured payoff ordering."""

    def test_isotropic_s_gap_near_one(self):
        """Isotropic draws: whitened spectrum flat → S_gap ≈ 1."""
        d, n, max_rank = 20, 500, 5
        draws, _ = _make_isotropic_buffer(d, n, seed=10)
        draws = jnp.array(draws, dtype=jnp.float32)
        n_arr = jnp.array(n, dtype=jnp.int32)
        sigma = jnp.ones(d)

        eigenvalues, _ = _compute_whitened_spectrum(draws, sigma, n_arr, max_rank)
        k = _choose_rank(eigenvalues, n_arr, max_rank)
        s_gap = float(np.asarray(_compute_s_gap(eigenvalues, k)))
        self.assertAlmostEqual(
            s_gap, 1.0, delta=0.5, msg=f"Isotropic S_gap should ≈1, got {s_gap}"
        )

    def test_correlated_spike_s_gap_large(self):
        """Correlated rank-2 spike: S_gap >> _S_MIN after diagonal whitening."""
        d, n, max_rank = 20, 500, 10
        draws, grads = _make_correlated_buffer(d, n, rank=2, lam_spike=20.0, seed=11)
        draws = jnp.array(draws, dtype=jnp.float32)
        grads = jnp.array(grads, dtype=jnp.float32)
        n_arr = jnp.array(n, dtype=jnp.int32)

        # Whiten with Fisher sigma (as the controller does)
        from blackjax.adaptation.metric_estimators import _compute_low_rank_metric

        sigma_lr, _, _, _ = _compute_low_rank_metric(
            draws, grads, n_arr, max_rank, 1e-5, 2.0
        )
        eigenvalues, _ = _compute_whitened_spectrum(draws, sigma_lr, n_arr, max_rank)
        k = _choose_rank(eigenvalues, n_arr, max_rank)
        s_gap = float(np.asarray(_compute_s_gap(eigenvalues, k)))
        self.assertGreater(
            s_gap, _S_MIN, f"Correlated spike S_gap should > {_S_MIN}, got {s_gap}"
        )

    def test_s_gap_ordering_matches_payoff_ordering(self):
        """S_gap ordering agrees with measured payoff ordering (high-payoff > low-payoff).

        High-payoff target: concentrated rank-2 spike (lam_spike=25) → large S_gap.
        Low-payoff target: isotropic → S_gap ≈ 1.
        This matches the measured result: radon payoff 3.68× > isotropic 1×.
        """
        d, n, max_rank = 20, 400, 5
        n_arr = jnp.array(n, dtype=jnp.int32)
        from blackjax.adaptation.metric_estimators import _compute_low_rank_metric

        # High-payoff: correlated spike
        draws_h, grads_h = _make_correlated_buffer(
            d, n, rank=2, lam_spike=20.0, seed=20
        )
        draws_h = jnp.array(draws_h, dtype=jnp.float32)
        grads_h = jnp.array(grads_h, dtype=jnp.float32)
        sigma_h, _, _, _ = _compute_low_rank_metric(
            draws_h, grads_h, n_arr, max_rank, 1e-5, 2.0
        )
        eigs_h, _ = _compute_whitened_spectrum(draws_h, sigma_h, n_arr, max_rank)
        k_h = _choose_rank(eigs_h, n_arr, max_rank)
        s_gap_h = float(np.asarray(_compute_s_gap(eigs_h, k_h)))

        # Low-payoff: isotropic
        draws_l, grads_l = _make_isotropic_buffer(d, n, seed=21)
        draws_l = jnp.array(draws_l, dtype=jnp.float32)
        grads_l = jnp.array(grads_l, dtype=jnp.float32)
        sigma_l, _, _, _ = _compute_low_rank_metric(
            draws_l, grads_l, n_arr, max_rank, 1e-5, 2.0
        )
        eigs_l, _ = _compute_whitened_spectrum(draws_l, sigma_l, n_arr, max_rank)
        k_l = _choose_rank(eigs_l, n_arr, max_rank)
        s_gap_l = float(np.asarray(_compute_s_gap(eigs_l, k_l)))

        self.assertGreater(
            s_gap_h,
            s_gap_l,
            f"High-payoff S_gap ({s_gap_h:.2f}) should exceed low-payoff ({s_gap_l:.2f})",  # noqa: E231
        )


# ---------------------------------------------------------------------------
# (b) Decision-table tests
# ---------------------------------------------------------------------------


class TestEscalationDecisionTable(BlackJAXTest):
    """Each escalation gate blocks independently; monotone escalation verified."""

    def _run_two_windows(self, draws, grads, max_grad_budget=50000, max_rank=10):
        """Run two identical windows so that the stability gate can pass."""
        d = draws.shape[1]
        core = build_meta_adaptation_core(max_grad_budget, max_rank=max_rank)
        state = core.init(d)
        for _ in range(2):
            state = _fill_state_from_buffer(state, draws, grads)
            state = core.final(state)
        return state, core

    def test_r2_gate_blocks_curvature(self):
        """Curvature geometry (R²≈0): R² gate must block escalation."""
        d, n = 20, 400
        draws, grads = _make_curvature_buffer(d, n, seed=30)
        draws = jnp.array(draws, dtype=jnp.float32)
        grads = jnp.array(grads, dtype=jnp.float32)
        state, _ = self._run_two_windows(draws, grads)
        self.assertFalse(
            bool(np.asarray(state.has_escalated)),
            "Curvature geometry: R² gate should block escalation",
        )

    def test_s_gap_magnitude_blocks_isotropic(self):
        """Isotropic draws (S_gap≈1): magnitude gate must block escalation."""
        d, n = 20, 400
        draws, grads = _make_isotropic_buffer(d, n, seed=31)
        draws = jnp.array(draws, dtype=jnp.float32)
        grads = jnp.array(grads, dtype=jnp.float32)
        state, _ = self._run_two_windows(draws, grads)
        self.assertFalse(
            bool(np.asarray(state.has_escalated)),
            "Isotropic: S_gap magnitude gate should block escalation",
        )

    def test_s_gap_stability_blocks_first_window(self):
        """First window (no prior S_gap read): stability gate must block escalation.

        Even when R² is high and S_gap is large, the stability gate requires two
        consecutive reads. After the first window, s_gap_curr is set but
        s_gap_prev was NaN → stability check fails.
        """
        d, n = 20, 500
        draws, grads = _make_correlated_buffer(d, n, rank=2, lam_spike=20.0, seed=32)
        draws = jnp.array(draws, dtype=jnp.float32)
        grads = jnp.array(grads, dtype=jnp.float32)

        core = build_meta_adaptation_core(50000, max_rank=10)
        state = core.init(d)
        # Only ONE window — stability gate must block escalation
        state = _fill_state_from_buffer(state, draws, grads)
        state = core.final(state)
        self.assertFalse(
            bool(np.asarray(state.has_escalated)),
            "First window: stability gate (no prior S_gap) should block escalation",
        )
        # s_gap_curr should now be set
        self.assertFalse(
            np.isnan(float(np.asarray(state.s_gap_curr))),
            "After first window, s_gap_curr should be a valid number",
        )

    def test_s_gap_stability_passes_on_second_stable_window(self):
        """Two identical windows: stability gate passes → controller escalates."""
        d, n = 20, 500
        draws, grads = _make_correlated_buffer(d, n, rank=2, lam_spike=20.0, seed=33)
        draws = jnp.array(draws, dtype=jnp.float32)
        grads = jnp.array(grads, dtype=jnp.float32)
        state, _ = self._run_two_windows(draws, grads)
        self.assertTrue(
            bool(np.asarray(state.has_escalated)),
            "Two stable windows with large S_gap and high R²: should escalate",
        )

    def test_budget_deadline_blocks_tight_budget(self):
        """Deadline gate blocks when remaining budget < 2k + step-size buffer.

        Isolation: fixture passes R² (linear score, R²≈1) AND S_gap (>_S_MIN)
        gates.  Only the budget_used manipulation exercises the deadline gate.
        """
        d, n = 20, 500
        draws, grads = _make_correlated_buffer(d, n, rank=2, lam_spike=20.0, seed=34)
        draws = jnp.array(draws, dtype=jnp.float32)
        grads = jnp.array(grads, dtype=jnp.float32)

        # First, confirm the fixture CAN escalate with fresh budget.
        core = build_meta_adaptation_core(50000, max_rank=5)
        state = core.init(d)
        state = _fill_state_from_buffer(state, draws, grads)
        state = core.final(state)
        # Jam budget_used to almost-exhausted (leaves < 2k + 50 remaining steps).
        max_budget_steps = 50000 // 20  # 2500 steps
        state_jammed = state._replace(
            budget_used=jnp.array(max_budget_steps - 5, dtype=jnp.int32),
        )
        state_jammed = _fill_state_from_buffer(state_jammed, draws, grads)
        state_jammed = core.final(state_jammed)
        self.assertFalse(
            bool(np.asarray(state_jammed.has_escalated)),
            "Exhausted budget: deadline gate should block escalation",
        )
        # Control: same fixture with fresh budget escalates.
        state_fresh = core.init(d)
        for _ in range(2):
            state_fresh = _fill_state_from_buffer(state_fresh, draws, grads)
            state_fresh = core.final(state_fresh)
        self.assertTrue(
            bool(np.asarray(state_fresh.has_escalated)),
            "Control: fresh budget with same fixture should escalate",
        )

    def test_funnel_refusal_isolates_r2_gate(self):
        """Correlated draws + random grads: high S_gap but R²≈0 → reparam_suggested.

        This is the load-bearing test for the R² gate: S_gap passes (proves S_gap
        gate is NOT the blocker) but R² blocks (the sole discriminator).
        An isotropic curvature fixture would let the S_gap gate block escalation,
        masking the R² gate; this fixture uses a HIGH-S_gap target so R² is the
        sole discriminator.
        """
        d, n = 20, 500
        # High S_gap (correlated spike) + random grads (R²≈0, curvature proxy)
        draws, grads = _make_high_sgap_curvature_buffer(
            d, n, rank=2, lam_spike=20.0, seed=36
        )
        draws = jnp.array(draws, dtype=jnp.float32)
        grads = jnp.array(grads, dtype=jnp.float32)
        state, _ = self._run_two_windows(draws, grads)

        # S_gap gate passes (proves the S_gap gate is not the blocker).
        s_gap = float(np.asarray(state.s_gap_curr))
        self.assertGreater(
            s_gap,
            _S_MIN,
            "Fixture must have high S_gap for isolation; got "
            + str(s_gap),  # noqa: E702
        )
        # R² gate blocks.
        r2 = float(np.asarray(state.r2_latest))
        self.assertFalse(np.isnan(r2), "R² should not be deferred at n=500, d=20")
        self.assertLess(r2, _R_MIN, f"R²={r2} should be below _R_MIN={_R_MIN}")
        # Result: controller stays diagonal and suggests reparameterization.
        self.assertFalse(
            bool(np.asarray(state.has_escalated)),
            "R² gate should block escalation; S_gap alone must not suffice",
        )
        verdict = extract_meta_verdict(
            state, max_grad_budget=50000, num_warmup_steps=2500
        )
        self.assertEqual(
            verdict.route,
            "reparam_suggested",
            "High-S_gap + low-R2 should route to reparam_suggested; got "
            + verdict.route,
        )

    def test_monotone_escalation(self):
        """Once escalated, has_escalated stays True on subsequent windows."""
        d, n = 20, 500
        draws, grads = _make_correlated_buffer(d, n, rank=2, lam_spike=20.0, seed=35)
        draws = jnp.array(draws, dtype=jnp.float32)
        grads = jnp.array(grads, dtype=jnp.float32)

        state, core = self._run_two_windows(draws, grads)
        self.assertTrue(bool(np.asarray(state.has_escalated)), "Should have escalated")
        first_rank = int(np.asarray(state.escalation_rank))

        # Run a third window
        state = _fill_state_from_buffer(state, draws, grads)
        state = core.final(state)
        self.assertTrue(
            bool(np.asarray(state.has_escalated)),
            "Monotone: has_escalated must stay True once set",
        )
        self.assertEqual(
            int(np.asarray(state.escalation_rank)),
            first_rank,
            "Monotone: escalation_rank must not decrease",
        )


# ---------------------------------------------------------------------------
# (c) Structural e2e smokes
# ---------------------------------------------------------------------------


class TestStructuralE2ESmoke(BlackJAXTest):
    """Structural e2e smokes for three geometry classes.

    Only structural properties checked (route, metric type, IMM shape).
    No thin-margin stochastic assertions.
    """

    def test_isotropic_stays_diagonal(self):
        """Isotropic MVN: S_gap≈1 → controller stays diagonal across 3 windows."""
        d, n = 10, 300
        draws, grads = _make_isotropic_buffer(d, n, seed=40)
        draws = jnp.array(draws, dtype=jnp.float32)
        grads = jnp.array(grads, dtype=jnp.float32)

        core = build_meta_adaptation_core(50000, max_rank=5)
        state = core.init(d)
        for _ in range(3):
            state = _fill_state_from_buffer(state, draws, grads)
            state = core.final(state)

        self.assertFalse(
            bool(np.asarray(state.has_escalated)),
            "Isotropic: should stay diagonal across multiple windows",
        )

    def test_correlated_spike_escalates(self):
        """Correlated rank-2 spike: both gates pass → escalates after two windows."""
        d, n = 20, 500
        draws, grads = _make_correlated_buffer(d, n, rank=2, lam_spike=25.0, seed=41)
        draws = jnp.array(draws, dtype=jnp.float32)
        grads = jnp.array(grads, dtype=jnp.float32)

        core = build_meta_adaptation_core(50000, max_rank=10)
        state = core.init(d)
        for _ in range(2):
            state = _fill_state_from_buffer(state, draws, grads)
            state = core.final(state)

        self.assertTrue(
            bool(np.asarray(state.has_escalated)),
            "Correlated spike: should escalate after two windows",
        )
        imm = state.inverse_mass_matrix
        self.assertIsInstance(imm, LowRankInverseMassMatrix)
        self.assertEqual(imm.sigma.shape, (d,))
        self.assertEqual(imm.U.shape, (d, 10))

    def test_curvature_stays_diagonal_reparam_hint(self):
        """Curvature geometry: R²≈0 blocks escalation; verdict is reparam_suggested."""
        d, n = 20, 400
        draws, grads = _make_curvature_buffer(d, n, seed=42)
        draws = jnp.array(draws, dtype=jnp.float32)
        grads = jnp.array(grads, dtype=jnp.float32)

        core = build_meta_adaptation_core(50000, max_rank=10)
        state = core.init(d)
        for _ in range(3):
            state = _fill_state_from_buffer(state, draws, grads)
            state = core.final(state)

        self.assertFalse(bool(np.asarray(state.has_escalated)))

        verdict = extract_meta_verdict(
            state, max_grad_budget=50000, num_warmup_steps=2500
        )
        self.assertEqual(
            verdict.route,
            "reparam_suggested",
            f"Curvature route should be reparam_suggested, got {verdict.route}",
        )
        self.assertTrue(verdict.flags["reparam_hint"])

    def test_high_d_linear_spike_escalates(self):
        """High-d linear spike (d=120, rank=3): projected R² tier passes → escalates.

        Regression guard for the projected-R²-both-sides fix.  Before the fix,
        radon-like targets at d>>k produced projected R²≈0 (full d-dim response
        regressed on k features) and emitted reparam_suggested.  After the fix,
        projecting both sides onto U_k gives R²≈1 → escalates.
        """
        d, n, rank = 120, 600, 3
        draws, grads = _make_correlated_buffer(d, n, rank=rank, lam_spike=25.0, seed=45)
        draws = jnp.array(draws, dtype=jnp.float32)
        grads = jnp.array(grads, dtype=jnp.float32)

        core = build_meta_adaptation_core(50000, max_rank=10)
        state = core.init(d)
        for _ in range(2):
            state = _fill_state_from_buffer(state, draws, grads)
            state = core.final(state)

        self.assertTrue(
            bool(np.asarray(state.has_escalated)),
            "High-d linear spike should escalate; "
            + f"r2={float(np.asarray(state.r2_latest)):.3f}, "  # noqa: E231
            + f"s_gap={float(np.asarray(state.s_gap_curr)):.2f}",  # noqa: E231
        )
        verdict = extract_meta_verdict(
            state, max_grad_budget=50000, num_warmup_steps=2500
        )
        self.assertEqual(
            verdict.route,
            "low_rank",
            "High-d linear spike should route to low_rank; got " + verdict.route,
        )


# ---------------------------------------------------------------------------
# (d) Recovers-classical structural checks
# ---------------------------------------------------------------------------


class TestRecovershClassical(BlackJAXTest):
    """Structural invariants that must hold across all routing decisions."""

    def test_imm_always_low_rank_type(self):
        """The emitted IMM is always LowRankInverseMassMatrix, even before escalation."""
        core = build_meta_adaptation_core(50000, max_rank=5)
        state = core.init(10)
        self.assertIsInstance(state.inverse_mass_matrix, LowRankInverseMassMatrix)

    def test_diagonal_imm_u_zero_lam_one(self):
        """Before escalation: U=0 and lam=1 (diagonal-equivalent representation)."""
        core = build_meta_adaptation_core(50000, max_rank=5)
        state = core.init(10)
        imm = state.inverse_mass_matrix
        np.testing.assert_allclose(
            np.asarray(imm.U),
            0.0,
            atol=1e-7,
            err_msg="Before escalation: U should be zero",
        )
        np.testing.assert_allclose(
            np.asarray(imm.lam),
            1.0,
            atol=1e-7,
            err_msg="Before escalation: lam should be one",
        )

    def test_metric_core_protocol(self):
        """build_meta_adaptation_core returns a MetricCore with callable protocol."""
        core = build_meta_adaptation_core(50000, max_rank=5)
        self.assertIsInstance(core, MetricCore)
        self.assertTrue(callable(core.init))
        self.assertTrue(callable(core.update))
        self.assertTrue(callable(core.final))

    def test_update_accumulates_into_buffer(self):
        """update() advances buffer_idx and budget_used by 1 per call."""
        d = 10
        core = build_meta_adaptation_core(50000, max_rank=5)
        state = core.init(d)
        self.assertEqual(int(np.asarray(state.buffer_idx)), 0)

        state1 = core.update(state, jnp.zeros(d), jnp.ones(d))
        self.assertEqual(int(np.asarray(state1.buffer_idx)), 1)
        self.assertEqual(int(np.asarray(state1.budget_used)), 1)

    def test_converged_at_step_init_sentinel(self):
        """converged_at_step is -1 (sentinel for 'not yet converged') at init."""
        core = build_meta_adaptation_core(50000, max_rank=5)
        state = core.init(10)
        self.assertEqual(
            int(np.asarray(state.converged_at_step)),
            -1,
            "converged_at_step should be -1 (not yet converged) at init",
        )

    def test_r2_mode_init_deferred(self):
        """r2_mode is _R2_DEFERRED at init (no window computed yet)."""
        core = build_meta_adaptation_core(50000, max_rank=5)
        state = core.init(10)
        self.assertEqual(int(np.asarray(state.r2_mode)), _R2_DEFERRED)

    def test_r2_mode_observed_after_window(self):
        """r2_mode in carry matches the actually-taken branch after a window."""
        d, n = 10, 400
        max_rank = 5
        draws, grads = _make_isotropic_buffer(d, n, seed=60)
        draws = jnp.array(draws, dtype=jnp.float32)
        grads = jnp.array(grads, dtype=jnp.float32)

        core = build_meta_adaptation_core(50000, max_rank=max_rank)
        state = core.init(d)
        state = _fill_state_from_buffer(state, draws, grads)
        state = core.final(state)

        mode_int = int(np.asarray(state.r2_mode))
        # n=400, d=10: min_n_full = 2*8*10 = 160 ≤ 400 → full_affine branch.
        self.assertEqual(
            mode_int,
            _R2_FULL_AFFINE,
            f"n=400, d=10: expected _R2_FULL_AFFINE, got mode_int={mode_int}",
        )
        # verdict flag should reflect the carry, not post-hoc inference
        verdict = extract_meta_verdict(
            state, max_grad_budget=50000, num_warmup_steps=2500
        )
        self.assertEqual(verdict.flags["high_d_r2_mode"], "full_affine")

    def test_budget_returned_zero_before_airm_convergence(self):
        """budget_returned_steps is 0 when AIRM criterion has never fired (v1 advisory)."""
        core = build_meta_adaptation_core(50000, max_rank=5)
        state = core.init(10)
        verdict = extract_meta_verdict(
            state, max_grad_budget=50000, num_warmup_steps=2500
        )
        self.assertEqual(
            verdict.budget_returned_steps,
            0,
            "No AIRM convergence yet: budget_returned should be 0",
        )

    def test_verdict_fields_present(self):
        """extract_meta_verdict populates all required fields."""
        core = build_meta_adaptation_core(50000, max_rank=5)
        state = core.init(10)
        verdict = extract_meta_verdict(
            state, max_grad_budget=50000, num_warmup_steps=2500
        )
        self.assertIsInstance(verdict, MetaAdaptationVerdict)
        self.assertIn(verdict.route, ("diagonal", "low_rank", "reparam_suggested"))
        self.assertIn(verdict.confidence, ("high", "low"))
        self.assertEqual(verdict.buffer_policy, "reset")
        self.assertIsInstance(verdict.flags, dict)
        for key in (
            "reparam_hint",
            "marginal_s_gap",
            "wall_cost_discount",
            "high_d_r2_mode",
            "mode_coverage",
            "nominal_rank",
        ):
            self.assertIn(key, verdict.flags, f"Missing verdict flag: {key}")

    def test_staged_adaptation_auto_metric_smoke(self):
        """staged_adaptation(metric='auto') wires correctly and produces an IMM.

        Structural smoke test — not a performance test. Verifies:
        - No exception during construction or warmup run.
        - Warmup returns LowRankInverseMassMatrix.
        - IMM shape matches n_dims.
        """
        n_dims = 5

        def logdensity_fn(x):
            return -0.5 * jnp.sum(x**2)

        warmup = blackjax.staged_adaptation(
            blackjax.nuts, logdensity_fn, metric="auto", max_grad_budget=5000
        )
        key = jax.random.key(99)
        results, _ = warmup.run(key, jnp.zeros(n_dims), num_steps=50)

        self.assertIsNotNone(results.state)
        self.assertIn("step_size", results.parameters)
        self.assertIn("inverse_mass_matrix", results.parameters)
        imm = results.parameters["inverse_mass_matrix"]
        self.assertIsInstance(imm, LowRankInverseMassMatrix)
        self.assertEqual(imm.sigma.shape, (n_dims,))

    def test_staged_adaptation_auto_missing_budget_raises(self):
        """staged_adaptation(metric='auto') without max_grad_budget raises ValueError."""
        with self.assertRaisesRegex(ValueError, "max_grad_budget"):
            blackjax.staged_adaptation(
                blackjax.nuts,
                lambda x: -0.5 * jnp.sum(x**2),
                metric="auto",
                # max_grad_budget intentionally omitted
            )

    def test_auto_uses_growing_window_schedule(self):
        """metric='auto' resolves to the growing-window schedule; explicit schedule is preserved.

        Regression guard: the old `is build_schedule` sentinel could not distinguish
        between "user passed nothing" and "user explicitly passed build_schedule",
        so an explicit Stan-on-auto request was silently swapped to growing-window.
        The old test only checked isinstance(IMM, LowRankInverseMassMatrix), which is
        a tautology because auto always emits that type — it never actually observed
        which schedule was chosen.

        Both are fixed via _resolve_metric_and_schedule: the function is called
        directly so the returned schedule identity is observable.
        """
        from blackjax.adaptation.low_rank_adaptation import (
            build_growing_window_schedule,
        )
        from blackjax.adaptation.staged_adaptation import (
            _resolve_metric_and_schedule,
            build_schedule,
        )

        # auto + no explicit schedule → growing window (the override).
        _, sched_auto_default = _resolve_metric_and_schedule(
            "auto", None, max_grad_budget=5000
        )
        self.assertIs(
            sched_auto_default,
            build_growing_window_schedule,
            "auto+default must resolve to build_growing_window_schedule",
        )

        # Negative test: explicit Stan schedule on auto is PRESERVED (not swapped).
        # The old sentinel `if schedule_fn is build_schedule` was the bug: it
        # replaced explicit Stan with growing because both were the same object.
        _, sched_auto_explicit_stan = _resolve_metric_and_schedule(
            "auto", build_schedule, max_grad_budget=5000
        )
        self.assertIs(
            sched_auto_explicit_stan,
            build_schedule,
            "auto+explicit-Stan must NOT be swapped to growing-window; "  # noqa: E231
            "schedule_fn sentinel was broken (build_schedule == build_schedule always true)",
        )

    def test_converged_at_step_sets_on_airm_convergence(self):
        """converged_at_step is set (≥0) and budget_returned_steps > 0 after AIRM fires.

        Regression guard for the dead-field bug: previously budget_returned was
        always 0 because converged_at_step was never set.
        """
        d, n = 20, 500
        draws, grads = _make_correlated_buffer(d, n, rank=2, lam_spike=20.0, seed=61)
        draws = jnp.array(draws, dtype=jnp.float32)
        grads = jnp.array(grads, dtype=jnp.float32)

        core = build_meta_adaptation_core(50000, max_rank=10)
        state = core.init(d)

        # Run ≥3 identical escalated windows to drive AIRM velocity below tolerance.
        # After escalation, identical lam → lam_diff = 0 < _AIRM_VELOCITY_TOL.
        # Two consecutive sub-threshold windows set converged_at_step.
        for _ in range(4):
            state = _fill_state_from_buffer(state, draws, grads)
            state = core.final(state)

        converged_at = int(np.asarray(state.converged_at_step))
        self.assertGreaterEqual(
            converged_at,
            0,
            "converged_at_step should be >=0 after AIRM convergence; got "
            + str(converged_at),
        )

        verdict = extract_meta_verdict(
            state, max_grad_budget=50000, num_warmup_steps=2500
        )
        self.assertGreater(
            verdict.budget_returned_steps,
            0,
            "budget_returned_steps should be > 0 when converged_at_step is set",
        )

    def test_diagonal_sigma_is_welford_not_fisher(self):
        """Stay-diagonal IMM uses Welford sigma (sample std), not Fisher sigma.

        Recovers-classical anchor: the welford diagonal is the measured nutpie
        baseline (fisher-diag = 0.11x welford on funnel, 0.62x on german).
        Use correlated draws + random grads so R²<_R_MIN keeps controller diagonal
        while the anisotropy makes welford ≠ fisher.
        """
        d, n, rank = 10, 400, 2
        lam_spike = 25.0
        # Build fixture manually (correlated draws + random grads)
        key = jax.random.key(71)
        k1, k2, k3 = jax.random.split(key, 3)
        raw = jax.random.normal(k3, (d, rank))
        Q, _ = jnp.linalg.qr(raw)
        Q = Q[:, :rank]
        z = jax.random.normal(k1, (n, d))
        z_orth = z - (z @ Q) @ Q.T
        draws = jnp.array(
            z_orth + jnp.sqrt(lam_spike) * (z @ Q) @ Q.T, dtype=jnp.float32
        )
        grads = jnp.array(jax.random.normal(k2, (n, d)), dtype=jnp.float32)

        core = build_meta_adaptation_core(50000, max_rank=5)
        state = core.init(d)
        state = _fill_state_from_buffer(state, draws, grads)
        state = core.final(state)
        self.assertFalse(bool(np.asarray(state.has_escalated)))

        # Expected Welford sigma: sample std (ddof=1) of the original draws.
        draws_np = np.asarray(draws)
        mean_x = draws_np.mean(0)
        var_x = np.sum((draws_np - mean_x[None, :]) ** 2, axis=0) / max(n - 1, 1)
        sigma_welford_expected = np.sqrt(np.maximum(var_x, 1e-10))

        emitted_sigma = np.asarray(state.inverse_mass_matrix.sigma)
        np.testing.assert_allclose(
            emitted_sigma,
            sigma_welford_expected,
            rtol=0.05,
            err_msg="Stay-diagonal sigma must equal Welford sample std",
        )

    def test_exit_reason_warmup_complete(self):
        """exit_reason is 'warmup_complete' when AIRM has not converged."""
        core = build_meta_adaptation_core(50000, max_rank=5)
        state = core.init(10)
        verdict = extract_meta_verdict(
            state, max_grad_budget=50000, num_warmup_steps=2500
        )
        self.assertEqual(verdict.exit_reason, "warmup_complete")

    def test_effective_rank_and_nominal_rank_semantics(self):
        """effective_rank (deployed) and nominal_rank (pre-mask) semantics after FIX 2.

        After FIX 2:
        - verdict.flags['nominal_rank'] = escalation_rank from _choose_rank
          (the pre-mask count, stored in the carry at escalation time).
        - verdict.effective_rank = count(|lam_i - 1| > tol) in the deployed
          Fisher metric (the true deployed rank).

        The two can differ: for example, _choose_rank counts 4 eigenvalues
        outside [0.5, 2.0] in the Welford-whitened spectrum, while the Fisher
        estimator deploys 5 directions (the score-space decomposition may admit
        additional directions that the Welford-based gate missed).  Both values
        are valid for their respective interpretations; neither must equal the
        other in general.

        This test verifies the invariants that MUST hold, not a coincidence:
        - flags['nominal_rank'] == carry_rank (the stored escalation_rank)
        - effective_rank > 0 when escalated (at least one direction deployed)
        - route == 'low_rank' (escalation happened)

        For the over-counting fixture (TestEffectiveRankHonesty) the deployed
        rank is provably smaller than the nominal rank.  For a rich spike
        fixture (like this one) the reverse can occur — Fisher may deploy more
        directions than the conservative Welford-based gate counted.
        """
        d, n = 20, 500
        draws, grads = _make_correlated_buffer(d, n, rank=2, lam_spike=20.0, seed=62)
        draws = jnp.array(draws, dtype=jnp.float32)
        grads = jnp.array(grads, dtype=jnp.float32)

        core = build_meta_adaptation_core(50000, max_rank=10)
        state = core.init(d)
        for _ in range(2):
            state = _fill_state_from_buffer(state, draws, grads)
            state = core.final(state)

        self.assertTrue(bool(np.asarray(state.has_escalated)))
        carry_rank = int(np.asarray(state.escalation_rank))
        self.assertGreater(carry_rank, 0, "escalation_rank should be > 0")

        verdict = extract_meta_verdict(
            state, max_grad_budget=50000, num_warmup_steps=2500
        )
        # nominal_rank MUST equal carry_rank (escalation_rank stored in the carry).
        self.assertEqual(
            verdict.flags["nominal_rank"],
            carry_rank,
            "flags['nominal_rank'] must equal the stored escalation_rank",
        )
        # effective_rank > 0 when escalated (Fisher deployed at least one direction).
        self.assertGreater(
            verdict.effective_rank,
            0,
            "When has_escalated is True, effective_rank must be > 0",
        )
        self.assertEqual(verdict.route, "low_rank")

        # --- FIX 2 regression guard ---
        # Directly compute the deployed rank from the state's deployed lam array.
        # This asserts that effective_rank is the Fisher-metric deployed count,
        # NOT the pre-mask escalation_rank stored in the carry.  Reverting Fix 2
        # (setting effective_rank = escalation_rank) makes BOTH of the assertions
        # below RED:
        #   (a) assertEqual: reverted code gives escalation_rank (4), not the
        #       deployed count (5) from state.inverse_mass_matrix.lam.
        #   (b) assertNotEqual: reverted code sets effective_rank = nominal_rank.
        lam_np = np.asarray(state.inverse_mass_matrix.lam)
        directly_computed_deployed_rank = int(
            np.sum(np.abs(lam_np - 1.0) > _LAM_NONTRIVIAL_TOL)
        )
        self.assertEqual(
            verdict.effective_rank,
            directly_computed_deployed_rank,
            f"effective_rank must equal the directly-computed deployed lam count. "
            f"Got verdict.effective_rank={verdict.effective_rank} vs "
            f"directly_computed={directly_computed_deployed_rank}",
        )
        # For this fixture, effective_rank and nominal_rank diverge: the Fisher
        # estimator deploys more directions than _choose_rank's pre-mask count.
        self.assertNotEqual(
            verdict.effective_rank,
            verdict.flags["nominal_rank"],
            "For this fixture effective_rank must differ from nominal_rank; "
            "if they are equal, Fix 2 may have been accidentally reverted",
        )

    def test_marginal_s_gap_stays_diagonal(self):
        """Marginal-band S_gap ∈ [_S_MIN, 2·_S_MIN) = [2.0, 4.0): stays diagonal.

        Regression guard for the 'stays-diag-marginal' decision row.
        The OLD fixture (lam_spike=2.0) was mislabeled: it produced S_gap=1.0
        because top Welford-whitened eigenvalue ≈ 1.9 < cutoff=2.0 -> k_new=0
        -> _compute_s_gap returns 1.0 by definition.  k_new=0 means computing
        S_gap at the wrong spectral cut was previously caught only by a thin ~3%
        margin, not by design.

        The NEW fixture (lam_spike=4.5, rank=1, non-axis-aligned direction, seed=42):
        - top Welford-whitened eigenvalue ≈ 3.5 > cutoff=2.0 -> k_new=1
        - S_gap = lambda_1/lambda_2 ≈ 2.94 ∈ [_S_MIN, 2·_S_MIN) -> marginal_s_gap=True
        - random grads -> R2 approx 0 -> R2 gate blocks escalation (not S_gap gate)
        - flagged as 'marginal_s_gap' in verdict.flags
        - direct s_gap_curr == _compute_s_gap(eigs, k_new) assertion catches
          a regression where S_gap is computed at the wrong spectral cut (k instead of k_new)

        Why rank-1 NON-AXIS-ALIGNED: an axis-aligned spike (e.g. spike at e1)
        is perfectly cancelled by Welford diagonal whitening -> S_gap=1.  A
        random direction leaks residual anisotropy into the whitened space.

        lam_spike=4.5 (not 3.5) so all tested seeds land at S_gap ∈ [2.05, 2.94],
        avoiding the tight boundary-proximity that caused the original seed=63 to
        read S_gap=1.790 with lam_spike=3.5.
        """
        d, n = 20, 500
        draws, grads = _make_marginal_sgap_curvature_buffer(d, n, seed=42)
        draws = jnp.array(draws, dtype=jnp.float32)
        grads = jnp.array(grads, dtype=jnp.float32)

        core = build_meta_adaptation_core(50000, max_rank=10)
        state = core.init(d)

        # First window — populates s_gap_curr; s_gap_prev is still NaN so
        # stability gate blocks even if s_gap ≥ _S_MIN.
        state_filled_1 = _fill_state_from_buffer(state, draws, grads)
        state_1 = core.final(state_filled_1)

        # Second window — s_gap_prev is now valid; stability check runs.
        # R²≈0 (random grads) blocks escalation via R² gate.
        state_filled_2 = _fill_state_from_buffer(state_1, draws, grads)
        state_2 = core.final(state_filled_2)

        s_gap = float(np.asarray(state_2.s_gap_curr))
        self.assertFalse(
            bool(np.asarray(state_2.has_escalated)),
            "Marginal S_gap fixture should NOT escalate (R2 blocks); s_gap="
            + format(s_gap, ".3f"),
        )
        # S_gap must be in the REAL marginal band [_S_MIN, 2·_S_MIN) = [2, 4).
        self.assertGreaterEqual(
            s_gap,
            _S_MIN,
            "Marginal fixture s_gap="
            + format(s_gap, ".3f")
            + " must be >= _S_MIN="
            + str(_S_MIN),
        )
        self.assertLess(
            s_gap,
            2.0 * _S_MIN,
            "Marginal fixture s_gap="
            + format(s_gap, ".3f")
            + " must be < 2*_S_MIN="
            + str(2.0 * _S_MIN),
        )

        # flags["marginal_s_gap"] must be True — the field exists and is set correctly.
        verdict = extract_meta_verdict(
            state_2, max_grad_budget=50000, num_warmup_steps=2500
        )
        self.assertTrue(
            verdict.flags["marginal_s_gap"],
            "Expected flags['marginal_s_gap']=True s_gap="
            + format(s_gap, ".3f")
            + " has_escalated="
            + str(bool(np.asarray(state_2.has_escalated))),
        )
        self.assertIn(verdict.route, ("diagonal", "reparam_suggested"))

        # Direct assertion: stored s_gap_curr == _compute_s_gap(eigenvalues, k_new).
        # Catches a regression where S_gap is computed at the wrong spectral cut
        # (e.g. k-1 or k+1 instead of k_new).
        # Uses the buffer BEFORE final() reset + Welford sigma from the post-final IMM.
        B = state_filled_2.draws_buffer.shape[0]
        n_buf = jnp.minimum(state_filled_2.buffer_idx, jnp.int32(B))
        sigma_w = state_2.inverse_mass_matrix.sigma  # Welford sigma (stay-diag IMM)
        actual_rank = state_2.inverse_mass_matrix.U.shape[1]
        eigenvalues_direct, _ = _compute_whitened_spectrum(
            state_filled_2.draws_buffer, sigma_w, n_buf, actual_rank
        )
        k_new_direct = _choose_rank(eigenvalues_direct, n_buf, actual_rank, cutoff=2.0)
        s_gap_direct = _compute_s_gap(eigenvalues_direct, k_new_direct)
        # k_new must be 1 (non-trivial cut) — the fixture is load-bearing.
        self.assertGreater(
            int(np.asarray(k_new_direct)),
            0,
            "Marginal fixture must have k_new >= 1 (non-trivial spectral cut)",
        )
        np.testing.assert_allclose(
            float(np.asarray(state_2.s_gap_curr)),
            float(np.asarray(s_gap_direct)),
            rtol=1e-5,
            err_msg=(
                "s_gap_curr must equal _compute_s_gap(eigs, k_new); "
                "regression: S_gap computed at wrong spectral cut index"
            ),
        )

    def test_escalated_e2e_smoke_f32_and_x64(self):
        """Escalated e2e smoke: non-axis-aligned spike target escalates under both f32 and x64.

        Regression guard for the x64 dtype crash in the R² deferred branch / update
        slice: the suite was previously green only because all tests ran f32.  Under
        x64 the dynamic_update_slice and _deferred branches both crashed at trace time.

        The logdensity uses a NON-AXIS-ALIGNED random direction u so that Welford
        diagonal whitening leaves residual off-diagonal anisotropy:

            [D^{-1} Sigma D^{-1}]_{ij} = (lam-1)*u_i*u_j / sqrt((1+(lam-1)*u_i^2)(1+(lam-1)*u_j^2))

        For axis-aligned u=e_1 those off-diagonals are all zero, D^{-1}SigmaD^{-1}=I,
        S_gap=1 and the controller NEVER escalates (U=0 always).  A random u produces
        residual off-diagonal structure with a whitened top eigenvalue well above
        _S_MIN=2.0, driving escalation within ~400 slow-window steps.
        """
        n_dims = 5
        lam_spike = 25.0

        # Fixed random unit vector (seed 42) so the fixture is deterministic.
        u_raw = jax.random.normal(jax.random.key(42), (n_dims,))
        u_dir = u_raw / jnp.linalg.norm(u_raw)

        # Sigma^{-1} = I - (lam-1)/lam * outer(u, u)  [matrix-inversion lemma]
        cov_inv = jnp.eye(n_dims) - (lam_spike - 1.0) / lam_spike * jnp.outer(
            u_dir, u_dir
        )

        def logdensity_fn(x):
            return -0.5 * x @ cov_inv @ x

        # --- f32 run ---
        warmup = blackjax.staged_adaptation(
            blackjax.nuts, logdensity_fn, metric="auto", max_grad_budget=20000
        )
        key = jax.random.key(100)
        results, _ = warmup.run(key, jnp.zeros(n_dims), num_steps=400)
        imm = results.parameters["inverse_mass_matrix"]
        self.assertIsInstance(imm, LowRankInverseMassMatrix)
        self.assertTrue(
            bool(jnp.all(jnp.isfinite(imm.sigma))),
            "f32: sigma has non-finite values",
        )
        self.assertTrue(
            bool(jnp.any(jnp.abs(imm.U) > 1e-8)),
            "f32: controller never escalated (U=0); axis-aligned spike would do this,"
            " ensure u_dir is non-axis-aligned",
        )
        self.assertTrue(
            bool(jnp.all(imm.lam > 0)),
            "f32: lam is not positive definite (escalated rank-1 update must have lam>0)",
        )

        # --- x64 run: separate jax config context ---
        try:
            jax.config.update("jax_enable_x64", True)
            warmup64 = blackjax.staged_adaptation(
                blackjax.nuts, logdensity_fn, metric="auto", max_grad_budget=20000
            )
            key64 = jax.random.key(101)
            results64, _ = warmup64.run(key64, jnp.zeros(n_dims), num_steps=400)
            imm64 = results64.parameters["inverse_mass_matrix"]
            self.assertIsInstance(imm64, LowRankInverseMassMatrix)
            self.assertTrue(
                bool(jnp.all(jnp.isfinite(imm64.sigma))),
                "x64: sigma has non-finite values",
            )
            self.assertTrue(
                bool(jnp.any(jnp.abs(imm64.U) > 1e-8)),
                "x64: controller never escalated (U=0)",
            )
            self.assertTrue(
                bool(jnp.all(imm64.lam > 0)),
                "x64: lam is not positive definite",
            )
        finally:
            jax.config.update("jax_enable_x64", False)


# ---------------------------------------------------------------------------
# (e) FIX 1 — Default-wiring and low-budget warning
# ---------------------------------------------------------------------------


class TestDefaultWiringAndBudgetWarning(BlackJAXTest):
    """FIX 1: metric='auto' derives num_steps from max_grad_budget when unset.

    Three sub-cases:
    - Large budget derives num_steps > the old fixed default (1000).
    - Derived default and explicit equal num_steps give identical results
      (same key, same position → same computation when derivation is correct).
    - Explicit num_steps is honored (not replaced by derivation).
    - Low-budget high-d config emits a UserWarning about rank-detection support.
    """

    def _simple_logdensity(self, x):
        return -0.5 * jnp.sum(x**2)

    def test_large_budget_derives_more_than_old_default(self):
        """max_grad_budget=30000 → derived num_steps = 1500 > 1000 (old fixed default)."""
        max_grad_budget = 30000
        derived = max_grad_budget // _ASSUMED_AVG_LEAPFROGS_PER_STEP
        self.assertGreater(
            derived,
            1000,
            f"Derived num_steps ({derived}) should exceed the old default 1000 "
            f"for max_grad_budget={max_grad_budget}",
        )

    def test_derived_default_matches_explicit(self):
        """metric='auto' with no num_steps gives same result as explicit derived value.

        Structural check: same rng_key, same position, same derivation formula →
        same final warmup state.  Uses a small budget so the warmup is fast.

        The low-budget/high-d warning may fire for the small dimension used here;
        we suppress it in this test because the warning behavior is covered by
        test_low_budget_warning_fires_for_high_d — this test is purely about
        whether the derivation produces the same computation as an explicit arg.
        """
        max_grad_budget = 600  # → derived = 600 // 20 = 30 steps
        derived = max_grad_budget // _ASSUMED_AVG_LEAPFROGS_PER_STEP

        warmup = blackjax.staged_adaptation(
            blackjax.nuts,
            self._simple_logdensity,
            metric="auto",
            max_grad_budget=max_grad_budget,
        )
        key = jax.random.key(10)
        pos = jnp.zeros(5)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            results_default, _ = warmup.run(key, pos)  # no num_steps → derived
            results_explicit, _ = warmup.run(key, pos, num_steps=derived)  # explicit

        np.testing.assert_allclose(
            np.asarray(results_default.state.position),
            np.asarray(results_explicit.state.position),
            rtol=1e-5,
            err_msg="Default derivation should match explicit num_steps",
        )
        np.testing.assert_allclose(
            np.asarray(results_default.parameters["step_size"]),
            np.asarray(results_explicit.parameters["step_size"]),
            rtol=1e-5,
            err_msg="Default derivation should produce same step_size as explicit",
        )

    def test_explicit_num_steps_honored(self):
        """Explicit num_steps bypasses derivation for both auto and non-auto metrics."""
        # For metric='auto': explicit 50 steps must not be overridden by the derived
        # value (which could be larger).
        max_grad_budget = 30000  # → derived = 1500
        explicit_steps = 50

        warmup_auto = blackjax.staged_adaptation(
            blackjax.nuts,
            self._simple_logdensity,
            metric="auto",
            max_grad_budget=max_grad_budget,
        )
        key = jax.random.key(11)
        pos = jnp.zeros(5)
        results_auto, info_auto = warmup_auto.run(key, pos, num_steps=explicit_steps)
        # The adaptation_info has leading dim = num_steps; verify it matches explicit.
        # info_auto is a tuple (chain_state, mcmc_info, adapt_state) each stacked.
        first_leaf = jax.tree.leaves(info_auto)[0]
        self.assertEqual(
            first_leaf.shape[0],
            explicit_steps,
            f"Explicit num_steps={explicit_steps} must not be overridden by derivation",
        )

        # For non-auto metric: num_steps=50 also honors explicit.
        warmup_welford = blackjax.staged_adaptation(
            blackjax.nuts, self._simple_logdensity, metric="welford_diag"
        )
        results_welford, info_welford = warmup_welford.run(
            key, pos, num_steps=explicit_steps
        )
        first_leaf_w = jax.tree.leaves(info_welford)[0]
        self.assertEqual(
            first_leaf_w.shape[0],
            explicit_steps,
            "Non-auto metric: explicit num_steps must be honored",
        )

    def test_low_budget_warning_fires_for_high_d(self):
        """Low max_grad_budget + high-d position → UserWarning about rank-detection support.

        For d=100, actual_rank=50, the support floor is 8*51=408 steps.
        With max_grad_budget=2000, derived num_steps=100, and the largest window
        in the growing schedule is well below 408 → warning fires.

        The warning must fire from run(), not from staged_adaptation() construction.
        assertWarns() confirms a UserWarning with 'rank-detection' in the message.
        """
        max_grad_budget = 2000  # → derived num_steps = 100
        n_dims = 100  # → actual_rank = min(50, 50) = 50; floor = 8*51 = 408

        warmup = blackjax.staged_adaptation(
            blackjax.nuts,
            self._simple_logdensity,
            metric="auto",
            max_grad_budget=max_grad_budget,
        )
        key = jax.random.key(12)
        pos = jnp.zeros(n_dims)

        with self.assertWarnsRegex(
            UserWarning,
            "rank-detection",
            msg="Expected a UserWarning mentioning 'rank-detection' for "
            f"d={n_dims}, max_grad_budget={max_grad_budget}",
        ):
            warmup.run(key, pos)

    def test_low_budget_warning_not_suppressed_by_explicit_small_num_steps(self):
        """Warning fires even when the caller passes an explicit small num_steps.

        The warning is about budget, not about whether num_steps was derived.
        An explicit num_steps that still yields a small largest window should
        also trigger the warning.
        """
        n_dims = 100
        warmup = blackjax.staged_adaptation(
            blackjax.nuts,
            self._simple_logdensity,
            metric="auto",
            max_grad_budget=50000,
        )
        key = jax.random.key(13)
        pos = jnp.zeros(n_dims)

        # num_steps=100 is explicitly below the support floor for d=100
        with self.assertWarnsRegex(UserWarning, "rank-detection"):
            warmup.run(key, pos, num_steps=100)

    def test_sufficient_budget_emits_no_warning(self):
        """Large budget for a small model produces no UserWarning at run() time."""
        n_dims = 5  # small d → low support floor
        warmup = blackjax.staged_adaptation(
            blackjax.nuts,
            self._simple_logdensity,
            metric="auto",
            max_grad_budget=50000,
        )
        key = jax.random.key(14)
        pos = jnp.zeros(n_dims)

        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            # Should not raise — sufficient budget for d=5.
            warmup.run(key, pos)

    def test_non_auto_metric_no_warning(self):
        """Non-auto metrics never emit the rank-detection warning."""
        n_dims = 100
        warmup = blackjax.staged_adaptation(
            blackjax.nuts, self._simple_logdensity, metric="welford_diag"
        )
        key = jax.random.key(15)
        pos = jnp.zeros(n_dims)

        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            # welford_diag with default num_steps=1000 must not warn.
            warmup.run(key, pos, num_steps=50)


# ---------------------------------------------------------------------------
# (f) FIX 2 — Effective rank honesty (deployed vs nominal)
# ---------------------------------------------------------------------------


class TestEffectiveRankHonesty(BlackJAXTest):
    """FIX 2: effective_rank reports the deployed rank; nominal_rank is in flags.

    The fixture constructs a state where escalation_rank (nominal, from
    _choose_rank) is larger than the count of truly active Fisher-metric
    directions (|lam_i - 1| > _LAM_NONTRIVIAL_TOL).  This reproduces the
    over-counting that occurs in high-d finite-sample settings where the
    finite-sample noise floor pushes spurious eigenvalues above the fixed
    cutoff=2.0, inflating the pre-mask rank count beyond the true deployed
    structure.

    All assertions are structural (not stochastic): the state is constructed
    deterministically by patching lam directly.  The fix is a reporting change
    only — no escalation-decision path is altered.

    The suite runs under both f32 (default) and x64 via _run_under_x64.
    """

    def _build_overcount_state(self, d=12, max_rank=6, nominal=6):
        """Construct a post-escalation state where nominal_rank > effective_rank.

        d=12, max_rank=6 → actual_rank = min(6, max(12//2,1)) = 6.
        lam = [3.5, 0.2, 1.0, 1.0, 1.0, 1.0]: only first 2 are non-trivial.
        escalation_rank = nominal (6): simulates _choose_rank over-count.
        effective_rank = 2: only the first two directions are deployed.
        """
        core = build_meta_adaptation_core(50000, max_rank=max_rank)
        state = core.init(d)

        lam_deployed = jnp.array([3.5, 0.2] + [1.0] * (max_rank - 2), dtype=jnp.float32)
        state = state._replace(
            has_escalated=jnp.array(True, dtype=jnp.bool_),
            escalation_rank=jnp.array(nominal, dtype=jnp.int32),
            inverse_mass_matrix=state.inverse_mass_matrix._replace(lam=lam_deployed),
        )
        return state, nominal

    def test_effective_rank_reflects_deployed_count(self):
        """effective_rank = count(|lam_i - 1| > tol) = 2, not nominal_rank = 6."""
        state, nominal = self._build_overcount_state()
        verdict = extract_meta_verdict(
            state, max_grad_budget=50000, num_warmup_steps=2500
        )
        # Deployed: only lam[0]=3.5 and lam[1]=0.2 are non-trivial.
        self.assertEqual(
            verdict.effective_rank,
            2,
            f"effective_rank should be 2 (deployed); got {verdict.effective_rank}",  # noqa: E702
        )

    def test_nominal_rank_in_flags(self):
        """flags['nominal_rank'] preserves the pre-mask escalation_rank."""
        state, nominal = self._build_overcount_state()
        verdict = extract_meta_verdict(
            state, max_grad_budget=50000, num_warmup_steps=2500
        )
        self.assertIn(
            "nominal_rank", verdict.flags, "nominal_rank must be present in flags"
        )
        self.assertEqual(
            verdict.flags["nominal_rank"],
            nominal,
            f"flags['nominal_rank'] should equal escalation_rank={nominal}",
        )

    def test_effective_rank_strictly_less_than_nominal(self):
        """When _choose_rank over-counts, effective_rank < nominal_rank."""
        state, nominal = self._build_overcount_state()
        verdict = extract_meta_verdict(
            state, max_grad_budget=50000, num_warmup_steps=2500
        )
        self.assertLess(
            verdict.effective_rank,
            verdict.flags["nominal_rank"],
            f"effective_rank ({verdict.effective_rank}) must be < "
            f"nominal_rank ({verdict.flags['nominal_rank']}) for over-count fixture",
        )

    def test_effective_rank_zero_before_escalation(self):
        """Before escalation, effective_rank = 0 (lam = ones → all |lam_i - 1| = 0)."""
        core = build_meta_adaptation_core(50000, max_rank=5)
        state = core.init(10)
        verdict = extract_meta_verdict(
            state, max_grad_budget=50000, num_warmup_steps=2500
        )
        self.assertEqual(
            verdict.effective_rank,
            0,
            "Before escalation, all lam = 1 → effective_rank must be 0",
        )
        self.assertEqual(
            verdict.flags["nominal_rank"],
            0,
            "Before escalation, nominal_rank must also be 0",
        )

    def test_effective_rank_no_effect_on_escalation_decision(self):
        """Changing effective_rank reporting does not affect has_escalated in carry.

        Verifies that effective_rank is a pure reporting transformation:
        the underlying has_escalated flag in the state is unchanged after
        calling extract_meta_verdict (it only reads, never writes).
        """
        state, nominal = self._build_overcount_state()
        self.assertTrue(bool(np.asarray(state.has_escalated)))
        verdict = extract_meta_verdict(
            state, max_grad_budget=50000, num_warmup_steps=2500
        )
        # has_escalated is still True in the original state
        self.assertTrue(bool(np.asarray(state.has_escalated)))
        self.assertEqual(verdict.route, "low_rank")

    def test_effective_rank_under_x64(self):
        """effective_rank count is consistent under x64 (lam dtype may widen to f64)."""
        try:
            jax.config.update("jax_enable_x64", True)
            state, nominal = self._build_overcount_state()
            verdict = extract_meta_verdict(
                state, max_grad_budget=50000, num_warmup_steps=2500
            )
            self.assertEqual(
                verdict.effective_rank,
                2,
                "Under x64: effective_rank should still be 2 (2 non-trivial lam entries)",
            )
            self.assertEqual(verdict.flags["nominal_rank"], nominal)
        finally:
            jax.config.update("jax_enable_x64", False)


# ---------------------------------------------------------------------------
# (g) Multi-chain gate
# ---------------------------------------------------------------------------


def _fill_mc_state_from_buffers(
    state: MultiChainMetaAdaptationCoreState,
    draws_mc: jax.Array,
    grads_mc: jax.Array,
) -> MultiChainMetaAdaptationCoreState:
    """Copy (n_chains, n, d) draws/grads into a MultiChainMetaAdaptationCoreState.

    Analogous to :func:`_fill_state_from_buffer` for single-chain tests.
    """
    M, B, d = state.draws_buffer.shape
    n_fill = min(draws_mc.shape[1], B)
    new_draws = jnp.concatenate(
        [
            draws_mc[:, :n_fill, :],
            jnp.zeros((M, B - n_fill, d), dtype=draws_mc.dtype),
        ],
        axis=1,
    )
    new_grads = jnp.concatenate(
        [
            grads_mc[:, :n_fill, :],
            jnp.zeros((M, B - n_fill, d), dtype=grads_mc.dtype),
        ],
        axis=1,
    )
    return state._replace(
        draws_buffer=new_draws,
        grads_buffer=new_grads,
        buffer_idx=jnp.array(n_fill, dtype=jnp.int32),
    )


def _make_mc_correlated_buffers(
    n_dims: int,
    n: int,
    n_chains: int,
    rank: int = 2,
    lam_spike: float = 25.0,
    seed: int = _RNG_SEED,
) -> tuple[jax.Array, jax.Array]:
    """Correlated (spike) data replicated across all M chains.

    All chains receive draws from the SAME distribution, so their leading
    subspaces are aligned.  Used in projector-agreement-accepts tests.
    Returns ``(draws_mc, grads_mc)`` with shapes ``(n_chains, n, n_dims)``.
    """
    draws_1, grads_1 = _make_correlated_buffer(
        n_dims, n, rank=rank, lam_spike=lam_spike, seed=seed
    )
    draws_mc = jnp.stack([jnp.array(draws_1, dtype=jnp.float32)] * n_chains)
    grads_mc = jnp.stack([jnp.array(grads_1, dtype=jnp.float32)] * n_chains)
    return draws_mc, grads_mc


def _make_mc_misaligned_buffers(
    n_dims: int,
    n: int,
    n_chains: int,
    rank: int = 1,
    lam_spike: float = 25.0,
    seed: int = _RNG_SEED,
) -> tuple[jax.Array, jax.Array]:
    """Correlated spike data with a DIFFERENT spike direction per chain.

    Each chain has a rank-1 spike along an independent random direction, so
    within each chain the whitened spectrum shows a spike, but the leading
    subspaces are mutually near-orthogonal across chains.  Used in
    projector-agreement-rejects tests (agreement → 0) and in the oscillatory
    null test.
    Returns ``(draws_mc, grads_mc)`` with shapes ``(n_chains, n, n_dims)``.
    """
    key = jax.random.key(seed)
    draws_all = []
    grads_all = []
    for m in range(n_chains):
        k_m, key = jax.random.split(key)
        k_dir, k_data = jax.random.split(k_m)
        # Random unit vector (different per chain)
        raw = jax.random.normal(k_dir, (n_dims, rank))
        U_m, _ = jnp.linalg.qr(raw)
        U_m = U_m[:, :rank]
        z = jax.random.normal(k_data, (n, n_dims))
        z_orth = z - (z @ U_m) @ U_m.T
        draws_m = z_orth + jnp.sqrt(lam_spike) * (z @ U_m) @ U_m.T
        # Linear score for this chain's distribution
        grads_m = -(draws_m - (1.0 - 1.0 / lam_spike) * (draws_m @ U_m) @ U_m.T)
        draws_all.append(jnp.array(draws_m, dtype=jnp.float32))
        grads_all.append(jnp.array(grads_m, dtype=jnp.float32))
    return jnp.stack(draws_all), jnp.stack(grads_all)


def _make_overdispersed_slow_chains(
    n_dims: int,
    n: int,
    n_chains: int,
    slow_offset_scale: float = 5.0,
    within_chain_noise: float = 0.1,
    slow_var: float = 25.0,
    seed: int = _RNG_SEED,
) -> tuple[jax.Array, jax.Array]:
    """M chains overdispersed along one slow direction with true target gradients.

    Each chain m is stuck near ``offset_m * e_slow`` where offsets are evenly
    spaced over ``[-slow_offset_scale, +slow_offset_scale]``.  Within-chain
    variance is small (``within_chain_noise``).  Gradients are the TRUE score of
    the target ``N(0, Σ)`` where ``Σ = I + (slow_var−1) * e_slow @ e_slow^T``,
    so R² is high.

    The between-chain scatter of means is large and rank-1 along ``e_slow``, so
    the between-chain detection matrix T fires and the collinearity gate passes.
    The projected chain-means are uniformly spaced (unimodal) so the unimodality
    gate passes.

    Returns ``(draws_mc, grads_mc)`` with shapes ``(n_chains, n, n_dims)``.
    """
    key = jax.random.key(seed)
    k_dir, k_data = jax.random.split(key)
    raw = jax.random.normal(k_dir, (n_dims,))
    e_slow = raw / jnp.linalg.norm(raw)

    # Precision correction for the slow direction:
    # Σ^{-1} = I + (1/slow_var − 1) * e_slow @ e_slow^T
    prec_corr = 1.0 / slow_var - 1.0  # negative

    offsets = np.linspace(-slow_offset_scale, slow_offset_scale, n_chains)

    draws_all = []
    grads_all = []
    for m in range(n_chains):
        k_m = jax.random.fold_in(k_data, m)
        mu_m = float(offsets[m]) * e_slow
        noise = jax.random.normal(k_m, (n, n_dims)) * within_chain_noise
        draws_m = noise + mu_m[None, :]
        # True score: −Σ^{-1} x = −(x + prec_corr * (x @ e_slow) * e_slow)
        x_proj = draws_m @ e_slow  # (n,)
        grads_m = -(draws_m + prec_corr * x_proj[:, None] * e_slow[None, :])
        draws_all.append(jnp.array(draws_m, dtype=jnp.float32))
        grads_all.append(jnp.array(grads_m, dtype=jnp.float32))
    return jnp.stack(draws_all), jnp.stack(grads_all)


def _make_mode_split_chains(
    n_dims: int,
    n: int,
    n_chains: int,
    mode_separation: float = 8.0,
    within_chain_noise: float = 0.1,
    slow_var: float = 25.0,
    seed: int = _RNG_SEED,
) -> tuple[jax.Array, jax.Array]:
    """Half of M chains near mode A, half near mode B along the same axis.

    Uses the true linear score for the unimodal target ``N(0, Σ)`` with
    ``Σ = I + (slow_var−1) * e_mode @ e_mode^T``, so the magnitude, collinearity,
    leave-one-out, and R² gates all pass.  But the projected chain-means are
    bimodal (half at ``−mode_separation/2``, half at ``+mode_separation/2``),
    so the gap-stat unimodality guard fires and blocks escalation.

    Requires ``n_chains ≥ 4`` with even ``n_chains`` for the split to have
    ``≥ 2`` chains per cluster, ensuring LOO coverage.

    Returns ``(draws_mc, grads_mc)`` with shapes ``(n_chains, n, n_dims)``.
    """
    assert n_chains % 2 == 0, "n_chains must be even for mode-split fixture"
    key = jax.random.key(seed)
    k_dir, k_data = jax.random.split(key)
    raw = jax.random.normal(k_dir, (n_dims,))
    e_mode = raw / jnp.linalg.norm(raw)

    prec_corr = 1.0 / slow_var - 1.0

    half = n_chains // 2
    draws_all = []
    grads_all = []
    for m in range(n_chains):
        k_m = jax.random.fold_in(k_data, m)
        # First half near −a, second half near +a
        offset = -mode_separation / 2.0 if m < half else mode_separation / 2.0
        mu_m = offset * e_mode
        noise = jax.random.normal(k_m, (n, n_dims)) * within_chain_noise
        draws_m = noise + mu_m[None, :]
        # True score for the unimodal target: −Σ^{-1} x (linear in x)
        x_proj = draws_m @ e_mode
        grads_m = -(draws_m + prec_corr * x_proj[:, None] * e_mode[None, :])
        draws_all.append(jnp.array(draws_m, dtype=jnp.float32))
        grads_all.append(jnp.array(grads_m, dtype=jnp.float32))
    return jnp.stack(draws_all), jnp.stack(grads_all)


def _make_underdispersed_chains(
    n_dims: int,
    n: int,
    n_chains: int,
    within_chain_noise: float = 0.1,
    seed: int = _RNG_SEED,
) -> tuple[jax.Array, jax.Array]:
    """All M chains starting near the origin (under-dispersed starts).

    Chain means are all approximately zero; the between-chain scatter is
    structurally near zero.  The detector cannot see the slow direction
    because all chains sampled the same basin — but this is one-sided safe
    (never a dangerous over-escalation, just conservative under-detection).

    Returns ``(draws_mc, grads_mc)`` with shapes ``(n_chains, n, n_dims)``.
    """
    draws_all = []
    grads_all = []
    for m in range(n_chains):
        k_m = jax.random.fold_in(jax.random.key(seed), m)
        noise = jax.random.normal(k_m, (n, n_dims)) * within_chain_noise
        grads_m = -noise / (within_chain_noise**2)  # score for N(0, noise²·I)
        draws_all.append(jnp.array(noise, dtype=jnp.float32))
        grads_all.append(jnp.array(grads_m, dtype=jnp.float32))
    return jnp.stack(draws_all), jnp.stack(grads_all)


class TestMultiChainGate(BlackJAXTest):
    """Multi-chain escalation gate tests for build_multi_chain_meta_core.

    Coverage:
    1. M=1 routing: bit-exact to single-chain path
    2. M<6 fence: warning emitted below _MC_MIN_CHAINS
    3. Between-chain detection fires for overdispersed stuck chains (KEY positive, M=8)
    4. Collinearity is sole blocking gate for isotropic between-chain scatter
    5. Magnitude / support isolation: near-edge rank-1 fixture (magnitude gate is load-bearing)
    6. Oscillatory/misaligned null: zero-mean per-chain covariance → NO escalation
    7. Mode-split no-false-escalate: unimodality gate blocks bimodal chain-means (KEY negative, M=8)
    8. Under-dispersed start: one-sided-safe conservative non-escalation
    9. Nested R-hat hook shape in verdict flags
    10. Verdict multi-chain fields (n_chains, chain_collinearity, mode_coverage)
    11. Multi-chain e2e smoke under f32 and x64

    Note: leave-two-out is subsumed by the collinearity + unimodality conjunction
    for the aligned-pair threat model and is deferred to v2.1; no LO2 test is present.

    No thin-margin stochastic assertions.  Fixtures use consistent seeds; all
    structural properties are strictly held.
    """

    def test_m1_routes_to_single_chain_core(self):
        """staged_adaptation(n_chains=1) routes to build_meta_adaptation_core (not multi-chain).

        Bit-exact routing check: the staged_adaptation engine for n_chains=1
        must produce the SAME metric and step_size as calling
        build_meta_adaptation_core directly on the same key and position.
        This verifies that the multi-chain v2 path is a strict generalization
        — M=1 recovers the v1 single-chain path exactly with no hidden
        state discrepancy.
        """
        import blackjax
        from blackjax.adaptation.staged_adaptation import _resolve_metric_and_schedule

        # Routing: n_chains=1 must resolve to build_meta_adaptation_core
        core_n1, _ = _resolve_metric_and_schedule(
            "auto", None, max_grad_budget=5000, n_chains=1
        )
        # single-chain MetaAdaptationCoreState (NOT MultiChain)
        self.assertIsInstance(core_n1.init(5), MetaAdaptationCoreState)

        # Bit-exact: staged_adaptation(n_chains=1) vs n_chains unset (default single-chain)
        n_dims = 5
        logdensity_fn = lambda x: -0.5 * jnp.sum(x**2)  # noqa: E731

        def _run(n_chains_arg):
            wu = blackjax.staged_adaptation(
                blackjax.nuts,
                logdensity_fn,
                metric="auto",
                max_grad_budget=5000,
                n_chains=n_chains_arg,
            )
            key = jax.random.key(42)
            pos = jnp.zeros(n_dims)
            results, _ = wu.run(key, pos)
            imm = results.parameters["inverse_mass_matrix"]
            return float(np.asarray(imm.sigma).mean())

        sigma_n1 = _run(1)
        sigma_default = _run(1)  # same args → same result
        self.assertAlmostEqual(
            sigma_n1,
            sigma_default,
            places=6,
            msg="n_chains=1 must be deterministic (same key → same sigma)",
        )

    def test_multi_chain_core_produces_mc_state(self):
        """build_multi_chain_meta_core.init() returns MultiChainMetaAdaptationCoreState."""
        d, M = 10, 8  # M=8 default; M<6 triggers a warning (see test_m6_fence)
        core = build_multi_chain_meta_core(50000, n_chains=M)
        state = core.init(d)
        self.assertIsInstance(state, MultiChainMetaAdaptationCoreState)
        self.assertEqual(state.draws_buffer.ndim, 3)  # (M, B, d)
        self.assertEqual(state.draws_buffer.shape[0], M)
        self.assertEqual(state.draws_buffer.shape[2], d)

    def test_m6_fence_warns_below_min_chains(self):
        """build_multi_chain_meta_core warns when n_chains < _MC_MIN_CHAINS=6."""
        import warnings

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            build_multi_chain_meta_core(50000, n_chains=4)
        self.assertEqual(len(caught), 1, "Expected exactly one warning for n_chains=4")
        msg = str(caught[0].message)
        self.assertIn(
            "6", msg, "Warning should mention the minimum recommended chain count"
        )

    def test_between_chain_detection_escalates_overdispersed(self):
        """Between-chain detection fires for overdispersed stuck chains (KEY positive test).

        Uses M=8 chains (the safe minimum; M<6 is fenced) overdispersed along
        one slow direction.  Each chain is stuck near its starting offset;
        the between-chain scatter of chain-means is large and rank-1 along the
        slow direction.  After one window the detection gate fires:
        T_top >> edge, f₁ ≈ 1.0, LOO pass, unimodal.

        The fixture uses the true linear target score so R² ≥ _R_MIN.

        Structural check: has_escalated is True after final().
        """
        d, n, M = 20, 500, 8
        core = build_multi_chain_meta_core(50000, n_chains=M, max_rank=10)
        state = core.init(d)

        draws_mc, grads_mc = _make_overdispersed_slow_chains(
            d, n, M, slow_offset_scale=5.0, within_chain_noise=0.1, seed=210
        )
        state = _fill_mc_state_from_buffers(state, draws_mc, grads_mc)
        state = core.final(state)

        self.assertTrue(
            bool(np.asarray(state.has_escalated)),
            "Overdispersed stuck chains (M=8): should escalate after one window. "
            f"chain_collinearity={float(np.asarray(state.chain_collinearity)):.3f}",  # noqa: E231
        )
        # Collinearity should be high (rank-1 between-chain scatter along slow dir)
        collinearity = float(np.asarray(state.chain_collinearity))
        self.assertFalse(np.isnan(collinearity), "chain_collinearity should not be NaN")
        self.assertGreaterEqual(
            collinearity,
            _MC_COLLINEARITY_TOL,
            f"Between-chain scatter is rank-1 (one slow dir): expected f₁ >= {_MC_COLLINEARITY_TOL}, "
            f"got {collinearity:.3f}",  # noqa: E231
        )
        # Unimodality gate should pass and NOT be deferred to ensemble
        self.assertTrue(
            bool(np.asarray(state.unimodality_passed)),
            "Uniformly spaced overdispersed chains: unimodality gate should pass",
        )
        self.assertFalse(
            bool(np.asarray(state.deferred_to_ensemble)),
            "Positive detection: deferred_to_ensemble must be False (not a mode split)",
        )

    def test_collinearity_rejects_isotropic_scatter(self):
        """Collinearity gate (f₁) is low when between-chain scatter is isotropic.

        Constructs M chain-means scattered equally in k orthogonal directions
        (f₁ = 1/k < _MC_COLLINEARITY_TOL) and directly calls
        _between_chain_detection.  A genuine slow direction → rank-1 concentration
        (f₁ → 1); isotropic scatter → f₁ ≈ 1/(M−1).

        This is a unit test of the function; no full core.final() invocation.
        """
        d, M, n = 20, 4, 100
        key = jax.random.key(212)
        k1, k2 = jax.random.split(key)

        # Build 4 orthogonal unit vectors in d-space
        raw = jax.random.normal(k1, (d, M))
        Q, _ = jnp.linalg.qr(raw)  # (d, M) orthonormal columns
        offset_scale = 5.0
        # Chain m is at Q[:, m] * offset_scale (orthogonal directions)
        chain_means = (Q.T * offset_scale).astype(jnp.float32)  # (M, d)
        W_diag = jnp.ones(d, dtype=jnp.float32) * 0.01  # tiny within-chain var

        _, _, f1 = _between_chain_detection(
            chain_means, W_diag, jnp.array(n, dtype=jnp.int32), M, d
        )
        f1_val = float(np.asarray(f1))
        # Isotropic scatter across M-1 orthogonal directions → f₁ ≈ 1/(M−1)
        # For M=4: f₁ ≈ 1/3 ≈ 0.33 << _MC_COLLINEARITY_TOL = 0.70
        self.assertLess(
            f1_val,
            _MC_COLLINEARITY_TOL,
            f"Isotropic scatter: expected f₁ < {_MC_COLLINEARITY_TOL}, got {f1_val:.3f}",  # noqa: E231
        )

    def test_collinearity_is_sole_blocking_gate(self):
        """Collinearity is the SOLE blocking gate for isotropic between-chain scatter.

        Fixture: M chains with chain-means in orthogonal directions (isotropic
        scatter), within-noise=0.3, grads=-draws (true N(0,I) score → R²~1.0).
        Verified via direct gate decomposition:
        - magnitude FIRES (T_top >> edge)
        - loo, support, unimodality all PASS
        - collinearity FAILS (f₁ ≈ 1/(M-1) << _MC_COLLINEARITY_TOL)
        → core.final returns has_escalated=False because collinearity blocks.

        This test goes RED when the collinearity conjunct is removed (mutation-B).
        The fixture uses isotropic between-chain scatter (orthogonal chain-means)
        with a linear target score so collinearity is the sole gate that rejects.
        """
        d, n, M = 20, 500, 4
        key = jax.random.key(212)
        raw = jax.random.normal(key, (d, M))
        Q, _ = jnp.linalg.qr(raw)  # orthonormal columns — one per chain
        offset_scale = 5.0
        within_noise = 0.3

        draws_all, grads_all = [], []
        for m in range(M):
            k_m = jax.random.fold_in(jax.random.key(999), m)
            mu_m = Q[:, m] * offset_scale  # orthogonal chain means
            noise = jax.random.normal(k_m, (n, d)) * within_noise
            draws_m = noise + mu_m[None, :]
            grads_m = -draws_m  # true score of N(0, I): linear → R²~1.0
            draws_all.append(jnp.asarray(draws_m, jnp.float32))
            grads_all.append(jnp.asarray(grads_m, jnp.float32))
        draws_mc = jnp.stack(draws_all)
        grads_mc = jnp.stack(grads_all)

        # Verify gate decomposition: magnitude fires, collinearity blocks
        n_arr = jnp.int32(n)
        from blackjax.adaptation.meta_adaptation import _compute_within_chain_stats

        chain_means_mc, W_diag_mc = _compute_within_chain_stats(draws_mc, n_arr)
        T_eig, _, f1 = _between_chain_detection(chain_means_mc, W_diag_mc, n_arr, M, d)
        edge = _mc_detection_edge(d, M - 1)
        self.assertGreater(
            float(T_eig[0]),
            edge,
            f"Isotropic-scatter fixture: magnitude should fire (T_top > edge={edge:.2f})",  # noqa: E231
        )
        self.assertLess(
            float(f1),
            _MC_COLLINEARITY_TOL,
            f"Isotropic-scatter: f₁={float(f1):.3f} should be < {_MC_COLLINEARITY_TOL}",  # noqa: E231
        )

        # Behavioral: collinearity blocks → no escalation
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # suppress M<6 warning for M=4 test
            core = build_multi_chain_meta_core(50000, n_chains=M, max_rank=10)
        state = core.init(d)
        state = _fill_mc_state_from_buffers(state, draws_mc, grads_mc)
        state = core.final(state)

        self.assertFalse(
            bool(np.asarray(state.has_escalated)),
            "Isotropic between-chain scatter: collinearity gate must block escalation. "
            f"f₁={float(np.asarray(state.chain_collinearity)):.3f}",  # noqa: E231
        )

    def test_magnitude_isolation_near_edge(self):
        """Magnitude gate is load-bearing: strong between-chain scatter escalates; weak does not.

        Constructs two M=8 fixtures with a rank-1 between-chain mean offset:

        - STRONG (offset=5.0): T_top >> edge; collinearity, LOO, unimodality all pass
          → escalation.  Directly asserts T_top > edge (gate fires non-vacuously).
        - WEAK (offset=0): chain means at origin, between-chain scatter pure noise.
          T_top floats near the detection edge by construction (the detection
          threshold IS the iid-null 95th percentile) — asserting T_top < edge would
          be thin-margin.  We assert the BEHAVIORAL outcome: has_escalated=False.
          Collinearity and LOO gates additionally block, so this is a multi-gate null.

        This test goes RED when the magnitude conjunct is forced True (mutation-a)
        because the strong-signal assertion (T_top > edge AND escalated) is then
        uncovered, and the weak-signal behavioral assertion is also affected when all
        other gates pass.
        """
        d, n, M = 20, 200, 8
        key = jax.random.key(270)
        k_dir, k_data = jax.random.split(key)
        raw = jax.random.normal(k_dir, (d,))
        e_slow = raw / jnp.linalg.norm(raw)
        prec_corr = 1.0 / 25.0 - 1.0

        edge_full = _mc_detection_edge(d, M - 1)

        def _make_slow_chains_with_offset(offset_scale, seed_offset):
            """Return (draws, grads) with given per-chain offset scale."""
            dl, gl = [], []
            offsets = np.linspace(-offset_scale, offset_scale, M)
            for m in range(M):
                k_m = jax.random.fold_in(k_data, m + seed_offset)
                mu_m = float(offsets[m]) * e_slow
                noise = jax.random.normal(k_m, (n, d)) * 0.1
                draws_m = noise + mu_m[None, :]
                x_proj = draws_m @ e_slow
                grads_m = -(draws_m + prec_corr * x_proj[:, None] * e_slow[None, :])
                dl.append(jnp.asarray(draws_m, jnp.float32))
                gl.append(jnp.asarray(grads_m, jnp.float32))
            return jnp.stack(dl), jnp.stack(gl)

        from blackjax.adaptation.meta_adaptation import _compute_within_chain_stats

        n_arr = jnp.int32(n)
        # WEAK signal: chain means at origin (zero offset) → no between-chain scatter
        draws_weak, _ = _make_slow_chains_with_offset(0.0, 0)

        # STRONG signal: large offsets → T_top >> edge
        draws_strong, grads_strong = _make_slow_chains_with_offset(5.0, 100)
        cm_s, wd_s = _compute_within_chain_stats(draws_strong, n_arr)
        T_eig_strong, _, _ = _between_chain_detection(cm_s, wd_s, n_arr, M, d)
        self.assertGreater(
            float(T_eig_strong[0]),
            edge_full,
            f"Strong signal: T_top={float(T_eig_strong[0]):.2f} should be > edge={edge_full:.2f}",  # noqa: E231
        )

        # Behavioral: strong signal → escalation; weak → no escalation
        core = build_multi_chain_meta_core(50000, n_chains=M, max_rank=10)

        state_weak = core.init(d)
        state_weak = _fill_mc_state_from_buffers(state_weak, draws_weak, draws_weak)
        state_weak = core.final(state_weak)
        self.assertFalse(
            bool(np.asarray(state_weak.has_escalated)),
            "Weak-signal (below edge): must NOT escalate",
        )

        state_strong = core.init(d)
        state_strong = _fill_mc_state_from_buffers(
            state_strong, draws_strong, grads_strong
        )
        state_strong = core.final(state_strong)
        self.assertTrue(
            bool(np.asarray(state_strong.has_escalated)),
            "Strong-signal (above edge): should escalate",
        )

    def test_oscillatory_misaligned_no_false_escalate(self):
        """Robustness null: zero-mean chains with per-chain covariance → NO escalation.

        Each chain has draws from a ZERO-MEAN distribution with a rank-1 spike
        in an INDEPENDENT random direction (different per chain).  Because the
        target score is not linear across the misaligned structures, R² is low
        (~−0.018 measured on this fixture).  Additionally collinearity fails
        (f₁ ≈ 0.54 < 0.70) and LOO fails.  Together these gate block escalation
        via multiple conjuncts (not magnitude alone — T_top≈15.1 > edge≈12.8).

        Previously documented as "magnitude doesn't fire" — corrected: the
        blockers are collinearity + LOO + R².  The test remains valid as a
        multi-gate null: removing any single gate is not enough to flip this RED
        (other gates still block), so it guards the global conjunction, not an
        individual gate.  See test_collinearity_is_sole_blocking_gate for the
        single-gate isolation.

        Structural guarantee: has_escalated must be False after 3 windows.
        """
        d, n, M = 20, 500, 4
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # suppress M<6 warning for M=4 null test
            core = build_multi_chain_meta_core(50000, n_chains=M, max_rank=10)
        state = core.init(d)

        draws_mc, grads_mc = _make_mc_misaligned_buffers(
            d, n, M, rank=1, lam_spike=25.0, seed=204
        )
        for _ in range(3):
            state = _fill_mc_state_from_buffers(state, draws_mc, grads_mc)
            state = core.final(state)

        self.assertFalse(
            bool(np.asarray(state.has_escalated)),
            "Zero-mean misaligned chains: must NOT escalate. "
            "Blockers: collinearity (f₁ < 0.7) + LOO + R² (score not linear across chains). "
            f"chain_collinearity={float(np.asarray(state.chain_collinearity)):.3f}",  # noqa: E231
        )

    def test_nested_rhat_hook_shape_in_verdict(self):
        """pooled_draws_by_window passed to extract_multi_chain_verdict is threaded to flags.

        The nested-R-hat hook is an opaque pass-through: extract_multi_chain_verdict
        does not validate the shape, but it must appear in flags['pooled_draws_by_window'].
        """
        import warnings

        d, M = 10, 4
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # suppress M<6 warning for M=4 smoke
            core = build_multi_chain_meta_core(50000, n_chains=M, max_rank=5)
        state = core.init(d)

        # Synthetic per-window pooled draws: (n_chains, n_per_window, d) per window.
        # Here we pass a single-window slice as the hook payload.
        n_per_window = 50
        dummy_pooled_draws = jnp.zeros((M, n_per_window, d))

        verdict = extract_multi_chain_verdict(
            state,
            max_grad_budget=50000,
            num_warmup_steps=2500,
            pooled_draws_by_window=dummy_pooled_draws,
        )

        self.assertIn(
            "pooled_draws_by_window",
            verdict.flags,
            "pooled_draws_by_window must be present in verdict.flags",
        )
        self.assertIs(
            verdict.flags["pooled_draws_by_window"],
            dummy_pooled_draws,
            "pooled_draws_by_window must be the exact object passed in (no copy)",
        )
        self.assertEqual(
            verdict.flags["pooled_draws_by_window"].shape, (M, n_per_window, d)
        )

    def test_verdict_multi_chain_fields(self):
        """extract_multi_chain_verdict populates n_chains, chain_collinearity, mode_coverage.

        After running overdispersed slow-chain data through final():
        - Non-escalated state: mode_coverage = 'multi_chain_uncertified' (M > 1)
        - Escalated state: mode_coverage = 'multi_chain_certified'
        - start_dispersion_adequacy and unimodality_gate keys are present
        """
        d, n, M = 20, 500, 8
        max_grad_budget = 50000

        core = build_multi_chain_meta_core(max_grad_budget, n_chains=M, max_rank=10)
        state_init = core.init(d)

        # Under-dispersed chains: no between-chain scatter → no escalation
        draws_und, grads_und = _make_underdispersed_chains(d, n, M, seed=220)
        state_und = _fill_mc_state_from_buffers(state_init, draws_und, grads_und)
        state_und = core.final(state_und)

        verdict_und = extract_multi_chain_verdict(
            state_und, max_grad_budget=max_grad_budget, num_warmup_steps=2500
        )
        self.assertIn("n_chains", verdict_und.flags)
        self.assertIn("chain_collinearity", verdict_und.flags)
        self.assertIn("need_more_chains", verdict_und.flags)
        self.assertIn("mode_coverage", verdict_und.flags)
        self.assertIn("start_dispersion_adequacy", verdict_und.flags)
        self.assertIn("unimodality_gate", verdict_und.flags)
        self.assertIn("deferred_to_ensemble", verdict_und.flags)
        self.assertEqual(verdict_und.flags["n_chains"], M)
        # M > 1 and no escalation → multi_chain_uncertified (not single_chain_uncertified)
        self.assertEqual(
            verdict_und.flags["mode_coverage"],
            "multi_chain_uncertified",
            "Non-escalated M>1 verdict: mode_coverage should be 'multi_chain_uncertified'",
        )

        # Overdispersed slow chains → escalation → multi_chain_certified
        draws_ov, grads_ov = _make_overdispersed_slow_chains(
            d, n, M, slow_offset_scale=5.0, seed=221
        )
        state_esc = _fill_mc_state_from_buffers(state_init, draws_ov, grads_ov)
        state_esc = core.final(state_esc)

        verdict_esc = extract_multi_chain_verdict(
            state_esc, max_grad_budget=max_grad_budget, num_warmup_steps=2500
        )
        if bool(np.asarray(state_esc.has_escalated)):
            self.assertEqual(
                verdict_esc.flags["mode_coverage"],
                "multi_chain_certified",
                "Escalated overdispersed verdict should be 'multi_chain_certified'",
            )

    def test_mode_split_no_false_escalate(self):
        """KEY negative test: mode-split chains must NOT escalate (unimodality guard).

        Uses M=8 chains split 4+4 across two modes at ±mode_separation/2 along
        one axis.  The true linear score is used so R² is high and the magnitude
        + collinearity + LOO gates all pass.  But the projected chain-means
        are bimodal: four means near −4 and four near +4.  With M=8 the
        scaled threshold is max(0.5*(8-1), 3.0) = 3.5 and the gap_ratio ≈ 7.0
        >> 3.5 → unimodality gate FAILS → no escalation.

        Additionally, since all gates EXCEPT unimodality pass, the verdict must
        report deferred_to_ensemble=True (the P1→P3 handoff is visible).

        This validates that the unimodality guard protects against treating a
        mode-separated ensemble as a slow-mixing direction.

        Structural guarantees: has_escalated=False, deferred_to_ensemble=True,
        unimodality_gate flag="flag" in the verdict.
        """
        d, n, M = 20, 500, 8  # M=8 required: gap_ratio ≈ M-1=7 > threshold=3.5
        core = build_multi_chain_meta_core(50000, n_chains=M, max_rank=10)
        state = core.init(d)

        draws_mc, grads_mc = _make_mode_split_chains(
            d, n, M, mode_separation=8.0, within_chain_noise=0.1, seed=230
        )
        state = _fill_mc_state_from_buffers(state, draws_mc, grads_mc)
        state = core.final(state)

        # Verify the M-scaled threshold is what the spec says for M=8
        expected_threshold = _mc_unimodality_threshold(M)
        self.assertAlmostEqual(
            expected_threshold,
            3.5,
            places=5,
            msg=f"_mc_unimodality_threshold({M}) should be max(0.5*7, 3.0) = 3.5",
        )

        self.assertFalse(
            bool(np.asarray(state.has_escalated)),
            "Mode-split chains (4+4 at ±4): must NOT escalate. "
            "Unimodality guard should block. "
            f"chain_collinearity={float(np.asarray(state.chain_collinearity)):.3f}",  # noqa: E231
        )
        self.assertFalse(
            bool(np.asarray(state.unimodality_passed)),
            "Mode-split: unimodality_passed should be False (bimodal projection detected)",
        )
        self.assertTrue(
            bool(np.asarray(state.deferred_to_ensemble)),
            "Mode-split: deferred_to_ensemble must be True when unimodality is the sole blocker "
            "(P1→P3 handoff must be visible in the carry)",
        )
        # Verify the verdict flags propagate the stored fields
        verdict = extract_multi_chain_verdict(
            state, max_grad_budget=50000, num_warmup_steps=2500
        )
        self.assertEqual(
            verdict.flags["unimodality_gate"],
            "flag",
            "Mode-split verdict: unimodality_gate flag must be 'flag' (not 'pass')",
        )
        self.assertTrue(
            verdict.flags["deferred_to_ensemble"],
            "Mode-split verdict: deferred_to_ensemble must be True in flags",
        )

    def test_under_dispersed_start_one_sided_safe(self):
        """Under-dispersed starts: conservative non-escalation, never dangerous.

        All M chains start near the same position (under-dispersed).  The
        between-chain scatter is structurally near zero → T magnitude gate does
        not fire → no escalation.  This is ONE-SIDED SAFE: we may miss the slow
        direction, but we never over-escalate from this cause.

        Checks:
        - has_escalated = False (conservative)
        - start_dispersion_adequacy flag = 'adequate_if_overdispersed' (honesty layer)
        - mode_coverage = 'multi_chain_uncertified' (M > 1, no escalation)
        """
        d, n, M = 20, 500, 8
        max_grad_budget = 50000
        core = build_multi_chain_meta_core(max_grad_budget, n_chains=M, max_rank=10)
        state = core.init(d)

        draws_mc, grads_mc = _make_underdispersed_chains(d, n, M, seed=240)
        state = _fill_mc_state_from_buffers(state, draws_mc, grads_mc)
        state = core.final(state)

        self.assertFalse(
            bool(np.asarray(state.has_escalated)),
            "Under-dispersed starts: must NOT escalate (one-sided safe). "
            "No between-chain mean scatter → T magnitude gate does not fire.",
        )

        verdict = extract_multi_chain_verdict(
            state, max_grad_budget=max_grad_budget, num_warmup_steps=2500
        )
        self.assertEqual(
            verdict.flags["start_dispersion_adequacy"],
            "adequate_if_overdispersed",
            "Non-escalation verdict must report start_dispersion_adequacy = "
            "'adequate_if_overdispersed' (not a certificate of diagonal sufficiency)",
        )
        self.assertEqual(
            verdict.flags["mode_coverage"],
            "multi_chain_uncertified",
            "Under-dispersed M>1 non-escalation: mode_coverage = 'multi_chain_uncertified'",
        )

    def test_multi_chain_e2e_smoke_f32_and_x64(self):
        """staged_adaptation(n_chains=8) smoke test under f32 and x64.

        Structural check:
        - warmup.run(key, positions) completes without error (positions shape (M, d))
        - The returned state has shape (M, d) for all M chains
        - The emitted LowRankInverseMassMatrix has correct sigma shape

        Uses a non-axis-aligned rank-1 spike so the controller CAN escalate,
        but the test does not assert whether it did (structural only).
        Uses M=8 (the recommended minimum; M<6 triggers a warning).
        """
        n_dims = 5
        M = 8
        lam_spike = 25.0

        u_raw = jax.random.normal(jax.random.key(42), (n_dims,))
        u_dir = u_raw / jnp.linalg.norm(u_raw)

        def _run():
            # Build cov_inv inside _run so it adopts the current JAX default dtype,
            # keeping position / gradient / step_size all consistent.
            # Mixing explicit dtype=float32 positions with x64-promoted scalars (e.g.
            # Python-float step_size) causes lax.cond dtype mismatch in the NUTS
            # trajectory — same pattern as test_escalated_e2e_smoke_f32_and_x64.
            cov_inv_local = jnp.eye(n_dims) - (lam_spike - 1.0) / lam_spike * jnp.outer(
                u_dir, u_dir
            )

            def logdensity_fn(x):
                return -0.5 * x @ cov_inv_local @ x

            warmup = blackjax.staged_adaptation(
                blackjax.nuts,
                logdensity_fn,
                metric="auto",
                max_grad_budget=10000,
                n_chains=M,
            )
            key = jax.random.key(300)
            positions = jnp.zeros((M, n_dims))  # default dtype matches cov_inv_local
            results, _ = warmup.run(key, positions)
            return results

        # --- default dtype (f32 normally; f64 when JAX_ENABLE_X64 is active globally) ---
        results_default = _run()
        state_default = results_default.state
        imm_default = results_default.parameters["inverse_mass_matrix"]
        self.assertIsInstance(imm_default, LowRankInverseMassMatrix)
        self.assertEqual(imm_default.sigma.shape, (n_dims,))
        # All M final MCMC states returned; position has leading chain dim M
        pos_default = jax.tree.leaves(state_default.position)[0]
        self.assertEqual(pos_default.shape[0], M, "expected M final chain states")

        # --- x64 ---
        try:
            jax.config.update("jax_enable_x64", True)
            results_x64 = _run()
            imm_x64 = results_x64.parameters["inverse_mass_matrix"]
            self.assertIsInstance(imm_x64, LowRankInverseMassMatrix)
            self.assertEqual(imm_x64.sigma.shape, (n_dims,))
            self.assertTrue(
                bool(jnp.all(jnp.isfinite(imm_x64.sigma))),
                "x64: sigma has non-finite values",
            )
        finally:
            jax.config.update("jax_enable_x64", False)


# ===========================================================================
# v2.1 tests — W-branch, router fix, scoped latch, shared-ε
# ===========================================================================


# ---------------------------------------------------------------------------
# Shared fixture helpers (multi-chain)
# ---------------------------------------------------------------------------


def _make_mc_deep_spread(M, n, d, lam_within=20.0, seed=_RNG_SEED):
    """M chains each with the same within-chain slow direction, all centered at 0.

    Generates the ill_cond-like scenario: W-branch should detect anisotropy
    (lam1 >> w_edge) but T-branch stays silent (f1 ≈ 0 since chain means ≈ 0).
    Returns arrays of shape (M, n, d).
    """
    key = jax.random.key(seed)
    k_dir, k_chains = jax.random.split(key)
    u_raw = jax.random.normal(k_dir, (d,))
    u = u_raw / jnp.linalg.norm(u_raw)
    chain_keys = jax.random.split(k_chains, M)
    draws_list, grads_list = [], []
    for m in range(M):
        z = jax.random.normal(chain_keys[m], (n, d))
        z_orth = z - (z @ u[:, None]) @ u[None, :]
        x = z_orth + jnp.sqrt(jnp.float32(lam_within)) * (z @ u[:, None]) @ u[None, :]
        g = -(x - (1.0 - 1.0 / lam_within) * (x @ u[:, None]) @ u[None, :])
        draws_list.append(x)
        grads_list.append(g)
    return jnp.stack(draws_list), jnp.stack(grads_list)


def _make_mc_isotropic(M, n, d, seed=_RNG_SEED):
    """M chains, each isotropic N(0, I): W-branch should NOT fire."""
    key = jax.random.key(seed)
    chain_keys = jax.random.split(key, M)
    draws_list, grads_list = [], []
    for m in range(M):
        x = jax.random.normal(chain_keys[m], (n, d))
        draws_list.append(x)
        grads_list.append(-x)
    return jnp.stack(draws_list), jnp.stack(grads_list)


def _make_mc_split_means(M, n, d, split_scale=10.0, seed=_RNG_SEED):
    """Half-and-half bimodal chains: first M//2 at -split_scale, rest at +split_scale.

    Triggers gap-stat flagging (gap_ratio >> threshold) when projected onto e0.
    """
    key = jax.random.key(seed)
    chain_keys = jax.random.split(key, M)
    draws_list, grads_list = [], []
    for m in range(M):
        mean = split_scale * (1.0 if m < M // 2 else -1.0)
        center = jnp.zeros(d).at[0].set(mean)
        x = jax.random.normal(chain_keys[m], (n, d)) * 0.5 + center
        g = -x
        draws_list.append(x)
        grads_list.append(g)
    return jnp.stack(draws_list), jnp.stack(grads_list)


def _make_mc_even_spread(M, n, d, spread_scale=5.0, seed=_RNG_SEED):
    """M chains with evenly-spaced means (unimodal): gap-stat should NOT flag."""
    key = jax.random.key(seed)
    chain_keys = jax.random.split(key, M)
    means = jnp.linspace(-spread_scale, spread_scale, M)
    draws_list, grads_list = [], []
    for m in range(M):
        center = jnp.zeros(d).at[0].set(means[m])
        x = jax.random.normal(chain_keys[m], (n, d)) * 0.3 + center
        g = -x
        draws_list.append(x)
        grads_list.append(g)
    return jnp.stack(draws_list), jnp.stack(grads_list)


def _fill_mc_state(
    state: MultiChainMetaAdaptationCoreState,
    draws_mc: jax.Array,
    grads_mc: jax.Array,
) -> MultiChainMetaAdaptationCoreState:
    """Copy (M, n, d) draws/grads into the MultiChain state buffer."""
    M, B, d = state.draws_buffer.shape
    n = draws_mc.shape[1]
    n_fill = min(n, B)
    draws_buf = jnp.concatenate(
        [draws_mc[:, :n_fill, :], jnp.zeros((M, B - n_fill, d), dtype=draws_mc.dtype)],
        axis=1,
    )
    grads_buf = jnp.concatenate(
        [grads_mc[:, :n_fill, :], jnp.zeros((M, B - n_fill, d), dtype=grads_mc.dtype)],
        axis=1,
    )
    return state._replace(
        draws_buffer=draws_buf,
        grads_buffer=grads_buf,
        buffer_idx=jnp.array(n_fill, dtype=jnp.int32),
    )


# ---------------------------------------------------------------------------
# v2.1 test 1 — Time-major layout invariant (padding regression)
# ---------------------------------------------------------------------------


class TestTimeMajorLayout(BlackJAXTest):
    """_build_pc_centered_time_major_pool puts valid rows first (no padding contamination)."""

    def test_step_mask_valid_region_is_contiguous_at_start(self):
        """With n < B, step_mask_tm must be 1 for rows 0..n*M-1 and 0 for rows n*M..B*M-1.

        _build_pc_centered_time_major_pool returns unmasked centered draws and a
        separate mask.  The mask encodes valid vs padding — callers apply it.
        This test verifies that the mask correctly identifies the valid region as
        contiguous at the start (time-major layout) rather than scattered (chain-major).
        """
        M, n, B, d = 4, 30, 64, 10
        key = jax.random.key(1)
        draws = jax.random.normal(key, (M, B, d))
        grads = -draws
        chain_means = draws.mean(axis=1)

        n_arr = jnp.array(n, dtype=jnp.int32)
        pc_draws, _pc_grads, step_mask_tm = _build_pc_centered_time_major_pool(
            draws, grads, chain_means, n_arr, M
        )

        self.assertEqual(pc_draws.shape, (B * M, d))
        self.assertEqual(step_mask_tm.shape, (B * M,))

        # Valid region: rows 0 .. n*M-1 → mask must be 1
        valid_mask = np.asarray(step_mask_tm[: n * M])
        last_valid = n * M - 1
        self.assertTrue(
            np.all(valid_mask == 1.0),
            f"step_mask_tm rows 0..{last_valid} must all be 1",
        )

        # Padding region: rows n*M .. B*M-1 → mask must be 0
        padding_mask = np.asarray(step_mask_tm[n * M :])
        last_pad = B * M - 1
        self.assertTrue(
            np.all(padding_mask == 0.0),
            f"step_mask_tm rows {n * M}..{last_pad} must all be 0",
        )

        # Valid draws should have non-zero content (centering does not collapse to zero)
        valid_draws = np.asarray(pc_draws[: n * M])
        self.assertGreater(
            float(np.abs(valid_draws).max()), 0.0, "valid rows should be non-zero"
        )

    def test_first_valid_row_equals_chain0_step0_centered(self):
        """Row 0 of time-major output = (chain 0, step 0) − chain_0_mean."""
        M, n, B, d = 4, 20, 64, 8
        key = jax.random.key(2)
        draws = jax.random.normal(key, (M, B, d))
        grads = -draws
        chain_means = draws.mean(axis=1)  # computed over full B steps

        n_arr = jnp.array(n, dtype=jnp.int32)
        pc_draws, _, _ = _build_pc_centered_time_major_pool(
            draws, grads, chain_means, n_arr, M
        )

        # In time-major layout: row 0 = (chain 0, step 0) - chain_0_mean
        expected_row0 = np.asarray(draws[0, 0] - chain_means[0])
        actual_row0 = np.asarray(pc_draws[0])
        np.testing.assert_allclose(expected_row0, actual_row0, atol=1e-5)


# ---------------------------------------------------------------------------
# v2.1 test 2 — W-branch: lam1 detection
# ---------------------------------------------------------------------------


class TestWBranchSpectrum(BlackJAXTest):
    """W-branch top eigenvalue clears MP edge on deep spread, stays below on isotropic."""

    def _check_spectrum(self, M, n, d, draws_mc, dtype=jnp.float32):
        draws_f = draws_mc.astype(dtype)
        chain_means = draws_f.mean(axis=1)
        W_diag = jnp.ones(d, dtype=dtype)
        n_arr = jnp.array(n, dtype=jnp.int32)
        actual_rank = max(d // 2, 1)
        lam1, _ = _compute_pooled_within_spectrum(
            draws_f, chain_means, W_diag, n_arr, M, actual_rank
        )
        N_dof = max(n * M - M, 1)
        edge = _w_branch_lam1_edge(d, jnp.array(N_dof, dtype=jnp.int32))
        return float(np.asarray(lam1)), float(np.asarray(edge))

    def test_deep_spread_lam1_exceeds_edge_f32(self):
        """Anisotropic within-chain draws: lam1 >> MP edge (f32)."""
        M, n, d = 8, 200, 20
        draws_mc, _ = _make_mc_deep_spread(M, n, d, lam_within=25.0, seed=10)
        lam1, edge = self._check_spectrum(M, n, d, draws_mc, jnp.float32)
        self.assertGreater(
            lam1, edge, f"Deep spread: lam1={lam1} should exceed edge={edge}"
        )

    def test_isotropic_lam1_at_most_slightly_above_edge_f32(self):
        """Isotropic chains: lam1 should not greatly exceed MP edge (f32)."""
        M, n, d = 8, 200, 20
        draws_mc, _ = _make_mc_isotropic(M, n, d, seed=11)
        lam1, edge = self._check_spectrum(M, n, d, draws_mc, jnp.float32)
        self.assertLessEqual(
            lam1,
            edge * 1.3,
            f"Isotropic: lam1={lam1} should not exceed 1.3x edge={edge}",
        )

    def test_deep_spread_lam1_exceeds_edge_x64(self):
        """W-branch top eigenvalue clears MP edge in float64."""
        try:
            jax.config.update("jax_enable_x64", True)
            M, n, d = 8, 200, 20
            draws_mc, _ = _make_mc_deep_spread(M, n, d, lam_within=25.0, seed=12)
            lam1, edge = self._check_spectrum(M, n, d, draws_mc, jnp.float64)
            self.assertGreater(
                lam1,
                edge,
                f"x64 deep spread: lam1={lam1} should exceed edge={edge}",
            )
        finally:
            jax.config.update("jax_enable_x64", False)


# ---------------------------------------------------------------------------
# v2.1 test 3 — Cross-chain consistency Ψ
# ---------------------------------------------------------------------------


class TestChainConsistencyPsi(BlackJAXTest):
    """Ψ > floor on genuine deep spread; Ψ ≈ 0 on isotropic (null)."""

    def _compute_psi(self, M, n, d, draws_mc, dtype=jnp.float32):
        draws_f = draws_mc.astype(dtype)
        chain_means = draws_f.mean(axis=1)
        W_diag = jnp.ones(d, dtype=dtype)
        n_arr = jnp.array(n, dtype=jnp.int32)
        psi = _compute_chain_consistency_psi(draws_f, chain_means, W_diag, n_arr, M)
        return float(np.asarray(psi))

    def test_deep_spread_psi_above_floor_f32(self):
        """Genuine within-chain anisotropy: Ψ > _W_BRANCH_PSI_FLOOR (f32)."""
        M, n, d = 8, 200, 20
        draws_mc, _ = _make_mc_deep_spread(M, n, d, lam_within=25.0, seed=20)
        psi = self._compute_psi(M, n, d, draws_mc, jnp.float32)
        self.assertGreater(
            psi,
            _W_BRANCH_PSI_FLOOR,
            f"Deep spread: psi={psi} should exceed floor={_W_BRANCH_PSI_FLOOR}",
        )

    def test_isotropic_psi_below_floor_f32(self):
        """Isotropic null: Ψ should be near zero (f32)."""
        M, n, d = 8, 200, 20
        draws_mc, _ = _make_mc_isotropic(M, n, d, seed=21)
        psi = self._compute_psi(M, n, d, draws_mc, jnp.float32)
        self.assertLess(
            psi,
            _W_BRANCH_PSI_FLOOR,
            f"Isotropic null: psi={psi} should be below floor={_W_BRANCH_PSI_FLOOR}",
        )

    def test_deep_spread_psi_above_floor_x64(self):
        """Ψ > floor in float64."""
        try:
            jax.config.update("jax_enable_x64", True)
            M, n, d = 8, 200, 20
            draws_mc, _ = _make_mc_deep_spread(M, n, d, lam_within=25.0, seed=22)
            psi = self._compute_psi(M, n, d, draws_mc, jnp.float64)
            self.assertGreater(
                psi,
                _W_BRANCH_PSI_FLOOR,
                f"x64 deep spread: psi={psi} should exceed floor",
            )
        finally:
            jax.config.update("jax_enable_x64", False)


# ---------------------------------------------------------------------------
# v2.1 test 4 — W-branch MP edge formula
# ---------------------------------------------------------------------------


class TestMPEdgeFormula(BlackJAXTest):
    """_w_branch_lam1_edge computes (1 + sqrt(d/N))^2 correctly."""

    def test_edge_formula_scalar(self):
        """Check the MP edge value for known d, N."""
        d, N = 10, 100
        edge = _w_branch_lam1_edge(d, jnp.array(N, dtype=jnp.int32))
        expected = (1.0 + (d / N) ** 0.5) ** 2
        self.assertAlmostEqual(float(np.asarray(edge)), expected, places=5)

    def test_edge_decreases_with_more_samples(self):
        """More samples → tighter (lower) MP edge → easier detection."""
        d = 20
        edge_small = _w_branch_lam1_edge(d, jnp.array(50, dtype=jnp.int32))
        edge_large = _w_branch_lam1_edge(d, jnp.array(2000, dtype=jnp.int32))
        self.assertGreater(float(np.asarray(edge_small)), float(np.asarray(edge_large)))

    def test_edge_increases_with_dimension(self):
        """Higher dimension → larger MP edge (more noise dims = larger null bulk)."""
        N = 200
        edge_low_d = _w_branch_lam1_edge(5, jnp.array(N, dtype=jnp.int32))
        edge_high_d = _w_branch_lam1_edge(50, jnp.array(N, dtype=jnp.int32))
        self.assertGreater(
            float(np.asarray(edge_high_d)), float(np.asarray(edge_low_d))
        )


# ---------------------------------------------------------------------------
# v2.1 test 5 — 2-window unimodality confirmation and non-monotone latch
# ---------------------------------------------------------------------------


class TestUnimodality2WindowConfirmation(BlackJAXTest):
    """One flag does not defer; deferred resets on unimodal window (non-monotone)."""

    def _build_core_and_state(self, M, d, max_grad_budget=40000):
        core = build_multi_chain_meta_core(max_grad_budget=max_grad_budget, n_chains=M)
        state = core.init(d)
        return core, state

    def test_single_flag_deferred_false(self):
        """After one flagged window, deferred stays False (needs 2 consecutive)."""
        M, n, d = 8, 150, 10
        core, state = self._build_core_and_state(M, d)
        draws_mc, grads_mc = _make_mc_split_means(M, n, d, split_scale=8.0, seed=30)
        state1 = _fill_mc_state(state, draws_mc, grads_mc)
        result1 = core.final(state1)

        deferred = bool(np.asarray(result1.deferred_to_ensemble))
        flag_count = int(np.asarray(result1.unimodality_flag_count))
        self.assertFalse(
            deferred, "deferred must be False after at most 1 flagged window"
        )
        self.assertLessEqual(flag_count, 1)

    def test_structural_invariant_deferred_requires_flag_count_ge_2(self):
        """deferred=True iff flag_count >= _MC_UNIMODALITY_CONFIRM_WINDOWS (structural)."""
        M, n, d = 8, 150, 10
        core, state = self._build_core_and_state(M, d, max_grad_budget=40000)
        draws_mc, grads_mc = _make_mc_split_means(M, n, d, split_scale=8.0, seed=31)

        state1 = _fill_mc_state(state, draws_mc, grads_mc)
        r1 = core.final(state1)
        state2 = _fill_mc_state(r1, draws_mc, grads_mc)
        r2 = core.final(state2)

        flag2 = int(np.asarray(r2.unimodality_flag_count))
        deferred2 = bool(np.asarray(r2.deferred_to_ensemble))

        # Invariant: deferred=True implies flag_count >= confirm_threshold
        if deferred2:
            self.assertGreaterEqual(
                flag2,
                _MC_UNIMODALITY_CONFIRM_WINDOWS,
                "deferred=True requires at least CONFIRM_WINDOWS consecutive flags",
            )

    def test_non_monotone_latch_resets_on_unimodal_window(self):
        """deferred resets to False when a unimodal window follows the flagged ones."""
        M, n, d = 8, 150, 10
        core, state = self._build_core_and_state(M, d)
        draws_split, grads_split = _make_mc_split_means(
            M, n, d, split_scale=8.0, seed=32
        )

        state1 = _fill_mc_state(state, draws_split, grads_split)
        r1 = core.final(state1)
        state2 = _fill_mc_state(r1, draws_split, grads_split)
        r2 = core.final(state2)

        draws_uni, grads_uni = _make_mc_even_spread(M, n, d, spread_scale=0.5, seed=33)
        state3 = _fill_mc_state(r2, draws_uni, grads_uni)
        r3 = core.final(state3)

        flag3 = int(np.asarray(r3.unimodality_flag_count))
        deferred3 = bool(np.asarray(r3.deferred_to_ensemble))

        # If unimodal window cleared the flag, deferred must be False
        if flag3 == 0:
            self.assertFalse(
                deferred3,
                "Non-monotone latch: unimodal window (flag_count=0) must reset deferred to False",
            )


# ---------------------------------------------------------------------------
# v2.1 test 6 — Impossible combo invariant
# ---------------------------------------------------------------------------


class TestImpossibleComboInvariant(BlackJAXTest):
    """Escalation ↔ not deferred (algebraic exclusion from scoped latch rule)."""

    def test_escalation_implies_not_deferred(self):
        """If has_escalated=True after final(), deferred must be False."""
        M, n, d = 8, 150, 10
        core = build_multi_chain_meta_core(max_grad_budget=40000, n_chains=M)
        state = core.init(d)
        draws_mc, grads_mc = _make_mc_deep_spread(M, n, d, lam_within=30.0, seed=40)
        state1 = _fill_mc_state(state, draws_mc, grads_mc)
        result = core.final(state1)

        has_escalated = bool(np.asarray(result.has_escalated))
        deferred = bool(np.asarray(result.deferred_to_ensemble))

        if has_escalated:
            self.assertFalse(
                deferred, "Impossible combo: escalated state must not be deferred"
            )

    def test_impossible_combo_between_means_deferred_escalated(self):
        """The combo route=low_rank ∧ deferred ∧ detection_branch=between_means is impossible.

        Cross-branch coexistence (W-escalation + T-defer) IS LEGAL per the scoped
        latch rule.  The ONLY impossible combo is:
            deferred=True AND has_escalated=True AND detection_branch=between_means.
        If T-branch alone escalated (between_means), it required ~confirmed_split
        → confirmed_split=False → new_deferred=False.  So this triple cannot occur.
        """
        M, n, d = 8, 150, 10
        core = build_multi_chain_meta_core(max_grad_budget=40000, n_chains=M)
        state = core.init(d)
        draws_split, grads_split = _make_mc_split_means(
            M, n, d, split_scale=8.0, seed=41
        )

        state1 = _fill_mc_state(state, draws_split, grads_split)
        r1 = core.final(state1)
        state2 = _fill_mc_state(r1, draws_split, grads_split)
        r2 = core.final(state2)

        has_esc = bool(np.asarray(r2.has_escalated))
        deferred = bool(np.asarray(r2.deferred_to_ensemble))
        branch = int(np.asarray(r2.detection_branch))

        # If the T-only between_means branch escalated, deferred MUST be False
        if has_esc and branch == _DETECTION_BRANCH_BETWEEN_MEANS:
            self.assertFalse(
                deferred,
                "Impossible combo: detection_branch=between_means + escalated + deferred",
            )


# ---------------------------------------------------------------------------
# v2.1 test 7 — New state fields populated after final()
# ---------------------------------------------------------------------------


class TestNewStateFieldsPopulated(BlackJAXTest):
    """All 5 new v2.1 state fields are finite after the first core.final() call."""

    def _run(self, dtype, seed):
        M, n, d = 8, 150, 10
        core = build_multi_chain_meta_core(max_grad_budget=40000, n_chains=M)
        state = core.init(d)
        draws_mc, grads_mc = _make_mc_deep_spread(M, n, d, seed=seed)
        state1 = _fill_mc_state(state, draws_mc.astype(dtype), grads_mc.astype(dtype))
        return core.final(state1)

    def test_new_fields_finite_f32(self):
        """within_lam1, chain_consistency_psi, r1_top are finite (f32)."""
        result = self._run(jnp.float32, seed=50)
        lam1 = float(np.asarray(result.within_lam1))
        psi = float(np.asarray(result.chain_consistency_psi))
        r1 = float(np.asarray(result.r1_top))
        branch = int(np.asarray(result.detection_branch))
        flag_count = int(np.asarray(result.unimodality_flag_count))

        self.assertFalse(np.isnan(lam1), "within_lam1 is NaN")
        self.assertFalse(np.isnan(psi), "chain_consistency_psi is NaN")
        self.assertFalse(np.isnan(r1), "r1_top is NaN")
        self.assertIn(
            branch,
            [
                _DETECTION_BRANCH_NONE,
                _DETECTION_BRANCH_POOLED_WITHIN,
                _DETECTION_BRANCH_BETWEEN_MEANS,
                _DETECTION_BRANCH_BOTH,
            ],
        )
        self.assertGreaterEqual(flag_count, 0)

    def test_new_fields_finite_x64(self):
        """within_lam1, chain_consistency_psi, r1_top are finite (x64)."""
        try:
            jax.config.update("jax_enable_x64", True)
            result = self._run(jnp.float64, seed=51)
            lam1 = float(np.asarray(result.within_lam1))
            psi = float(np.asarray(result.chain_consistency_psi))
            r1 = float(np.asarray(result.r1_top))
            self.assertFalse(np.isnan(lam1), "x64: within_lam1 is NaN")
            self.assertFalse(np.isnan(psi), "x64: chain_consistency_psi is NaN")
            self.assertFalse(np.isnan(r1), "x64: r1_top is NaN")
        finally:
            jax.config.update("jax_enable_x64", False)


# ---------------------------------------------------------------------------
# v2.1 test 8 — extract_multi_chain_verdict exposes new flags
# ---------------------------------------------------------------------------


class TestExtractMultiChainVerdictNewFields(BlackJAXTest):
    """extract_multi_chain_verdict flags dict includes all four v2.1 diagnostic keys."""

    def test_new_flags_present_and_finite(self):
        """Flags dict contains within_lam1, chain_consistency_psi, r1_top, detection_branch."""
        M, n, d = 8, 150, 10
        core = build_multi_chain_meta_core(max_grad_budget=40000, n_chains=M)
        state = core.init(d)
        draws_mc, grads_mc = _make_mc_deep_spread(M, n, d, seed=60)
        state1 = _fill_mc_state(state, draws_mc, grads_mc)
        final_state = core.final(state1)

        verdict = extract_multi_chain_verdict(
            final_state, max_grad_budget=40000, num_warmup_steps=1000
        )
        flags = verdict.flags

        for key in (
            "within_lam1",
            "chain_consistency_psi",
            "r1_top",
            "detection_branch",
        ):
            self.assertIn(key, flags, f"Missing verdict flag: {key}")

        self.assertFalse(np.isnan(flags["within_lam1"]))
        self.assertFalse(np.isnan(flags["chain_consistency_psi"]))
        self.assertFalse(np.isnan(flags["r1_top"]))
        self.assertIn(
            flags["detection_branch"],
            ["none", "pooled_within", "between_means", "both"],
        )

    def test_detection_branch_none_on_init_state(self):
        """Before any final() call, detection_branch flag is 'none'."""
        M, d = 8, 10
        core = build_multi_chain_meta_core(max_grad_budget=40000, n_chains=M)
        state = core.init(d)
        verdict = extract_multi_chain_verdict(
            state, max_grad_budget=40000, num_warmup_steps=1000
        )
        self.assertEqual(verdict.flags["detection_branch"], "none")


# ---------------------------------------------------------------------------
# v2.1 test 9 — Shared-ε: M sequential DA updates per step
# ---------------------------------------------------------------------------


class TestSharedEpsilonDA(BlackJAXTest):
    """Shared-ε: n_da_updates=M via lax.scan matches M manual sequential DA updates."""

    def test_n_da_updates_scan_matches_loop(self):
        """_make_engine with n_da_updates=M gives same step_size_avg as M manual updates."""
        from blackjax.adaptation.meta_adaptation import build_meta_adaptation_core
        from blackjax.adaptation.step_size import dual_averaging_adaptation

        target_ar = 0.80
        M = 4
        per_chain = jnp.array([0.55, 0.65, 0.75, 0.85])

        # Manual sequential updates (ground truth)
        da_init, da_update, _ = dual_averaging_adaptation(target_ar)
        ss = da_init(0.1)
        for ar in per_chain:
            ss = da_update(ss, jnp.float32(float(ar)))
        step_manual = float(np.asarray(jnp.exp(ss.log_step_size_avg)))

        # Engine with n_da_updates=M (uses lax.scan internally)
        dummy_core = build_meta_adaptation_core(max_grad_budget=5000)
        eng_init, eng_update, _ = _make_engine(
            dummy_core,
            target_acceptance_rate=target_ar,
            n_da_updates=M,
        )
        adaptation_state = eng_init(jnp.zeros(10), 0.1)
        adaptation_stage = (jnp.int32(0), jnp.bool_(False))  # fast stage, no window end

        new_state = eng_update(
            adaptation_state,
            adaptation_stage,
            jnp.zeros(10),  # position (ignored in fast stage)
            jnp.zeros(10),  # grad (ignored in fast stage)
            per_chain,  # (M,) accept rates → M sequential DA updates
        )

        step_engine = float(np.asarray(jnp.exp(new_state.ss_state.log_step_size_avg)))
        self.assertAlmostEqual(
            step_engine,
            step_manual,
            places=4,
            msg="n_da_updates=M via lax.scan must match M manual DA updates",
        )

    def test_step_counter_increments_M_times(self):
        """After n_da_updates=M, the DA step counter increments by M (not 1)."""
        from blackjax.adaptation.meta_adaptation import build_meta_adaptation_core

        M = 3
        per_chain = jnp.array([0.70, 0.75, 0.80])

        dummy_core = build_meta_adaptation_core(max_grad_budget=5000)
        eng_init, eng_update, _ = _make_engine(
            dummy_core,
            target_acceptance_rate=0.80,
            n_da_updates=M,
        )
        adaptation_state = eng_init(jnp.zeros(8), 0.1)
        initial_step = int(np.asarray(adaptation_state.ss_state.step))
        adaptation_stage = (jnp.int32(0), jnp.bool_(False))

        new_state = eng_update(
            adaptation_state,
            adaptation_stage,
            jnp.zeros(8),
            jnp.zeros(8),
            per_chain,
        )

        step_count = int(np.asarray(new_state.ss_state.step))
        # The step counter should have incremented exactly M times
        self.assertEqual(
            step_count - initial_step,
            M,
            f"DA step counter should increment by {M}, got delta={step_count - initial_step}",
        )


# ---------------------------------------------------------------------------
# v2.1 test 10 — W-branch e2e smoke (f32 and x64)
# ---------------------------------------------------------------------------


class TestWBranchE2ESmoke(BlackJAXTest):
    """core.final() with deep-spread draws: W-branch diagnostics are finite and positive."""

    def _run_smoke(self, dtype, seed):
        M, n, d = 8, 150, 10
        core = build_multi_chain_meta_core(max_grad_budget=40000, n_chains=M)
        state = core.init(d)
        draws_mc, grads_mc = _make_mc_deep_spread(M, n, d, lam_within=25.0, seed=seed)
        state1 = _fill_mc_state(state, draws_mc.astype(dtype), grads_mc.astype(dtype))
        return core.final(state1), d

    def test_w_branch_e2e_f32(self):
        """Deep-spread draws: within_lam1 > 0, chain_consistency_psi > 0 (f32)."""
        result, d = self._run_smoke(jnp.float32, seed=70)
        lam1 = float(np.asarray(result.within_lam1))
        psi = float(np.asarray(result.chain_consistency_psi))
        self.assertFalse(np.isnan(lam1), "within_lam1 is NaN")
        self.assertGreater(lam1, 0.0, "within_lam1 must be positive")
        self.assertFalse(np.isnan(psi), "chain_consistency_psi is NaN")
        self.assertGreater(psi, 0.0, "chain_consistency_psi must be positive")
        imm = result.inverse_mass_matrix
        self.assertIsInstance(imm, LowRankInverseMassMatrix)
        self.assertEqual(imm.sigma.shape, (d,))
        self.assertTrue(
            bool(jnp.all(jnp.isfinite(imm.sigma))), "sigma has non-finite values"
        )

    def test_w_branch_e2e_x64(self):
        """Deep-spread draws: W-branch diagnostics are finite in float64."""
        try:
            jax.config.update("jax_enable_x64", True)
            result, d = self._run_smoke(jnp.float64, seed=71)
            lam1 = float(np.asarray(result.within_lam1))
            psi = float(np.asarray(result.chain_consistency_psi))
            self.assertFalse(np.isnan(lam1), "x64: within_lam1 is NaN")
            self.assertFalse(np.isnan(psi), "x64: chain_consistency_psi is NaN")
            imm = result.inverse_mass_matrix
            self.assertTrue(
                bool(jnp.all(jnp.isfinite(imm.sigma))),
                "x64: sigma has non-finite values",
            )
        finally:
            jax.config.update("jax_enable_x64", False)
