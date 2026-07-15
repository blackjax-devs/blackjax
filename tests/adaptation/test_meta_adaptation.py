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
import jax
import jax.numpy as jnp
import numpy as np

import blackjax
from blackjax.adaptation.meta_adaptation import (
    _R2_DEFERRED,
    _R2_FULL_AFFINE,
    _R2_PROJECTED,
    _R_MIN,
    _S_MIN,
    MetaAdaptationCoreState,
    MetaAdaptationVerdict,
    _choose_rank,
    _compute_r2_score_linearity,
    _compute_s_gap,
    _compute_whitened_spectrum,
    build_meta_adaptation_core,
    extract_meta_verdict,
)
from blackjax.adaptation.metric_recipes import MetricCore
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
        Pre-review MUT-a was green on the curvature test because S_gap also blocked;
        this fixture ensures only R² can block.
        """
        d, n = 20, 500
        # High S_gap (correlated spike) + random grads (R²≈0, curvature proxy)
        draws, grads = _make_high_sgap_curvature_buffer(
            d, n, rank=2, lam_spike=20.0, seed=36
        )
        draws = jnp.array(draws, dtype=jnp.float32)
        grads = jnp.array(grads, dtype=jnp.float32)
        state, _ = self._run_two_windows(draws, grads)

        # S_gap gate passes (proves it is not the blocker).
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

        Regression guard for two bugs fixed together:
        (1) MUT-j: the old `if schedule_fn is build_schedule` sentinel could not
            distinguish between "user passed nothing" and "user explicitly passed
            build_schedule", so explicit Stan was silently swapped to growing.
        (2) vacuous assertion: the old test only checked isinstance(IMM, LRK), a
            tautology because auto always emits LowRankInverseMassMatrix.

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

    def test_effective_rank_matches_escalation_rank(self):
        """effective_rank in verdict equals the chosen rank at escalation."""
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
        self.assertEqual(verdict.effective_rank, carry_rank)
        self.assertEqual(verdict.route, "low_rank")

    def test_marginal_s_gap_stays_diagonal(self):
        """Marginal-band S_gap ∈ [_S_MIN, 2·_S_MIN) = [2.0, 4.0): stays diagonal.

        Regression guard for the 'stays-diag-marginal' decision row.
        The OLD fixture (lam_spike=2.0) was mislabeled: it produced S_gap=1.0
        because top Welford-whitened eigenvalue ≈ 1.9 < cutoff=2.0 -> k_new=0
        -> _compute_s_gap returns 1.0 by definition.  k_new=0 means the 'wrong
        cut' MUT-e mutation was only caught by a 3% accident, not by design.

        The NEW fixture (lam_spike=4.5, rank=1, non-axis-aligned direction, seed=42):
        - top Welford-whitened eigenvalue ≈ 3.5 > cutoff=2.0 -> k_new=1
        - S_gap = lambda_1/lambda_2 ≈ 2.94 ∈ [_S_MIN, 2·_S_MIN) -> marginal_s_gap=True
        - random grads -> R2 approx 0 -> R2 gate blocks escalation (not S_gap gate)
        - flagged as 'marginal_s_gap' in verdict.flags
        - direct s_gap_curr == _compute_s_gap(eigs, k_new) assertion catches
          MUT-e (using k instead of k_new as the spectral cut)

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
        # Catches MUT-e (spectral cut at wrong index, e.g. k-1 or k+1 instead of k).
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
                "catches MUT-e (wrong spectral cut index)"
            ),
        )

    def test_escalated_e2e_smoke_f32_and_x64(self):
        """Escalated e2e smoke: non-axis-aligned spike target escalates under both f32 and x64.

        Regression guard for the x64 dtype crash (BLOCKER 1 in adversarial review):
        the suite was green 28/28 only because all tests ran f32.  Under x64 the
        dynamic_update_slice and _deferred branches both crashed at trace time.

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
