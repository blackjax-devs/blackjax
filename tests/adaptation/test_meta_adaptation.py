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
covariance has a rank-k spike (Σ = I + U(Λ-I)Uᵀ). The Fisher-diagonal sigma
captures the marginal variances, leaving a whitened residual spectrum with
S_gap ≈ Λ_spike at the spike directions. The score is linear in position
(Gaussian), giving R²=1.0. This is the minimal model for the "linear-residual,
high-S_gap" class that the controller should escalate on.
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
    """Correlated Gaussian with rank-k spike: R²=1.0, S_gap≈lam_spike.

    Σ = I + U*(lam_spike - 1)*Uᵀ, score = -Σ⁻¹ x (linear in position).
    After Fisher-diagonal whitening, the whitened residual has a rank-k spike
    with eigenvalue ≈ lam_spike / diag(Σ), giving S_gap >> _S_MIN.
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
        n = 10  # far below 2 * 8 * (max_rank+1) = 176
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
        """Projected fit is used (non-NaN, mode=_R2_PROJECTED) when n ≥ 2*8*(max_rank+1)."""
        d = 200  # too large for full-affine with n=106
        max_rank = 5
        n = 2 * 8 * (max_rank + 1) + 10  # = 106, above min_n_proj=96
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
        """Insufficient remaining budget: deadline gate blocks escalation."""
        d, n = 20, 500
        draws, grads = _make_correlated_buffer(d, n, rank=2, lam_spike=20.0, seed=34)
        draws = jnp.array(draws, dtype=jnp.float32)
        grads = jnp.array(grads, dtype=jnp.float32)
        # Budget so small that conversion to steps = 1 → no budget remaining
        state, _ = self._run_two_windows(draws, grads, max_grad_budget=20)
        self.assertFalse(
            bool(np.asarray(state.has_escalated)),
            "Exhausted budget: deadline gate should block escalation",
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
        with self.assertRaises(ValueError):
            blackjax.staged_adaptation(
                blackjax.nuts,
                lambda x: -0.5 * jnp.sum(x**2),
                metric="auto",
                # max_grad_budget intentionally omitted
            )
