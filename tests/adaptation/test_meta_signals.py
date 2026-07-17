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
"""Tests for single-chain signal computation (:mod:`blackjax.adaptation.meta._signals`).

Coverage:
- TestCriterionR2Gate: R² gate separates linear-residual from curvature geometry.
- TestCriterionSGap: S_gap ordering agrees with measured payoff ordering.
"""
import jax.numpy as jnp
import numpy as np

from blackjax.adaptation.meta._calibration import (
    _R2_DEFERRED,
    _R2_FULL_AFFINE,
    _R2_PROJECTED,
    _R_MIN,
    _S_MIN,
)
from blackjax.adaptation.meta._signals import (
    _choose_rank,
    _compute_r2_score_linearity,
    _compute_s_gap,
    _compute_whitened_spectrum,
)
from tests.adaptation._meta_fixtures import (
    _make_correlated_buffer,
    _make_curvature_buffer,
    _make_isotropic_buffer,
)
from tests.fixtures import BlackJAXTest


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
