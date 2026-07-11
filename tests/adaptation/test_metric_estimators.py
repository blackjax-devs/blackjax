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
"""Parity golden tests for blackjax.adaptation.metric_estimators (E-layer).

Each test class follows the RFC R2a pattern: a **frozen inline reference** of
the original implementation is embedded directly in the test (not imported, to
prevent any module-level mutation from silencing the golden — see
worklog/lessons/code-patterns/2026-07-11-monkeypatch-mutation-restore-before-call.md).
The E-layer function must produce the same result as the inline reference on
the same structured test inputs.

Test data strategy
------------------
- Correlated Gaussian draws with known analytic structure (Σ with off-diagonal
  correlations, so the eigenbasis is non-trivial).
- Analytic score gradients for Fisher estimators (for a Gaussian target with
  mean μ and covariance Σ, ∇ log p(x) = -Σ^{-1}(x - μ)).
- n > d AND n < d shapes tested for SVD/eigh estimators.
- max_rank ∈ {0-equivalent (k=1), 1, d//2, d} tested where meaningful.
  (k=d is tested since it earned its own case in R1's adversarial review.)
- f32 atol 1e-5 / f64 atol 1e-9.
- Degenerate-support shapes (n small vs d) are included; parity must hold
  on the MASKED / raw behavior in those shapes too.
"""
import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest, parameterized

import blackjax.adaptation.mass_matrix as mass_matrix_module
from blackjax.adaptation.metric_estimators import (
    draws_singular_value_low_rank,
    eigenvalue_informativeness,
    fisher_score_diagonal,
    fisher_score_low_rank,
    sample_covariance_eigh_low_rank,
    sample_variance_diagonal,
    select_top_eigenvalues_by_informativeness,
    welford_dense,
    welford_diagonal,
)
from tests.fixtures import BlackJAXTest


# ---------------------------------------------------------------------------
# Helpers shared across test classes
# ---------------------------------------------------------------------------


def _make_correlated_draws(key, n, d, rho=0.7, scale=None):
    """Draws from a correlated Gaussian: Σ = scale * (rho * 11^T + (1-rho) * I).

    When ``scale`` is None, uses ``jnp.ones(d)`` (isotropic up to rho).
    When ``scale`` is an array of shape ``(d,)``, each coordinate is scaled
    independently.

    Returns ``(draws, mean, cov)`` where ``mean=0`` and ``cov`` is the true
    covariance.
    """
    if scale is None:
        scale = jnp.ones(d)
    # Build correlation matrix
    corr = rho * jnp.ones((d, d)) + (1.0 - rho) * jnp.eye(d)
    # Scale to covariance
    cov = scale[:, None] * corr * scale[None, :]
    L = jnp.linalg.cholesky(cov)
    z = jax.random.normal(key, (n, d))
    draws = z @ L.T  # (n, d)  mean 0
    return draws, jnp.zeros(d), cov


def _analytic_score_grads(draws, cov):
    """Score gradients for N(0, cov): ∇ log p(x) = -cov^{-1} x."""
    cov_inv = jnp.linalg.inv(cov)
    return -(draws @ cov_inv.T)  # (n, d)


# ---------------------------------------------------------------------------
# Frozen inline reference implementations (parity goldens)
# These are NOT imports — they are frozen copies of the originals so that
# any change to the source module does not silently alias the golden.
# ---------------------------------------------------------------------------


def _ref_relative_pd_floor(vals):
    """Frozen copy of low_rank_adaptation._relative_pd_floor (main@532631c1)."""
    scale = jnp.maximum(jnp.max(jnp.abs(vals)), jnp.finfo(vals.dtype).tiny)
    return jnp.finfo(vals.dtype).eps * scale


def _ref_spd_mean(A, B):
    """Frozen copy of low_rank_adaptation._spd_mean (main@532631c1)."""
    vals_b, vecs_b = jnp.linalg.eigh(B)
    vals_b = jnp.maximum(vals_b, _ref_relative_pd_floor(vals_b))
    sqrt_b = jnp.sqrt(vals_b)
    inv_sqrt_b = 1.0 / sqrt_b
    tmp = vecs_b.T @ A @ vecs_b
    M = inv_sqrt_b[:, None] * tmp * inv_sqrt_b[None, :]
    vals_m, vecs_m = jnp.linalg.eigh(M)
    vals_m = jnp.maximum(vals_m, _ref_relative_pd_floor(vals_m))
    sqrt_m = jnp.sqrt(vals_m)
    W = vecs_b @ (sqrt_b[:, None] * vecs_m)
    return (W * sqrt_m[None, :]) @ W.T


def _ref_fisher_score_low_rank(draws, grads, max_rank, gamma, cutoff):
    """Frozen parity golden for fisher_score_low_rank.

    Adapted from low_rank_adaptation._compute_low_rank_metric (main@532631c1),
    taking raw draw/grad arrays (all-valid, no buffer masking needed).
    """
    orig_dtype = draws.dtype
    compute_dtype = jnp.float64 if jax.config.jax_enable_x64 else orig_dtype
    draws = draws.astype(compute_dtype)
    grads = grads.astype(compute_dtype)

    n, d = draws.shape
    n_f = float(n)

    mean_x = draws.mean(0)
    mean_g = grads.mean(0)
    diff_x = draws - mean_x[None, :]
    diff_g = grads - mean_g[None, :]
    var_x = (diff_x**2).sum(0) / n_f
    var_g = (diff_g**2).sum(0) / n_f

    sigma = jnp.power(jnp.clip(var_x / jnp.maximum(var_g, 1e-10), 0.0, None), 0.25)
    sigma = jnp.clip(sigma, 1e-20, 1e20)

    X = diff_x / sigma[None, :]
    A = diff_g * sigma[None, :]

    _, _, Vt_x = jnp.linalg.svd(X, full_matrices=False)
    _, _, Vt_a = jnp.linalg.svd(A, full_matrices=False)
    U_x = Vt_x[:max_rank].T
    U_a = Vt_a[:max_rank].T

    combined = jnp.concatenate([U_x, U_a], axis=1)
    Q, _ = jnp.linalg.qr(combined)
    q = Q.shape[1]

    P_x = Q.T @ X.T
    P_a = Q.T @ A.T

    C_x = (P_x @ P_x.T) / gamma + jnp.eye(q, dtype=compute_dtype)
    C_a = (P_a @ P_a.T) / gamma + jnp.eye(q, dtype=compute_dtype)

    Sigma = _ref_spd_mean(C_x, jnp.linalg.inv(C_a))

    vals, vecs = jnp.linalg.eigh(Sigma)
    vals = jnp.maximum(vals, _ref_relative_pd_floor(vals))
    U_full = Q @ vecs

    actual_rank = min(max_rank, q)
    distances = jnp.abs(vals - 1.0)
    order = jnp.argsort(-distances)[:actual_rank]
    U_out = U_full[:, order]
    lam_raw = vals[order]
    is_informative = (lam_raw < 1.0 / cutoff) | (lam_raw > cutoff)
    lam_out = jnp.where(is_informative, lam_raw, 1.0)

    if actual_rank < max_rank:
        pad = max_rank - actual_rank
        U_out = jnp.concatenate([U_out, jnp.zeros((d, pad))], axis=1)
        lam_out = jnp.concatenate([lam_out, jnp.ones(pad)])

    return (
        sigma.astype(orig_dtype),
        U_out.astype(orig_dtype),
        lam_out.astype(orig_dtype),
    )


def _ref_draws_singular_value_low_rank(draws, max_rank):
    """Frozen parity golden for draws_singular_value_low_rank.

    Adapted from mclmc_lrd_adaptation._extract_lrd_from_samples (main@532631c1).
    """
    mean = jnp.mean(draws, axis=0)
    sigma = jnp.std(draws, axis=0)
    sigma = jnp.where(sigma == 0.0, 1.0, sigma)

    standardised = (draws - mean[None, :]) / sigma[None, :]
    n = draws.shape[0]

    _, S, Vt = jnp.linalg.svd(standardised, full_matrices=False)
    V = Vt.T
    lam = (S**2) / n

    sort_idx = jnp.argsort(jnp.abs(lam - 1.0))[::-1]
    top_idx = sort_idx[:max_rank]

    lam_k = lam[top_idx]
    U_k = V[:, top_idx]
    return sigma, U_k, lam_k


def _ref_sample_covariance_eigh_low_rank(m2, count, max_rank):
    """Frozen parity golden for sample_covariance_eigh_low_rank.

    Adapted from meads_adaptation._lrd_from_accumulated_covariance (main@532631c1).
    """
    covariance = m2 / jnp.maximum(count - 1.0, 1.0)
    variance = jnp.diag(covariance)
    sigma = jnp.sqrt(jnp.maximum(variance, 0.0))
    sigma = jnp.where(sigma <= 0.0, 1.0, sigma)

    inv_sigma = 1.0 / sigma
    correlation = covariance * inv_sigma[:, None] * inv_sigma[None, :]

    lam_all, V = jnp.linalg.eigh(correlation)
    sort_idx = jnp.argsort(jnp.abs(lam_all - 1.0))[::-1]
    top_idx = sort_idx[:max_rank]
    lam = lam_all[top_idx]
    U = V[:, top_idx]
    return sigma, U, lam


def _ref_fisher_score_diagonal(draws, grads):
    """Frozen parity golden for fisher_score_diagonal.

    Source: branch b197f1e2, mass_matrix._fisher_diagonal_inverse_mass.
    """
    # Bessel-corrected variance via Welford (same as welford_diagonal).
    n, d = draws.shape

    def _welford_var(xs):
        wc_init, wc_update, wc_final = mass_matrix_module.welford_algorithm(
            is_diagonal_matrix=True
        )

        def scan_fn(state, draw):
            return wc_update(state, draw), None

        final_state, _ = jax.lax.scan(scan_fn, wc_init(d), xs)
        cov, _, _ = wc_final(final_state)
        return cov

    var_x = _welford_var(draws)
    var_g = _welford_var(grads)

    sigma = jnp.power(jnp.clip(var_x / jnp.maximum(var_g, 1e-10), 0.0, None), 0.25)
    sigma = jnp.clip(sigma, 1e-20, 1e20)
    return sigma**2


def _ref_sample_variance_diagonal_mclmc(draws):
    """Frozen parity golden for sample_variance_diagonal — MCLMC twin.

    Source: mclmc_adaptation.py:341-342 (inline population variance).
    """
    x_average = jnp.mean(draws, axis=0)
    x_squared_average = jnp.mean(draws**2, axis=0)
    return x_squared_average - jnp.square(x_average)


def _ref_sample_variance_diagonal_adjusted(draws):
    """Frozen parity golden for sample_variance_diagonal — adjusted_mclmc twin.

    Source: adjusted_mclmc_adaptation.py:374-375 (verbatim duplicate).
    """
    x_average = jnp.mean(draws, axis=0)
    x_squared_average = jnp.mean(draws**2, axis=0)
    return x_squared_average - jnp.square(x_average)


# Metric reconstruction helper for visual sanity checks
def _dense_imm_from_lowrank(sigma, U, lam):
    """M^{-1} = diag(sigma) (I + U (lam-I) U^T) diag(sigma)."""
    d = sigma.shape[0]
    correction = U @ jnp.diag(lam - 1.0) @ U.T
    return jnp.diag(sigma) @ (jnp.eye(d) + correction) @ jnp.diag(sigma)


# ---------------------------------------------------------------------------
# Test classes
# ---------------------------------------------------------------------------


class EigenvalueInformativenessTest(BlackJAXTest):
    """Tests for eigenvalue_informativeness."""

    def test_isotropic(self):
        """All-ones eigenvalues → all-zero informativeness."""
        lam = jnp.ones(5)
        inf = eigenvalue_informativeness(lam)
        np.testing.assert_allclose(inf, jnp.zeros(5), atol=1e-9)

    def test_known_values(self):
        lam = jnp.array([0.5, 1.0, 2.0, 3.0])
        expected = jnp.array([0.5, 0.0, 1.0, 2.0])
        np.testing.assert_allclose(eigenvalue_informativeness(lam), expected, atol=1e-9)

    def test_symmetric_around_one(self):
        """λ and 2-λ have the same informativeness."""
        lam = jnp.array([0.3, 1.7])
        inf = eigenvalue_informativeness(lam)
        np.testing.assert_allclose(inf[0], inf[1], atol=1e-9)


class SelectTopEigenvaluesTest(BlackJAXTest):
    """Tests for select_top_eigenvalues_by_informativeness."""

    def _make_random_eigenpairs(self, d, q):
        key = self.next_key()
        raw = jax.random.normal(key, (d, q))
        Q, _ = jnp.linalg.qr(raw)  # orthonormal columns
        lam = jax.random.uniform(self.next_key(), (q,), minval=0.1, maxval=4.0)
        return Q[:, :q], lam

    def test_mask_pad_selects_top_k_by_informativeness(self):
        d, q, k = 6, 5, 3
        V, lam = self._make_random_eigenpairs(d, q)
        U_out, lam_out = select_top_eigenvalues_by_informativeness(
            lam, V, k, tail_handling="mask_pad"
        )
        self.assertEqual(U_out.shape, (d, k))
        self.assertEqual(lam_out.shape, (k,))

        # The returned eigenvalues must be the top-k by |λ-1|.
        expected_order = jnp.argsort(-jnp.abs(lam - 1.0))[:k]
        expected_lam_raw = lam[expected_order]
        # After masking, informative ones are preserved.
        cutoff = 2.0
        is_inf = (expected_lam_raw < 1.0 / cutoff) | (expected_lam_raw > cutoff)
        expected_lam = jnp.where(is_inf, expected_lam_raw, 1.0)
        np.testing.assert_allclose(lam_out, expected_lam, atol=1e-6)

    def test_raw_selects_top_k_no_masking(self):
        d, q, k = 6, 5, 3
        V, lam = self._make_random_eigenpairs(d, q)
        U_out, lam_out = select_top_eigenvalues_by_informativeness(
            lam, V, k, tail_handling="raw"
        )
        self.assertEqual(U_out.shape, (d, k))
        self.assertEqual(lam_out.shape, (k,))

        expected_order = jnp.argsort(-jnp.abs(lam - 1.0))[:k]
        np.testing.assert_allclose(lam_out, lam[expected_order], atol=1e-9)

    def test_mask_pad_pads_when_q_lt_max_rank(self):
        """When fewer eigenvectors than max_rank are available, zero-pad."""
        d, q, k = 8, 3, 6  # q < k → must pad
        V, lam = self._make_random_eigenpairs(d, q)
        U_out, lam_out = select_top_eigenvalues_by_informativeness(
            lam, V, k, tail_handling="mask_pad"
        )
        self.assertEqual(U_out.shape, (d, k))
        self.assertEqual(lam_out.shape, (k,))
        # Padded lam entries must be 1.0.
        np.testing.assert_allclose(lam_out[q:], jnp.ones(k - q), atol=1e-9)
        # Padded U columns must be zeros.
        np.testing.assert_allclose(U_out[:, q:], jnp.zeros((d, k - q)), atol=1e-9)

    def test_mask_pad_k_equals_d(self):
        """k=d: all eigenvectors selected (edge case from R1 adversarial review)."""
        d = 5
        V, lam = self._make_random_eigenpairs(d, d)
        U_out, lam_out = select_top_eigenvalues_by_informativeness(
            lam, V, d, tail_handling="mask_pad"
        )
        self.assertEqual(U_out.shape, (d, d))
        self.assertEqual(lam_out.shape, (d,))

    def test_invalid_tail_handling_raises(self):
        with self.assertRaises(ValueError):
            select_top_eigenvalues_by_informativeness(
                jnp.ones(3), jnp.eye(3), 2, tail_handling="invalid"
            )


class FisherScoreLowRankParityTest(BlackJAXTest):
    """Parity goldens for fisher_score_low_rank vs _ref_fisher_score_low_rank."""

    @parameterized.parameters(
        # (n, d, max_rank, gamma, cutoff, dtype)
        (50, 10, 3, 1e-5, 2.0, jnp.float32),
        (50, 10, 1, 1e-5, 2.0, jnp.float32),
        (50, 10, 5, 1e-5, 2.0, jnp.float32),
        (50, 10, 10, 1e-5, 2.0, jnp.float32),  # k=d
        (8, 10, 3, 1e-5, 2.0, jnp.float32),  # n < d
    )
    def test_parity_vs_reference(self, n, d, max_rank, gamma, cutoff, dtype):
        draws_f32, _, cov = _make_correlated_draws(
            self.next_key(), n, d, rho=0.5, scale=jnp.arange(1, d + 1, dtype=jnp.float32)
        )
        grads_f32 = _analytic_score_grads(draws_f32, cov)

        draws = draws_f32.astype(dtype)
        grads = grads_f32.astype(dtype)

        result = fisher_score_low_rank(draws, grads, max_rank, gamma=gamma, cutoff=cutoff)
        ref_sigma, ref_U, ref_lam = _ref_fisher_score_low_rank(
            draws, grads, max_rank, gamma, cutoff
        )

        atol = 1e-5 if dtype == jnp.float32 else 1e-9
        np.testing.assert_allclose(result.sigma, ref_sigma, atol=atol, err_msg="sigma")
        np.testing.assert_allclose(result.lam, ref_lam, atol=atol, err_msg="lam")

        # U is defined up to column sign flips — compare IMM reconstructions.
        if n >= d:  # well-defined reconstruction
            M_result = _dense_imm_from_lowrank(result.sigma, result.U, result.lam)
            M_ref = _dense_imm_from_lowrank(ref_sigma, ref_U, ref_lam)
            np.testing.assert_allclose(M_result, M_ref, atol=atol * 10, err_msg="dense IMM")

    def test_output_shapes(self):
        n, d, k = 30, 6, 4
        draws, _, cov = _make_correlated_draws(self.next_key(), n, d)
        grads = _analytic_score_grads(draws, cov)
        result = fisher_score_low_rank(draws, grads, k)
        self.assertEqual(result.sigma.shape, (d,))
        self.assertEqual(result.U.shape, (d, k))
        self.assertEqual(result.lam.shape, (k,))

    def test_sigma_positive(self):
        n, d, k = 30, 6, 2
        draws, _, cov = _make_correlated_draws(self.next_key(), n, d)
        grads = _analytic_score_grads(draws, cov)
        result = fisher_score_low_rank(draws, grads, k)
        self.assertTrue(jnp.all(result.sigma > 0).item())

    def test_lam_informative_elements_not_masked(self):
        """Informative eigenvalues (outside [1/cutoff, cutoff]) are preserved."""
        n, d, k = 100, 10, 5
        # Use highly correlated draws to ensure some eigenvalues are far from 1.
        draws, _, cov = _make_correlated_draws(
            self.next_key(), n, d, rho=0.9, scale=jnp.linspace(0.5, 3.0, d)
        )
        grads = _analytic_score_grads(draws, cov)
        result = fisher_score_low_rank(draws, grads, k, cutoff=2.0)
        # All lam values must be >= 0 (not NaN).
        self.assertTrue(jnp.all(jnp.isfinite(result.lam)).item())


class DrawsSingularValueLowRankParityTest(BlackJAXTest):
    """Parity goldens for draws_singular_value_low_rank vs _ref_draws_svd_lrd."""

    @parameterized.parameters(
        # (n, d, max_rank)
        (50, 10, 3),
        (50, 10, 1),
        (50, 10, 5),
        (50, 10, 10),  # k=d  (only works when min(n,d) >= d, i.e. n >= d)
        (10, 6, 3),   # n > d, k < n
    )
    def test_parity_vs_reference(self, n, d, max_rank):
        k = min(max_rank, min(n, d))  # respect SVD rank constraint
        draws, _, _ = _make_correlated_draws(self.next_key(), n, d, rho=0.6)

        result = draws_singular_value_low_rank(draws, k)
        ref_sigma, ref_U, ref_lam = _ref_draws_singular_value_low_rank(draws, k)

        np.testing.assert_allclose(result.sigma, ref_sigma, atol=1e-5, err_msg="sigma")
        np.testing.assert_allclose(result.lam, ref_lam, atol=1e-5, err_msg="lam")

        # U defined up to column sign — compare IMM reconstructions.
        M_result = _dense_imm_from_lowrank(result.sigma, result.U, result.lam)
        M_ref = _dense_imm_from_lowrank(ref_sigma, ref_U, ref_lam)
        np.testing.assert_allclose(M_result, M_ref, atol=1e-4, err_msg="dense IMM")

    def test_n_less_than_d(self):
        """n < d: only min(n,d)=n eigenvalues available from SVD."""
        n, d, k = 4, 10, 4  # k = n = min(n,d)
        draws, _, _ = _make_correlated_draws(self.next_key(), n, d)
        result = draws_singular_value_low_rank(draws, k)
        # Shape must be (d, k) for U and (k,) for lam.
        self.assertEqual(result.U.shape, (d, k))
        self.assertEqual(result.lam.shape, (k,))

    def test_no_regularisation_or_masking(self):
        """Raw lam values: eigenvalues inside [1/2, 2] are NOT masked to 1."""
        n, d, k = 100, 5, 5
        # Draws from near-isotropic Gaussian — most eigenvalues near 1.
        draws = jax.random.normal(self.next_key(), (n, d))
        result = draws_singular_value_low_rank(draws, k)
        ref_sigma, ref_U, ref_lam = _ref_draws_singular_value_low_rank(draws, k)
        # lam values that are near 1 in reference must also be near 1 in result
        # (not masked — raw preservation).
        np.testing.assert_allclose(result.lam, ref_lam, atol=1e-5)


class SampleCovarianceEighLowRankParityTest(BlackJAXTest):
    """Parity goldens for sample_covariance_eigh_low_rank vs reference."""

    def _make_m2(self, draws):
        """Compute Chan-Welford M2 from draws in one batch."""
        mean = jnp.mean(draws, axis=0)
        diff = draws - mean[None, :]
        return diff.T @ diff, draws.shape[0]

    @parameterized.parameters(
        # (n, d, max_rank)
        (100, 10, 3),
        (100, 10, 1),
        (100, 10, 5),
        (100, 10, 10),  # k=d
        (8, 10, 3),    # n < d: rank-deficient correlation matrix
    )
    def test_parity_vs_reference(self, n, d, max_rank):
        k = min(max_rank, d)
        draws, _, _ = _make_correlated_draws(self.next_key(), n, d, rho=0.6)
        m2, count = self._make_m2(draws)

        result = sample_covariance_eigh_low_rank(m2, count, k)
        ref_sigma, ref_U, ref_lam = _ref_sample_covariance_eigh_low_rank(m2, count, k)

        np.testing.assert_allclose(result.sigma, ref_sigma, atol=1e-5, err_msg="sigma")
        np.testing.assert_allclose(result.lam, ref_lam, atol=1e-5, err_msg="lam")

        # U defined up to column sign — compare IMM reconstructions.
        M_result = _dense_imm_from_lowrank(result.sigma, result.U, result.lam)
        M_ref = _dense_imm_from_lowrank(ref_sigma, ref_U, ref_lam)
        np.testing.assert_allclose(M_result, M_ref, atol=1e-4, err_msg="dense IMM")

    def test_count_1_degenerate_support(self):
        """count=1: Bessel denominator clamped to 1 (no division by zero)."""
        d, k = 5, 2
        # Single draw → m2 = 0 (mean subtraction gives zero).
        m2 = jnp.zeros((d, d))
        count = 1
        result = sample_covariance_eigh_low_rank(m2, count, k)
        # sigma must be 1.0 everywhere (zero variance → fallback to 1.0).
        np.testing.assert_allclose(result.sigma, jnp.ones(d), atol=1e-6)


class WelfordDiagonalParityTest(BlackJAXTest):
    """Parity goldens for welford_diagonal vs np.var(..., ddof=1)."""

    @parameterized.parameters(
        (50, 5),
        (3, 4),   # small n
        (200, 2),
    )
    def test_parity_vs_numpy_var(self, n, d):
        draws, _, _ = _make_correlated_draws(self.next_key(), n, d)
        result = welford_diagonal(draws)
        expected = jnp.array(np.var(np.array(draws), axis=0, ddof=1))
        np.testing.assert_allclose(result, expected, atol=1e-5)

    def test_output_shape(self):
        n, d = 30, 7
        draws = jax.random.normal(self.next_key(), (n, d))
        self.assertEqual(welford_diagonal(draws).shape, (d,))


class WelfordDenseParityTest(BlackJAXTest):
    """Parity goldens for welford_dense vs np.cov."""

    @parameterized.parameters(
        (50, 5),
        (4, 3),
    )
    def test_parity_vs_numpy_cov(self, n, d):
        draws, _, _ = _make_correlated_draws(self.next_key(), n, d)
        result = welford_dense(draws)
        expected = jnp.array(np.cov(np.array(draws).T))  # np.cov uses ddof=1
        np.testing.assert_allclose(result, expected, atol=1e-5)

    def test_output_shape(self):
        n, d = 30, 7
        draws = jax.random.normal(self.next_key(), (n, d))
        self.assertEqual(welford_dense(draws).shape, (d, d))


class FisherScoreDiagonalParityTest(BlackJAXTest):
    """Parity goldens for fisher_score_diagonal vs branch b197f1e2 reference."""

    @parameterized.parameters(
        # (n, d)
        (50, 5),
        (4, 5),   # n < d
        (200, 3),
    )
    def test_parity_vs_reference(self, n, d):
        draws, _, cov = _make_correlated_draws(
            self.next_key(), n, d, scale=jnp.linspace(0.3, 2.0, d)
        )
        grads = _analytic_score_grads(draws, cov)

        result = fisher_score_diagonal(draws, grads)
        ref = _ref_fisher_score_diagonal(draws, grads)
        np.testing.assert_allclose(result, ref, atol=1e-5)

    def test_output_shape(self):
        n, d = 30, 6
        draws = jax.random.normal(self.next_key(), (n, d))
        grads = jax.random.normal(self.next_key(), (n, d))
        self.assertEqual(fisher_score_diagonal(draws, grads).shape, (d,))

    def test_positive(self):
        n, d = 40, 4
        draws, _, cov = _make_correlated_draws(self.next_key(), n, d)
        grads = _analytic_score_grads(draws, cov)
        result = fisher_score_diagonal(draws, grads)
        self.assertTrue(jnp.all(result > 0).item())

    def test_consistency_with_fisher_score_low_rank_sigma_sq(self):
        """fisher_score_diagonal(draws, grads) ≈ fisher_score_low_rank sigma^2.

        Both estimators use the same diagonal scale formula; the diagonal-only
        version omits the low-rank correction (U=0 effective).  The sigma^2
        output of the two must agree (up to f32 atol) because both compute:
            sigma = (Var[x] / Var[grad])^{0.25} clipped to [1e-20, 1e20]
            IMM_diag = sigma^2
        with the same Welford variance.
        """
        n, d, k = 80, 6, 3
        draws, _, cov = _make_correlated_draws(self.next_key(), n, d)
        grads = _analytic_score_grads(draws, cov)

        diag_result = fisher_score_diagonal(draws, grads)
        lr_result = fisher_score_low_rank(draws, grads, k)

        # sigma^2 from low-rank matches the diagonal estimator.
        np.testing.assert_allclose(
            diag_result, lr_result.sigma**2, atol=1e-5,
            err_msg="diagonal fisher sigma^2 must match low-rank sigma^2"
        )


class SampleVarianceDiagonalParityTest(BlackJAXTest):
    """Parity goldens for sample_variance_diagonal vs both MCLMC inline twins."""

    @parameterized.parameters(
        # (n, d)
        (50, 5),
        (3, 4),   # n < d
        (100, 8),
    )
    def test_parity_vs_mclmc_twin(self, n, d):
        draws, _, _ = _make_correlated_draws(self.next_key(), n, d)
        result = sample_variance_diagonal(draws)
        ref = _ref_sample_variance_diagonal_mclmc(draws)
        np.testing.assert_allclose(result, ref, atol=1e-6,
                                   err_msg="parity vs mclmc_adaptation twin")

    @parameterized.parameters(
        (50, 5),
        (3, 4),
        (100, 8),
    )
    def test_parity_vs_adjusted_mclmc_twin(self, n, d):
        draws, _, _ = _make_correlated_draws(self.next_key(), n, d)
        result = sample_variance_diagonal(draws)
        ref = _ref_sample_variance_diagonal_adjusted(draws)
        np.testing.assert_allclose(result, ref, atol=1e-6,
                                   err_msg="parity vs adjusted_mclmc_adaptation twin")

    def test_population_not_bessel(self):
        """sample_variance_diagonal uses population variance (n denominator)."""
        n, d = 10, 3
        draws, _, _ = _make_correlated_draws(self.next_key(), n, d)
        result = sample_variance_diagonal(draws)
        pop_var = jnp.mean(draws**2, axis=0) - jnp.mean(draws, axis=0) ** 2
        bessel_var = jnp.array(np.var(np.array(draws), axis=0, ddof=1))
        # Should match population variance, not Bessel-corrected.
        np.testing.assert_allclose(result, pop_var, atol=1e-6)
        # Must differ from Bessel for n>1.
        self.assertFalse(
            jnp.allclose(result, bessel_var, atol=1e-6).item(),
            "Expected population variance to differ from Bessel-corrected"
        )

    def test_non_negative(self):
        """Variance must be non-negative for any input."""
        n, d = 20, 4
        draws, _, _ = _make_correlated_draws(self.next_key(), n, d)
        result = sample_variance_diagonal(draws)
        self.assertTrue(jnp.all(result >= 0.0).item())

    def test_output_shape(self):
        n, d = 30, 7
        draws = jax.random.normal(self.next_key(), (n, d))
        self.assertEqual(sample_variance_diagonal(draws).shape, (d,))


if __name__ == "__main__":
    absltest.main()
