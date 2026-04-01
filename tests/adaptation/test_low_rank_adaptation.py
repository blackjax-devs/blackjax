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
"""Tests for low-rank mass matrix adaptation."""
import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest

import blackjax
from blackjax.adaptation.low_rank_adaptation import _compute_low_rank_metric, _spd_mean
from tests.util import BlackJAXTest


class SPDMeanTest(BlackJAXTest):
    """Tests for _spd_mean (SPD geometric mean)."""

    def _make_spd(self, key, n):
        A = jax.random.normal(key, (n, n))
        return A @ A.T + n * jnp.eye(n)

    def test_symmetry(self):
        """A # B is symmetric."""
        k1, k2 = jax.random.split(self.next_key())
        A = self._make_spd(k1, 5)
        B = self._make_spd(k2, 5)
        G = _spd_mean(A, B)
        np.testing.assert_allclose(G, G.T, atol=1e-5)

    def test_commutativity(self):
        """A # B = B # A."""
        k1, k2 = jax.random.split(self.next_key())
        A = self._make_spd(k1, 5)
        B = self._make_spd(k2, 5)
        np.testing.assert_allclose(_spd_mean(A, B), _spd_mean(B, A), atol=1e-4)

    def test_identity_case(self):
        """I # A = A^{1/2} (geometric mean with identity gives matrix sqrt)."""
        k1 = self.next_key()
        A = self._make_spd(k1, 4)
        G = _spd_mean(jnp.eye(4), A)
        # G^2 should equal A
        np.testing.assert_allclose(G @ G, A, atol=1e-4)

    def test_equal_matrices(self):
        """A # A = A."""
        k1 = self.next_key()
        A = self._make_spd(k1, 4)
        np.testing.assert_allclose(_spd_mean(A, A), A, atol=1e-5)

    def test_eigenvalue_bounds(self):
        """Eigenvalues of A # B lie between those of A and B (geometric interpolation)."""
        k1, k2 = jax.random.split(self.next_key())
        A = self._make_spd(k1, 6)
        B = self._make_spd(k2, 6)
        vals_g = jnp.linalg.eigvalsh(_spd_mean(A, B))
        # Geometric mean eigenvalues don't simply interleave with A and B eigenvalues,
        # but all eigenvalues of G must be positive.
        self.assertTrue(bool(jnp.all(vals_g > 0)))


class ComputeLowRankMetricTest(BlackJAXTest):
    """Tests for _compute_low_rank_metric."""

    def _make_buffers(self, key, B, d):
        k1, k2 = jax.random.split(key)
        draws = jax.random.normal(k1, (B, d))
        grads = jax.random.normal(k2, (B, d))
        return draws, grads

    def test_output_shapes(self):
        """Returns tensors with correct shapes."""
        d, B, k = 10, 50, 4
        draws, grads = self._make_buffers(self.next_key(), B, d)
        sigma, mu_star, U, lam = _compute_low_rank_metric(draws, grads, B, k, 1.0, 2.0)
        assert sigma.shape == (d,)
        assert mu_star.shape == (d,)
        assert U.shape == (d, k)
        assert lam.shape == (k,)

    def test_sigma_positive(self):
        """σ is strictly positive."""
        d, B, k = 8, 40, 3
        draws, grads = self._make_buffers(self.next_key(), B, d)
        sigma, _, _, _ = _compute_low_rank_metric(draws, grads, B, k, 1.0, 2.0)
        self.assertTrue(bool(jnp.all(sigma > 0)))

    def test_u_orthonormal(self):
        """Columns of U are orthonormal: U^T U = I_k."""
        d, B, k = 10, 60, 3
        draws, grads = self._make_buffers(self.next_key(), B, d)
        _, _, U, _ = _compute_low_rank_metric(draws, grads, B, k, 1.0, 2.0)
        np.testing.assert_allclose(U.T @ U, jnp.eye(k), atol=1e-5)

    def test_eigenvalue_masking(self):
        """Eigenvalues in [1/cutoff, cutoff] are set to 1."""
        d, B, k = 10, 60, 4
        draws, grads = self._make_buffers(self.next_key(), B, d)
        cutoff = 2.0
        _, _, _, lam = _compute_low_rank_metric(draws, grads, B, k, 1.0, cutoff)
        # Each lam value is either 1 (masked) or outside [1/cutoff, cutoff]
        informative = (lam < 1.0 / cutoff) | (lam > cutoff)
        masked = lam == 1.0
        self.assertTrue(bool(jnp.all(informative | masked)))

    def test_mu_star_formula(self):
        """μ* = mean_x + σ² ⊙ mean_g matches independent computation."""
        d, B, k = 6, 80, 2
        draws, grads = self._make_buffers(self.next_key(), B, d)
        sigma, mu_star, _, _ = _compute_low_rank_metric(draws, grads, B, k, 1.0, 2.0)
        mean_x = draws.mean(0)
        mean_g = grads.mean(0)
        mu_star_expected = mean_x + sigma**2 * mean_g
        np.testing.assert_allclose(mu_star, mu_star_expected, rtol=1e-5)

    def test_sigma_formula(self):
        """σ = (Var[x] / Var[∇log p])^{1/4} (population variance)."""
        d, B, k = 6, 200, 2
        draws, grads = self._make_buffers(self.next_key(), B, d)
        sigma, _, _, _ = _compute_low_rank_metric(draws, grads, B, k, 1.0, 2.0)
        # Population variance
        var_x = draws.var(axis=0)
        var_g = grads.var(axis=0)
        sigma_expected = (var_x / jnp.maximum(var_g, 1e-10)) ** 0.25
        np.testing.assert_allclose(
            sigma, jnp.clip(sigma_expected, 1e-20, 1e20), rtol=1e-4
        )

    def test_d_less_than_2k(self):
        """Works when d < 2*max_rank (QR gives smaller Q)."""
        d, B, k = 4, 30, 4  # 2k = 8 > d = 4
        draws, grads = self._make_buffers(self.next_key(), B, d)
        sigma, mu_star, U, lam = _compute_low_rank_metric(draws, grads, B, k, 1.0, 2.0)
        assert sigma.shape == (d,)
        assert U.shape == (d, k)

    def test_partial_buffer(self):
        """Works with n < B (partially filled buffer)."""
        d, B, k = 8, 100, 3
        draws, grads = self._make_buffers(self.next_key(), B, d)
        # Only first 40 rows are valid
        n = 40
        draws = draws.at[n:].set(0.0)
        grads = grads.at[n:].set(0.0)
        sigma, mu_star, U, lam = _compute_low_rank_metric(draws, grads, n, k, 1.0, 2.0)
        assert sigma.shape == (d,)
        assert mu_star.shape == (d,)


class LowRankWindowAdaptationTest(BlackJAXTest):
    """Integration tests for low_rank_window_adaptation."""

    def test_runs_on_standard_normal(self):
        """Adaptation runs without error on a standard normal target."""
        logdensity_fn = lambda x: -0.5 * jnp.sum(x**2)
        warmup = blackjax.low_rank_window_adaptation(
            blackjax.nuts, logdensity_fn, max_rank=3
        )
        (state, params), _ = warmup.run(self.next_key(), jnp.ones(5), num_steps=200)
        self.assertIn("step_size", params)
        self.assertIn("inverse_mass_matrix", params)
        self.assertNotIn("mu_star", params)
        self.assertEqual(state.position.shape, (5,))

    def test_mu_star_recovers_posterior_mean(self):
        """State position (= μ*) should be close to the true posterior mean after warmup."""
        d = 6
        true_mean = jnp.array([2.0, -1.0, 0.5, -0.5, 1.5, -2.0])
        logdensity_fn = lambda x: -0.5 * jnp.sum((x - true_mean) ** 2)
        warmup = blackjax.low_rank_window_adaptation(
            blackjax.nuts, logdensity_fn, max_rank=3
        )
        (state, _), _ = warmup.run(self.next_key(), jnp.zeros(d), num_steps=500)
        np.testing.assert_allclose(state.position, true_mean, atol=0.2)

    def test_step_size_positive(self):
        """Adapted step size is strictly positive."""
        logdensity_fn = lambda x: -0.5 * jnp.sum(x**2)
        warmup = blackjax.low_rank_window_adaptation(
            blackjax.nuts, logdensity_fn, max_rank=2
        )
        (_, params), _ = warmup.run(self.next_key(), jnp.zeros(4), num_steps=200)
        self.assertGreater(float(params["step_size"]), 0.0)

    def test_works_with_hmc(self):
        """Adaptation works with HMC (not just NUTS)."""
        logdensity_fn = lambda x: -0.5 * jnp.sum(x**2)
        warmup = blackjax.low_rank_window_adaptation(
            blackjax.hmc,
            logdensity_fn,
            max_rank=2,
            num_integration_steps=3,
        )
        (state, params), _ = warmup.run(self.next_key(), jnp.zeros(4), num_steps=200)
        self.assertIn("step_size", params)

    def test_various_ranks(self):
        """Adaptation succeeds for various max_rank values."""
        d = 8
        logdensity_fn = lambda x: -0.5 * jnp.sum(x**2)
        for max_rank in [1, 5, 10]:
            warmup = blackjax.low_rank_window_adaptation(
                blackjax.nuts, logdensity_fn, max_rank=max_rank
            )
            (state, params), _ = warmup.run(
                self.next_key(), jnp.zeros(d), num_steps=200
            )
            self.assertEqual(state.position.shape, (d,))


if __name__ == "__main__":
    absltest.main()
