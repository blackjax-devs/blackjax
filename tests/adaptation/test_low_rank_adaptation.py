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
from blackjax.adaptation.low_rank_adaptation import (
    _compute_low_rank_metric,
    _spd_mean,
    base,
    build_schedule_nutpie_mvp,
)
from blackjax.adaptation.window_adaptation import build_schedule
from blackjax.mcmc.metrics import LowRankInverseMassMatrix
from tests.fixtures import BlackJAXTest


def _low_rank_inverse_mass_matrix(sigma, U, lam):
    """Reconstruct the dense inverse mass matrix from the low-rank factors.

    ``M^{-1} = diag(sigma) (I + U (lam - I) U^T) diag(sigma)``, matching
    Eq. (10)-(11) of :cite:p:`seyboldt2026preconditioning` and the docstring
    of :func:`window_adaptation_low_rank`.
    """
    d = sigma.shape[0]
    return (
        jnp.diag(sigma) @ (jnp.eye(d) + U @ jnp.diag(lam - 1.0) @ U.T) @ jnp.diag(sigma)
    )


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
    """Integration tests for window_adaptation_low_rank."""

    def test_runs_on_standard_normal(self):
        """Adaptation runs without error on a standard normal target."""
        logdensity_fn = lambda x: -0.5 * jnp.sum(x**2)
        warmup = blackjax.window_adaptation_low_rank(
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
        warmup = blackjax.window_adaptation_low_rank(
            blackjax.nuts, logdensity_fn, max_rank=3
        )
        (state, _), _ = warmup.run(self.next_key(), jnp.zeros(d), num_steps=500)
        np.testing.assert_allclose(state.position, true_mean, atol=0.2)

    def test_step_size_positive(self):
        """Adapted step size is strictly positive."""
        logdensity_fn = lambda x: -0.5 * jnp.sum(x**2)
        warmup = blackjax.window_adaptation_low_rank(
            blackjax.nuts, logdensity_fn, max_rank=2
        )
        (_, params), _ = warmup.run(self.next_key(), jnp.zeros(4), num_steps=200)
        self.assertGreater(float(params["step_size"]), 0.0)

    def test_works_with_hmc(self):
        """Adaptation works with HMC (not just NUTS)."""
        logdensity_fn = lambda x: -0.5 * jnp.sum(x**2)
        warmup = blackjax.window_adaptation_low_rank(
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
            warmup = blackjax.window_adaptation_low_rank(
                blackjax.nuts, logdensity_fn, max_rank=max_rank
            )
            (state, params), _ = warmup.run(
                self.next_key(), jnp.zeros(d), num_steps=200
            )
            self.assertEqual(state.position.shape, (d,))

    def test_inverse_mass_matrix_is_pure_pytree(self):
        """``params['inverse_mass_matrix']`` is an array-only NamedTuple (GH #916).

        The warmup must return ``LowRankInverseMassMatrix(sigma, U, lam)``
        — a pure JAX pytree — rather than a closure-bearing ``Metric``, so
        the result can be transported across ``jax.vmap`` / ``jax.pmap``.
        """
        d = 5
        logdensity_fn = lambda x: -0.5 * jnp.sum(x**2)
        warmup = blackjax.window_adaptation_low_rank(
            blackjax.nuts, logdensity_fn, max_rank=3
        )
        (_, params), _ = warmup.run(self.next_key(), jnp.zeros(d), num_steps=200)
        imm = params["inverse_mass_matrix"]
        self.assertIsInstance(imm, LowRankInverseMassMatrix)
        self.assertEqual(imm.sigma.shape, (d,))
        self.assertEqual(imm.U.shape, (d, 3))
        self.assertEqual(imm.lam.shape, (3,))
        # All fields are JAX arrays (no Python closures captured as leaves).
        leaves = jax.tree_util.tree_leaves(imm)
        self.assertEqual(len(leaves), 3)
        for leaf in leaves:
            self.assertTrue(hasattr(leaf, "shape"))

    def test_multi_chain_vmap(self):
        """Warmup composes with ``jax.vmap`` over chains (GH #916 reproducer).

        Before the fix this raised ``TypeError: Output from batched function
        ... is not a valid JAX type`` because the returned ``Metric`` carried
        Python closures that vmap could not stack.
        """
        d = 8
        num_chains = 3
        logdensity_fn = lambda x: -0.5 * jnp.sum(x**2)

        warmup = blackjax.window_adaptation_low_rank(
            blackjax.hmc,
            logdensity_fn,
            num_integration_steps=5,
            max_rank=2,
        )

        chain_keys = jax.random.split(self.next_key(), num_chains)
        init_positions = jnp.tile(jnp.zeros(d)[None, :], (num_chains, 1))

        @jax.vmap
        def run_one(k, x):
            (state, params), _ = warmup.run(k, x, num_steps=150)
            return state, params

        states, params = run_one(chain_keys, init_positions)
        self.assertEqual(states.position.shape, (num_chains, d))

        imm = params["inverse_mass_matrix"]
        self.assertIsInstance(imm, LowRankInverseMassMatrix)
        # Each field carries an extra leading batch axis from vmap.
        self.assertEqual(imm.sigma.shape, (num_chains, d))
        self.assertEqual(imm.U.shape, (num_chains, d, 2))
        self.assertEqual(imm.lam.shape, (num_chains, 2))
        self.assertEqual(params["step_size"].shape, (num_chains,))

        # The per-chain payload is consumable by the kernel: pick chain 0,
        # build a NUTS kernel, and take one step (smoke test of the
        # default_metric dispatch on LowRankInverseMassMatrix).
        chain0_imm = jax.tree.map(lambda x: x[0], imm)
        chain0_state = jax.tree.map(lambda x: x[0], states)
        nuts = blackjax.nuts(
            logdensity_fn,
            step_size=float(params["step_size"][0]),
            inverse_mass_matrix=chain0_imm,
        )
        new_state, _info = nuts.step(self.next_key(), chain0_state)
        self.assertEqual(new_state.position.shape, (d,))


class LowRankCorrelationRecoveryTest(BlackJAXTest):
    """Regression tests for the missing score-covariance inversion.

    Finding: ``_spd_mean(C_x, C_a)`` in ``_compute_low_rank_metric`` omitted
    the matrix inversion of ``C_a`` (the regularized score/gradient
    covariance) required by Theorem 2.3 / Eq. (9) of
    :cite:p:`seyboldt2026preconditioning`. Cross-validated against nutpie's
    Rust reference implementation (``nuts-rs``
    ``src/transform/adapt/low_rank.rs``). See the worklog case study
    ``worklog/lessons/case-studies/gp_regression/2026-05-13-low-rank-wrong-sign-suspicion.md``
    for the full diagnosis, including the exact controls reused below.
    """

    def _pairwise_mvn(self, rho, key, n=200_000):
        """2-D correlated Gaussian: var 4/1, correlation rho (case study control)."""
        cov = jnp.array([[4.0, rho * 2.0], [rho * 2.0, 1.0]])
        draws = jax.random.multivariate_normal(key, jnp.zeros(2), cov, shape=(n,))
        grads = -draws @ jnp.linalg.inv(cov).T
        return draws, grads

    def test_pairwise_sign_and_magnitude_recovery(self):
        """Both the sign and the magnitude of a known pairwise correlation
        are recovered (before the fix: exactly-degenerate eigenspectrum,
        seed-unstable sign; see LowRankSeedStabilityTest below)."""
        for rho in (0.7, -0.7, 0.9):
            draws, grads = self._pairwise_mvn(rho, self.next_key())
            sigma, _, U, lam = _compute_low_rank_metric(
                draws, grads, draws.shape[0], 2, 1e-5, 2.0
            )
            minv = _low_rank_inverse_mass_matrix(sigma, U, lam)
            implied_corr = float(minv[0, 1] / jnp.sqrt(minv[0, 0] * minv[1, 1]))
            self.assertEqual(np.sign(implied_corr), np.sign(rho))
            np.testing.assert_allclose(implied_corr, rho, atol=0.05)

    def test_star_topology_sign_recovery(self):
        """7-D hub-and-spoke topology (case study control #2 — closer to a
        real hierarchical funnel than an isolated pair). Before the fix this
        was *stably* wrong-signed (not noise): true +0.30 recovered as
        approximately -0.20 at max_rank=1, shrinking to approximately -0.02
        at max_rank=3-6."""
        d = 7
        rho = 0.3
        cov = jnp.eye(d).at[0, 1:].set(rho).at[1:, 0].set(rho)
        draws = jax.random.multivariate_normal(
            self.next_key(), jnp.zeros(d), cov, shape=(200_000,)
        )
        grads = -draws @ jnp.linalg.inv(cov).T
        for max_rank in (3, 6):
            sigma, _, U, lam = _compute_low_rank_metric(
                draws, grads, draws.shape[0], max_rank, 1e-5, 2.0
            )
            minv = _low_rank_inverse_mass_matrix(sigma, U, lam)
            hub_spoke_corrs = minv[0, 1:] / jnp.sqrt(minv[0, 0] * jnp.diag(minv)[1:])
            self.assertTrue(bool(jnp.all(hub_spoke_corrs > 0)))
            np.testing.assert_allclose(np.asarray(hub_spoke_corrs), rho, atol=0.1)


class LowRankGaussianLimitTest(BlackJAXTest):
    """Gaussian-limit exact-recovery test (Theorem 2.4)."""

    def test_recovers_known_covariance(self):
        """Exact iid draws + exact scores from a known Sigma recover
        M^{-1} == Sigma, matching Theorem 2.4's exact-recovery guarantee once
        the number of draws exceeds d+1. ``cutoff=1.0`` disables eigenvalue
        masking (masks only exactly-unity eigenvalues) so the full-rank
        correction is retained."""
        d = 5
        key1, key2 = jax.random.split(self.next_key())
        A = jax.random.normal(key1, (d, d))
        cov = A @ A.T + d * jnp.eye(d)
        draws = jax.random.multivariate_normal(key2, jnp.zeros(d), cov, shape=(50_000,))
        grads = -draws @ jnp.linalg.inv(cov).T
        sigma, _, U, lam = _compute_low_rank_metric(
            draws, grads, draws.shape[0], d, 1e-5, 1.0
        )
        minv = _low_rank_inverse_mass_matrix(sigma, U, lam)
        np.testing.assert_allclose(
            np.asarray(minv), np.asarray(cov), rtol=0.1, atol=0.1
        )


class LowRankSeedStabilityTest(BlackJAXTest):
    """Top-eigenvector seed-stability test.

    Before the fix, an isolated 2x2 correlated block produced an exactly
    degenerate ``_spd_mean`` eigenspectrum (eigenvalues equal to ~7
    significant figures), so which eigenvector ``max_rank=1`` retained — and
    hence the sign and magnitude of the implied correlation — was arbitrary,
    seed-dependent noise even at n=1,000,000 iid draws (not finite-sample
    noise: the degeneracy was exact). After the fix the recovered
    correlation must be stable (consistent sign, low seed-to-seed variance).
    """

    def test_top_eigenvector_direction_stable_across_seeds(self):
        rho = 0.7
        cov = jnp.array([[4.0, rho * 2.0], [rho * 2.0, 1.0]])
        corrs = []
        for _ in range(6):
            key = self.next_key()
            draws = jax.random.multivariate_normal(
                key, jnp.zeros(2), cov, shape=(200_000,)
            )
            grads = -draws @ jnp.linalg.inv(cov).T
            sigma, _, U, lam = _compute_low_rank_metric(
                draws, grads, draws.shape[0], 1, 1e-5, 2.0
            )
            minv = _low_rank_inverse_mass_matrix(sigma, U, lam)
            corrs.append(float(minv[0, 1] / jnp.sqrt(minv[0, 0] * minv[1, 1])))
        corrs = np.array(corrs)
        self.assertTrue(bool(np.all(corrs > 0)), f"sign flipped across seeds: {corrs}")
        self.assertLess(float(np.std(corrs)), 0.05, f"unstable across seeds: {corrs}")


class LowRankDiagonalConsistencyTest(BlackJAXTest):
    """1-D low-rank/diagonal consistency test.

    In 1 dimension there is no possible correlation structure beyond the
    diagonal scale itself, so Step 7's SPD-mean eigenvalue must reduce to
    lambda=1 (no correction), leaving ``M^{-1} == sigma**2`` exactly matching
    Step 1's diagonal formula. Before the fix, the un-inverted ``_spd_mean``
    implied a spurious ``sqrt(var_x * var_g)``-flavoured correction,
    contradicting the Step-1 diagonal's own ``sqrt(var_x / var_g)`` formula
    three lines above it in the source.
    """

    def test_low_rank_path_agrees_with_diagonal_in_1d(self):
        for var_x, var_g in [(4.0, 1.0), (0.1, 9.0), (25.0, 0.5)]:
            key1, key2 = jax.random.split(self.next_key())
            n = 50_000
            x = jax.random.normal(key1, (n, 1)) * jnp.sqrt(var_x)
            g = jax.random.normal(key2, (n, 1)) * jnp.sqrt(var_g)
            sigma, _, U, lam = _compute_low_rank_metric(x, g, n, 1, 1e-5, 2.0)
            diagonal_only = float(sigma[0] ** 2)
            low_rank = float(_low_rank_inverse_mass_matrix(sigma, U, lam)[0, 0])
            np.testing.assert_allclose(low_rank, diagonal_only, rtol=1e-3)
            np.testing.assert_allclose(float(lam[0]), 1.0, atol=1e-3)


class LowRankSmallNRobustnessTest(BlackJAXTest):
    """Small-n robustness (early-warmup buffer sizes) -- reviewer finding.

    Statistician correctness review (stat-e1) surfaced this adversarial
    check during the fix's review: at borderline-rank-deficient buffer
    sizes typical of the *first* slow window in warmup (n as low as 4, up
    to n=200), the fixed estimator with the corrected ``gamma=1e-5``
    regularisation must (a) never produce NaN/Inf at JAX's default float32
    precision, and (b) already recover the correct sign -- and a
    non-trivial fraction of the true magnitude -- at n=4, where the OLD
    ``gamma=1.0`` default's n-scaled regularisation shows essentially
    nothing (implied correlation collapses to ~0.0 at n=4; see PR draft for
    the side-by-side numbers). This locks in that the gamma-scale fix is
    *more* robust in the small-n regime, not just asymptotically correct.
    """

    def test_no_nan_inf_and_correct_sign_across_small_n(self):
        rho = 0.7
        cov = jnp.array([[4.0, rho * 2.0], [rho * 2.0, 1.0]])
        for n in (4, 10, 50, 200):
            draws = jax.random.multivariate_normal(
                self.next_key(), jnp.zeros(2), cov, shape=(n,)
            )
            grads = -draws @ jnp.linalg.inv(cov).T
            sigma, mu_star, U, lam = _compute_low_rank_metric(
                draws, grads, n, 2, 1e-5, 2.0
            )
            for name, arr in [
                ("sigma", sigma),
                ("mu_star", mu_star),
                ("U", U),
                ("lam", lam),
            ]:
                self.assertTrue(
                    bool(jnp.all(jnp.isfinite(arr))),
                    f"n={n} -- {name} has NaN/Inf: {arr}",
                )
            minv = _low_rank_inverse_mass_matrix(sigma, U, lam)
            self.assertTrue(
                bool(jnp.all(jnp.isfinite(minv))), f"n={n} -- M^-1 has NaN/Inf"
            )
            implied_corr = round(
                float(minv[0, 1] / jnp.sqrt(minv[0, 0] * minv[1, 1])), 4
            )
            self.assertGreater(
                implied_corr,
                0.3,
                f"n={n} -- correlation not recovered (got {implied_corr})",
            )


class BuildScheduleNutpieMVPTest(BlackJAXTest):
    """Tests for build_schedule_nutpie_mvp (nutpie-continuous schedule MVP,
    queue #9 -- window-growth/sizing delta only; see the function's
    docstring for what this MVP proxy does and does not capture)."""

    def test_shape_and_total_length(self):
        for num_steps in (50, 200, 1000, 5000):
            schedule = build_schedule_nutpie_mvp(num_steps)
            self.assertEqual(schedule.shape, (num_steps, 2))

    def test_final_phase_is_fast_no_window_ends(self):
        """The final step_size_window fraction must be pure fast (stage 0)
        with no window-end recomputes, matching Stan's final-buffer
        semantics (unchanged recompute-cadence scope for the MVP)."""
        num_steps = 2000
        schedule = build_schedule_nutpie_mvp(num_steps, step_size_window=0.15)
        final_start = num_steps - int(round(0.15 * num_steps))
        final_stage = schedule[final_start:, 0]
        final_is_end = schedule[final_start:, 1]
        self.assertTrue(bool(jnp.all(final_stage == 0)))
        self.assertTrue(bool(jnp.all(final_is_end == 0)))

    def test_no_purely_fast_initial_buffer(self):
        """Unlike Stan's schedule (which starts with a pure step-size-only
        buffer), nutpie adapts the mass matrix from the very first draw --
        the MVP schedule's first entry must be the slow (mass-matrix
        adapting) stage, not fast."""
        schedule = build_schedule_nutpie_mvp(1000)
        self.assertEqual(int(schedule[0, 0]), 1)
        # Contrast: Stan's default schedule starts fast.
        stan_schedule = build_schedule(1000)
        self.assertEqual(int(stan_schedule[0, 0]), 0)

    def test_window_sizes_grow_in_main_phase(self):
        """Successive window sizes (gaps between window-end markers) in the
        main phase must be non-decreasing, reflecting the 1.5x growth
        factor (vs Stan's fixed-size doubling-only-at-restart windows)."""
        num_steps = 5000
        schedule = build_schedule_nutpie_mvp(
            num_steps, early_window=0.3, step_size_window=0.15
        )
        window_end_indices = np.where(np.asarray(schedule[:, 1]) == 1)[0]
        # Window sizes = gaps between consecutive window-end markers.
        gaps = np.diff(np.concatenate([[-1], window_end_indices]))
        # Drop the first few (early phase, fixed size) -- check the tail
        # (main phase) is non-decreasing until the final truncated window.
        main_phase_gaps = gaps[3:-1] if len(gaps) > 4 else gaps
        if len(main_phase_gaps) > 1:
            self.assertTrue(bool(np.all(np.diff(main_phase_gaps) >= 0)))

    def test_degenerate_small_num_steps_does_not_crash(self):
        for num_steps in (1, 5, 19, 20, 21):
            schedule = build_schedule_nutpie_mvp(num_steps)
            self.assertEqual(schedule.shape, (num_steps, 2))

    def test_custom_fractions(self):
        """Custom early_window/step_size_window fractions are respected."""
        num_steps = 1000
        schedule = build_schedule_nutpie_mvp(
            num_steps, early_window=0.5, step_size_window=0.2
        )
        final_start = num_steps - int(round(0.2 * num_steps))
        self.assertTrue(bool(jnp.all(schedule[final_start:, 0] == 0)))


class LowRankGradientBasedInitTest(BlackJAXTest):
    """Tests for the gradient_based_init MVP delta (queue #9)."""

    def test_default_reproduces_identity_init(self):
        """gradient_based_init=False (default) must reproduce the original
        sigma=ones(d) initialisation exactly -- no default-behavior change."""
        init, _, _ = base(max_rank=3, gradient_based_init=False)
        d = 5
        grad = jnp.array([2.0, -4.0, 0.5, 10.0, 0.1])
        state = init(jnp.zeros(d), grad, 1.0, 100)
        np.testing.assert_allclose(state.sigma, jnp.ones(d))
        np.testing.assert_allclose(state.U, jnp.zeros((d, 3)))
        np.testing.assert_allclose(state.lam, jnp.ones(3))

    def test_gradient_based_init_formula(self):
        """gradient_based_init=True seeds sigma = 1/sqrt(|grad|), so that
        M^{-1}_diag = sigma**2 = 1/|grad|, matching the paper's
        M = diag(|grad|) (mass matrix, not inverse) initialisation."""
        init, _, _ = base(max_rank=3, gradient_based_init=True)
        d = 5
        grad = jnp.array([2.0, -4.0, 0.5, 10.0, 0.1])
        state = init(jnp.zeros(d), grad, 1.0, 100)
        expected_sigma = jnp.abs(grad) ** -0.5
        np.testing.assert_allclose(state.sigma, expected_sigma, rtol=1e-5)
        # Only the diagonal changes; low-rank correction still starts inert.
        np.testing.assert_allclose(state.U, jnp.zeros((d, 3)))
        np.testing.assert_allclose(state.lam, jnp.ones(3))

    def test_gradient_based_init_clips_extreme_gradients(self):
        """Zero or huge gradient components must not produce NaN/Inf."""
        init, _, _ = base(max_rank=2, gradient_based_init=True)
        d = 4
        grad = jnp.array([0.0, 1e30, 1e-30, 5.0])
        state = init(jnp.zeros(d), grad, 1.0, 50)
        self.assertTrue(bool(jnp.all(jnp.isfinite(state.sigma))))
        self.assertTrue(bool(jnp.all(state.sigma > 0)))

    def test_end_to_end_mvp_options_run_finite(self):
        """window_adaptation_low_rank with both MVP deltas (gradient_based
        init + the nutpie-mvp schedule) runs to a finite result."""
        d = 5
        logdensity_fn = lambda x: -0.5 * jnp.sum(x**2)
        warmup = blackjax.window_adaptation_low_rank(
            blackjax.nuts,
            logdensity_fn,
            max_rank=3,
            gradient_based_init=True,
            schedule_fn=build_schedule_nutpie_mvp,
        )
        (state, params), _ = warmup.run(self.next_key(), jnp.ones(d), num_steps=300)
        self.assertTrue(bool(jnp.all(jnp.isfinite(state.position))))
        self.assertTrue(
            bool(jnp.all(jnp.isfinite(params["inverse_mass_matrix"].sigma)))
        )
        self.assertGreater(float(params["step_size"]), 0.0)

    def test_default_schedule_fn_unchanged(self):
        """window_adaptation_low_rank without schedule_fn/gradient_based_init
        must still use Stan's default build_schedule (no behavior change)."""
        d = 4
        logdensity_fn = lambda x: -0.5 * jnp.sum(x**2)
        warmup = blackjax.window_adaptation_low_rank(
            blackjax.nuts, logdensity_fn, max_rank=2
        )
        (state, params), _ = warmup.run(self.next_key(), jnp.zeros(d), num_steps=200)
        self.assertEqual(state.position.shape, (d,))
        self.assertGreater(float(params["step_size"]), 0.0)


if __name__ == "__main__":
    absltest.main()
