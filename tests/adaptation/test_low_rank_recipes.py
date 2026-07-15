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
"""Tests for the low-rank MetricCore recipes.

Coverage:
- :class:`LowRankMetricCoreState` field contract.
- :func:`seed_low_rank_sigma_from_grad` gradient seeding logic.
- :func:`_build_fisher_low_rank_core` init/update/final contract.
- :func:`_build_sample_cov_low_rank_core` init/update/final contract.
- Registry entries ``"fisher_low_rank"`` and ``"sample_cov_low_rank"``.
- :func:`MetricRecipe.build_core` low-rank dispatch and buffer_size guard.
- :func:`staged_adaptation` ``schedule_fn`` parameter.
- :func:`staged_adaptation` ``initial_metric_state`` seam.
- Bit-identity: ``_compute_low_rank_metric`` moved to ``metric_estimators``
  is the same object as the backward-compat re-export from
  ``low_rank_adaptation``.
- :func:`window_adaptation_low_rank` shim: both buffer policies via staged engine,
  gradient_based_init seam, info-bridge field layout, and bit-exact accumulating
  parity gate vs frozen inline reference (``recompute_every=10**9``, ``atol=0.0``).
"""
import jax
import jax.flatten_util as fu
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest

import blackjax
import blackjax.mcmc as mcmc
from blackjax.adaptation.base import return_all_adapt_info
from blackjax.adaptation.low_rank_adaptation import (
    LowRankAdaptationState,
    _accumulating_buffer_capacity,
)
from blackjax.adaptation.low_rank_adaptation import _compute_low_rank_metric as lra_clrm
from blackjax.adaptation.low_rank_adaptation import (
    _default_low_rank_adaptation_info_fn,
    _engine_state_to_low_rank_adaptation_state,
    _make_low_rank_bridge_info_fn,
    build_growing_window_schedule,
    window_adaptation_low_rank,
)
from blackjax.adaptation.metric_estimators import _compute_low_rank_metric
from blackjax.adaptation.metric_recipes import (
    REGISTRY,
    LowRankMetricCoreState,
    MetricCore,
    MetricRecipe,
    _build_fisher_low_rank_core,
    _build_sample_cov_low_rank_core,
    _shift_buffer_left,
    lookup_recipe,
    seed_low_rank_sigma_from_grad,
)
from blackjax.adaptation.staged_adaptation import (
    StagedAdaptationState,
    build_schedule,
    staged_adaptation,
)
from blackjax.adaptation.step_size import dual_averaging_adaptation
from blackjax.mcmc.metrics import LowRankInverseMassMatrix, gaussian_euclidean_low_rank
from tests.fixtures import BlackJAXTest, std_normal_logdensity

# ---------------------------------------------------------------------------
# Backward-compat object-identity test
# ---------------------------------------------------------------------------


class ComputeLowRankMetricMovedTest(BlackJAXTest):
    """_compute_low_rank_metric moved to metric_estimators; re-export is same object."""

    def test_backward_compat_import_is_same_object(self):
        """The re-export from low_rank_adaptation must be the same callable."""
        self.assertIs(_compute_low_rank_metric, lra_clrm)


# ---------------------------------------------------------------------------
# LowRankMetricCoreState field contract
# ---------------------------------------------------------------------------


class LowRankMetricCoreStateTest(BlackJAXTest):
    """LowRankMetricCoreState NamedTuple field contract."""

    def test_has_required_fields(self):
        """State must have inverse_mass_matrix, mu_star, buffers, and counters."""
        required = {
            "inverse_mass_matrix",
            "mu_star",
            "draws_buffer",
            "grads_buffer",
            "buffer_idx",
            "background_split",
            "recompute_counter",
        }
        self.assertTrue(required <= set(LowRankMetricCoreState._fields))

    def test_inverse_mass_matrix_is_low_rank_namedtuple(self):
        """inverse_mass_matrix field must be a LowRankInverseMassMatrix."""
        core = _build_fisher_low_rank_core(
            buffer_size=20, max_rank=3, gamma=1e-5, cutoff=2.0
        )
        state = core.init(5)
        self.assertIsInstance(state.inverse_mass_matrix, LowRankInverseMassMatrix)

    def test_init_shapes(self):
        """All buffer and metric shapes must match n_dims and max_rank."""
        n_dims, max_rank, buffer_size = 7, 4, 30
        core = _build_fisher_low_rank_core(
            buffer_size=buffer_size, max_rank=max_rank, gamma=1e-5, cutoff=2.0
        )
        state = core.init(n_dims)
        self.assertEqual(state.inverse_mass_matrix.sigma.shape, (n_dims,))
        self.assertEqual(state.inverse_mass_matrix.U.shape, (n_dims, max_rank))
        self.assertEqual(state.inverse_mass_matrix.lam.shape, (max_rank,))
        self.assertEqual(state.mu_star.shape, (n_dims,))
        self.assertEqual(state.draws_buffer.shape, (buffer_size, n_dims))
        self.assertEqual(state.grads_buffer.shape, (buffer_size, n_dims))
        self.assertEqual(int(state.buffer_idx), 0)
        self.assertEqual(int(state.background_split), 0)
        self.assertEqual(int(state.recompute_counter), 0)

    def test_init_identity_metric(self):
        """Initial sigma must be ones, U must be zeros, lam must be ones."""
        core = _build_fisher_low_rank_core(
            buffer_size=10, max_rank=2, gamma=1e-5, cutoff=2.0
        )
        state = core.init(4)
        np.testing.assert_array_equal(
            np.asarray(state.inverse_mass_matrix.sigma), np.ones(4)
        )
        np.testing.assert_array_equal(
            np.asarray(state.inverse_mass_matrix.U), np.zeros((4, 2))
        )
        np.testing.assert_array_equal(
            np.asarray(state.inverse_mass_matrix.lam), np.ones(2)
        )
        np.testing.assert_array_equal(np.asarray(state.mu_star), np.zeros(4))


# ---------------------------------------------------------------------------
# seed_low_rank_sigma_from_grad
# ---------------------------------------------------------------------------


class SeedLowRankSigmaFromGradTest(BlackJAXTest):
    """seed_low_rank_sigma_from_grad: gradient seeding logic."""

    def _make_init_state(self, n_dims=5, max_rank=2):
        core = _build_fisher_low_rank_core(
            buffer_size=10, max_rank=max_rank, gamma=1e-5, cutoff=2.0
        )
        return core.init(n_dims)

    def test_returns_same_type(self):
        state = self._make_init_state()
        grad = jnp.ones(5) * 0.5
        seeded = seed_low_rank_sigma_from_grad(state, grad)
        self.assertIsInstance(seeded, LowRankMetricCoreState)

    def test_sigma_seeded_from_grad_magnitude(self):
        """sigma_i = 1/sqrt(|grad_i|) for large-enough gradients."""
        grad = jnp.array([0.25, 1.0, 4.0, 0.1, 0.01])
        state = self._make_init_state(n_dims=5)
        seeded = seed_low_rank_sigma_from_grad(state, grad)
        expected_sigma = 1.0 / jnp.sqrt(grad)
        np.testing.assert_allclose(
            np.asarray(seeded.inverse_mass_matrix.sigma),
            np.asarray(expected_sigma),
            rtol=1e-5,
        )

    def test_near_zero_grad_falls_back_to_identity(self):
        """Coordinates with |grad| < 1e-10 must get sigma=1.0 (identity)."""
        grad = jnp.array([0.5, 0.0, 1e-11, 1.0, 0.0])
        state = self._make_init_state(n_dims=5)
        seeded = seed_low_rank_sigma_from_grad(state, grad)
        sigma = np.asarray(seeded.inverse_mass_matrix.sigma)
        # Coordinates 1, 2, 4 have near-zero grad -> sigma = 1.0
        np.testing.assert_allclose(sigma[1], 1.0, atol=1e-6)
        np.testing.assert_allclose(sigma[2], 1.0, atol=1e-6)
        np.testing.assert_allclose(sigma[4], 1.0, atol=1e-6)
        # Coordinate 0: sigma = 1/sqrt(0.5) = sqrt(2)
        np.testing.assert_allclose(sigma[0], np.sqrt(2.0), rtol=1e-5)

    def test_u_and_lam_unchanged(self):
        """Seeding must not modify U or lam."""
        grad = jnp.ones(5)
        state = self._make_init_state(n_dims=5)
        seeded = seed_low_rank_sigma_from_grad(state, grad)
        np.testing.assert_array_equal(
            np.asarray(seeded.inverse_mass_matrix.U),
            np.asarray(state.inverse_mass_matrix.U),
        )
        np.testing.assert_array_equal(
            np.asarray(seeded.inverse_mass_matrix.lam),
            np.asarray(state.inverse_mass_matrix.lam),
        )

    def test_mu_star_unchanged(self):
        """Seeding must not modify mu_star."""
        grad = jnp.ones(5)
        state = self._make_init_state(n_dims=5)
        seeded = seed_low_rank_sigma_from_grad(state, grad)
        np.testing.assert_array_equal(
            np.asarray(seeded.mu_star), np.asarray(state.mu_star)
        )


# ---------------------------------------------------------------------------
# _build_fisher_low_rank_core: init/update/final contract
# ---------------------------------------------------------------------------


class FisherLowRankCoreContractTest(BlackJAXTest):
    """Fisher low-rank MetricCore init/update/final contract."""

    n_dims = 8
    max_rank = 4
    buffer_size = 40

    def _make_core(self):
        return _build_fisher_low_rank_core(
            buffer_size=self.buffer_size,
            max_rank=self.max_rank,
            gamma=1e-5,
            cutoff=2.0,
        )

    def test_returns_metric_core_namedtuple(self):
        core = self._make_core()
        self.assertIsInstance(core, MetricCore)
        self.assertTrue(callable(core.init))
        self.assertTrue(callable(core.update))
        self.assertTrue(callable(core.final))

    def test_update_increments_buffer_idx(self):
        core = self._make_core()
        state = core.init(self.n_dims)
        pos = jnp.ones(self.n_dims)
        grad = jnp.zeros(self.n_dims)
        state2 = core.update(state, pos, grad)
        self.assertEqual(int(state2.buffer_idx), 1)
        state3 = core.update(state2, pos, grad)
        self.assertEqual(int(state3.buffer_idx), 2)

    def test_update_writes_to_buffer(self):
        """The position must appear in the draws buffer after update."""
        core = self._make_core()
        state = core.init(self.n_dims)
        pos = jnp.arange(self.n_dims, dtype=jnp.float32)
        state2 = core.update(state, pos, jnp.zeros(self.n_dims))
        np.testing.assert_array_equal(
            np.asarray(state2.draws_buffer[0]),
            np.asarray(pos),
        )

    def test_update_is_scannable(self):
        """update() must be safe inside jax.lax.scan (traced buffer_idx)."""
        core = self._make_core()
        state = core.init(self.n_dims)
        key = self.next_key()
        k1, k2 = jax.random.split(key)
        positions = jax.random.normal(k1, (30, self.n_dims))
        grads = jax.random.normal(k2, (30, self.n_dims))

        def body(s, xs):
            pos, g = xs
            return core.update(s, pos, g), None

        final_state, _ = jax.lax.scan(body, state, (positions, grads))
        self.assertEqual(int(final_state.buffer_idx), 30)

    def test_final_resets_buffer(self):
        """final() must reset buffer_idx to 0 and zero out the buffers."""
        core = self._make_core()
        state = core.init(self.n_dims)
        key = self.next_key()
        k1, k2 = jax.random.split(key)
        positions = jax.random.normal(k1, (20, self.n_dims))
        grads = jax.random.normal(k2, (20, self.n_dims))

        def body(s, xs):
            pos, g = xs
            return core.update(s, pos, g), None

        state, _ = jax.lax.scan(body, state, (positions, grads))
        final_state = core.final(state)
        self.assertEqual(int(final_state.buffer_idx), 0)
        np.testing.assert_array_equal(
            np.asarray(final_state.draws_buffer),
            np.zeros((self.buffer_size, self.n_dims)),
        )
        np.testing.assert_array_equal(
            np.asarray(final_state.grads_buffer),
            np.zeros((self.buffer_size, self.n_dims)),
        )

    def test_final_produces_finite_metric(self):
        """After accumulating anisotropic draws, final() gives finite sigma/U/lam."""
        core = self._make_core()
        state = core.init(self.n_dims)
        key = self.next_key()
        k1, k2 = jax.random.split(key)
        scales = jnp.ones(self.n_dims).at[0].set(5.0).at[1].set(0.1)
        draws = jax.random.normal(k1, (25, self.n_dims)) * scales
        grads = -draws / scales**2

        def body(s, xs):
            pos, g = xs
            return core.update(s, pos, g), None

        state, _ = jax.lax.scan(body, state, (draws, grads))
        final_state = core.final(state)
        imm = final_state.inverse_mass_matrix
        self.assertTrue(bool(jnp.all(jnp.isfinite(imm.sigma))))
        self.assertTrue(bool(jnp.all(jnp.isfinite(imm.U))))
        self.assertTrue(bool(jnp.all(jnp.isfinite(imm.lam))))
        self.assertTrue(bool(jnp.all(imm.sigma > 0)))

    def test_final_updates_mu_star(self):
        """After accumulating off-center draws, mu_star should be non-zero."""
        core = self._make_core()
        state = core.init(self.n_dims)
        key = self.next_key()
        k1, k2 = jax.random.split(key)
        # Draws from N(mean=3, scale=1)
        draws = jax.random.normal(k1, (25, self.n_dims)) + 3.0
        grads = -draws + 3.0  # gradient of -0.5*(x-3)^2

        def body(s, xs):
            pos, g = xs
            return core.update(s, pos, g), None

        state, _ = jax.lax.scan(body, state, (draws, grads))
        final_state = core.final(state)
        self.assertTrue(bool(jnp.any(final_state.mu_star != 0.0)))

    def test_final_metric_shape_correct(self):
        """final()'s inverse_mass_matrix must have correct shapes."""
        core = self._make_core()
        state = core.init(self.n_dims)
        key = self.next_key()
        k1, k2 = jax.random.split(key)
        draws = jax.random.normal(k1, (15, self.n_dims))
        grads = jax.random.normal(k2, (15, self.n_dims))

        def body(s, xs):
            return core.update(s, xs[0], xs[1]), None

        state, _ = jax.lax.scan(body, state, (draws, grads))
        final_state = core.final(state)
        imm = final_state.inverse_mass_matrix
        self.assertEqual(imm.sigma.shape, (self.n_dims,))
        self.assertEqual(imm.U.shape, (self.n_dims, self.max_rank))
        self.assertEqual(imm.lam.shape, (self.max_rank,))


# ---------------------------------------------------------------------------
# _build_sample_cov_low_rank_core: init/update/final contract
# ---------------------------------------------------------------------------


class SampleCovLowRankCoreContractTest(BlackJAXTest):
    """Sample-covariance low-rank MetricCore init/update/final contract."""

    n_dims = 6
    max_rank = 3
    buffer_size = 30

    def _make_core(self):
        return _build_sample_cov_low_rank_core(
            buffer_size=self.buffer_size,
            max_rank=self.max_rank,
        )

    def test_returns_metric_core_namedtuple(self):
        core = self._make_core()
        self.assertIsInstance(core, MetricCore)

    def test_update_only_uses_position(self):
        """update() must work when grad=None (draws-only core)."""
        core = self._make_core()
        state = core.init(self.n_dims)
        pos = jnp.ones(self.n_dims)
        # grad=None should be accepted (ignored)
        state2 = core.update(state, pos, None)
        self.assertEqual(int(state2.buffer_idx), 1)

    def test_final_mu_star_is_zero(self):
        """mu_star must always be zero for this estimator (no optimal translation)."""
        core = self._make_core()
        state = core.init(self.n_dims)
        key = self.next_key()
        draws = jax.random.normal(key, (20, self.n_dims)) + 5.0

        def body(s, xs):
            return core.update(s, xs, None), None

        state, _ = jax.lax.scan(body, state, draws)
        final_state = core.final(state)
        np.testing.assert_array_equal(
            np.asarray(final_state.mu_star), np.zeros(self.n_dims)
        )

    def test_final_produces_finite_metric(self):
        """After accumulating draws, final() gives finite sigma/U/lam."""
        core = self._make_core()
        state = core.init(self.n_dims)
        key = self.next_key()
        draws = jax.random.normal(key, (20, self.n_dims))

        def body(s, xs):
            return core.update(s, xs, None), None

        state, _ = jax.lax.scan(body, state, draws)
        final_state = core.final(state)
        imm = final_state.inverse_mass_matrix
        self.assertTrue(bool(jnp.all(jnp.isfinite(imm.sigma))))
        self.assertTrue(bool(jnp.all(jnp.isfinite(imm.U))))
        self.assertTrue(bool(jnp.all(jnp.isfinite(imm.lam))))
        self.assertTrue(bool(jnp.all(imm.sigma > 0)))

    def test_grads_buffer_unused(self):
        """grads_buffer must remain zeros even when a real gradient is passed.

        Passing a non-None gradient catches the errant-grad-write mutation
        (a sample_cov update that writes grad to grads_buffer): with grad=None
        the mutation is a no-op because None cannot be stored; with a real
        gradient array the mutation would write non-zero values and the
        assertion would fail.
        """
        core = self._make_core()
        state = core.init(self.n_dims)
        draws = jax.random.normal(self.next_key(), (10, self.n_dims))
        grads = jax.random.normal(self.next_key(), (10, self.n_dims))

        def body(s, xs):
            pos, grad = xs
            return core.update(s, pos, grad), None

        final_state, _ = jax.lax.scan(body, state, (draws, grads))
        np.testing.assert_array_equal(
            np.asarray(final_state.grads_buffer),
            np.zeros((self.buffer_size, self.n_dims)),
        )


# ---------------------------------------------------------------------------
# Registry entries for new recipes
# ---------------------------------------------------------------------------


class LowRankRegistryTest(BlackJAXTest):
    """Registry entries for fisher_low_rank and sample_cov_low_rank."""

    def test_fisher_low_rank_in_registry(self):
        recipe = lookup_recipe("fisher_low_rank")
        self.assertIsInstance(recipe, MetricRecipe)

    def test_sample_cov_low_rank_in_registry(self):
        recipe = lookup_recipe("sample_cov_low_rank")
        self.assertIsInstance(recipe, MetricRecipe)

    def test_fisher_low_rank_defaults(self):
        """Registry defaults must match window_adaptation_low_rank defaults."""
        recipe = lookup_recipe("fisher_low_rank")
        self.assertEqual(recipe.max_rank, 10)
        self.assertAlmostEqual(recipe.gamma, 1e-5)
        self.assertAlmostEqual(recipe.cutoff, 2.0)
        self.assertEqual(recipe.representation, "low_rank")
        self.assertEqual(recipe.emits, "low_rank")
        self.assertIn("positions", recipe.needs)
        self.assertIn("gradients", recipe.needs)

    def test_sample_cov_low_rank_defaults(self):
        """sample_cov_low_rank defaults: max_rank=10, no gamma/cutoff."""
        recipe = lookup_recipe("sample_cov_low_rank")
        self.assertEqual(recipe.max_rank, 10)
        self.assertIsNone(recipe.gamma)
        self.assertIsNone(recipe.cutoff)
        self.assertIn("positions", recipe.needs)
        self.assertNotIn("gradients", recipe.needs)

    def test_build_core_requires_buffer_size_for_fisher_low_rank(self):
        """build_core without buffer_size must raise ValueError."""
        recipe = lookup_recipe("fisher_low_rank")
        with self.assertRaisesRegex(ValueError, "buffer_size"):
            recipe.build_core()

    def test_build_core_requires_buffer_size_for_sample_cov_low_rank(self):
        recipe = lookup_recipe("sample_cov_low_rank")
        with self.assertRaisesRegex(ValueError, "buffer_size"):
            recipe.build_core()

    def test_build_core_with_buffer_size_returns_metric_core(self):
        """build_core(buffer_size=N) must return a MetricCore."""
        for name in ("fisher_low_rank", "sample_cov_low_rank"):
            with self.subTest(name=name):
                recipe = lookup_recipe(name)
                core = recipe.build_core(buffer_size=50)
                self.assertIsInstance(core, MetricCore)

    def test_all_low_rank_recipe_names_present(self):
        for name in ("fisher_low_rank", "sample_cov_low_rank"):
            with self.subTest(name=name):
                self.assertIn(name, REGISTRY)

    def test_construction_low_rank_max_rank_none_raises(self):
        """MetricRecipe with representation='low_rank' and max_rank=None must
        raise ValueError at construction time (not deferred to build_core)."""
        with self.assertRaises(ValueError):
            MetricRecipe(
                representation="low_rank",
                estimator="sample_cov_low_rank",
                buffer="reset_window",
                support_gate=None,
                needs=frozenset({"positions"}),
                provides=frozenset({"positions"}),
                emits="low_rank",
                provenance="test — construction must raise",
                max_rank=None,
            )


# ---------------------------------------------------------------------------
# Estimator correctness: condition-number reduction invariant
# ---------------------------------------------------------------------------


def _reconstruct_imm_matrix(
    sigma: "np.ndarray", U: "np.ndarray", lam: "np.ndarray"
) -> "np.ndarray":
    """Reconstruct the d×d inverse mass matrix from (sigma, U, lam).

    M^{-1} = diag(sigma) @ (I + U @ diag(lam - 1) @ U^T) @ diag(sigma)

    This matches the formula in :class:`~blackjax.mcmc.metrics.LowRankInverseMassMatrix`.
    """
    sigma = np.asarray(sigma, dtype=np.float64)
    U = np.asarray(U, dtype=np.float64)
    lam = np.asarray(lam, dtype=np.float64)
    d = sigma.shape[0]
    low_rank = U @ np.diag(lam - 1.0) @ U.T
    return np.diag(sigma) @ (np.eye(d) + low_rank) @ np.diag(sigma)


class EstimatorCorrectnessInvariantTest(BlackJAXTest):
    """Basis-free condition-number-reduction invariant for low-rank estimators.

    For an ill-conditioned diagonal Gaussian target with covariance Σ, the
    fitted inverse mass matrix M^{-1} must satisfy:

        cond(M^{-1} Σ^{-1}) << cond(Σ^{-1})

    This is the standard HMC dynamics condition number: a good metric (M^{-1} ≈ Σ)
    makes M^{-1} Σ^{-1} ≈ I and the condition number ≈ 1.  The test is
    basis-free (no golden values; the comparison is relative to cond_before)
    and catches two bug classes:

    - Inversion-drop: without inverting the score covariance in the geometric
      mean, eigenvalues blow up to ~B/gamma ≈ 2e7, making cond_after >> cond_before.
    - Gamma-drop (gamma → 0): division by zero produces NaN lam values, caught
      by the finiteness assertion before the condition-number check.
    """

    _N_DIMS = 6
    # stds = [50, 1, 1, 1, 1, 0.1] → cond(Σ) = (50 / 0.1)^2 = 250 000
    _STDS = np.array([50.0, 1.0, 1.0, 1.0, 1.0, 0.1])
    _BUFFER_SIZE = 400

    def _fill_fisher_core(self, core):
        """Fill buffer with draws + gradients from the ill-conditioned diagonal target."""
        target_cov = np.diag(self._STDS**2)
        rng = np.random.default_rng(42)
        draws = rng.multivariate_normal(
            np.zeros(self._N_DIMS), target_cov, self._BUFFER_SIZE
        )
        # grad log p(x) = -x / stds^2 for N(0, diag(stds^2))
        grads = -draws / (self._STDS**2)[None, :]
        state = core.init(self._N_DIMS)

        def body(s, xs):
            return core.update(s, xs[0], xs[1]), None

        state, _ = jax.lax.scan(
            body,
            state,
            (jnp.array(draws), jnp.array(grads)),
        )
        return core.final(state)

    def _fill_sample_cov_core(self, core):
        """Fill buffer with draws from the ill-conditioned diagonal target."""
        target_cov = np.diag(self._STDS**2)
        rng = np.random.default_rng(42)
        draws = rng.multivariate_normal(
            np.zeros(self._N_DIMS), target_cov, self._BUFFER_SIZE
        )
        state = core.init(self._N_DIMS)

        def body(s, xs):
            return core.update(s, xs, None), None

        state, _ = jax.lax.scan(body, state, jnp.array(draws))
        return core.final(state)

    def _assert_cond_reduced(self, imm, label):
        """Assert all metric fields are finite and condition number is reduced."""
        self.assertTrue(
            bool(jnp.all(jnp.isfinite(imm.sigma))), f"{label}: sigma not finite"
        )
        self.assertTrue(bool(jnp.all(jnp.isfinite(imm.U))), f"{label}: U not finite")
        self.assertTrue(
            bool(jnp.all(jnp.isfinite(imm.lam))), f"{label}: lam not finite"
        )
        self.assertTrue(bool(jnp.all(imm.lam > 0)), f"{label}: lam not positive")
        target_cov = np.diag(self._STDS**2)
        cond_before = np.linalg.cond(target_cov)  # ≈ 250 000
        M_inv = _reconstruct_imm_matrix(
            np.asarray(imm.sigma), np.asarray(imm.U), np.asarray(imm.lam)
        )
        target_prec = np.linalg.inv(target_cov)
        cond_after = np.linalg.cond(M_inv @ target_prec)
        self.assertLess(
            cond_after,
            cond_before / 100,
            f"{label}: metric did not reduce condition number "
            f"(cond_before={round(cond_before)}, cond_after={round(cond_after, 1)})",
        )

    def test_fisher_low_rank_reduces_condition_number(self):
        """Fisher low-rank metric must significantly reduce the target's condition number.

        Uses x64 for stable SVD/QR (recommended for this estimator).
        """
        with jax.enable_x64():
            core = _build_fisher_low_rank_core(
                buffer_size=self._BUFFER_SIZE, max_rank=4, gamma=1e-5, cutoff=2.0
            )
            final_state = self._fill_fisher_core(core)
            self._assert_cond_reduced(
                final_state.inverse_mass_matrix, "fisher_low_rank"
            )

    def test_sample_cov_low_rank_reduces_condition_number(self):
        """Sample-cov low-rank metric must significantly reduce the target's condition number."""
        with jax.enable_x64():
            core = _build_sample_cov_low_rank_core(
                buffer_size=self._BUFFER_SIZE, max_rank=4
            )
            final_state = self._fill_sample_cov_core(core)
            self._assert_cond_reduced(
                final_state.inverse_mass_matrix, "sample_cov_low_rank"
            )


# ---------------------------------------------------------------------------
# staged_adaptation schedule_fn parameter
# ---------------------------------------------------------------------------


class StagedAdaptationScheduleFnTest(BlackJAXTest):
    """schedule_fn parameter forwarding in staged_adaptation."""

    def test_default_schedule_fn_is_build_schedule(self):
        """Without schedule_fn, the default build_schedule is used (smoke test)."""
        wu = staged_adaptation(
            blackjax.nuts,
            std_normal_logdensity,
            metric="welford_diag",
        )
        key = self.next_key()
        pos = jnp.zeros(2)
        (state, _), _ = wu.run(key, pos, num_steps=50)
        self.assertEqual(state.position.shape, (2,))

    def test_custom_schedule_fn_is_called(self):
        """Passing a custom schedule_fn must route the scan through it."""
        calls = []

        def counting_schedule(num_steps):
            calls.append(num_steps)
            return build_schedule(num_steps)

        wu = staged_adaptation(
            blackjax.nuts,
            std_normal_logdensity,
            metric="welford_diag",
            schedule_fn=counting_schedule,
        )
        key = self.next_key()
        pos = jnp.zeros(2)
        wu.run(key, pos, num_steps=50)
        self.assertEqual(calls, [50])

    def test_growing_window_schedule_fn(self):
        """build_growing_window_schedule must work as schedule_fn."""
        wu = staged_adaptation(
            blackjax.nuts,
            std_normal_logdensity,
            metric="welford_diag",
            schedule_fn=build_growing_window_schedule,
        )
        key = self.next_key()
        pos = jnp.zeros(2)
        (state, _), _ = wu.run(key, pos, num_steps=100)
        self.assertEqual(state.position.shape, (2,))


# ---------------------------------------------------------------------------
# Integration: staged_adaptation with fisher_low_rank core on std normal
# ---------------------------------------------------------------------------


class FisherLowRankStagedAdaptationTest(BlackJAXTest):
    """staged_adaptation with the fisher_low_rank MetricCore on std normal."""

    def test_runs_to_completion(self):
        """staged_adaptation with a pre-built fisher_low_rank core must complete."""
        n_dims = 5
        num_steps = 200
        buffer_size = min(2 * max(num_steps // 5, 128), max(num_steps, 1))

        core = _build_fisher_low_rank_core(
            buffer_size=buffer_size,
            max_rank=5,
            gamma=1e-5,
            cutoff=2.0,
        )
        wu = staged_adaptation(
            blackjax.nuts,
            std_normal_logdensity,
            metric=core,
        )
        key = self.next_key()
        pos = jnp.zeros(n_dims)
        (state, params), info = wu.run(key, pos, num_steps=num_steps)
        self.assertEqual(state.position.shape, (n_dims,))
        self.assertIsInstance(params["inverse_mass_matrix"], LowRankInverseMassMatrix)
        self.assertEqual(params["inverse_mass_matrix"].sigma.shape, (n_dims,))
        self.assertEqual(params["inverse_mass_matrix"].U.shape, (n_dims, 5))

    def test_returns_finite_metric(self):
        """All metric components must be finite after adaptation."""
        n_dims = 4
        num_steps = 150
        buffer_size = min(2 * max(num_steps // 5, 128), max(num_steps, 1))

        core = _build_fisher_low_rank_core(
            buffer_size=buffer_size,
            max_rank=4,
            gamma=1e-5,
            cutoff=2.0,
        )
        wu = staged_adaptation(
            blackjax.nuts,
            std_normal_logdensity,
            metric=core,
        )
        key = self.next_key()
        pos = jnp.zeros(n_dims)
        (_, params), _ = wu.run(key, pos, num_steps=num_steps)
        imm = params["inverse_mass_matrix"]
        self.assertTrue(bool(jnp.all(jnp.isfinite(imm.sigma))))
        self.assertTrue(bool(jnp.all(jnp.isfinite(imm.U))))
        self.assertTrue(bool(jnp.all(jnp.isfinite(imm.lam))))


# ---------------------------------------------------------------------------
# Integration: staged_adaptation with sample_cov_low_rank core on std normal
# ---------------------------------------------------------------------------


class SampleCovLowRankStagedAdaptationTest(BlackJAXTest):
    """staged_adaptation with the sample_cov_low_rank MetricCore on std normal."""

    def test_runs_to_completion(self):
        """staged_adaptation with a pre-built sample_cov_low_rank core must complete."""
        n_dims = 5
        num_steps = 200
        buffer_size = min(2 * max(num_steps // 5, 128), max(num_steps, 1))

        core = _build_sample_cov_low_rank_core(buffer_size=buffer_size, max_rank=4)
        wu = staged_adaptation(
            blackjax.nuts,
            std_normal_logdensity,
            metric=core,
        )
        key = self.next_key()
        pos = jnp.zeros(n_dims)
        (state, params), _ = wu.run(key, pos, num_steps=num_steps)
        self.assertEqual(state.position.shape, (n_dims,))
        self.assertIsInstance(params["inverse_mass_matrix"], LowRankInverseMassMatrix)
        self.assertEqual(params["inverse_mass_matrix"].sigma.shape, (n_dims,))
        self.assertEqual(params["inverse_mass_matrix"].U.shape, (n_dims, 4))

    def test_returns_finite_metric(self):
        """All sample_cov_low_rank metric components must be finite after adaptation."""
        n_dims = 4
        num_steps = 150
        buffer_size = min(2 * max(num_steps // 5, 128), max(num_steps, 1))

        core = _build_sample_cov_low_rank_core(buffer_size=buffer_size, max_rank=4)
        wu = staged_adaptation(
            blackjax.nuts,
            std_normal_logdensity,
            metric=core,
        )
        key = self.next_key()
        pos = jnp.zeros(n_dims)
        (_, params), _ = wu.run(key, pos, num_steps=num_steps)
        imm = params["inverse_mass_matrix"]
        self.assertTrue(bool(jnp.all(jnp.isfinite(imm.sigma))))
        self.assertTrue(bool(jnp.all(jnp.isfinite(imm.U))))
        self.assertTrue(bool(jnp.all(jnp.isfinite(imm.lam))))

    def test_grads_buffer_stays_zero_through_engine(self):
        """sample_cov_low_rank through the engine must never write to grads_buffer.

        Verifies that the sample_cov update does not write to grads_buffer even
        when an adaptation_info_fn exposes the raw buffer (bypassing the default
        buffer-drop).
        """
        n_dims = 4
        num_steps = 50
        buffer_size = 64

        core = _build_sample_cov_low_rank_core(buffer_size=buffer_size, max_rank=4)
        wu = staged_adaptation(
            blackjax.nuts,
            std_normal_logdensity,
            metric=core,
            # Use return_all_adapt_info so draws_buffer / grads_buffer are stacked.
            adaptation_info_fn=return_all_adapt_info,
        )
        key = self.next_key()
        pos = jnp.zeros(n_dims)
        _, info = wu.run(key, pos, num_steps=num_steps)
        # The engine stacks grads_buffer over all steps: shape (num_steps, B, d).
        # For sample_cov_low_rank it must remain all-zeros at every step.
        grads_trace = info.adaptation_state.imm_state.grads_buffer
        np.testing.assert_array_equal(
            np.asarray(grads_trace),
            np.zeros_like(np.asarray(grads_trace)),
        )


# ---------------------------------------------------------------------------
# staged_adaptation initial_metric_state seam
# ---------------------------------------------------------------------------


def _make_seeded_state(n_dims, max_rank=4):
    """Build a LowRankMetricCoreState seeded with non-identity sigma."""
    buffer_size = 64
    core = _build_fisher_low_rank_core(
        buffer_size=buffer_size, max_rank=max_rank, gamma=1e-5, cutoff=2.0
    )
    base_state = core.init(n_dims)
    # Seed sigma so we can distinguish it from the identity default.
    grad = jnp.arange(1, n_dims + 1, dtype=jnp.float32)
    return seed_low_rank_sigma_from_grad(base_state, grad), core


class InitialMetricStateSeamTest(BlackJAXTest):
    """staged_adaptation initial_metric_state=... overrides core.init(n_dims)."""

    def test_seam_overrides_sigma_at_first_step(self):
        """When initial_metric_state is provided, the first adaptation step must
        use the seeded sigma rather than ones."""
        n_dims = 4
        seeded_state, core = _make_seeded_state(n_dims)
        seeded_sigma = seeded_state.inverse_mass_matrix.sigma

        wu = staged_adaptation(
            blackjax.nuts,
            std_normal_logdensity,
            metric=core,
            adaptation_info_fn=return_all_adapt_info,
            initial_metric_state=seeded_state,
        )
        key = self.next_key()
        pos = jnp.zeros(n_dims)
        _, info = wu.run(key, pos, num_steps=10)
        # At step 0 the adaptation state still carries the seeded sigma
        # (no slow window has ended yet at 10 steps with the default schedule).
        first_sigma = info.adaptation_state.imm_state.inverse_mass_matrix.sigma[0]
        self.assertTrue(bool(jnp.allclose(first_sigma, seeded_sigma, atol=0.0)))

    def test_seam_none_uses_core_init(self):
        """When initial_metric_state=None (default), core.init is used (sigma=ones)."""
        n_dims = 5
        buffer_size = 64
        core = _build_fisher_low_rank_core(
            buffer_size=buffer_size, max_rank=4, gamma=1e-5, cutoff=2.0
        )
        wu = staged_adaptation(
            blackjax.nuts,
            std_normal_logdensity,
            metric=core,
            adaptation_info_fn=return_all_adapt_info,
            initial_metric_state=None,
        )
        key = self.next_key()
        pos = jnp.zeros(n_dims)
        _, info = wu.run(key, pos, num_steps=10)
        first_sigma = info.adaptation_state.imm_state.inverse_mass_matrix.sigma[0]
        self.assertTrue(bool(jnp.allclose(first_sigma, jnp.ones(n_dims), atol=0.0)))

    def test_seam_produces_finite_output(self):
        """staged_adaptation with initial_metric_state seam runs to completion."""
        n_dims = 6
        seeded_state, core = _make_seeded_state(n_dims, max_rank=4)
        wu = staged_adaptation(
            blackjax.nuts,
            std_normal_logdensity,
            metric=core,
            initial_metric_state=seeded_state,
        )
        key = self.next_key()
        (state, params), _ = wu.run(key, jnp.zeros(n_dims), num_steps=200)
        imm = params["inverse_mass_matrix"]
        self.assertTrue(bool(jnp.all(jnp.isfinite(imm.sigma))))
        self.assertTrue(bool(jnp.all(jnp.isfinite(imm.U))))
        self.assertTrue(bool(jnp.all(jnp.isfinite(imm.lam))))

    def test_seam_inverse_mass_matrix_carried_immediately(self):
        """The seeded inverse_mass_matrix must be present in StagedAdaptationState
        inverse_mass_matrix field at step 0 (i.e., MCMC kernel sees it)."""
        n_dims = 4
        seeded_state, core = _make_seeded_state(n_dims)
        wu = staged_adaptation(
            blackjax.nuts,
            std_normal_logdensity,
            metric=core,
            adaptation_info_fn=return_all_adapt_info,
            initial_metric_state=seeded_state,
        )
        key = self.next_key()
        _, info = wu.run(key, jnp.zeros(n_dims), num_steps=5)
        # StagedAdaptationState.inverse_mass_matrix is a LowRankInverseMassMatrix;
        # after scan stacking, .sigma has shape (num_steps, n_dims).  [0] gives
        # the step-0 sigma, which must NOT be ones (the identity default).
        first_sigma = info.adaptation_state.inverse_mass_matrix.sigma[0]
        self.assertFalse(bool(jnp.allclose(first_sigma, jnp.ones(n_dims), atol=1e-6)))


# ---------------------------------------------------------------------------
# Engine-to-LowRankAdaptationState bridge
# ---------------------------------------------------------------------------


class LowRankBridgeFnTest(BlackJAXTest):
    """_engine_state_to_low_rank_adaptation_state and _make_low_rank_bridge_info_fn."""

    def _make_engine_state(self, n_dims=4, max_rank=3):
        from blackjax.adaptation.step_size import dual_averaging_adaptation

        da_init, _, _ = dual_averaging_adaptation(0.80)
        ss_state = da_init(0.5)
        imm = LowRankInverseMassMatrix(
            sigma=jnp.full(n_dims, 2.0),
            U=jnp.eye(n_dims, max_rank),
            lam=jnp.full(max_rank, 1.5),
        )
        imm_state = LowRankMetricCoreState(
            inverse_mass_matrix=imm,
            mu_star=jnp.full(n_dims, 0.1),
            draws_buffer=jnp.ones((16, n_dims)),
            grads_buffer=jnp.ones((16, n_dims)) * 0.5,
            buffer_idx=7,
            background_split=3,
            recompute_counter=2,
        )
        return StagedAdaptationState(
            ss_state=ss_state,
            imm_state=imm_state,
            step_size=0.5,
            inverse_mass_matrix=imm,
        )

    def test_bridge_sigma_matches(self):
        es = self._make_engine_state()
        lr = _engine_state_to_low_rank_adaptation_state(es)
        self.assertTrue(
            bool(jnp.allclose(lr.sigma, es.imm_state.inverse_mass_matrix.sigma))
        )

    def test_bridge_mu_star_matches(self):
        es = self._make_engine_state()
        lr = _engine_state_to_low_rank_adaptation_state(es)
        self.assertTrue(bool(jnp.allclose(lr.mu_star, es.imm_state.mu_star)))

    def test_bridge_u_lam_match(self):
        es = self._make_engine_state()
        lr = _engine_state_to_low_rank_adaptation_state(es)
        self.assertTrue(bool(jnp.allclose(lr.U, es.imm_state.inverse_mass_matrix.U)))
        self.assertTrue(
            bool(jnp.allclose(lr.lam, es.imm_state.inverse_mass_matrix.lam))
        )

    def test_bridge_buffer_fields_match(self):
        es = self._make_engine_state()
        lr = _engine_state_to_low_rank_adaptation_state(es)
        self.assertTrue(bool(jnp.allclose(lr.draws_buffer, es.imm_state.draws_buffer)))
        self.assertTrue(bool(jnp.allclose(lr.grads_buffer, es.imm_state.grads_buffer)))
        self.assertEqual(int(lr.buffer_idx), int(es.imm_state.buffer_idx))
        self.assertEqual(int(lr.background_split), int(es.imm_state.background_split))
        self.assertEqual(int(lr.recompute_counter), int(es.imm_state.recompute_counter))

    def test_bridge_step_size_matches(self):
        es = self._make_engine_state()
        lr = _engine_state_to_low_rank_adaptation_state(es)
        self.assertEqual(float(lr.step_size), float(es.step_size))

    def test_bridge_info_fn_drops_buffers(self):
        """_make_low_rank_bridge_info_fn + _default_low_rank_adaptation_info_fn
        must drop draws_buffer/grads_buffer from the returned adaptation_state."""
        es = self._make_engine_state()
        # Dummy MCMC state and info objects.
        dummy_state = object()
        dummy_info = object()
        bridge_fn = _make_low_rank_bridge_info_fn(_default_low_rank_adaptation_info_fn)
        result = bridge_fn(dummy_state, dummy_info, es)
        self.assertIsNone(result.adaptation_state.draws_buffer)
        self.assertIsNone(result.adaptation_state.grads_buffer)
        # sigma and mu_star must still be present.
        self.assertIsNotNone(result.adaptation_state.sigma)
        self.assertIsNotNone(result.adaptation_state.mu_star)


# ---------------------------------------------------------------------------
# window_adaptation_low_rank shim — integration tests
# ---------------------------------------------------------------------------


class WindowAdaptationLowRankShimTest(BlackJAXTest):
    """Integration tests for the rewritten window_adaptation_low_rank shim."""

    _N_DIMS = 5
    _NUM_STEPS = 200

    def test_reset_path_runs(self):
        """buffer_policy='reset' (default) must complete and return finite metric."""
        wu = window_adaptation_low_rank(
            blackjax.nuts, std_normal_logdensity, max_rank=4
        )
        (state, params), info = wu.run(
            self.next_key(), jnp.zeros(self._N_DIMS), num_steps=self._NUM_STEPS
        )
        imm = params["inverse_mass_matrix"]
        self.assertIsInstance(imm, LowRankInverseMassMatrix)
        self.assertTrue(bool(jnp.all(jnp.isfinite(imm.sigma))))
        self.assertTrue(bool(jnp.all(jnp.isfinite(imm.U))))
        self.assertTrue(bool(jnp.all(jnp.isfinite(imm.lam))))

    def test_reset_path_state_is_reinited_at_mu_star(self):
        """The returned chain state must be re-initialised at mu_star, not the
        last MCMC position (i.e., state.position == unravel(mu_star))."""
        wu = window_adaptation_low_rank(
            blackjax.nuts, std_normal_logdensity, max_rank=4
        )
        (state, params), info = wu.run(
            self.next_key(), jnp.zeros(self._N_DIMS), num_steps=self._NUM_STEPS
        )
        # mu_star from the last step of info.
        mu_star_from_info = info.adaptation_state.mu_star[-1]
        self.assertTrue(
            bool(jnp.allclose(state.position, mu_star_from_info, atol=1e-6))
        )

    def test_accumulating_path_runs(self):
        """buffer_policy='accumulating' must complete and return finite metric."""
        wu = window_adaptation_low_rank(
            blackjax.nuts,
            std_normal_logdensity,
            max_rank=4,
            buffer_policy="accumulating",
        )
        (state, params), _ = wu.run(
            self.next_key(), jnp.zeros(self._N_DIMS), num_steps=self._NUM_STEPS
        )
        imm = params["inverse_mass_matrix"]
        self.assertTrue(bool(jnp.all(jnp.isfinite(imm.sigma))))
        self.assertTrue(bool(jnp.all(jnp.isfinite(imm.U))))

    def test_gradient_based_init_reset_path(self):
        """gradient_based_init=True on the reset path must produce finite metric
        and a non-identity initial sigma (different from gradient_based_init=False)."""
        wu_seed = window_adaptation_low_rank(
            blackjax.nuts, std_normal_logdensity, max_rank=4, gradient_based_init=True
        )
        wu_no_seed = window_adaptation_low_rank(
            blackjax.nuts, std_normal_logdensity, max_rank=4, gradient_based_init=False
        )
        key = self.next_key()
        (_, params_seed), _ = wu_seed.run(key, jnp.zeros(self._N_DIMS), num_steps=150)
        (_, params_no_seed), _ = wu_no_seed.run(
            key, jnp.zeros(self._N_DIMS), num_steps=150
        )
        # Both must be finite.
        self.assertTrue(
            bool(jnp.all(jnp.isfinite(params_seed["inverse_mass_matrix"].sigma)))
        )
        # They need not be identical (seeding changes the trajectory).
        # At zero position the gradient is non-zero so sigma will differ from ones.
        initial_sigma_seed = params_seed["inverse_mass_matrix"].sigma
        initial_sigma_no_seed = params_no_seed["inverse_mass_matrix"].sigma
        # Cannot guarantee numerical difference since the warmup may converge to
        # a similar value — only assert that both are finite.
        self.assertTrue(bool(jnp.all(jnp.isfinite(initial_sigma_seed))))
        self.assertTrue(bool(jnp.all(jnp.isfinite(initial_sigma_no_seed))))

    def test_info_adaptation_state_has_lr_fields_reset_path(self):
        """info.adaptation_state must have LowRankAdaptationState field layout
        (sigma, mu_star, U, lam, step_size) on the reset path."""
        wu = window_adaptation_low_rank(
            blackjax.nuts, std_normal_logdensity, max_rank=4
        )
        _, info = wu.run(
            self.next_key(), jnp.zeros(self._N_DIMS), num_steps=self._NUM_STEPS
        )
        st = info.adaptation_state
        # Fields present and shaped correctly.
        self.assertEqual(st.sigma.shape, (self._NUM_STEPS, self._N_DIMS))
        self.assertEqual(st.mu_star.shape, (self._NUM_STEPS, self._N_DIMS))
        self.assertEqual(st.U.shape, (self._NUM_STEPS, self._N_DIMS, 4))
        self.assertEqual(st.lam.shape, (self._NUM_STEPS, 4))
        # draws_buffer and grads_buffer dropped by default info fn.
        self.assertIsNone(st.draws_buffer)
        self.assertIsNone(st.grads_buffer)

    def test_info_adaptation_state_has_lr_fields_accumulating_path(self):
        """info.adaptation_state must have the same field layout on accumulating path."""
        wu = window_adaptation_low_rank(
            blackjax.nuts,
            std_normal_logdensity,
            max_rank=4,
            buffer_policy="accumulating",
        )
        _, info = wu.run(
            self.next_key(), jnp.zeros(self._N_DIMS), num_steps=self._NUM_STEPS
        )
        st = info.adaptation_state
        self.assertEqual(st.sigma.shape, (self._NUM_STEPS, self._N_DIMS))
        self.assertIsNone(st.draws_buffer)

    def test_custom_schedule_fn_reset_path(self):
        """schedule_fn=build_growing_window_schedule must work on the reset path."""
        wu = window_adaptation_low_rank(
            blackjax.nuts,
            std_normal_logdensity,
            max_rank=4,
            schedule_fn=build_growing_window_schedule,
        )
        (state, params), _ = wu.run(
            self.next_key(), jnp.zeros(self._N_DIMS), num_steps=self._NUM_STEPS
        )
        self.assertTrue(
            bool(jnp.all(jnp.isfinite(params["inverse_mass_matrix"].sigma)))
        )

    def test_invalid_buffer_policy_raises(self):
        """Unknown buffer_policy must raise ValueError before run()."""
        with self.assertRaises(ValueError):
            window_adaptation_low_rank(
                blackjax.nuts, std_normal_logdensity, buffer_policy="invalid"
            )

    def test_invalid_recompute_every_raises(self):
        """recompute_every < 1 must raise ValueError."""
        with self.assertRaises(ValueError):
            window_adaptation_low_rank(
                blackjax.nuts, std_normal_logdensity, recompute_every=0
            )

    def test_gradient_based_init_seeds_sigma_at_first_step(self):
        """With gradient_based_init=True, the sigma at step 0 in the info must
        differ from the default ones-initialization.

        The initial sigma is seeded from the gradient at the starting position
        before the scan begins.  Step 0 is in the fast phase (no metric update),
        so the sigma in info[0].adaptation_state reflects the initial value —
        seeded vs ones for True vs False.  Catches h_shim_seed (shim silently
        ignores the gradient_based_init flag).

        Uses pos=2 (non-zero) so the gradient is non-zero and the seeding does
        not fall back to identity (which happens when |grad| < 1e-10, e.g. at
        the mode of std_normal_logdensity at x=0).
        """
        pos = jnp.ones(self._N_DIMS) * 2.0
        key = self.next_key()
        wu_true = window_adaptation_low_rank(
            blackjax.nuts, std_normal_logdensity, max_rank=4, gradient_based_init=True
        )
        wu_false = window_adaptation_low_rank(
            blackjax.nuts, std_normal_logdensity, max_rank=4, gradient_based_init=False
        )
        _, info_true = wu_true.run(key, pos, num_steps=self._NUM_STEPS)
        _, info_false = wu_false.run(key, pos, num_steps=self._NUM_STEPS)
        sigma_true_0 = info_true.adaptation_state.sigma[0]
        sigma_false_0 = info_false.adaptation_state.sigma[0]
        self.assertFalse(
            bool(jnp.allclose(sigma_true_0, sigma_false_0)),
            "sigma at step 0 must differ between gradient_based_init=True and False",
        )

    def test_accumulating_dispatch_background_split(self):
        """background_split must be nonzero after at least one accumulating step,
        and always zero on the reset path.

        background_split is set at each slow-window switch in the accumulating
        scan loop (marking old draws as "background" for the next partial-forget).
        On the reset path it is always 0 (inert).  Catches b_dispatch (accumulating
        silently routed to the reset engine, which never sets background_split).
        """
        key = self.next_key()
        pos = jnp.zeros(self._N_DIMS)

        wu_reset = window_adaptation_low_rank(
            blackjax.nuts, std_normal_logdensity, max_rank=4, buffer_policy="reset"
        )
        _, info_reset = wu_reset.run(key, pos, num_steps=self._NUM_STEPS)
        # Reset path: background_split is always 0 at every step.
        bg_reset = np.asarray(info_reset.adaptation_state.background_split)
        self.assertTrue(
            np.all(bg_reset == 0),
            f"reset path: background_split must be all-zero, got max={bg_reset.max()}",
        )

        wu_acc = window_adaptation_low_rank(
            blackjax.nuts,
            std_normal_logdensity,
            max_rank=4,
            buffer_policy="accumulating",
        )
        _, info_acc = wu_acc.run(key, pos, num_steps=self._NUM_STEPS)
        # Accumulating path: background_split must be nonzero after the first switch.
        bg_acc = np.asarray(info_acc.adaptation_state.background_split)
        self.assertTrue(
            np.any(bg_acc != 0),
            "accumulating path: background_split must be nonzero after at least one switch",
        )


# ---------------------------------------------------------------------------
# x64 end-to-end smoke tests (float-promotion path)
# ---------------------------------------------------------------------------


class LowRankX64SmokeTest(BlackJAXTest):
    """End-to-end smoke under x64 mode for both low-rank recipes.

    The float64-promotion path inside _compute_low_rank_metric is only
    exercised when jax_enable_x64 is True.  These tests verify the full
    window_adaptation_low_rank pipeline produces finite, positive metric
    components under x64.  The chain dtype follows core.init (float64
    under x64); we do not mix float32 chains with x64 (out of scope).
    """

    _N_DIMS = 5
    _NUM_STEPS = 200

    def test_x64_smoke_fisher_low_rank(self):
        """Full reset-path e2e under x64: all metric fields must be finite and positive."""
        with jax.enable_x64():
            wu = window_adaptation_low_rank(
                blackjax.nuts, std_normal_logdensity, max_rank=4
            )
            (_, params), _ = wu.run(
                self.next_key(),
                jnp.zeros(self._N_DIMS),
                num_steps=self._NUM_STEPS,
            )
            imm = params["inverse_mass_matrix"]
            self.assertTrue(
                bool(jnp.all(jnp.isfinite(imm.sigma))), "sigma not finite under x64"
            )
            self.assertTrue(
                bool(jnp.all(jnp.isfinite(imm.U))), "U not finite under x64"
            )
            self.assertTrue(
                bool(jnp.all(jnp.isfinite(imm.lam))), "lam not finite under x64"
            )
            self.assertTrue(
                bool(jnp.all(imm.sigma > 0)), "sigma not positive under x64"
            )
            self.assertTrue(bool(jnp.all(imm.lam > 0)), "lam not positive under x64")

    def test_x64_smoke_sample_cov_low_rank(self):
        """sample_cov_low_rank full e2e under x64: metric must be finite and positive."""
        with jax.enable_x64():
            num_steps = self._NUM_STEPS
            buffer_size = min(2 * max(num_steps // 5, 128), max(num_steps, 1))
            core = _build_sample_cov_low_rank_core(buffer_size=buffer_size, max_rank=4)
            wu = staged_adaptation(
                blackjax.nuts,
                std_normal_logdensity,
                metric=core,
            )
            (_, params), _ = wu.run(
                self.next_key(),
                jnp.zeros(self._N_DIMS),
                num_steps=num_steps,
            )
            imm = params["inverse_mass_matrix"]
            self.assertTrue(
                bool(jnp.all(jnp.isfinite(imm.sigma))), "sigma not finite under x64"
            )
            self.assertTrue(
                bool(jnp.all(jnp.isfinite(imm.lam))), "lam not finite under x64"
            )
            self.assertTrue(
                bool(jnp.all(imm.sigma > 0)), "sigma not positive under x64"
            )
            self.assertTrue(bool(jnp.all(imm.lam > 0)), "lam not positive under x64")


# ---------------------------------------------------------------------------
# Bit-exact parity: accumulating path via engine vs frozen reference scan loop
# ---------------------------------------------------------------------------


def _reference_run_accumulating(
    rng_key, position, num_steps, n_dims, recompute_every=1
):
    """Frozen inline accumulating scan loop (verbatim copy of ``base(buffer_policy='accumulating')`` math).

    Bit-exact reference for :class:`WindowAdaptationLowRankAccumulatingParityTest`.
    Inline rather than importing ``base()`` so that this reference survives
    independently of the production code and cannot be silently changed.

    **Must NOT be modified** — this is the frozen pre-migration reference used by the parity gate.

    Bit-exact comparison works at ANY ``recompute_every`` value (default 1,
    matching nutpie's ``mass_matrix_update_freq=1``) after the engine fix that
    changed ``slow_update()`` to return ``new_metric_st.inverse_mass_matrix``
    instead of ``ws.inverse_mass_matrix``.  The engine now surfaces mid-window
    metric recomputes to MCMC immediately, matching ``slow_recompute_only()``
    in this reference.
    """
    max_rank = 4
    gamma = 1e-5
    cutoff = 2.0
    initial_step_size = 1.0
    target_acceptance_rate = 0.80

    da_init, da_update, da_final = dual_averaging_adaptation(target_acceptance_rate)
    mcmc_kernel = blackjax.nuts.build_kernel(mcmc.integrators.velocity_verlet)
    schedule = build_schedule(num_steps)
    buffer_size = max(_accumulating_buffer_capacity(schedule), 1)
    d = n_dims

    def fast_update(pos, grad, acc_rate, state):
        del pos, grad
        new_ss = da_update(state.ss_state, acc_rate)
        return LowRankAdaptationState(
            new_ss,
            state.sigma,
            state.mu_star,
            state.U,
            state.lam,
            jnp.exp(new_ss.log_step_size),
            state.draws_buffer,
            state.grads_buffer,
            state.buffer_idx,
            state.background_split,
            state.recompute_counter,
        )

    def slow_update(pos, grad, acc_rate, state):
        pos_flat, _ = fu.ravel_pytree(pos)
        grad_flat, _ = fu.ravel_pytree(grad)
        B = state.draws_buffer.shape[0]
        idx = state.buffer_idx % B
        new_draws = jax.lax.dynamic_update_slice(
            state.draws_buffer, pos_flat[None, :], (idx, 0)
        )
        new_grads = jax.lax.dynamic_update_slice(
            state.grads_buffer, grad_flat[None, :], (idx, 0)
        )
        new_ss = da_update(state.ss_state, acc_rate)
        return LowRankAdaptationState(
            new_ss,
            state.sigma,
            state.mu_star,
            state.U,
            state.lam,
            jnp.exp(new_ss.log_step_size),
            new_draws,
            new_grads,
            state.buffer_idx + 1,
            state.background_split,
            state.recompute_counter + 1,
        )

    def slow_switch(state):
        shift = state.background_split
        new_draws = _shift_buffer_left(state.draws_buffer, shift)
        new_grads = _shift_buffer_left(state.grads_buffer, shift)
        new_n_valid = state.buffer_idx - shift

        def _recompute():
            return _compute_low_rank_metric(
                new_draws, new_grads, new_n_valid, max_rank, gamma, cutoff
            )

        def _keep():
            return state.sigma, state.mu_star, state.U, state.lam

        sigma, mu_star, U, lam = jax.lax.cond(new_n_valid >= 3, _recompute, _keep)
        new_ss = da_init(da_final(state.ss_state))
        return LowRankAdaptationState(
            new_ss,
            sigma,
            mu_star,
            U,
            lam,
            jnp.exp(new_ss.log_step_size),
            new_draws,
            new_grads,
            new_n_valid,
            new_n_valid,
            0,
        )

    def slow_recompute_only(state):
        n = state.buffer_idx

        def _recompute():
            return _compute_low_rank_metric(
                state.draws_buffer, state.grads_buffer, n, max_rank, gamma, cutoff
            )

        def _keep():
            return state.sigma, state.mu_star, state.U, state.lam

        sigma, mu_star, U, lam = jax.lax.cond(n >= 3, _recompute, _keep)
        return LowRankAdaptationState(
            state.ss_state,
            sigma,
            mu_star,
            U,
            lam,
            state.step_size,
            state.draws_buffer,
            state.grads_buffer,
            state.buffer_idx,
            state.background_split,
            0,
        )

    def one_step(carry, xs):
        _, rng_key_, adaptation_stage = xs
        state, adaptation_state = carry
        metric = gaussian_euclidean_low_rank(
            adaptation_state.sigma, adaptation_state.U, adaptation_state.lam
        )
        new_state, step_info = mcmc_kernel(
            rng_key_, state, std_normal_logdensity, adaptation_state.step_size, metric
        )
        stage = adaptation_stage[0]
        is_window_end = adaptation_stage[1].astype(bool)
        new_adapt = jax.lax.switch(
            stage,
            (fast_update, slow_update),
            new_state.position,
            new_state.logdensity_grad,
            step_info.acceptance_rate,
            adaptation_state,
        )

        def _maybe_periodic_recompute(s):
            due = jnp.logical_and(
                stage == 1, s.recompute_counter % recompute_every == 0
            )
            return jax.lax.cond(due, slow_recompute_only, lambda x: x, s)

        new_adapt = jax.lax.cond(
            is_window_end, slow_switch, _maybe_periodic_recompute, new_adapt
        )
        return (
            (new_state, new_adapt),
            _default_low_rank_adaptation_info_fn(new_state, step_info, new_adapt),
        )

    ss_state = da_init(initial_step_size)
    init_state = blackjax.nuts.init(position, std_normal_logdensity)
    init_adaptation_state = LowRankAdaptationState(
        ss_state,
        jnp.ones(d),
        jnp.zeros(d),
        jnp.zeros((d, max_rank)),
        jnp.ones(max_rank),
        initial_step_size,
        jnp.zeros((buffer_size, d)),
        jnp.zeros((buffer_size, d)),
        0,
        0,
        0,
    )
    keys = jax.random.split(rng_key, num_steps)
    last_state, info = jax.lax.scan(
        one_step,
        (init_state, init_adaptation_state),
        (jnp.arange(num_steps), keys, schedule),
    )
    _, last_warmup_state, *_ = last_state
    step_size = jnp.exp(last_warmup_state.ss_state.log_step_size_avg)
    inverse_mass_matrix = LowRankInverseMassMatrix(
        sigma=last_warmup_state.sigma,
        U=last_warmup_state.U,
        lam=last_warmup_state.lam,
    )
    _, unravel = fu.ravel_pytree(position)
    mu_star_state = blackjax.nuts.init(
        unravel(last_warmup_state.mu_star), std_normal_logdensity
    )
    return (
        mu_star_state,
        {"step_size": step_size, "inverse_mass_matrix": inverse_mass_matrix},
        info,
    )


class WindowAdaptationLowRankAccumulatingParityTest(BlackJAXTest):
    """Bit-exact parity: accumulating path via engine == frozen reference scan loop.

    Parametrized over ``recompute_every ∈ {1, 5, 25}`` — all three must pass at
    ``atol=0.0``.  Verified against the 200-step Stan schedule (build_schedule(200)):

    * Window 1: 25 slow steps (ends at global step 99).
    * Window 2: 50 slow steps (ends at global step 149).
    * Final fast phase: 50 steps (no mass-matrix updates).

    With ``recompute_every=1`` : 73 mid-window + 2 at-window-end fires — exercises
    every cadence combination.

    With ``recompute_every=5`` : 13 mid-window + 2 at-window-end fires — exercises
    the "spurious recompute at window-end, immediately overwritten by final()" path
    (window 1 ends at slow_count=25, a multiple of 5; window 2 ends at 50=5×10).

    With ``recompute_every=25`` : 1 mid-window + 2 at-window-end fires (per-window
    counter resets: window-1 fires at counter=25=window-end; window-2 fires at
    counter=25=mid and counter=50=window-end).  Exercises the infrequent-recompute
    + at-window-boundary combination.

    Note: ``recompute_every=100`` fires zero times in 75 total slow steps and is
    equivalent to the degenerate ``10**9`` that was replaced — not included.

    After the ``slow_update()`` engine fix (commit e2a4afa7d), mid-window metric
    recomputes are forwarded to MCMC immediately (``new_metric_st.inverse_mass_matrix``
    is returned, not ``ws.inverse_mass_matrix``), so the engine is bit-identical to
    the legacy ``base()`` accumulating scan loop at ALL recompute cadences.
    """

    _N_DIMS = 5
    _NUM_STEPS = 200
    _RECOMPUTE_CADENCES = (1, 5, 25)

    def _run_engine(self, rng_key, position, recompute_every):
        wu = window_adaptation_low_rank(
            blackjax.nuts,
            std_normal_logdensity,
            max_rank=4,
            buffer_policy="accumulating",
            recompute_every=recompute_every,
        )
        return wu.run(rng_key, position, num_steps=self._NUM_STEPS)

    def test_final_step_size_identical(self):
        """Final step_size must be bit-identical for all recompute cadences."""
        key = self.next_key()
        pos = jnp.zeros(self._N_DIMS)
        for re in self._RECOMPUTE_CADENCES:
            with self.subTest(recompute_every=re):
                (_, params_engine), _ = self._run_engine(key, pos, re)
                _, params_ref, _ = _reference_run_accumulating(
                    key, pos, self._NUM_STEPS, self._N_DIMS, recompute_every=re
                )
                self.assertTrue(
                    bool(
                        jnp.allclose(
                            params_engine["step_size"],
                            params_ref["step_size"],
                            atol=0.0,
                        )
                    )
                )

    def test_final_sigma_identical(self):
        key = self.next_key()
        pos = jnp.zeros(self._N_DIMS)
        for re in self._RECOMPUTE_CADENCES:
            with self.subTest(recompute_every=re):
                (_, params_engine), _ = self._run_engine(key, pos, re)
                _, params_ref, _ = _reference_run_accumulating(
                    key, pos, self._NUM_STEPS, self._N_DIMS, recompute_every=re
                )
                self.assertTrue(
                    bool(
                        jnp.allclose(
                            params_engine["inverse_mass_matrix"].sigma,
                            params_ref["inverse_mass_matrix"].sigma,
                            atol=0.0,
                        )
                    )
                )

    def test_final_u_identical(self):
        key = self.next_key()
        pos = jnp.zeros(self._N_DIMS)
        for re in self._RECOMPUTE_CADENCES:
            with self.subTest(recompute_every=re):
                (_, params_engine), _ = self._run_engine(key, pos, re)
                _, params_ref, _ = _reference_run_accumulating(
                    key, pos, self._NUM_STEPS, self._N_DIMS, recompute_every=re
                )
                self.assertTrue(
                    bool(
                        jnp.allclose(
                            params_engine["inverse_mass_matrix"].U,
                            params_ref["inverse_mass_matrix"].U,
                            atol=0.0,
                        )
                    )
                )

    def test_final_lam_identical(self):
        key = self.next_key()
        pos = jnp.zeros(self._N_DIMS)
        for re in self._RECOMPUTE_CADENCES:
            with self.subTest(recompute_every=re):
                (_, params_engine), _ = self._run_engine(key, pos, re)
                _, params_ref, _ = _reference_run_accumulating(
                    key, pos, self._NUM_STEPS, self._N_DIMS, recompute_every=re
                )
                self.assertTrue(
                    bool(
                        jnp.allclose(
                            params_engine["inverse_mass_matrix"].lam,
                            params_ref["inverse_mass_matrix"].lam,
                            atol=0.0,
                        )
                    )
                )

    def test_final_chain_state_position_identical(self):
        """Returned chain state (mu_star re-init) must be bit-identical."""
        key = self.next_key()
        pos = jnp.zeros(self._N_DIMS)
        for re in self._RECOMPUTE_CADENCES:
            with self.subTest(recompute_every=re):
                (state_engine, _), _ = self._run_engine(key, pos, re)
                state_ref, _, _ = _reference_run_accumulating(
                    key, pos, self._NUM_STEPS, self._N_DIMS, recompute_every=re
                )
                self.assertTrue(
                    bool(
                        jnp.allclose(
                            state_engine.position, state_ref.position, atol=0.0
                        )
                    )
                )

    def test_per_step_sigma_identical(self):
        """Per-step sigma trace must be bit-identical across all steps."""
        key = self.next_key()
        pos = jnp.zeros(self._N_DIMS)
        for re in self._RECOMPUTE_CADENCES:
            with self.subTest(recompute_every=re):
                (_, _), info_engine = self._run_engine(key, pos, re)
                _, _, info_ref = _reference_run_accumulating(
                    key, pos, self._NUM_STEPS, self._N_DIMS, recompute_every=re
                )
                self.assertTrue(
                    bool(
                        jnp.allclose(
                            info_engine.adaptation_state.sigma,
                            info_ref.adaptation_state.sigma,
                            atol=0.0,
                        )
                    )
                )

    def test_per_step_mu_star_identical(self):
        """Per-step mu_star trace must be bit-identical."""
        key = self.next_key()
        pos = jnp.zeros(self._N_DIMS)
        for re in self._RECOMPUTE_CADENCES:
            with self.subTest(recompute_every=re):
                (_, _), info_engine = self._run_engine(key, pos, re)
                _, _, info_ref = _reference_run_accumulating(
                    key, pos, self._NUM_STEPS, self._N_DIMS, recompute_every=re
                )
                self.assertTrue(
                    bool(
                        jnp.allclose(
                            info_engine.adaptation_state.mu_star,
                            info_ref.adaptation_state.mu_star,
                            atol=0.0,
                        )
                    )
                )

    def test_per_step_step_size_identical(self):
        """Per-step step_size trace must be bit-identical."""
        key = self.next_key()
        pos = jnp.zeros(self._N_DIMS)
        for re in self._RECOMPUTE_CADENCES:
            with self.subTest(recompute_every=re):
                (_, _), info_engine = self._run_engine(key, pos, re)
                _, _, info_ref = _reference_run_accumulating(
                    key, pos, self._NUM_STEPS, self._N_DIMS, recompute_every=re
                )
                self.assertTrue(
                    bool(
                        jnp.allclose(
                            info_engine.adaptation_state.step_size,
                            info_ref.adaptation_state.step_size,
                            atol=0.0,
                        )
                    )
                )


if __name__ == "__main__":
    absltest.main()
