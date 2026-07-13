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
"""Tests for the slice-2 low-rank MetricCore recipes.

Coverage:
- :class:`LowRankMetricCoreState` field contract.
- :func:`seed_low_rank_sigma_from_grad` gradient seeding logic.
- :func:`_build_fisher_low_rank_core` init/update/final contract.
- :func:`_build_sample_cov_low_rank_core` init/update/final contract.
- Registry entries ``"fisher_low_rank"`` and ``"sample_cov_low_rank"``.
- :func:`MetricRecipe.build_core` low-rank dispatch and buffer_size guard.
- :func:`staged_adaptation` ``schedule_fn`` parameter.
- Bit-identity: ``_compute_low_rank_metric`` moved to ``metric_estimators``
  is the same object as the backward-compat re-export from
  ``low_rank_adaptation``.
"""
import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest

import blackjax
from blackjax.adaptation.low_rank_adaptation import (
    _compute_low_rank_metric as lra_clrm,
    build_growing_window_schedule,
)
from blackjax.adaptation.metric_estimators import _compute_low_rank_metric
from blackjax.adaptation.metric_recipes import (
    LowRankMetricCoreState,
    MetricCore,
    MetricRecipe,
    REGISTRY,
    lookup_recipe,
    seed_low_rank_sigma_from_grad,
    _build_fisher_low_rank_core,
    _build_sample_cov_low_rank_core,
)
from blackjax.adaptation.staged_adaptation import build_schedule, staged_adaptation
from blackjax.mcmc.metrics import LowRankInverseMassMatrix
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
            np.asarray(final_state.draws_buffer), np.zeros((self.buffer_size, self.n_dims))
        )
        np.testing.assert_array_equal(
            np.asarray(final_state.grads_buffer), np.zeros((self.buffer_size, self.n_dims))
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
        """grads_buffer must remain zeros after update() (draws-only core)."""
        core = self._make_core()
        state = core.init(self.n_dims)
        key = self.next_key()
        draws = jax.random.normal(key, (10, self.n_dims))

        def body(s, xs):
            return core.update(s, xs, None), None

        final_state, _ = jax.lax.scan(body, state, draws)
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

    def test_all_slice2_names_present(self):
        for name in ("fisher_low_rank", "sample_cov_low_rank"):
            with self.subTest(name=name):
                self.assertIn(name, REGISTRY)


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


if __name__ == "__main__":
    absltest.main()
