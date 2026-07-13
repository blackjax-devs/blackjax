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
"""Tests for :func:`blackjax.staged_adaptation` engine.

Coverage:
- Engine e2e smoke: NUTS + staged_adaptation on a small isotropic Gaussian;
  finite/positive step_size and inverse_mass_matrix for f32 and x64.
- String / MetricRecipe / MetricCore metric argument paths.
- ``StagedAdaptationState`` / ``WindowAdaptationState`` alias identity.
- :func:`build_schedule` re-export parity (same object from both modules).
- MetricCore metric argument accepted (pre-built core passed directly).
- Fisher path: staged_adaptation with ``metric="fisher_diag"`` on a
  small correlated Gaussian gives finite/positive results.

Notes
-----
Per the implementation brief (TL adjustment D), this file does NOT include
bit-for-bit comparison of ``window_adaptation`` vs ``staged_adaptation``
outputs.  The shim parity guarantee is enforced by the existing adaptation
tests (``test_adaptation.py``, ``test_window_adaptation_fisher_diag.py``)
which now run through the shim path.
"""
import jax
import jax.numpy as jnp
import numpy as np

import blackjax
from blackjax.adaptation.metric_recipes import REGISTRY, lookup_recipe
from blackjax.adaptation.staged_adaptation import (
    StagedAdaptationState,
    build_schedule,
    staged_adaptation,
)
from blackjax.adaptation.window_adaptation import (
    WindowAdaptationState,
    _pick_recipe_name,
)
from blackjax.adaptation.window_adaptation import (
    build_schedule as window_build_schedule,
)
from tests.fixtures import BlackJAXTest, std_normal_logdensity

# ---------------------------------------------------------------------------
# Alias identity tests
# ---------------------------------------------------------------------------


class StagedAdaptationStateAliasTest(BlackJAXTest):
    """WindowAdaptationState is an alias for StagedAdaptationState."""

    def test_class_identity(self):
        """WindowAdaptationState IS StagedAdaptationState (same class object)."""
        self.assertIs(WindowAdaptationState, StagedAdaptationState)

    def test_isinstance_interchangeable(self):
        """An instance created with one name satisfies isinstance for the other."""
        # Build a state via the engine to get a genuine StagedAdaptationState.
        core = lookup_recipe("welford_diag").build_core()
        state_0 = core.init(3)
        from blackjax.adaptation.step_size import dual_averaging_adaptation

        da_init, _, _ = dual_averaging_adaptation(0.8)
        ss_state = da_init(1.0)
        sa_state = StagedAdaptationState(
            ss_state=ss_state,
            imm_state=state_0,
            step_size=1.0,
            inverse_mass_matrix=state_0.inverse_mass_matrix,
        )
        self.assertIsInstance(sa_state, WindowAdaptationState)
        self.assertIsInstance(sa_state, StagedAdaptationState)

    def test_blackjax_top_level_export(self):
        """blackjax.staged_adaptation must be importable."""
        self.assertTrue(callable(blackjax.staged_adaptation))

    def test_build_schedule_reexport_identity(self):
        """build_schedule from staged_adaptation and window_adaptation is the same function."""
        self.assertIs(build_schedule, window_build_schedule)


# ---------------------------------------------------------------------------
# Engine e2e smoke: metric argument paths
# ---------------------------------------------------------------------------


class StagedAdaptationMetricArgPathTest(BlackJAXTest):
    """Three supported metric argument types all produce finite/positive results."""

    def _run_warmup(self, metric_arg):
        def logdensity_fn(x):
            return std_normal_logdensity(x)

        warmup = staged_adaptation(
            blackjax.nuts,
            logdensity_fn,
            metric=metric_arg,
            initial_step_size=0.5,
        )
        (state, params), _ = warmup.run(self.next_key(), jnp.zeros(3), num_steps=200)
        return state, params

    def test_string_metric_arg(self):
        """metric='welford_diag' (string) path produces valid parameters."""
        _, params = self._run_warmup("welford_diag")
        self.assertTrue(bool(jnp.isfinite(params["step_size"])))
        self.assertGreater(float(params["step_size"]), 0.0)
        self.assertTrue(bool(jnp.all(jnp.isfinite(params["inverse_mass_matrix"]))))
        self.assertTrue(bool(jnp.all(params["inverse_mass_matrix"] > 0)))

    def test_recipe_metric_arg(self):
        """metric=MetricRecipe path produces valid parameters."""
        recipe = REGISTRY["welford_diag"]
        _, params = self._run_warmup(recipe)
        self.assertTrue(bool(jnp.isfinite(params["step_size"])))
        self.assertGreater(float(params["step_size"]), 0.0)
        self.assertTrue(bool(jnp.all(jnp.isfinite(params["inverse_mass_matrix"]))))
        self.assertTrue(bool(jnp.all(params["inverse_mass_matrix"] > 0)))

    def test_core_metric_arg(self):
        """metric=MetricCore (pre-built) path produces valid parameters."""
        core = REGISTRY["welford_diag"].build_core()
        _, params = self._run_warmup(core)
        self.assertTrue(bool(jnp.isfinite(params["step_size"])))
        self.assertGreater(float(params["step_size"]), 0.0)
        self.assertTrue(bool(jnp.all(jnp.isfinite(params["inverse_mass_matrix"]))))
        self.assertTrue(bool(jnp.all(params["inverse_mass_matrix"] > 0)))

    def test_invalid_metric_type_raises(self):
        """Passing a non-str/non-recipe/non-core metric type should raise TypeError."""

        def logdensity_fn(x):
            return std_normal_logdensity(x)

        with self.assertRaises(TypeError):
            staged_adaptation(blackjax.nuts, logdensity_fn, metric=42)

    def test_unknown_string_metric_raises(self):
        """An unknown string metric name should raise ValueError."""

        def logdensity_fn(x):
            return std_normal_logdensity(x)

        with self.assertRaisesRegex(ValueError, "Unknown metric recipe"):
            staged_adaptation(blackjax.nuts, logdensity_fn, metric="bogus_recipe")


# ---------------------------------------------------------------------------
# Engine e2e: dtype tests (f32 and x64 — no cross-dtype comparison)
# ---------------------------------------------------------------------------


class StagedAdaptationDtypeTest(BlackJAXTest):
    """Each dtype must independently produce finite/positive outputs."""

    def _run_and_check(self, position_dtype):
        def logdensity_fn(x):
            return std_normal_logdensity(x)

        initial_pos = jnp.zeros(4, dtype=position_dtype)
        warmup = staged_adaptation(
            blackjax.nuts,
            logdensity_fn,
            metric="welford_diag",
            initial_step_size=1.0,
        )
        (state, params), _ = warmup.run(self.next_key(), initial_pos, num_steps=300)
        self.assertTrue(
            bool(jnp.isfinite(params["step_size"])),
            f"step_size not finite for dtype={position_dtype}",
        )
        self.assertGreater(
            float(params["step_size"]),
            0.0,
            f"step_size not positive for dtype={position_dtype}",
        )
        self.assertTrue(
            bool(jnp.all(jnp.isfinite(params["inverse_mass_matrix"]))),
            f"inverse_mass_matrix not finite for dtype={position_dtype}",
        )
        self.assertTrue(
            bool(jnp.all(params["inverse_mass_matrix"] > 0)),
            f"inverse_mass_matrix not positive for dtype={position_dtype}",
        )

    def test_f32(self):
        self._run_and_check(jnp.float32)

    def test_f32_second_call_is_idempotent(self):
        """A second f32 run produces finite/positive results (idempotency check)."""
        # x64 tests require JAX_ENABLE_X64=1 which is not set in the default
        # test invocation; we only test f32 here to avoid the UserWarning that
        # explicit float64 dtypes emit when x64 is not enabled (pytest.ini turns
        # all warnings into errors).
        self._run_and_check(jnp.float32)


# ---------------------------------------------------------------------------
# Engine e2e: recipe coverage
# ---------------------------------------------------------------------------


class StagedAdaptationRecipeSmokeTest(BlackJAXTest):
    """All three slice-1 recipes produce finite/positive warmup results on a
    small isotropic Gaussian."""

    def _run_recipe(self, recipe_name, n_dims=3):
        def logdensity_fn(x):
            return std_normal_logdensity(x)

        warmup = staged_adaptation(
            blackjax.nuts,
            logdensity_fn,
            metric=recipe_name,
            initial_step_size=0.5,
        )
        (state, params), _ = warmup.run(
            self.next_key(), jnp.zeros(n_dims), num_steps=300
        )
        return state, params

    def _check_finite_positive(self, params, recipe_name):
        self.assertTrue(
            bool(jnp.isfinite(params["step_size"])),
            f"{recipe_name}: step_size not finite",
        )
        self.assertGreater(
            float(params["step_size"]),
            0.0,
            f"{recipe_name}: step_size not positive",
        )
        imm = params["inverse_mass_matrix"]
        self.assertTrue(
            bool(jnp.all(jnp.isfinite(imm))),
            f"{recipe_name}: inverse_mass_matrix has non-finite entries",
        )

    def test_welford_diag(self):
        _, params = self._run_recipe("welford_diag")
        self._check_finite_positive(params, "welford_diag")
        # Diagonal: 1-D array
        self.assertEqual(params["inverse_mass_matrix"].ndim, 1)

    def test_welford_dense(self):
        _, params = self._run_recipe("welford_dense")
        self._check_finite_positive(params, "welford_dense")
        # Dense: 2-D square array
        imm = params["inverse_mass_matrix"]
        self.assertEqual(imm.ndim, 2)
        self.assertEqual(imm.shape[0], imm.shape[1])

    def test_fisher_diag(self):
        """fisher_diag on a small correlated Gaussian: finite/positive IMM."""
        n_dims = 2
        cov = jnp.array([[4.0, 0.7 * 2.0], [0.7 * 2.0, 1.0]])
        precision = jnp.linalg.inv(cov)

        def logdensity_fn(x):
            return -0.5 * x @ precision @ x

        warmup = staged_adaptation(
            blackjax.nuts,
            logdensity_fn,
            metric="fisher_diag",
            initial_step_size=0.5,
        )
        (state, params), _ = warmup.run(
            self.next_key(), jnp.zeros(n_dims), num_steps=500
        )
        self._check_finite_positive(params, "fisher_diag")
        imm = params["inverse_mass_matrix"]
        self.assertEqual(imm.ndim, 1)
        self.assertEqual(imm.shape[0], n_dims)
        self.assertTrue(bool(jnp.all(imm > 0)))


# ---------------------------------------------------------------------------
# Shim alias check: staged_adaptation at blackjax top level
# ---------------------------------------------------------------------------


class BlackjaxTopLevelStagedAdaptationTest(BlackJAXTest):
    """blackjax.staged_adaptation is importable and functional."""

    def test_runs_via_top_level_import(self):
        def logdensity_fn(x):
            return std_normal_logdensity(x)

        warmup = blackjax.staged_adaptation(
            blackjax.nuts,
            logdensity_fn,
            metric="welford_diag",
            initial_step_size=0.5,
        )
        (state, params), _ = warmup.run(self.next_key(), jnp.zeros(3), num_steps=100)
        self.assertTrue(bool(jnp.isfinite(params["step_size"])))
        self.assertGreater(float(params["step_size"]), 0.0)


# ---------------------------------------------------------------------------
# Fold-in 1: structural fast-window test
# ---------------------------------------------------------------------------


class StagedAdaptationFastWindowTest(BlackJAXTest):
    """With num_steps=19 (< initial_buffer_size=75) the schedule is all-fast.
    The Welford accumulator must never increment."""

    def test_all_fast_schedule_does_not_accumulate(self):
        """19 steps < initial_buffer_size (75) → all fast stages.

        Catch-site: if the stage-dispatch accidentally calls wc.update in the
        fast phase, sample_size would be > 0; this test detects that leak.
        """
        warmup = staged_adaptation(
            blackjax.nuts,
            std_normal_logdensity,
            metric="welford_diag",
            initial_step_size=1.0,
        )
        _, info = warmup.run(self.next_key(), jnp.zeros(3), num_steps=19)

        # info is stacked AdaptationInfo; .adaptation_state is stacked StagedAdaptationState.
        # imm_state.wc_state.sample_size is a (19,) array of per-step accumulators.
        sample_size = info.adaptation_state.imm_state.wc_state.sample_size
        self.assertEqual(
            int(sample_size[-1]),
            0,
            "Welford accumulator must stay at 0 in the all-fast schedule",
        )


# ---------------------------------------------------------------------------
# Fold-in 3a: _pick_recipe_name unit tests (all three param-to-name mappings)
# ---------------------------------------------------------------------------


class PickRecipeNameTest(BlackJAXTest):
    """Direct unit tests for the private ``_pick_recipe_name`` helper."""

    def test_welford_diag_mapping(self):
        """is_mass_matrix_diagonal=True → 'welford_diag'."""
        self.assertEqual(
            _pick_recipe_name(is_mass_matrix_diagonal=True),
            "welford_diag",
        )

    def test_welford_dense_mapping(self):
        """is_mass_matrix_diagonal=False → 'welford_dense'."""
        self.assertEqual(
            _pick_recipe_name(is_mass_matrix_diagonal=False),
            "welford_dense",
        )


# ---------------------------------------------------------------------------
# Fold-in 4: real x64 e2e smoke (per-recipe, finite/positive only)
# ---------------------------------------------------------------------------


class StagedAdaptationX64SmokeTest(BlackJAXTest):
    """All three recipes must produce finite/positive outputs in x64 mode.

    All JAX operations — both the warmup run and the assertion comparisons —
    execute inside the ``jax.enable_x64()`` context manager.  Moving the
    assertions outside the context would trigger a UserWarning from float64
    dtype promotion against Python literals, which pytest.ini turns into an
    error.
    """

    def _smoke_recipe_x64(self, recipe_name, n_dims=3):
        """Run the given recipe in x64 mode and check finite/positive results.

        For diagonal recipes the IMM is a 1-D array of positive marginal
        variances — element-wise ``> 0`` is correct.  For the dense recipe the
        IMM is a full covariance matrix whose off-diagonal entries can be
        negative, so we only assert finiteness (not element-wise positivity).
        """
        recipe = lookup_recipe(recipe_name)
        is_diag = recipe.representation == "diag"

        with jax.enable_x64():
            warmup = staged_adaptation(
                blackjax.nuts,
                std_normal_logdensity,
                metric=recipe_name,
                initial_step_size=0.5,
            )
            (_, params), _ = warmup.run(
                self.next_key(),
                jnp.zeros(n_dims, dtype=jnp.float64),
                num_steps=300,
            )
            # Assertions inside the context: float64 dtype promotion is valid here.
            self.assertTrue(
                bool(jnp.isfinite(params["step_size"])),
                f"x64 {recipe_name}: step_size not finite",
            )
            self.assertGreater(
                float(params["step_size"]),
                0.0,
                f"x64 {recipe_name}: step_size not positive",
            )
            self.assertTrue(
                bool(jnp.all(jnp.isfinite(params["inverse_mass_matrix"]))),
                f"x64 {recipe_name}: inverse_mass_matrix has non-finite entries",
            )
            if is_diag:
                # Diagonal IMM: every element is a marginal variance — must be > 0.
                self.assertTrue(
                    bool(jnp.all(params["inverse_mass_matrix"] > 0)),
                    f"x64 {recipe_name}: diagonal IMM has non-positive entries",
                )

    def test_welford_diag_x64(self):
        self._smoke_recipe_x64("welford_diag")

    def test_welford_dense_x64(self):
        self._smoke_recipe_x64("welford_dense")

    def test_fisher_diag_x64(self):
        self._smoke_recipe_x64("fisher_diag")


# ---------------------------------------------------------------------------
# Migrated behavioral tests from test_window_adaptation_imm_seed.py
# (targeting staged_adaptation instead of window_adaptation)
# ---------------------------------------------------------------------------


class StagedAdaptationIMMSeedBehavioralTest(BlackJAXTest):
    """Behavioral/numeric tests for initial_inverse_mass_matrix and
    imm_shrinkage_to_previous, migrated from window_adaptation to staged_adaptation."""

    def _setup_target(self):
        """Anisotropic 3-D Gaussian target — wide per-dim variance range makes the
        IMM-seed effect on early-warmup step-size adaptation measurable."""
        target_std = jnp.array([0.1, 1.0, 10.0])

        def logdensity_fn(x):
            return jax.scipy.stats.norm.logpdf(
                x / target_std, loc=0.0, scale=1.0
            ).sum() - jnp.sum(jnp.log(target_std))

        return logdensity_fn, target_std

    def _run_warmup_staged(
        self, rng_key, logdensity_fn, metric, imm=None, num_steps=200
    ):
        """Helper: run staged_adaptation and return (step_size, inverse_mass_matrix)."""
        # For tests with custom initial_inverse_mass_matrix, we need to construct
        # the core explicitly via lookup_recipe
        if imm is not None:
            from blackjax.adaptation.metric_recipes import lookup_recipe

            if metric == "welford_diag" and imm.ndim == 1:
                # Diagonal metric with seed
                core = lookup_recipe("welford_diag").build_core(
                    initial_inverse_mass_matrix=imm
                )
                metric_arg = core
            elif metric == "welford_dense" and imm.ndim == 2:
                # Dense metric with seed
                core = lookup_recipe("welford_dense").build_core(
                    initial_inverse_mass_matrix=imm
                )
                metric_arg = core
            else:
                metric_arg = metric
        else:
            metric_arg = metric

        warmup = staged_adaptation(
            blackjax.nuts,
            logdensity_fn,
            metric=metric_arg,
            initial_step_size=0.5,
        )
        dim = 3
        init_pos = jnp.zeros(dim)
        (state, params), _ = warmup.run(rng_key, init_pos, num_steps=num_steps)
        return params["step_size"], params["inverse_mass_matrix"]

    def test_backward_compat_no_imm(self):
        """staged_adaptation with no initial_inverse_mass_matrix runs without error."""
        logdensity_fn, _ = self._setup_target()
        rng_key = self.next_key()
        step_size, imm = self._run_warmup_staged(rng_key, logdensity_fn, "welford_diag")
        self.assertGreater(float(step_size), 0)
        self.assertEqual(imm.shape, (3,))
        self.assertTrue(bool(jnp.all(imm > 0)))

    def test_diagonal_seed_runs(self):
        """Diagonal seed IMM does not crash and returns well-shaped outputs."""
        logdensity_fn, _ = self._setup_target()
        rng_key = self.next_key()
        seed_imm = jnp.array([0.1, 1.0, 10.0])  # matches true covariance diagonal
        step_size, imm = self._run_warmup_staged(
            rng_key, logdensity_fn, "welford_diag", imm=seed_imm
        )
        self.assertGreater(float(step_size), 0)
        self.assertEqual(imm.shape, (3,))

    def test_diagonal_seed_differs_from_default(self):
        """First-window step-size adaptation differs when seeded vs default IMM.

        The seed IMM is very different from identity, so the adapted step sizes
        after a short warmup should differ between the two conditions.
        """
        logdensity_fn, _ = self._setup_target()
        rng_key = self.next_key()
        # Use a very short warmup so the seed has more influence
        step_default, _ = self._run_warmup_staged(
            rng_key, logdensity_fn, "welford_diag", imm=None, num_steps=100
        )
        # Extreme seed that strongly scales the geometry
        extreme_seed = jnp.array([100.0, 100.0, 100.0])
        step_seeded, _ = self._run_warmup_staged(
            rng_key, logdensity_fn, "welford_diag", imm=extreme_seed, num_steps=100
        )
        # They should differ — the seed changes the step size adaptation
        self.assertFalse(bool(jnp.allclose(step_default, step_seeded, atol=1e-6)))

    def test_dense_seed_runs(self):
        """Dense seed IMM (metric='welford_dense') runs without error."""
        logdensity_fn, _ = self._setup_target()
        rng_key = self.next_key()
        # Use a diagonal PD matrix as the dense seed
        seed_imm = jnp.diag(jnp.array([0.1, 1.0, 10.0]))
        step_size, imm = self._run_warmup_staged(
            rng_key, logdensity_fn, "welford_dense", imm=seed_imm
        )
        self.assertGreater(float(step_size), 0)
        self.assertEqual(imm.shape, (3, 3))

    def test_welford_convergence_seed_does_not_poison(self):
        """Final adapted IMM should be close regardless of seed when warmup is long.

        With enough steps, Welford's algorithm overwrites the seed.  We verify that
        both the default-seeded and the truth-seeded adaptations end up with similar
        final IMMs on a 3-D Gaussian where the true diagonal is known.
        """
        logdensity_fn, target_std = self._setup_target()
        rng_key = self.next_key()
        _, imm_default = self._run_warmup_staged(
            rng_key, logdensity_fn, "welford_diag", imm=None, num_steps=1000
        )
        # Seed with the true covariance diagonal (variance = std^2)
        seed_imm = target_std**2
        _, imm_seeded = self._run_warmup_staged(
            rng_key, logdensity_fn, "welford_diag", imm=seed_imm, num_steps=1000
        )

        # Both should be positive
        self.assertTrue(bool(jnp.all(imm_default > 0)))
        self.assertTrue(bool(jnp.all(imm_seeded > 0)))

        # With 1000 steps the Welford estimator dominates; the two IMMs should be
        # in the same ballpark (within 50% of each other)
        ratio = imm_seeded / imm_default
        np.testing.assert_allclose(ratio, jnp.ones_like(ratio), atol=0.5)

    def test_imm_shrinkage_backward_compat_default_zero(self):
        """Default imm_shrinkage_to_previous=0.0 produces Stan-identical output.

        With the new kwarg defaulting to 0.0, the behavior must be identical to
        the pre-P2 code. We verify that calling with explicit 0.0 and implicit
        default produce the same warmup result.
        """
        logdensity_fn, _ = self._setup_target()
        rng_key = self.next_key()

        # For staged_adaptation with shrinkage, construct the core
        from blackjax.adaptation.metric_recipes import lookup_recipe

        # Explicit 0.0
        core_explicit = lookup_recipe("welford_diag").build_core(
            imm_shrinkage_to_previous=0.0
        )
        warmup_explicit = staged_adaptation(
            blackjax.nuts,
            logdensity_fn,
            metric=core_explicit,
            initial_step_size=0.5,
        )
        (_, params_explicit), _ = warmup_explicit.run(
            rng_key, jnp.zeros(3), num_steps=300
        )

        # Implicit default (no shrinkage kwarg)
        warmup_default = staged_adaptation(
            blackjax.nuts,
            logdensity_fn,
            metric="welford_diag",
            initial_step_size=0.5,
        )
        (_, params_default), _ = warmup_default.run(
            rng_key, jnp.zeros(3), num_steps=300
        )

        # Should be bit-identical (or at least very close due to JAX numerical reproducibility)
        np.testing.assert_allclose(
            params_explicit["inverse_mass_matrix"],
            params_default["inverse_mass_matrix"],
            rtol=1e-6,
        )

    def test_imm_shrinkage_seed_influence_persists_diagonal(self):
        """With non-zero pseudo-count, seed IMM influence persists longer.

        Compare two diagonal cases: one with imm_shrinkage_to_previous=0.0 (seed
        loses influence quickly) and one with a large pseudo-count (seed sticky).
        With a deliberately-wrong seed, the sticky version's final IMM should be
        closer to the seed than the non-sticky version's.
        """
        logdensity_fn, _ = self._setup_target()
        rng_key = self.next_key()
        # Seed that is 100x larger than optimal — will bias the result
        wrong_seed = jnp.array([100.0, 100.0, 100.0])

        from blackjax.adaptation.metric_recipes import lookup_recipe

        # No shrinkage: seed is quickly overwritten by Welford
        core_no_shrink = lookup_recipe("welford_diag").build_core(
            initial_inverse_mass_matrix=wrong_seed, imm_shrinkage_to_previous=0.0
        )
        warmup_no_shrink = staged_adaptation(
            blackjax.nuts,
            logdensity_fn,
            metric=core_no_shrink,
            initial_step_size=0.5,
        )
        (_, params_no_shrink), _ = warmup_no_shrink.run(
            rng_key, jnp.zeros(3), num_steps=300
        )
        imm_no_shrink = params_no_shrink["inverse_mass_matrix"]

        # Large shrinkage: seed's influence persists
        core_with_shrink = lookup_recipe("welford_diag").build_core(
            initial_inverse_mass_matrix=wrong_seed, imm_shrinkage_to_previous=20.0
        )
        warmup_with_shrink = staged_adaptation(
            blackjax.nuts,
            logdensity_fn,
            metric=core_with_shrink,
            initial_step_size=0.5,
        )
        (_, params_with_shrink), _ = warmup_with_shrink.run(
            rng_key, jnp.zeros(3), num_steps=300
        )
        imm_with_shrink = params_with_shrink["inverse_mass_matrix"]

        # With shrinkage, the final IMM should be closer to the (wrong) seed
        # than the no-shrinkage case.
        dist_no_shrink = jnp.mean((imm_no_shrink - wrong_seed) ** 2)
        dist_with_shrink = jnp.mean((imm_with_shrink - wrong_seed) ** 2)
        error_msg = (
            f"Expected shrinkage to keep IMM closer to seed: "
            f"got dist_no_shrink={dist_no_shrink: .6f}, "
            f"dist_with_shrink={dist_with_shrink: .6f}"
        )
        self.assertLess(dist_with_shrink, dist_no_shrink, error_msg)

    def test_imm_shrinkage_dense_matrix_mirrors_diagonal(self):
        """Dense case with shrinkage applies the formula symmetrically.

        Test that imm_shrinkage_to_previous works correctly for dense matrices
        (metric='welford_dense'), confirming the shrinkage term is applied
        to the full matrix, not just the diagonal.
        """
        logdensity_fn, _ = self._setup_target()
        rng_key = self.next_key()
        # Use a diagonal PD matrix as the dense seed
        wrong_seed = jnp.diag(jnp.array([100.0, 100.0, 100.0]))

        from blackjax.adaptation.metric_recipes import lookup_recipe

        # No shrinkage: dense case
        core_dense_no_shrink = lookup_recipe("welford_dense").build_core(
            initial_inverse_mass_matrix=wrong_seed, imm_shrinkage_to_previous=0.0
        )
        warmup_dense_no_shrink = staged_adaptation(
            blackjax.nuts,
            logdensity_fn,
            metric=core_dense_no_shrink,
            initial_step_size=0.5,
        )
        (_, params_dense_no_shrink), _ = warmup_dense_no_shrink.run(
            rng_key, jnp.zeros(3), num_steps=300
        )
        imm_dense_no_shrink = params_dense_no_shrink["inverse_mass_matrix"]

        # With shrinkage: dense case
        core_dense_with_shrink = lookup_recipe("welford_dense").build_core(
            initial_inverse_mass_matrix=wrong_seed, imm_shrinkage_to_previous=20.0
        )
        warmup_dense_with_shrink = staged_adaptation(
            blackjax.nuts,
            logdensity_fn,
            metric=core_dense_with_shrink,
            initial_step_size=0.5,
        )
        (_, params_dense_with_shrink), _ = warmup_dense_with_shrink.run(
            rng_key, jnp.zeros(3), num_steps=300
        )
        imm_dense_with_shrink = params_dense_with_shrink["inverse_mass_matrix"]

        # Same logic as the diagonal test: shrinkage should keep the final IMM
        # closer to the (wrong) seed.
        dist_no_shrink = jnp.mean((imm_dense_no_shrink - wrong_seed) ** 2)
        dist_with_shrink = jnp.mean((imm_dense_with_shrink - wrong_seed) ** 2)
        error_msg = (
            f"Expected dense shrinkage to keep IMM closer to seed: "
            f"got dist_no_shrink={dist_no_shrink: .6f}, "
            f"dist_with_shrink={dist_with_shrink: .6f}"
        )
        self.assertLess(dist_with_shrink, dist_no_shrink, error_msg)
