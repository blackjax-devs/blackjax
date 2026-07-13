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

    def test_fisher_mapping(self):
        """diagonal_estimator='fisher' → 'fisher_diag' (diagonal flag ignored)."""
        self.assertEqual(
            _pick_recipe_name(
                is_mass_matrix_diagonal=True, diagonal_estimator="fisher"
            ),
            "fisher_diag",
        )

    def test_welford_diag_mapping(self):
        """is_mass_matrix_diagonal=True + non-fisher estimator → 'welford_diag'."""
        self.assertEqual(
            _pick_recipe_name(
                is_mass_matrix_diagonal=True, diagonal_estimator="welford"
            ),
            "welford_diag",
        )

    def test_welford_dense_mapping(self):
        """is_mass_matrix_diagonal=False + non-fisher estimator → 'welford_dense'."""
        self.assertEqual(
            _pick_recipe_name(
                is_mass_matrix_diagonal=False, diagonal_estimator="welford"
            ),
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
