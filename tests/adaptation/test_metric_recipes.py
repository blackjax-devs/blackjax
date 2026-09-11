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
"""Tests for :mod:`blackjax.adaptation.metric_recipes`.

Coverage:
- :class:`MetricRecipe` coupling-contract validation at construction time.
- :func:`lookup_recipe` registry access and error handling.
- :class:`MetricCore` init/update/final embeddable-core contract for all three
  slice-1 recipes (``welford_diag``, ``welford_dense``, ``fisher_diag``).
- State-type identities (``MassMatrixAdaptationState`` vs
  ``FisherMassMatrixAdaptationState``).
"""
import jax
import jax.numpy as jnp
import numpy as np

from blackjax.adaptation.mass_matrix import (
    FisherMassMatrixAdaptationState,
    MassMatrixAdaptationState,
)
from blackjax.adaptation.metric_recipes import (
    REGISTRY,
    MetricCore,
    MetricRecipe,
    lookup_recipe,
)
from tests.fixtures import BlackJAXTest

# ---------------------------------------------------------------------------
# MetricRecipe coupling-contract validation
# ---------------------------------------------------------------------------


class MetricRecipeCouplingContractTest(BlackJAXTest):
    """Construction-time validation of the needs ⊆ provides and emits == repr
    coupling contract."""

    def test_needs_not_subset_of_provides_raises(self):
        """needs ⊄ provides must raise ValueError with a clear message."""
        with self.assertRaisesRegex(ValueError, "coupling violation"):
            MetricRecipe(
                representation="diag",
                estimator="welford",
                buffer="reset_window",
                support_gate=None,
                needs=frozenset(
                    {"positions", "gradients"}
                ),  # gradients NOT in provides
                provides=frozenset({"positions"}),
                emits="diag",
                provenance="test",
            )

    def test_emits_mismatches_representation_raises(self):
        """emits != representation must raise ValueError with a clear message."""
        with self.assertRaisesRegex(ValueError, "coupling violation"):
            MetricRecipe(
                representation="dense",
                estimator="welford",
                buffer="reset_window",
                support_gate=None,
                needs=frozenset({"positions"}),
                provides=frozenset({"positions"}),
                emits="diag",  # mismatch: declares dense but emits diag
                provenance="test",
            )

    def test_valid_recipe_constructs_without_error(self):
        """A fully-consistent recipe should not raise."""
        recipe = MetricRecipe(
            representation="diag",
            estimator="welford",
            buffer="reset_window",
            support_gate=None,
            needs=frozenset({"positions"}),
            provides=frozenset({"positions", "extra"}),  # superset is valid
            emits="diag",
            provenance="test provenance",
        )
        self.assertEqual(recipe.representation, "diag")
        self.assertEqual(recipe.estimator, "welford")

    def test_frozen_dataclass_is_immutable(self):
        """MetricRecipe is frozen; attribute assignment must raise."""
        recipe = REGISTRY["welford_diag"]
        with self.assertRaises((AttributeError, TypeError)):
            recipe.representation = "dense"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Registry lookup
# ---------------------------------------------------------------------------


class MetricRecipeRegistryTest(BlackJAXTest):
    """Registry access and error handling."""

    def test_all_slice1_names_present(self):
        """All three slice-1 registry names must be present."""
        for name in ("welford_diag", "welford_dense", "fisher_diag"):
            with self.subTest(name=name):
                recipe = lookup_recipe(name)
                self.assertIsInstance(recipe, MetricRecipe)

    def test_unknown_name_raises_with_known_names(self):
        """lookup_recipe should raise ValueError and list known names."""
        with self.assertRaisesRegex(ValueError, "Unknown metric recipe"):
            lookup_recipe("bogus_recipe")

    def test_welford_diag_recipe_fields(self):
        recipe = lookup_recipe("welford_diag")
        self.assertEqual(recipe.representation, "diag")
        self.assertEqual(recipe.emits, "diag")
        self.assertEqual(recipe.estimator, "welford")
        self.assertIn("positions", recipe.needs)
        self.assertNotIn("gradients", recipe.needs)

    def test_welford_dense_recipe_fields(self):
        recipe = lookup_recipe("welford_dense")
        self.assertEqual(recipe.representation, "dense")
        self.assertEqual(recipe.emits, "dense")
        self.assertEqual(recipe.estimator, "welford")

    def test_fisher_diag_recipe_needs_gradients(self):
        recipe = lookup_recipe("fisher_diag")
        self.assertEqual(recipe.representation, "diag")
        self.assertIn("gradients", recipe.needs)
        self.assertIn("positions", recipe.needs)

    def test_registry_values_are_frozen(self):
        """Registry values are MetricRecipe instances (frozen dataclasses)."""
        for name, recipe in REGISTRY.items():
            with self.subTest(name=name):
                self.assertIsInstance(recipe, MetricRecipe)


# ---------------------------------------------------------------------------
# MetricCore — embeddable-core contract tests
# ---------------------------------------------------------------------------


class MetricCoreContractWelfordDiagTest(BlackJAXTest):
    """welford_diag MetricCore: init/update/final protocol contract."""

    recipe_name = "welford_diag"
    n_dims = 3

    def _make_core(self, **kwargs):
        return lookup_recipe(self.recipe_name).build_core(**kwargs)

    def test_core_is_namedtuple_with_three_callables(self):
        core = self._make_core()
        self.assertIsInstance(core, MetricCore)
        self.assertTrue(callable(core.init))
        self.assertTrue(callable(core.update))
        self.assertTrue(callable(core.final))

    def test_init_returns_mass_matrix_adaptation_state(self):
        core = self._make_core()
        state = core.init(self.n_dims)
        self.assertIsInstance(state, MassMatrixAdaptationState)

    def test_init_inverse_mass_matrix_shape(self):
        """Diagonal IMM should have shape (n_dims,)."""
        core = self._make_core()
        state = core.init(self.n_dims)
        self.assertEqual(state.inverse_mass_matrix.shape, (self.n_dims,))

    def test_update_returns_same_type(self):
        core = self._make_core()
        state = core.init(self.n_dims)
        pos = jnp.ones(self.n_dims)
        new_state = core.update(state, pos, grad=None)
        self.assertIsInstance(new_state, MassMatrixAdaptationState)

    def test_update_with_grad_kwarg_ignored(self):
        """welford path: passing grad should not cause errors (ignored silently)."""
        core = self._make_core()
        state = core.init(self.n_dims)
        pos = jnp.ones(self.n_dims)
        grad = jnp.zeros(self.n_dims)
        new_state = core.update(state, pos, grad)
        self.assertIsInstance(new_state, MassMatrixAdaptationState)

    def test_final_returns_finite_positive_diag_imm(self):
        """After accumulating anisotropic samples, final() gives finite+positive IMM."""
        core = self._make_core()
        state = core.init(self.n_dims)

        # Anisotropic: dim 0 has scale 5, dim 1 has scale 1, dim 2 has scale 0.2.
        scales = jnp.array([5.0, 1.0, 0.2])
        key = self.next_key()
        positions = jax.random.normal(key, (100, self.n_dims)) * scales
        grads = jnp.zeros((100, self.n_dims))  # welford ignores grads

        def body(st, xs):
            pos, g = xs
            return core.update(st, pos, g), None

        state, _ = jax.lax.scan(body, state, (positions, grads))
        final_state = core.final(state)

        imm = final_state.inverse_mass_matrix
        self.assertEqual(imm.shape, (self.n_dims,))
        self.assertTrue(bool(jnp.all(jnp.isfinite(imm))))
        self.assertTrue(bool(jnp.all(imm > 0)))


class MetricCoreContractWelfordDenseTest(BlackJAXTest):
    """welford_dense MetricCore: init/update/final protocol contract."""

    recipe_name = "welford_dense"
    n_dims = 4

    def _make_core(self, **kwargs):
        return lookup_recipe(self.recipe_name).build_core(**kwargs)

    def test_init_inverse_mass_matrix_shape(self):
        """Dense IMM should have shape (n_dims, n_dims)."""
        core = self._make_core()
        state = core.init(self.n_dims)
        self.assertEqual(state.inverse_mass_matrix.shape, (self.n_dims, self.n_dims))

    def test_final_returns_genuinely_dense_imm(self):
        """After accumulating CORRELATED samples, final() must produce a matrix
        with non-zero off-diagonals — not just the right shape.

        Catch-site: mapping welford_dense→welford_diag via a recipe-name bug
        would broadcast a 1-D welford estimate to (d,d) via the initial seed,
        leaving shape correct but off-diagonals zero.  This test detects that.
        """
        core = self._make_core()
        state = core.init(self.n_dims)

        # Correlated samples: Cholesky factor with rho_{0,1} ≈ 0.87.
        # Welford dense estimate of the covariance must have imm[0,1] ≠ 0.
        L = jnp.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.87, 0.49, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        key = self.next_key()
        z = jax.random.normal(key, (200, self.n_dims))
        positions = z @ L.T  # correlated, shape (200, 4)
        grads = jnp.zeros((200, self.n_dims))

        def body(st, xs):
            pos, g = xs
            return core.update(st, pos, g), None

        state, _ = jax.lax.scan(body, state, (positions, grads))
        final_state = core.final(state)

        imm = final_state.inverse_mass_matrix
        self.assertEqual(imm.shape, (self.n_dims, self.n_dims))
        self.assertTrue(bool(jnp.all(jnp.isfinite(imm))))
        # Must be genuinely dense: off-diagonal [0,1] must differ from zero.
        self.assertFalse(
            bool(jnp.allclose(imm, jnp.diag(jnp.diag(imm)))),
            "welford_dense IMM should have non-zero off-diagonals with correlated samples",
        )


class MetricCoreContractFisherDiagTest(BlackJAXTest):
    """fisher_diag MetricCore: init/update/final protocol contract."""

    recipe_name = "fisher_diag"
    n_dims = 3

    def _make_core(self, **kwargs):
        return lookup_recipe(self.recipe_name).build_core(**kwargs)

    def test_init_returns_fisher_state_type(self):
        core = self._make_core()
        state = core.init(self.n_dims)
        self.assertIsInstance(state, FisherMassMatrixAdaptationState)

    def test_update_returns_fisher_state_type(self):
        core = self._make_core()
        state = core.init(self.n_dims)
        pos = jnp.ones(self.n_dims)
        grad = jnp.ones(self.n_dims)
        new_state = core.update(state, pos, grad)
        self.assertIsInstance(new_state, FisherMassMatrixAdaptationState)

    def test_final_resets_block_count(self):
        """final() must reset the fisher_block count to 0 (memoryless windows)."""
        core = self._make_core()
        state = core.init(self.n_dims)
        state = core.update(state, jnp.ones(self.n_dims), jnp.ones(self.n_dims))
        final_state = core.final(state)
        self.assertIsInstance(final_state, FisherMassMatrixAdaptationState)
        self.assertEqual(int(final_state.fisher_block.count), 0)

    def test_final_returns_finite_positive_imm(self):
        """After accumulating samples, final() gives finite+positive diagonal IMM.

        Catch-site: a var_x / var_g swap in the Fisher stitch would produce
        IMM_i = 1 / true_var_i instead of true_var_i.  The assert_allclose with
        rtol=1e-3 catches that swap, whereas a shape+sign check would not.
        """
        core = self._make_core()
        state = core.init(self.n_dims)

        # Anisotropic: true_var = [4, 1, 0.25].
        # Draws ~ N(0, true_var) => grad_i = -draw_i / true_var_i.
        # Fisher estimator: IMM_i = Var[draw_i] / Var[grad_i]
        #                          = true_var_i / (1 / true_var_i) = true_var_i.
        true_var = jnp.array([4.0, 1.0, 0.25])
        key = self.next_key()
        draws = jax.random.normal(key, (500, self.n_dims)) * jnp.sqrt(true_var)
        grads = -draws / true_var  # gradient of diagonal normal: -x / true_var

        def body(st, xs):
            pos, g = xs
            return core.update(st, pos, g), None

        state, _ = jax.lax.scan(body, state, (draws, grads))
        final_state = core.final(state)

        imm = final_state.inverse_mass_matrix
        self.assertEqual(imm.shape, (self.n_dims,))
        self.assertTrue(bool(jnp.all(jnp.isfinite(imm))))
        self.assertTrue(bool(jnp.all(imm > 0)))
        np.testing.assert_allclose(
            np.asarray(imm),
            np.asarray(true_var),
            rtol=1e-3,
            err_msg="Fisher IMM should equal true_var; a var_x/var_g swap would invert it",
        )


class MetricCoreBuildCoreParamsTest(BlackJAXTest):
    """build_core() parameter forwarding tests."""

    def test_welford_diag_accepts_imm_shrinkage(self):
        """build_core with imm_shrinkage_to_previous != 0 should work on welford path."""
        core = lookup_recipe("welford_diag").build_core(imm_shrinkage_to_previous=5.0)
        state = core.init(3)
        self.assertIsInstance(state, MassMatrixAdaptationState)

    def test_welford_diag_accepts_initial_imm(self):
        """build_core with initial_inverse_mass_matrix should seed the IMM."""
        seed_imm = jnp.array([2.0, 3.0, 4.0])
        core = lookup_recipe("welford_diag").build_core(
            initial_inverse_mass_matrix=seed_imm
        )
        state = core.init(3)
        # The initial IMM (before any updates) should be the seed.
        np.testing.assert_array_equal(
            np.asarray(state.inverse_mass_matrix),
            np.asarray(seed_imm),
        )

    def test_fisher_diag_unknown_estimator_raises(self):
        """build_core should raise for an unknown estimator tag."""
        recipe = MetricRecipe(
            representation="diag",
            estimator="bogus_estimator",
            buffer="reset_window",
            support_gate=None,
            needs=frozenset({"positions"}),
            provides=frozenset({"positions"}),
            emits="diag",
            provenance="test",
        )
        with self.assertRaisesRegex(ValueError, "Unknown estimator tag"):
            recipe.build_core()
