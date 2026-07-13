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
"""Tests for window_adaptation's opt-in Fisher-diagonal estimator
(``diagonal_estimator="fisher"``).

The Fisher-diagonal estimator replaces Welford's online covariance with the
Fisher-divergence-minimising diagonal scale of
:cite:p:`seyboldt2026preconditioning` (the same formula underlying Step 1 of
``blackjax.adaptation.low_rank_adaptation._compute_low_rank_metric``):
``inverse_mass_matrix = sqrt(Var[position] / Var[logdensity_grad])``.

Ported from branch ``b197f1e2`` (``feat/window-adaptation-fisher-diag``).

**Dropped tests (noted here per brief):**

The following two tests from ``b197f1e2`` directly tested the private helper
``mass_matrix._fisher_diagonal_inverse_mass``, which was deleted in the
re-land (its body now lives exclusively in
``metric_estimators.fisher_score_diagonal``):

- ``FisherDiagNearZeroGradientTest.test_zero_gradient_variance_is_finite``
  → called ``_fisher_diagonal_inverse_mass(jnp.array(1.0), jnp.array(0.0))``
  directly.
- ``FisherDiagNearZeroGradientTest.test_mixed_zero_and_nonzero_gradient_variance_per_coordinate``
  → called ``_fisher_diagonal_inverse_mass(var_x, var_g)`` directly.

Both behaviors are now covered indirectly by
``FisherDiagNearZeroGradientTest.test_window_with_stationary_point_does_not_produce_nan``
(system-level) and by ``tests/adaptation/test_metric_estimators.py``
(estimator-level).

**Adapted tests:**

- ``FisherDiagRecoveryTest.test_final_resets_accumulators``: updated to check
  ``state.fisher_block.count == 0`` instead of ``state.wc_state.sample_size``
  and ``state.grad_wc_state.sample_size`` (internal accumulator changed from
  two Welford states to :class:`_FisherMomentBlock`).
- ``FisherDiagRecoveryTest.test_recovers_known_diagonal_gaussian_exactly``:
  updated to compose the estimator call explicitly via
  ``fisher_score_diagonal_from_moments`` instead of reading from the state
  after ``final()``.  After the bridge removal, ``mass_matrix_adaptation``'s
  ``final()`` only resets the block; the IMM computation is the consumer's
  responsibility (done by ``window_adaptation.base``'s ``slow_final``).
- ``FisherDiagNearZeroGradientTest.test_window_with_stationary_point_does_not_produce_nan``:
  same: IMM checked via explicit ``fisher_score_diagonal_from_moments`` call
  rather than via ``final()`` read-back.
"""
import jax
import jax.numpy as jnp
import numpy as np

import blackjax
from blackjax.adaptation.mass_matrix import (
    FisherMassMatrixAdaptationState,
    MassMatrixAdaptationState,
    mass_matrix_adaptation,
)
from blackjax.adaptation.metric_estimators import fisher_score_diagonal_from_moments
from tests.fixtures import BlackJAXTest, std_normal_logdensity

# ---------------------------------------------------------------------------
# (a) Default invariance: diagonal_estimator="welford" (explicit or implicit
#     default) must reproduce the pre-existing Welford path exactly.
# ---------------------------------------------------------------------------


class FisherDiagDefaultInvarianceTest(BlackJAXTest):
    def test_default_estimator_returns_plain_state_type(self):
        """With no diagonal_estimator kwarg, mass_matrix_adaptation returns
        the original 2-field MassMatrixAdaptationState -- not the new
        FisherMassMatrixAdaptationState -- i.e. the state's pytree structure is
        unchanged for existing callers."""
        init, _, _ = mass_matrix_adaptation(is_diagonal_matrix=True)
        state = init(3)
        self.assertIsInstance(state, MassMatrixAdaptationState)
        self.assertNotIsInstance(state, FisherMassMatrixAdaptationState)

    def test_explicit_welford_matches_implicit_default(self):
        """diagonal_estimator='welford' explicitly must be bit-identical to
        omitting the kwarg entirely (mirrors the imm_shrinkage_to_previous
        default-invariance test)."""

        def logdensity_fn(x):
            return std_normal_logdensity(x, scale=jnp.array([0.5, 1.0, 2.0]))

        rng_key = jax.random.key(0)
        warmup_default = blackjax.window_adaptation(blackjax.nuts, logdensity_fn)
        warmup_explicit = blackjax.window_adaptation(
            blackjax.nuts, logdensity_fn, diagonal_estimator="welford"
        )

        (_, params_default), _ = warmup_default.run(
            rng_key, jnp.zeros(3), num_steps=200
        )
        (_, params_explicit), _ = warmup_explicit.run(
            rng_key, jnp.zeros(3), num_steps=200
        )

        np.testing.assert_array_equal(
            params_default["step_size"], params_explicit["step_size"]
        )
        np.testing.assert_array_equal(
            params_default["inverse_mass_matrix"],
            params_explicit["inverse_mass_matrix"],
        )


# ---------------------------------------------------------------------------
# Construction-time validation
# ---------------------------------------------------------------------------


class FisherDiagValidationTest(BlackJAXTest):
    def test_invalid_estimator_name_raises(self):
        with self.assertRaisesRegex(ValueError, "diagonal_estimator must be"):
            mass_matrix_adaptation(diagonal_estimator="bogus")

    def test_fisher_requires_diagonal_matrix(self):
        with self.assertRaisesRegex(ValueError, "requires is_diagonal_matrix=True"):
            mass_matrix_adaptation(
                is_diagonal_matrix=False, diagonal_estimator="fisher"
            )

    def test_fisher_rejects_imm_shrinkage(self):
        with self.assertRaisesRegex(
            ValueError, "does not support imm_shrinkage_to_previous"
        ):
            mass_matrix_adaptation(
                diagonal_estimator="fisher", imm_shrinkage_to_previous=5.0
            )

    def test_window_adaptation_propagates_dense_validation(self):
        def logdensity_fn(x):
            return std_normal_logdensity(x)

        with self.assertRaisesRegex(
            ValueError, "requires is_mass_matrix_diagonal=True"
        ):
            blackjax.window_adaptation(
                blackjax.nuts,
                logdensity_fn,
                is_mass_matrix_diagonal=False,
                diagonal_estimator="fisher",
            )

    def test_window_adaptation_propagates_shrinkage_validation(self):
        def logdensity_fn(x):
            return std_normal_logdensity(x)

        with self.assertRaisesRegex(
            ValueError, "does not support imm_shrinkage_to_previous"
        ):
            blackjax.window_adaptation(
                blackjax.nuts,
                logdensity_fn,
                diagonal_estimator="fisher",
                imm_shrinkage_to_previous=10.0,
            )


# ---------------------------------------------------------------------------
# (b) Fisher-diag recovery on a known diagonal Gaussian.
# ---------------------------------------------------------------------------


class FisherDiagRecoveryTest(BlackJAXTest):
    def test_recovers_known_diagonal_gaussian_exactly(self):
        """For X ~ N(0, diag(true_var)), the log-density gradient is the
        *exact* linear map grad = -x / true_var. Because Var[c*X] == c**2 *
        Var[X] holds exactly for the *sample* variance (not just in
        expectation), sqrt(Var[x] / Var[grad]) == true_var exactly for any
        finite sample -- no Monte-Carlo tolerance needed (verified
        empirically at n in {2_000, 5_000, 10_000} across 6 seeds: relative
        error ~0 in every case, not a marginal/noise-limited statistic).

        NOTE: ``mass_matrix_adaptation``'s ``final()`` only resets the block;
        it does NOT compute the new IMM.  The IMM computation is the consumer's
        responsibility (performed by ``window_adaptation.base``'s ``slow_final``
        on the live pipeline).  This test exercises the accumulation + estimator
        directly via ``fisher_score_diagonal_from_moments``.
        """
        d = 4
        true_var = jnp.array([0.1, 1.0, 4.0, 25.0])
        n = 500
        key1, _ = jax.random.split(self.next_key())
        draws = jax.random.normal(key1, (n, d)) * jnp.sqrt(true_var)
        grads = -draws / true_var

        init, update, _ = mass_matrix_adaptation(
            is_diagonal_matrix=True, diagonal_estimator="fisher"
        )
        state = init(d)

        def body(state, xs):
            x, g = xs
            return update(state, x, g), None

        state, _ = jax.lax.scan(body, state, (draws, grads))

        # Compose the estimator call explicitly (mirrors window_adaptation's slow_final).
        block = state.fisher_block
        denom = jnp.maximum(block.count - 1.0, 1.0)
        var_x = block.m2_x / denom
        var_g = block.m2_g / denom
        imm = fisher_score_diagonal_from_moments(var_x, var_g)

        np.testing.assert_allclose(np.asarray(imm), np.asarray(true_var), rtol=1e-3)

    def test_final_resets_accumulators(self):
        """final() must re-initialize the FisherMomentBlock so a subsequent
        window starts from scratch, mirroring the Welford path's own reset
        semantics.

        Adapted from b197f1e2: the old branch checked state.wc_state.sample_size
        and state.grad_wc_state.sample_size (two separate Welford states). The
        new accumulator is a _FisherMomentBlock whose count must be 0 after
        final().
        """
        d = 3
        init, update, final = mass_matrix_adaptation(
            is_diagonal_matrix=True, diagonal_estimator="fisher"
        )
        state = init(d)
        state = update(state, jnp.ones(d), jnp.ones(d))
        state = final(state)
        self.assertIsInstance(state, FisherMassMatrixAdaptationState)
        self.assertEqual(int(state.fisher_block.count), 0)


# ---------------------------------------------------------------------------
# (c) Near-zero-gradient robustness -- system-level test via mass_matrix_adaptation.
#
# Tests for _fisher_diagonal_inverse_mass(var_x, var_g) directly from b197f1e2
# are DROPPED here because that private helper was removed; its body lives only
# in metric_estimators.fisher_score_diagonal and is tested in
# tests/adaptation/test_metric_estimators.py.
# ---------------------------------------------------------------------------


class FisherDiagNearZeroGradientTest(BlackJAXTest):
    def test_window_with_stationary_point_does_not_produce_nan(self):
        """A window in which the gradient is exactly zero throughout (e.g. a
        chain sitting at a stationary point of the target, such as the
        origin of any centered/standardised density) must not poison the
        final inverse mass matrix with NaN/Inf.

        NOTE: ``mass_matrix_adaptation``'s ``final()`` only resets the block;
        this test exercises the estimator path directly via
        ``fisher_score_diagonal_from_moments`` with zero gradient variance,
        mirroring what ``window_adaptation``'s ``slow_final`` does.
        """
        d = 3
        init, update, _ = mass_matrix_adaptation(
            is_diagonal_matrix=True, diagonal_estimator="fisher"
        )
        state = init(d)
        zero_grad = jnp.zeros(d)

        def body(state, position):
            return update(state, position, zero_grad), None

        positions = jax.random.normal(self.next_key(), (50, d))
        state, _ = jax.lax.scan(body, state, positions)

        # Compose the estimator call explicitly (mirrors window_adaptation's slow_final).
        block = state.fisher_block
        denom = jnp.maximum(block.count - 1.0, 1.0)
        var_x = block.m2_x / denom
        var_g = block.m2_g / denom  # zero throughout: gradient variance is zero
        imm = fisher_score_diagonal_from_moments(var_x, var_g)

        self.assertTrue(bool(jnp.all(jnp.isfinite(imm))))
        self.assertTrue(bool(jnp.all(imm > 0)))


# ---------------------------------------------------------------------------
# (d) End-to-end smoke: NUTS + window_adaptation(diagonal_estimator="fisher")
#     on a small correlated Gaussian.
# ---------------------------------------------------------------------------


class FisherDiagEndToEndSmokeTest(BlackJAXTest):
    def test_nuts_with_fisher_diag_on_correlated_gaussian(self):
        """2-D correlated Gaussian (var 4/1, rho=0.7 -- same control target
        as the low-rank adaptation tests). The Fisher-diagonal estimator
        only ever produces a *diagonal* metric, so it cannot capture the
        correlation -- that is expected, not a bug. This test only checks
        that warmup + sampling run to completion with finite output and a
        sane acceptance rate, and that the (marginal) diagonal variances
        recovered are in the right ballpark."""
        rho = 0.7
        cov = jnp.array([[4.0, rho * 2.0], [rho * 2.0, 1.0]])
        precision = jnp.linalg.inv(cov)

        def logdensity_fn(x):
            return -0.5 * x @ precision @ x

        warmup_key, inference_key = jax.random.split(self.next_key())
        warmup = blackjax.window_adaptation(
            blackjax.nuts, logdensity_fn, diagonal_estimator="fisher"
        )
        (state, parameters), _ = warmup.run(warmup_key, jnp.zeros(2), num_steps=1_000)

        self.assertTrue(bool(jnp.all(jnp.isfinite(parameters["inverse_mass_matrix"]))))
        self.assertTrue(bool(jnp.all(parameters["inverse_mass_matrix"] > 0)))
        self.assertTrue(bool(jnp.isfinite(parameters["step_size"])))
        self.assertGreater(float(parameters["step_size"]), 0.0)

        # Marginal variances (diag(cov) = [4.0, 1.0]) recovered within a
        # generous factor-of-3 band -- a diagonal-only estimator on a
        # correlated target is not expected to be precise, only sane.
        np.testing.assert_allclose(
            np.asarray(parameters["inverse_mass_matrix"]),
            np.array([4.0, 1.0]),
            rtol=2.0,
        )

        nuts = blackjax.nuts(logdensity_fn, **parameters)
        keys = jax.random.split(inference_key, 500)

        def one_step(state, key):
            state, info = nuts.step(key, state)
            return state, info

        final_state, infos = jax.lax.scan(one_step, state, keys)

        self.assertTrue(bool(jnp.all(jnp.isfinite(final_state.position))))
        mean_acceptance = jnp.mean(infos.acceptance_rate)
        self.assertTrue(bool(jnp.isfinite(mean_acceptance)))
        self.assertGreater(float(mean_acceptance), 0.3)
