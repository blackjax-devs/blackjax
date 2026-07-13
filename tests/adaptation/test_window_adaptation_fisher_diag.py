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
from blackjax.adaptation.window_adaptation import base as window_adaptation_base
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

    def test_gradient_wire_accumulates_into_m2_g_not_m2_x(self):
        """The gradient plumbing must feed grad into m2_g and position into
        m2_x -- not the same buffer.  Failure mode: if grad is silently
        replaced by position (e.g. ``update(state, position, position)``), the
        block would hold Var[x] in BOTH m2_x and m2_g, so the ratio would be
        identically 1.0 and the test would still pass -- this test closes that
        gap by constructing a batch where Var[x] ≠ Var[g] by a known factor
        and asserting the accumulated m2_g matches Var[g], not Var[x].

        Failure modes caught:
        - Severed gradient wire (grads := draws) → m2_g == m2_x → ratio = 1.
        - Inverted ratio (m2_x used as m2_g) → imm ≈ 1/true_var instead of
          true_var.
        - Read-after-reset (m2_g = 0 from a reset block) → ratio = inf/max(0,
          1e-10) → clipped constant, not true_var.
        """
        d = 3
        # Use Var[x] = 4 * Var[g] so the ratio Var[x]/Var[g] = 4 per coordinate
        # and IMM = sqrt(4) = 2 per coordinate -- distinct from 1 and from 1/2.
        n = 1_000
        key = self.next_key()
        base = jax.random.normal(key, (n, d))
        draws = base * 2.0  # Var[x] ≈ 4 per coord
        grads = base * 1.0  # Var[g] ≈ 1 per coord  (same base → ratio exact = 4)

        init, update, _ = mass_matrix_adaptation(
            is_diagonal_matrix=True, diagonal_estimator="fisher"
        )
        state = init(d)

        def body(state, xs):
            x, g = xs
            return update(state, x, g), None

        state, _ = jax.lax.scan(body, state, (draws, grads))

        block = state.fisher_block
        denom = jnp.maximum(block.count - 1.0, 1.0)
        var_x_acc = block.m2_x / denom
        var_g_acc = block.m2_g / denom

        # Since draws = 2 * base and grads = 1 * base (same base), the
        # sample variance ratio is ALGEBRAICALLY exact: var_x = 4 * var_g
        # regardless of the realised sample variance of base.
        np.testing.assert_allclose(
            np.asarray(var_x_acc / var_g_acc),
            np.ones(d) * 4.0,
            rtol=1e-4,
            err_msg="var_x / var_g must be exactly 4 (draws = 2 * grads)",
        )

        # Confirm the gradient buffer is the smaller one (~1), not the larger
        # one (~4).  rtol=0.2 handles n=1_000 sampling noise (std error ~0.045)
        # while excluding the wrong-buffer failure (which would give ~4 here).
        np.testing.assert_allclose(
            np.asarray(var_g_acc),
            np.ones(d),
            rtol=0.2,
            err_msg="m2_g must accumulate gradient variance (≈1), not position variance (≈4)",
        )

        # The IMM must be sqrt(var_x/var_g) = sqrt(4) = 2 exactly.
        imm = fisher_score_diagonal_from_moments(var_x_acc, var_g_acc)
        np.testing.assert_allclose(
            np.asarray(imm),
            np.ones(d) * 2.0,
            rtol=1e-4,
            err_msg="IMM should be sqrt(Var[x]/Var[g]) = sqrt(4) = 2.0 exactly",
        )


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
        as the low-rank adaptation tests).

        The Fisher-diagonal estimator converges to
            IMM[i] = sqrt(Var[x_i] / Var[∇log p_i])
        For X ~ N(0, cov) with precision = cov^{-1}:
            Var[x_i] = cov[i,i]
            Var[(precision @ x)[i]] = (precision @ cov @ precision)[i,i] = precision[i,i]
            → IMM[i] = sqrt(cov[i,i] / precision[i,i])

        For rho=0.7: cov=[[4, 1.4],[1.4, 1]], det=2.04,
        precision[0,0]=1/2.04≈0.490, precision[1,1]=4/2.04≈1.961.
        Analytic target: [sqrt(4/0.490), sqrt(1/1.961)] ≈ [2.857, 0.714].

        **Why not diag(cov)=[4,1]:** the Fisher estimator minimises the
        Fisher divergence between the true density and a Gaussian with the
        given diagonal metric, which is NOT the same as matching marginal
        variances. Asserting against diag(cov) is wrong and vacuous (the
        rtol=2.0 band covers [1,12] × [0.33, 3] -- too wide to catch
        severed gradient wires, inverted ratios, or read-after-reset).
        """
        rho = 0.7
        cov = jnp.array([[4.0, rho * 2.0], [rho * 2.0, 1.0]])
        precision = jnp.linalg.inv(cov)

        # Analytic Fisher-diagonal target: sqrt(diag(cov) / diag(precision)).
        # For symmetric precision matrix P: Var[Px]_i = (P cov P)[i,i] = P[i,i]
        # (since P·cov = I → P·cov·P = P), so the analytic target is below.
        analytic_imm = jnp.sqrt(jnp.diag(cov) / jnp.diag(precision))

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

        # Assert against the correct analytic target with rtol=0.3.
        # This band is tight enough to catch: grads:=draws (gives [1,1]),
        # inverted ratio (gives [0.35,1.4]), and read-after-reset (gives a
        # clipped constant) — all of which lie outside [0.7×analytic, 1.3×analytic].
        np.testing.assert_allclose(
            np.asarray(parameters["inverse_mass_matrix"]),
            np.asarray(analytic_imm),
            rtol=0.3,
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


# ---------------------------------------------------------------------------
# (e) Direct stitch #2 test: base(diagonal_estimator='fisher') path
# ---------------------------------------------------------------------------


class BaseFisherStitchDirectTest(BlackJAXTest):
    """Direct test of ``window_adaptation.base`` with
    ``diagonal_estimator='fisher'``.

    The ``base()`` function contains Fisher stitch #2 — the slow_final closure
    that computes the Fisher-diagonal IMM from accumulated Fisher block state.
    This test exercises that stitch directly (not via the MetricCore path) and
    is the canonical coverage for stitch #2.

    Catch-site: if someone removes the MetricCore fallback in ``slow_final``
    while ``base()`` still delegates to ``staged_adaptation``, only this test
    would catch the regression — the existing MetricCore-path tests would pass.
    """

    def test_base_fisher_diag_gives_finite_positive_imm(self):
        """base(diagonal_estimator='fisher') produces finite/positive IMM.

        This exercises stitch #2 directly: the ``slow_final`` closure in
        ``window_adaptation.base`` that reads the Fisher block state and writes
        the IMM.  Stitch #1 lives in ``metric_recipes._build_fisher_diag_core``
        and is tested separately by ``MetricCoreContractFisherDiagTest``.
        """
        n = 3
        init_fn, update_fn, final_fn = window_adaptation_base(
            is_mass_matrix_diagonal=True,
            diagonal_estimator="fisher",
        )
        # base().init takes (position: ArrayLikeTree, initial_step_size: float)
        position = jnp.zeros(n)
        state = init_fn(position, 1.0)

        key = self.next_key()
        draws = jax.random.normal(key, (200, n))
        grads = -draws  # gradient of std-normal log density: -x

        # Run in slow-window mode (stage=1) without triggering end-of-window
        def body(st, xs):
            pos, g = xs
            # stage=1 (slow), is_middle_window_end=False
            return update_fn(st, (jnp.array(1), jnp.array(False)), pos, g, 0.8), None

        state, _ = jax.lax.scan(body, state, (draws, grads))

        # Trigger slow_final: end-of-slow-window update that stitches in new IMM
        state = update_fn(
            state, (jnp.array(1), jnp.array(True)), draws[0], grads[0], 0.8
        )
        step_size, imm = final_fn(state)

        self.assertEqual(imm.shape, (n,))
        self.assertTrue(bool(jnp.all(jnp.isfinite(imm))))
        self.assertTrue(bool(jnp.all(imm > 0)))
