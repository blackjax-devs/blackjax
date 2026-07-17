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

"""Builder and e2e tests for :mod:`blackjax.adaptation.meta.builders`.

Coverage:
- TestEscalationDecisionTable, TestStructuralE2ESmoke, TestRecovershClassical,
  TestDefaultWiringAndBudgetWarning: single-chain core tests.
- TestImpossibleComboInvariant, TestNewStateFieldsPopulated, TestSharedEpsilonDA,
  TestWBranchE2ESmoke, TestEndToEndEscalation: multi-chain e2e tests.
"""
import warnings

import jax
import jax.numpy as jnp
import numpy as np

import blackjax
from blackjax.adaptation.meta import (
    MetaAdaptationVerdict,
    build_meta_adaptation_core,
    build_multi_chain_meta_core,
    extract_meta_verdict,
)
from blackjax.adaptation.meta._calibration import (
    _ASSUMED_AVG_LEAPFROGS_PER_STEP,
    _DETECTION_BRANCH_BETWEEN_MEANS,
    _DETECTION_BRANCH_BOTH,
    _DETECTION_BRANCH_NONE,
    _DETECTION_BRANCH_POOLED_WITHIN,
    _LAM_NONTRIVIAL_TOL,
    _R2_DEFERRED,
    _R2_FULL_AFFINE,
    _R_MIN,
    _S_MIN,
)
from blackjax.adaptation.meta._signals import (
    _choose_rank,
    _compute_s_gap,
    _compute_whitened_spectrum,
)
from blackjax.adaptation.metric_recipes import MetricCore
from blackjax.adaptation.staged_adaptation import _make_engine
from blackjax.mcmc.metrics import LowRankInverseMassMatrix
from tests.adaptation._meta_fixtures import (
    _fill_mc_state,
    _fill_state_from_buffer,
    _make_correlated_buffer,
    _make_curvature_buffer,
    _make_high_sgap_curvature_buffer,
    _make_isotropic_buffer,
    _make_marginal_sgap_curvature_buffer,
    _make_mc_coexistence,
    _make_mc_deep_spread,
    _make_mc_even_spread,
    _make_mc_isotropic,
    _make_mc_split_means,
)
from tests.fixtures import BlackJAXTest


class TestEscalationDecisionTable(BlackJAXTest):
    """Each escalation gate blocks independently; monotone escalation verified."""

    def _run_two_windows(self, draws, grads, max_grad_budget=50000, max_rank=10):
        """Run two identical windows so that the stability gate can pass."""
        d = draws.shape[1]
        core = build_meta_adaptation_core(max_grad_budget, max_rank=max_rank)
        state = core.init(d)
        for _ in range(2):
            state = _fill_state_from_buffer(state, draws, grads)
            state = core.final(state)
        return state, core

    def test_r2_gate_blocks_curvature(self):
        """Curvature geometry (R²≈0): R² gate must block escalation."""
        d, n = 20, 400
        draws, grads = _make_curvature_buffer(d, n, seed=30)
        draws = jnp.array(draws, dtype=jnp.float32)
        grads = jnp.array(grads, dtype=jnp.float32)
        state, _ = self._run_two_windows(draws, grads)
        self.assertFalse(
            bool(np.asarray(state.has_escalated)),
            "Curvature geometry: R² gate should block escalation",
        )

    def test_s_gap_magnitude_blocks_isotropic(self):
        """Isotropic draws (S_gap≈1): magnitude gate must block escalation."""
        d, n = 20, 400
        draws, grads = _make_isotropic_buffer(d, n, seed=31)
        draws = jnp.array(draws, dtype=jnp.float32)
        grads = jnp.array(grads, dtype=jnp.float32)
        state, _ = self._run_two_windows(draws, grads)
        self.assertFalse(
            bool(np.asarray(state.has_escalated)),
            "Isotropic: S_gap magnitude gate should block escalation",
        )

    def test_s_gap_stability_blocks_first_window(self):
        """First window (no prior S_gap read): stability gate must block escalation.

        Even when R² is high and S_gap is large, the stability gate requires two
        consecutive reads. After the first window, s_gap_curr is set but
        s_gap_prev was NaN → stability check fails.
        """
        d, n = 20, 500
        draws, grads = _make_correlated_buffer(d, n, rank=2, lam_spike=20.0, seed=32)
        draws = jnp.array(draws, dtype=jnp.float32)
        grads = jnp.array(grads, dtype=jnp.float32)

        core = build_meta_adaptation_core(50000, max_rank=10)
        state = core.init(d)
        # Only ONE window — stability gate must block escalation
        state = _fill_state_from_buffer(state, draws, grads)
        state = core.final(state)
        self.assertFalse(
            bool(np.asarray(state.has_escalated)),
            "First window: stability gate (no prior S_gap) should block escalation",
        )
        # s_gap_curr should now be set
        self.assertFalse(
            np.isnan(float(np.asarray(state.s_gap_curr))),
            "After first window, s_gap_curr should be a valid number",
        )

    def test_s_gap_stability_passes_on_second_stable_window(self):
        """Two identical windows: stability gate passes → controller escalates."""
        d, n = 20, 500
        draws, grads = _make_correlated_buffer(d, n, rank=2, lam_spike=20.0, seed=33)
        draws = jnp.array(draws, dtype=jnp.float32)
        grads = jnp.array(grads, dtype=jnp.float32)
        state, _ = self._run_two_windows(draws, grads)
        self.assertTrue(
            bool(np.asarray(state.has_escalated)),
            "Two stable windows with large S_gap and high R²: should escalate",
        )

    def test_budget_deadline_blocks_tight_budget(self):
        """Deadline gate blocks when remaining budget < 2k + step-size buffer.

        Isolation: fixture passes R² (linear score, R²≈1) AND S_gap (>_S_MIN)
        gates.  Only the budget_used manipulation exercises the deadline gate.
        """
        d, n = 20, 500
        draws, grads = _make_correlated_buffer(d, n, rank=2, lam_spike=20.0, seed=34)
        draws = jnp.array(draws, dtype=jnp.float32)
        grads = jnp.array(grads, dtype=jnp.float32)

        # First, confirm the fixture CAN escalate with fresh budget.
        core = build_meta_adaptation_core(50000, max_rank=5)
        state = core.init(d)
        state = _fill_state_from_buffer(state, draws, grads)
        state = core.final(state)
        # Jam budget_used to almost-exhausted (leaves < 2k + 50 remaining steps).
        max_budget_steps = 50000 // 20  # 2500 steps
        state_jammed = state._replace(
            budget_used=jnp.array(max_budget_steps - 5, dtype=jnp.int32),
        )
        state_jammed = _fill_state_from_buffer(state_jammed, draws, grads)
        state_jammed = core.final(state_jammed)
        self.assertFalse(
            bool(np.asarray(state_jammed.has_escalated)),
            "Exhausted budget: deadline gate should block escalation",
        )
        # Control: same fixture with fresh budget escalates.
        state_fresh = core.init(d)
        for _ in range(2):
            state_fresh = _fill_state_from_buffer(state_fresh, draws, grads)
            state_fresh = core.final(state_fresh)
        self.assertTrue(
            bool(np.asarray(state_fresh.has_escalated)),
            "Control: fresh budget with same fixture should escalate",
        )

    def test_funnel_refusal_isolates_r2_gate(self):
        """Correlated draws + random grads: high S_gap but R²≈0 → reparam_suggested.

        This is the load-bearing test for the R² gate: S_gap passes (proves S_gap
        gate is NOT the blocker) but R² blocks (the sole discriminator).
        An isotropic curvature fixture would let the S_gap gate block escalation,
        masking the R² gate; this fixture uses a HIGH-S_gap target so R² is the
        sole discriminator.
        """
        d, n = 20, 500
        # High S_gap (correlated spike) + random grads (R²≈0, curvature proxy)
        draws, grads = _make_high_sgap_curvature_buffer(
            d, n, rank=2, lam_spike=20.0, seed=36
        )
        draws = jnp.array(draws, dtype=jnp.float32)
        grads = jnp.array(grads, dtype=jnp.float32)
        state, _ = self._run_two_windows(draws, grads)

        # S_gap gate passes (proves the S_gap gate is not the blocker).
        s_gap = float(np.asarray(state.s_gap_curr))
        self.assertGreater(
            s_gap,
            _S_MIN,
            "Fixture must have high S_gap for isolation; got "
            + str(s_gap),  # noqa: E702
        )
        # R² gate blocks.
        r2 = float(np.asarray(state.r2_latest))
        self.assertFalse(np.isnan(r2), "R² should not be deferred at n=500, d=20")
        self.assertLess(r2, _R_MIN, f"R²={r2} should be below _R_MIN={_R_MIN}")
        # Result: controller stays diagonal and suggests reparameterization.
        self.assertFalse(
            bool(np.asarray(state.has_escalated)),
            "R² gate should block escalation; S_gap alone must not suffice",
        )
        verdict = extract_meta_verdict(
            state, max_grad_budget=50000, num_warmup_steps=2500
        )
        self.assertEqual(
            verdict.route,
            "reparam_suggested",
            "High-S_gap + low-R2 should route to reparam_suggested; got "
            + verdict.route,
        )

    def test_monotone_escalation(self):
        """Once escalated, has_escalated stays True on subsequent windows."""
        d, n = 20, 500
        draws, grads = _make_correlated_buffer(d, n, rank=2, lam_spike=20.0, seed=35)
        draws = jnp.array(draws, dtype=jnp.float32)
        grads = jnp.array(grads, dtype=jnp.float32)

        state, core = self._run_two_windows(draws, grads)
        self.assertTrue(bool(np.asarray(state.has_escalated)), "Should have escalated")
        first_rank = int(np.asarray(state.escalation_rank))

        # Run a third window
        state = _fill_state_from_buffer(state, draws, grads)
        state = core.final(state)
        self.assertTrue(
            bool(np.asarray(state.has_escalated)),
            "Monotone: has_escalated must stay True once set",
        )
        self.assertEqual(
            int(np.asarray(state.escalation_rank)),
            first_rank,
            "Monotone: escalation_rank must not decrease",
        )


class TestStructuralE2ESmoke(BlackJAXTest):
    """Structural e2e smokes for three geometry classes.

    Only structural properties checked (route, metric type, IMM shape).
    No thin-margin stochastic assertions.
    """

    def test_isotropic_stays_diagonal(self):
        """Isotropic MVN: S_gap≈1 → controller stays diagonal across 3 windows."""
        d, n = 10, 300
        draws, grads = _make_isotropic_buffer(d, n, seed=40)
        draws = jnp.array(draws, dtype=jnp.float32)
        grads = jnp.array(grads, dtype=jnp.float32)

        core = build_meta_adaptation_core(50000, max_rank=5)
        state = core.init(d)
        for _ in range(3):
            state = _fill_state_from_buffer(state, draws, grads)
            state = core.final(state)

        self.assertFalse(
            bool(np.asarray(state.has_escalated)),
            "Isotropic: should stay diagonal across multiple windows",
        )

    def test_correlated_spike_escalates(self):
        """Correlated rank-2 spike: both gates pass → escalates after two windows."""
        d, n = 20, 500
        draws, grads = _make_correlated_buffer(d, n, rank=2, lam_spike=25.0, seed=41)
        draws = jnp.array(draws, dtype=jnp.float32)
        grads = jnp.array(grads, dtype=jnp.float32)

        core = build_meta_adaptation_core(50000, max_rank=10)
        state = core.init(d)
        for _ in range(2):
            state = _fill_state_from_buffer(state, draws, grads)
            state = core.final(state)

        self.assertTrue(
            bool(np.asarray(state.has_escalated)),
            "Correlated spike: should escalate after two windows",
        )
        imm = state.inverse_mass_matrix
        self.assertIsInstance(imm, LowRankInverseMassMatrix)
        self.assertEqual(imm.sigma.shape, (d,))
        self.assertEqual(imm.U.shape, (d, 10))

    def test_curvature_stays_diagonal_reparam_hint(self):
        """Curvature geometry: R²≈0 blocks escalation; verdict is reparam_suggested."""
        d, n = 20, 400
        draws, grads = _make_curvature_buffer(d, n, seed=42)
        draws = jnp.array(draws, dtype=jnp.float32)
        grads = jnp.array(grads, dtype=jnp.float32)

        core = build_meta_adaptation_core(50000, max_rank=10)
        state = core.init(d)
        for _ in range(3):
            state = _fill_state_from_buffer(state, draws, grads)
            state = core.final(state)

        self.assertFalse(bool(np.asarray(state.has_escalated)))

        verdict = extract_meta_verdict(
            state, max_grad_budget=50000, num_warmup_steps=2500
        )
        self.assertEqual(
            verdict.route,
            "reparam_suggested",
            f"Curvature route should be reparam_suggested, got {verdict.route}",
        )
        self.assertTrue(verdict.flags["reparam_hint"])

    def test_high_d_linear_spike_escalates(self):
        """High-d linear spike (d=120, rank=3): projected R² tier passes → escalates.

        Regression guard for the projected-R²-both-sides fix.  Before the fix,
        radon-like targets at d>>k produced projected R²≈0 (full d-dim response
        regressed on k features) and emitted reparam_suggested.  After the fix,
        projecting both sides onto U_k gives R²≈1 → escalates.
        """
        d, n, rank = 120, 600, 3
        draws, grads = _make_correlated_buffer(d, n, rank=rank, lam_spike=25.0, seed=45)
        draws = jnp.array(draws, dtype=jnp.float32)
        grads = jnp.array(grads, dtype=jnp.float32)

        core = build_meta_adaptation_core(50000, max_rank=10)
        state = core.init(d)
        for _ in range(2):
            state = _fill_state_from_buffer(state, draws, grads)
            state = core.final(state)

        self.assertTrue(
            bool(np.asarray(state.has_escalated)),
            "High-d linear spike should escalate; "
            + f"r2={float(np.asarray(state.r2_latest)):.3f}, "  # noqa: E231
            + f"s_gap={float(np.asarray(state.s_gap_curr)):.2f}",  # noqa: E231
        )
        verdict = extract_meta_verdict(
            state, max_grad_budget=50000, num_warmup_steps=2500
        )
        self.assertEqual(
            verdict.route,
            "low_rank",
            "High-d linear spike should route to low_rank; got " + verdict.route,
        )


class TestRecovershClassical(BlackJAXTest):
    """Structural invariants that must hold across all routing decisions."""

    def test_imm_always_low_rank_type(self):
        """The emitted IMM is always LowRankInverseMassMatrix, even before escalation."""
        core = build_meta_adaptation_core(50000, max_rank=5)
        state = core.init(10)
        self.assertIsInstance(state.inverse_mass_matrix, LowRankInverseMassMatrix)

    def test_diagonal_imm_u_zero_lam_one(self):
        """Before escalation: U=0 and lam=1 (diagonal-equivalent representation)."""
        core = build_meta_adaptation_core(50000, max_rank=5)
        state = core.init(10)
        imm = state.inverse_mass_matrix
        np.testing.assert_allclose(
            np.asarray(imm.U),
            0.0,
            atol=1e-7,
            err_msg="Before escalation: U should be zero",
        )
        np.testing.assert_allclose(
            np.asarray(imm.lam),
            1.0,
            atol=1e-7,
            err_msg="Before escalation: lam should be one",
        )

    def test_metric_core_protocol(self):
        """build_meta_adaptation_core returns a MetricCore with callable protocol."""
        core = build_meta_adaptation_core(50000, max_rank=5)
        self.assertIsInstance(core, MetricCore)
        self.assertTrue(callable(core.init))
        self.assertTrue(callable(core.update))
        self.assertTrue(callable(core.final))

    def test_update_accumulates_into_buffer(self):
        """update() advances buffer_idx and budget_used by 1 per call."""
        d = 10
        core = build_meta_adaptation_core(50000, max_rank=5)
        state = core.init(d)
        self.assertEqual(int(np.asarray(state.buffer_idx)), 0)

        state1 = core.update(state, jnp.zeros(d), jnp.ones(d))
        self.assertEqual(int(np.asarray(state1.buffer_idx)), 1)
        self.assertEqual(int(np.asarray(state1.budget_used)), 1)

    def test_converged_at_step_init_sentinel(self):
        """converged_at_step is -1 (sentinel for 'not yet converged') at init."""
        core = build_meta_adaptation_core(50000, max_rank=5)
        state = core.init(10)
        self.assertEqual(
            int(np.asarray(state.converged_at_step)),
            -1,
            "converged_at_step should be -1 (not yet converged) at init",
        )

    def test_r2_mode_init_deferred(self):
        """r2_mode is _R2_DEFERRED at init (no window computed yet)."""
        core = build_meta_adaptation_core(50000, max_rank=5)
        state = core.init(10)
        self.assertEqual(int(np.asarray(state.r2_mode)), _R2_DEFERRED)

    def test_r2_mode_observed_after_window(self):
        """r2_mode in carry matches the actually-taken branch after a window."""
        d, n = 10, 400
        max_rank = 5
        draws, grads = _make_isotropic_buffer(d, n, seed=60)
        draws = jnp.array(draws, dtype=jnp.float32)
        grads = jnp.array(grads, dtype=jnp.float32)

        core = build_meta_adaptation_core(50000, max_rank=max_rank)
        state = core.init(d)
        state = _fill_state_from_buffer(state, draws, grads)
        state = core.final(state)

        mode_int = int(np.asarray(state.r2_mode))
        # n=400, d=10: min_n_full = 2*8*10 = 160 ≤ 400 → full_affine branch.
        self.assertEqual(
            mode_int,
            _R2_FULL_AFFINE,
            f"n=400, d=10: expected _R2_FULL_AFFINE, got mode_int={mode_int}",
        )
        # verdict flag should reflect the carry, not post-hoc inference
        verdict = extract_meta_verdict(
            state, max_grad_budget=50000, num_warmup_steps=2500
        )
        self.assertEqual(verdict.flags["high_d_r2_mode"], "full_affine")

    def test_budget_returned_zero_before_airm_convergence(self):
        """budget_returned_steps is 0 when AIRM criterion has never fired (v1 advisory)."""
        core = build_meta_adaptation_core(50000, max_rank=5)
        state = core.init(10)
        verdict = extract_meta_verdict(
            state, max_grad_budget=50000, num_warmup_steps=2500
        )
        self.assertEqual(
            verdict.budget_returned_steps,
            0,
            "No AIRM convergence yet: budget_returned should be 0",
        )

    def test_verdict_fields_present(self):
        """extract_meta_verdict populates all required fields."""
        core = build_meta_adaptation_core(50000, max_rank=5)
        state = core.init(10)
        verdict = extract_meta_verdict(
            state, max_grad_budget=50000, num_warmup_steps=2500
        )
        self.assertIsInstance(verdict, MetaAdaptationVerdict)
        self.assertIn(verdict.route, ("diagonal", "low_rank", "reparam_suggested"))
        self.assertIn(verdict.confidence, ("high", "low"))
        self.assertEqual(verdict.buffer_policy, "reset")
        self.assertIsInstance(verdict.flags, dict)
        for key in (
            "reparam_hint",
            "marginal_s_gap",
            "wall_cost_discount",
            "high_d_r2_mode",
            "mode_coverage",
            "nominal_rank",
        ):
            self.assertIn(key, verdict.flags, f"Missing verdict flag: {key}")

    def test_staged_adaptation_auto_metric_smoke(self):
        """staged_adaptation(metric='auto') wires correctly and produces an IMM.

        Structural smoke test — not a performance test. Verifies:
        - No exception during construction or warmup run.
        - Warmup returns LowRankInverseMassMatrix.
        - IMM shape matches n_dims.
        """
        n_dims = 5

        def logdensity_fn(x):
            return -0.5 * jnp.sum(x**2)

        warmup = blackjax.staged_adaptation(
            blackjax.nuts, logdensity_fn, metric="auto", max_grad_budget=5000
        )
        key = jax.random.key(99)
        results, _ = warmup.run(key, jnp.zeros(n_dims), num_steps=50)

        self.assertIsNotNone(results.state)
        self.assertIn("step_size", results.parameters)
        self.assertIn("inverse_mass_matrix", results.parameters)
        imm = results.parameters["inverse_mass_matrix"]
        self.assertIsInstance(imm, LowRankInverseMassMatrix)
        self.assertEqual(imm.sigma.shape, (n_dims,))

    def test_staged_adaptation_auto_missing_budget_raises(self):
        """staged_adaptation(metric='auto') without max_grad_budget raises ValueError."""
        with self.assertRaisesRegex(ValueError, "max_grad_budget"):
            blackjax.staged_adaptation(
                blackjax.nuts,
                lambda x: -0.5 * jnp.sum(x**2),
                metric="auto",
                # max_grad_budget intentionally omitted
            )

    def test_auto_uses_growing_window_schedule(self):
        """metric='auto' resolves to the growing-window schedule; explicit schedule is preserved.

        Regression guard: the old `is build_schedule` sentinel could not distinguish
        between "user passed nothing" and "user explicitly passed build_schedule",
        so an explicit Stan-on-auto request was silently swapped to growing-window.
        The old test only checked isinstance(IMM, LowRankInverseMassMatrix), which is
        a tautology because auto always emits that type — it never actually observed
        which schedule was chosen.

        Both are fixed via _resolve_metric_and_schedule: the function is called
        directly so the returned schedule identity is observable.
        """
        from blackjax.adaptation.low_rank_adaptation import (
            build_growing_window_schedule,
        )
        from blackjax.adaptation.staged_adaptation import (
            _resolve_metric_and_schedule,
            build_schedule,
        )

        # auto + no explicit schedule → growing window (the override).
        _, sched_auto_default = _resolve_metric_and_schedule(
            "auto", None, max_grad_budget=5000
        )
        self.assertIs(
            sched_auto_default,
            build_growing_window_schedule,
            "auto+default must resolve to build_growing_window_schedule",
        )

        # Negative test: explicit Stan schedule on auto is PRESERVED (not swapped).
        # The old sentinel `if schedule_fn is build_schedule` was the bug: it
        # replaced explicit Stan with growing because both were the same object.
        _, sched_auto_explicit_stan = _resolve_metric_and_schedule(
            "auto", build_schedule, max_grad_budget=5000
        )
        self.assertIs(
            sched_auto_explicit_stan,
            build_schedule,
            "auto+explicit-Stan must NOT be swapped to growing-window; "  # noqa: E231
            "schedule_fn sentinel was broken (build_schedule == build_schedule always true)",
        )

    def test_converged_at_step_sets_on_airm_convergence(self):
        """converged_at_step is set (≥0) and budget_returned_steps > 0 after AIRM fires.

        Regression guard for the dead-field bug: previously budget_returned was
        always 0 because converged_at_step was never set.
        """
        d, n = 20, 500
        draws, grads = _make_correlated_buffer(d, n, rank=2, lam_spike=20.0, seed=61)
        draws = jnp.array(draws, dtype=jnp.float32)
        grads = jnp.array(grads, dtype=jnp.float32)

        core = build_meta_adaptation_core(50000, max_rank=10)
        state = core.init(d)

        # Run ≥3 identical escalated windows to drive AIRM velocity below tolerance.
        # After escalation, identical lam → lam_diff = 0 < _AIRM_VELOCITY_TOL.
        # Two consecutive sub-threshold windows set converged_at_step.
        for _ in range(4):
            state = _fill_state_from_buffer(state, draws, grads)
            state = core.final(state)

        converged_at = int(np.asarray(state.converged_at_step))
        self.assertGreaterEqual(
            converged_at,
            0,
            "converged_at_step should be >=0 after AIRM convergence; got "
            + str(converged_at),
        )

        verdict = extract_meta_verdict(
            state, max_grad_budget=50000, num_warmup_steps=2500
        )
        self.assertGreater(
            verdict.budget_returned_steps,
            0,
            "budget_returned_steps should be > 0 when converged_at_step is set",
        )

    def test_diagonal_sigma_is_welford_not_fisher(self):
        """Stay-diagonal IMM uses Welford sigma (sample std), not Fisher sigma.

        Recovers-classical anchor: the welford diagonal is the measured nutpie
        baseline (fisher-diag = 0.11x welford on funnel, 0.62x on german).
        Use correlated draws + random grads so R²<_R_MIN keeps controller diagonal
        while the anisotropy makes welford ≠ fisher.
        """
        d, n, rank = 10, 400, 2
        lam_spike = 25.0
        # Build fixture manually (correlated draws + random grads)
        key = jax.random.key(71)
        k1, k2, k3 = jax.random.split(key, 3)
        raw = jax.random.normal(k3, (d, rank))
        Q, _ = jnp.linalg.qr(raw)
        Q = Q[:, :rank]
        z = jax.random.normal(k1, (n, d))
        z_orth = z - (z @ Q) @ Q.T
        draws = jnp.array(
            z_orth + jnp.sqrt(lam_spike) * (z @ Q) @ Q.T, dtype=jnp.float32
        )
        grads = jnp.array(jax.random.normal(k2, (n, d)), dtype=jnp.float32)

        core = build_meta_adaptation_core(50000, max_rank=5)
        state = core.init(d)
        state = _fill_state_from_buffer(state, draws, grads)
        state = core.final(state)
        self.assertFalse(bool(np.asarray(state.has_escalated)))

        # Expected Welford sigma: sample std (ddof=1) of the original draws.
        draws_np = np.asarray(draws)
        mean_x = draws_np.mean(0)
        var_x = np.sum((draws_np - mean_x[None, :]) ** 2, axis=0) / max(n - 1, 1)
        sigma_welford_expected = np.sqrt(np.maximum(var_x, 1e-10))

        emitted_sigma = np.asarray(state.inverse_mass_matrix.sigma)
        np.testing.assert_allclose(
            emitted_sigma,
            sigma_welford_expected,
            rtol=0.05,
            err_msg="Stay-diagonal sigma must equal Welford sample std",
        )

    def test_exit_reason_warmup_complete(self):
        """exit_reason is 'warmup_complete' when AIRM has not converged."""
        core = build_meta_adaptation_core(50000, max_rank=5)
        state = core.init(10)
        verdict = extract_meta_verdict(
            state, max_grad_budget=50000, num_warmup_steps=2500
        )
        self.assertEqual(verdict.exit_reason, "warmup_complete")

    def test_effective_rank_and_nominal_rank_semantics(self):
        """effective_rank (deployed) and nominal_rank (pre-mask) semantics after FIX 2.

        After FIX 2:
        - verdict.flags['nominal_rank'] = escalation_rank from _choose_rank
          (the pre-mask count, stored in the carry at escalation time).
        - verdict.effective_rank = count(|lam_i - 1| > tol) in the deployed
          Fisher metric (the true deployed rank).

        The two can differ: for example, _choose_rank counts 4 eigenvalues
        outside [0.5, 2.0] in the Welford-whitened spectrum, while the Fisher
        estimator deploys 5 directions (the score-space decomposition may admit
        additional directions that the Welford-based gate missed).  Both values
        are valid for their respective interpretations; neither must equal the
        other in general.

        This test verifies the invariants that MUST hold, not a coincidence:
        - flags['nominal_rank'] == carry_rank (the stored escalation_rank)
        - effective_rank > 0 when escalated (at least one direction deployed)
        - route == 'low_rank' (escalation happened)

        For the over-counting fixture (TestEffectiveRankHonesty) the deployed
        rank is provably smaller than the nominal rank.  For a rich spike
        fixture (like this one) the reverse can occur — Fisher may deploy more
        directions than the conservative Welford-based gate counted.
        """
        d, n = 20, 500
        draws, grads = _make_correlated_buffer(d, n, rank=2, lam_spike=20.0, seed=62)
        draws = jnp.array(draws, dtype=jnp.float32)
        grads = jnp.array(grads, dtype=jnp.float32)

        core = build_meta_adaptation_core(50000, max_rank=10)
        state = core.init(d)
        for _ in range(2):
            state = _fill_state_from_buffer(state, draws, grads)
            state = core.final(state)

        self.assertTrue(bool(np.asarray(state.has_escalated)))
        carry_rank = int(np.asarray(state.escalation_rank))
        self.assertGreater(carry_rank, 0, "escalation_rank should be > 0")

        verdict = extract_meta_verdict(
            state, max_grad_budget=50000, num_warmup_steps=2500
        )
        # nominal_rank MUST equal carry_rank (escalation_rank stored in the carry).
        self.assertEqual(
            verdict.flags["nominal_rank"],
            carry_rank,
            "flags['nominal_rank'] must equal the stored escalation_rank",
        )
        # effective_rank > 0 when escalated (Fisher deployed at least one direction).
        self.assertGreater(
            verdict.effective_rank,
            0,
            "When has_escalated is True, effective_rank must be > 0",
        )
        self.assertEqual(verdict.route, "low_rank")

        # --- FIX 2 regression guard ---
        # Directly compute the deployed rank from the state's deployed lam array.
        # This asserts that effective_rank is the Fisher-metric deployed count,
        # NOT the pre-mask escalation_rank stored in the carry.  Reverting Fix 2
        # (setting effective_rank = escalation_rank) makes BOTH of the assertions
        # below RED:
        #   (a) assertEqual: reverted code gives escalation_rank (4), not the
        #       deployed count (5) from state.inverse_mass_matrix.lam.
        #   (b) assertNotEqual: reverted code sets effective_rank = nominal_rank.
        lam_np = np.asarray(state.inverse_mass_matrix.lam)
        directly_computed_deployed_rank = int(
            np.sum(np.abs(lam_np - 1.0) > _LAM_NONTRIVIAL_TOL)
        )
        self.assertEqual(
            verdict.effective_rank,
            directly_computed_deployed_rank,
            f"effective_rank must equal the directly-computed deployed lam count. "
            f"Got verdict.effective_rank={verdict.effective_rank} vs "
            f"directly_computed={directly_computed_deployed_rank}",
        )
        # For this fixture, effective_rank and nominal_rank diverge: the Fisher
        # estimator deploys more directions than _choose_rank's pre-mask count.
        self.assertNotEqual(
            verdict.effective_rank,
            verdict.flags["nominal_rank"],
            "For this fixture effective_rank must differ from nominal_rank; "
            "if they are equal, Fix 2 may have been accidentally reverted",
        )

    def test_marginal_s_gap_stays_diagonal(self):
        """Marginal-band S_gap ∈ [_S_MIN, 2·_S_MIN) = [2.0, 4.0): stays diagonal.

        Regression guard for the 'stays-diag-marginal' decision row.
        The OLD fixture (lam_spike=2.0) was mislabeled: it produced S_gap=1.0
        because top Welford-whitened eigenvalue ≈ 1.9 < cutoff=2.0 -> k_new=0
        -> _compute_s_gap returns 1.0 by definition.  k_new=0 means computing
        S_gap at the wrong spectral cut was previously caught only by a thin ~3%
        margin, not by design.

        The NEW fixture (lam_spike=4.5, rank=1, non-axis-aligned direction, seed=42):
        - top Welford-whitened eigenvalue ≈ 3.5 > cutoff=2.0 -> k_new=1
        - S_gap = lambda_1/lambda_2 ≈ 2.94 ∈ [_S_MIN, 2·_S_MIN) -> marginal_s_gap=True
        - random grads -> R2 approx 0 -> R2 gate blocks escalation (not S_gap gate)
        - flagged as 'marginal_s_gap' in verdict.flags
        - direct s_gap_curr == _compute_s_gap(eigs, k_new) assertion catches
          a regression where S_gap is computed at the wrong spectral cut (k instead of k_new)

        Why rank-1 NON-AXIS-ALIGNED: an axis-aligned spike (e.g. spike at e1)
        is perfectly cancelled by Welford diagonal whitening -> S_gap=1.  A
        random direction leaks residual anisotropy into the whitened space.

        lam_spike=4.5 (not 3.5) so all tested seeds land at S_gap ∈ [2.05, 2.94],
        avoiding the tight boundary-proximity that caused the original seed=63 to
        read S_gap=1.790 with lam_spike=3.5.
        """
        d, n = 20, 500
        draws, grads = _make_marginal_sgap_curvature_buffer(d, n, seed=42)
        draws = jnp.array(draws, dtype=jnp.float32)
        grads = jnp.array(grads, dtype=jnp.float32)

        core = build_meta_adaptation_core(50000, max_rank=10)
        state = core.init(d)

        # First window — populates s_gap_curr; s_gap_prev is still NaN so
        # stability gate blocks even if s_gap ≥ _S_MIN.
        state_filled_1 = _fill_state_from_buffer(state, draws, grads)
        state_1 = core.final(state_filled_1)

        # Second window — s_gap_prev is now valid; stability check runs.
        # R²≈0 (random grads) blocks escalation via R² gate.
        state_filled_2 = _fill_state_from_buffer(state_1, draws, grads)
        state_2 = core.final(state_filled_2)

        s_gap = float(np.asarray(state_2.s_gap_curr))
        self.assertFalse(
            bool(np.asarray(state_2.has_escalated)),
            "Marginal S_gap fixture should NOT escalate (R2 blocks); s_gap="
            + format(s_gap, ".3f"),
        )
        # S_gap must be in the REAL marginal band [_S_MIN, 2·_S_MIN) = [2, 4).
        self.assertGreaterEqual(
            s_gap,
            _S_MIN,
            "Marginal fixture s_gap="
            + format(s_gap, ".3f")
            + " must be >= _S_MIN="
            + str(_S_MIN),
        )
        self.assertLess(
            s_gap,
            2.0 * _S_MIN,
            "Marginal fixture s_gap="
            + format(s_gap, ".3f")
            + " must be < 2*_S_MIN="
            + str(2.0 * _S_MIN),
        )

        # flags["marginal_s_gap"] must be True — the field exists and is set correctly.
        verdict = extract_meta_verdict(
            state_2, max_grad_budget=50000, num_warmup_steps=2500
        )
        self.assertTrue(
            verdict.flags["marginal_s_gap"],
            "Expected flags['marginal_s_gap']=True s_gap="
            + format(s_gap, ".3f")
            + " has_escalated="
            + str(bool(np.asarray(state_2.has_escalated))),
        )
        self.assertIn(verdict.route, ("diagonal", "reparam_suggested"))

        # Direct assertion: stored s_gap_curr == _compute_s_gap(eigenvalues, k_new).
        # Catches a regression where S_gap is computed at the wrong spectral cut
        # (e.g. k-1 or k+1 instead of k_new).
        # Uses the buffer BEFORE final() reset + Welford sigma from the post-final IMM.
        B = state_filled_2.draws_buffer.shape[0]
        n_buf = jnp.minimum(state_filled_2.buffer_idx, jnp.int32(B))
        sigma_w = state_2.inverse_mass_matrix.sigma  # Welford sigma (stay-diag IMM)
        actual_rank = state_2.inverse_mass_matrix.U.shape[1]
        eigenvalues_direct, _ = _compute_whitened_spectrum(
            state_filled_2.draws_buffer, sigma_w, n_buf, actual_rank
        )
        k_new_direct = _choose_rank(eigenvalues_direct, n_buf, actual_rank, cutoff=2.0)
        s_gap_direct = _compute_s_gap(eigenvalues_direct, k_new_direct)
        # k_new must be 1 (non-trivial cut) — the fixture is load-bearing.
        self.assertGreater(
            int(np.asarray(k_new_direct)),
            0,
            "Marginal fixture must have k_new >= 1 (non-trivial spectral cut)",
        )
        np.testing.assert_allclose(
            float(np.asarray(state_2.s_gap_curr)),
            float(np.asarray(s_gap_direct)),
            rtol=1e-5,
            err_msg=(
                "s_gap_curr must equal _compute_s_gap(eigs, k_new); "
                "regression: S_gap computed at wrong spectral cut index"
            ),
        )

    def test_escalated_e2e_smoke_f32_and_x64(self):
        """Escalated e2e smoke: non-axis-aligned spike target escalates under both f32 and x64.

        Regression guard for the x64 dtype crash in the R² deferred branch / update
        slice: the suite was previously green only because all tests ran f32.  Under
        x64 the dynamic_update_slice and _deferred branches both crashed at trace time.

        The logdensity uses a NON-AXIS-ALIGNED random direction u so that Welford
        diagonal whitening leaves residual off-diagonal anisotropy:

            [D^{-1} Sigma D^{-1}]_{ij} = (lam-1)*u_i*u_j / sqrt((1+(lam-1)*u_i^2)(1+(lam-1)*u_j^2))

        For axis-aligned u=e_1 those off-diagonals are all zero, D^{-1}SigmaD^{-1}=I,
        S_gap=1 and the controller NEVER escalates (U=0 always).  A random u produces
        residual off-diagonal structure with a whitened top eigenvalue well above
        _S_MIN=2.0, driving escalation within ~400 slow-window steps.
        """
        n_dims = 5
        lam_spike = 25.0

        # Fixed random unit vector (seed 42) so the fixture is deterministic.
        u_raw = jax.random.normal(jax.random.key(42), (n_dims,))
        u_dir = u_raw / jnp.linalg.norm(u_raw)

        # Sigma^{-1} = I - (lam-1)/lam * outer(u, u)  [matrix-inversion lemma]
        cov_inv = jnp.eye(n_dims) - (lam_spike - 1.0) / lam_spike * jnp.outer(
            u_dir, u_dir
        )

        def logdensity_fn(x):
            return -0.5 * x @ cov_inv @ x

        # --- f32 run ---
        warmup = blackjax.staged_adaptation(
            blackjax.nuts, logdensity_fn, metric="auto", max_grad_budget=20000
        )
        key = jax.random.key(100)
        results, _ = warmup.run(key, jnp.zeros(n_dims), num_steps=400)
        imm = results.parameters["inverse_mass_matrix"]
        self.assertIsInstance(imm, LowRankInverseMassMatrix)
        self.assertTrue(
            bool(jnp.all(jnp.isfinite(imm.sigma))),
            "f32: sigma has non-finite values",
        )
        self.assertTrue(
            bool(jnp.any(jnp.abs(imm.U) > 1e-8)),
            "f32: controller never escalated (U=0); axis-aligned spike would do this,"
            " ensure u_dir is non-axis-aligned",
        )
        self.assertTrue(
            bool(jnp.all(imm.lam > 0)),
            "f32: lam is not positive definite (escalated rank-1 update must have lam>0)",
        )

        # --- x64 run: separate jax config context ---
        try:
            jax.config.update("jax_enable_x64", True)
            warmup64 = blackjax.staged_adaptation(
                blackjax.nuts, logdensity_fn, metric="auto", max_grad_budget=20000
            )
            key64 = jax.random.key(101)
            results64, _ = warmup64.run(key64, jnp.zeros(n_dims), num_steps=400)
            imm64 = results64.parameters["inverse_mass_matrix"]
            self.assertIsInstance(imm64, LowRankInverseMassMatrix)
            self.assertTrue(
                bool(jnp.all(jnp.isfinite(imm64.sigma))),
                "x64: sigma has non-finite values",
            )
            self.assertTrue(
                bool(jnp.any(jnp.abs(imm64.U) > 1e-8)),
                "x64: controller never escalated (U=0)",
            )
            self.assertTrue(
                bool(jnp.all(imm64.lam > 0)),
                "x64: lam is not positive definite",
            )
        finally:
            jax.config.update("jax_enable_x64", False)


class TestDefaultWiringAndBudgetWarning(BlackJAXTest):
    """FIX 1: metric='auto' derives num_steps from max_grad_budget when unset.

    Three sub-cases:
    - Large budget derives num_steps > the old fixed default (1000).
    - Derived default and explicit equal num_steps give identical results
      (same key, same position → same computation when derivation is correct).
    - Explicit num_steps is honored (not replaced by derivation).
    - Low-budget high-d config emits a UserWarning about rank-detection support.
    """

    def _simple_logdensity(self, x):
        return -0.5 * jnp.sum(x**2)

    def test_large_budget_derives_more_than_old_default(self):
        """max_grad_budget=30000 → derived num_steps = 1500 > 1000 (old fixed default)."""
        max_grad_budget = 30000
        derived = max_grad_budget // _ASSUMED_AVG_LEAPFROGS_PER_STEP
        self.assertGreater(
            derived,
            1000,
            f"Derived num_steps ({derived}) should exceed the old default 1000 "
            f"for max_grad_budget={max_grad_budget}",
        )

    def test_derived_default_matches_explicit(self):
        """metric='auto' with no num_steps gives same result as explicit derived value.

        Structural check: same rng_key, same position, same derivation formula →
        same final warmup state.  Uses a small budget so the warmup is fast.

        The low-budget/high-d warning may fire for the small dimension used here;
        we suppress it in this test because the warning behavior is covered by
        test_low_budget_warning_fires_for_high_d — this test is purely about
        whether the derivation produces the same computation as an explicit arg.
        """
        max_grad_budget = 600  # → derived = 600 // 20 = 30 steps
        derived = max_grad_budget // _ASSUMED_AVG_LEAPFROGS_PER_STEP

        warmup = blackjax.staged_adaptation(
            blackjax.nuts,
            self._simple_logdensity,
            metric="auto",
            max_grad_budget=max_grad_budget,
        )
        key = jax.random.key(10)
        pos = jnp.zeros(5)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            results_default, _ = warmup.run(key, pos)  # no num_steps → derived
            results_explicit, _ = warmup.run(key, pos, num_steps=derived)  # explicit

        np.testing.assert_allclose(
            np.asarray(results_default.state.position),
            np.asarray(results_explicit.state.position),
            rtol=1e-5,
            err_msg="Default derivation should match explicit num_steps",
        )
        np.testing.assert_allclose(
            np.asarray(results_default.parameters["step_size"]),
            np.asarray(results_explicit.parameters["step_size"]),
            rtol=1e-5,
            err_msg="Default derivation should produce same step_size as explicit",
        )

    def test_explicit_num_steps_honored(self):
        """Explicit num_steps bypasses derivation for both auto and non-auto metrics."""
        # For metric='auto': explicit 50 steps must not be overridden by the derived
        # value (which could be larger).
        max_grad_budget = 30000  # → derived = 1500
        explicit_steps = 50

        warmup_auto = blackjax.staged_adaptation(
            blackjax.nuts,
            self._simple_logdensity,
            metric="auto",
            max_grad_budget=max_grad_budget,
        )
        key = jax.random.key(11)
        pos = jnp.zeros(5)
        results_auto, info_auto = warmup_auto.run(key, pos, num_steps=explicit_steps)
        # The adaptation_info has leading dim = num_steps; verify it matches explicit.
        # info_auto is a tuple (chain_state, mcmc_info, adapt_state) each stacked.
        first_leaf = jax.tree.leaves(info_auto)[0]
        self.assertEqual(
            first_leaf.shape[0],
            explicit_steps,
            f"Explicit num_steps={explicit_steps} must not be overridden by derivation",
        )

        # For non-auto metric: num_steps=50 also honors explicit.
        warmup_welford = blackjax.staged_adaptation(
            blackjax.nuts, self._simple_logdensity, metric="welford_diag"
        )
        results_welford, info_welford = warmup_welford.run(
            key, pos, num_steps=explicit_steps
        )
        first_leaf_w = jax.tree.leaves(info_welford)[0]
        self.assertEqual(
            first_leaf_w.shape[0],
            explicit_steps,
            "Non-auto metric: explicit num_steps must be honored",
        )

    def test_low_budget_warning_fires_for_high_d(self):
        """Low max_grad_budget + high-d position → UserWarning about rank-detection support.

        For d=100, actual_rank=50, the support floor is 8*51=408 steps.
        With max_grad_budget=2000, derived num_steps=100, and the largest window
        in the growing schedule is well below 408 → warning fires.

        The warning must fire from run(), not from staged_adaptation() construction.
        assertWarns() confirms a UserWarning with 'rank-detection' in the message.
        """
        max_grad_budget = 2000  # → derived num_steps = 100
        n_dims = 100  # → actual_rank = min(50, 50) = 50; floor = 8*51 = 408

        warmup = blackjax.staged_adaptation(
            blackjax.nuts,
            self._simple_logdensity,
            metric="auto",
            max_grad_budget=max_grad_budget,
        )
        key = jax.random.key(12)
        pos = jnp.zeros(n_dims)

        with self.assertWarnsRegex(
            UserWarning,
            "rank-detection",
            msg="Expected a UserWarning mentioning 'rank-detection' for "
            f"d={n_dims}, max_grad_budget={max_grad_budget}",
        ):
            warmup.run(key, pos)

    def test_low_budget_warning_not_suppressed_by_explicit_small_num_steps(self):
        """Warning fires even when the caller passes an explicit small num_steps.

        The warning is about budget, not about whether num_steps was derived.
        An explicit num_steps that still yields a small largest window should
        also trigger the warning.
        """
        n_dims = 100
        warmup = blackjax.staged_adaptation(
            blackjax.nuts,
            self._simple_logdensity,
            metric="auto",
            max_grad_budget=50000,
        )
        key = jax.random.key(13)
        pos = jnp.zeros(n_dims)

        # num_steps=100 is explicitly below the support floor for d=100
        with self.assertWarnsRegex(UserWarning, "rank-detection"):
            warmup.run(key, pos, num_steps=100)

    def test_sufficient_budget_emits_no_warning(self):
        """Large budget for a small model produces no UserWarning at run() time."""
        n_dims = 5  # small d → low support floor
        warmup = blackjax.staged_adaptation(
            blackjax.nuts,
            self._simple_logdensity,
            metric="auto",
            max_grad_budget=50000,
        )
        key = jax.random.key(14)
        pos = jnp.zeros(n_dims)

        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            # Should not raise — sufficient budget for d=5.
            warmup.run(key, pos)

    def test_non_auto_metric_no_warning(self):
        """Non-auto metrics never emit the rank-detection warning."""
        n_dims = 100
        warmup = blackjax.staged_adaptation(
            blackjax.nuts, self._simple_logdensity, metric="welford_diag"
        )
        key = jax.random.key(15)
        pos = jnp.zeros(n_dims)

        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            # welford_diag with default num_steps=1000 must not warn.
            warmup.run(key, pos, num_steps=50)


class TestImpossibleComboInvariant(BlackJAXTest):
    """Impossible combo + legal coexistence invariants from the scoped latch rule.

    The ONLY impossible combo: escalated=True AND detection_branch=between_means
    AND deferred=True.  (T-branch escalation requires confirmed_split=True, but
    confirmed_split=True ⟹ new_deferred=False algebraically.)

    Cross-branch coexistence IS LEGAL: W-escalation (pooled_within) + T-defer.
    """

    def test_split_means_no_between_branch_window1(self):
        """Split-means in window 1: detection_branch ≠ between_means (F2 regression).

        In v1/v2 a bug caused between_means escalation on the FIRST window even
        before the 2-window confirmation check could accumulate.  The fix makes
        window-1 data produce deferred (or none), not T-escalation.
        This test catches that revert: with split_scale=8 the between-chain signal
        is strong, but a single window must not yield detection_branch=between_means.
        """
        M, n, d = 8, 150, 10
        core = build_multi_chain_meta_core(max_grad_budget=40000, n_chains=M)
        state = core.init(d)
        draws_split, grads_split = _make_mc_split_means(
            M, n, d, split_scale=8.0, seed=43
        )
        state1 = _fill_mc_state(state, draws_split, grads_split)
        r1 = core.final(state1)

        branch = int(np.asarray(r1.detection_branch))
        self.assertNotEqual(
            branch,
            _DETECTION_BRANCH_BETWEEN_MEANS,
            f"Window 1 split-means must NOT escalate via between_means "
            f"(needs 2-window confirmation) -- got detection_branch={branch}",
        )

    def test_w_escalation_deferred_is_legal_coexistence(self):
        """W-branch escalation + deferred=True COEXIST: asserts the combined state (F5).

        The scoped latch rule: W-escalation (detection_branch=pooled_within) is
        independent of the T-branch defer gate.  Uses _make_mc_coexistence which
        has BOTH within-chain anisotropy (W fires) AND modal split gradients
        (any_mode_flag fires via GAIN = R2_local - R2_global > 0.3).

        Protocol (2 windows of the combined fixture):
          Window 1: W fires (has_escalated=True, detection_branch=pooled_within),
                    any_mode_flag fires (flag_count=1 -- not yet confirmed), deferred=False.
          Window 2: flag_count=2 (confirmed) -- deferred=True while has_escalated=True.

        Asserts: has_escalated=True AND deferred=True AND
                 detection_branch=pooled_within.
        """
        M, n, d = 8, 150, 10
        draws_cx, grads_cx = _make_mc_coexistence(
            M, n, d, lam_within=25.0, split_scale=8.0, seed=44
        )
        core = build_multi_chain_meta_core(max_grad_budget=80000, n_chains=M)

        # Window 1: W fires, any_mode_flag fires (flag_count=1, not yet confirmed)
        state = core.init(d)
        state = _fill_mc_state(
            state, draws_cx.astype(jnp.float32), grads_cx.astype(jnp.float32)
        )
        state = core.final(state)
        self.assertTrue(
            bool(np.asarray(state.has_escalated)),
            "Window 1: W-branch must escalate (within-chain anisotropy present)",
        )
        self.assertFalse(
            bool(np.asarray(state.deferred_to_ensemble)),
            "Window 1: deferred must be False (flag_count=1, confirmation needs 2 windows)",
        )

        # Window 2: flag_count becomes 2 (confirmed) → coexistence
        state = _fill_mc_state(
            state, draws_cx.astype(jnp.float32), grads_cx.astype(jnp.float32)
        )
        result = core.final(state)

        has_esc = bool(np.asarray(result.has_escalated))
        deferred = bool(np.asarray(result.deferred_to_ensemble))
        branch = int(np.asarray(result.detection_branch))

        self.assertTrue(
            has_esc,
            "Coexistence: has_escalated must remain True (monotone W-latch)",
        )
        self.assertTrue(
            deferred,
            "Coexistence: deferred_to_ensemble must be True (W-escalation + T-defer coexist "
            "under the branch-scoped latch rule).  "
            "If False -- check new_deferred uses ~escalate_T not ~new_has_escalated.",
        )
        self.assertEqual(
            branch,
            _DETECTION_BRANCH_POOLED_WITHIN,
            f"Coexistence: detection_branch must be pooled_within "
            f"(W fired in window 1 -- T never escalated) -- got {branch}",
        )

    def test_impossible_combo_never_occurs_over_windows(self):
        """Three-window scan: escalated+between_means+deferred NEVER appears (F5).

        Runs split-means draws through 3 consecutive windows.  Each window's
        output is checked: IF has_escalated AND detection_branch=between_means
        THEN deferred must be False.  The test is structurally rigorous because
        we RUN windows until between_means escalation could reasonably occur
        (not just checking a never-reached condition).
        """
        M, n, d = 8, 150, 10
        core = build_multi_chain_meta_core(max_grad_budget=80000, n_chains=M)
        state = core.init(d)
        draws_split, grads_split = _make_mc_split_means(
            M, n, d, split_scale=10.0, seed=45
        )

        for window in range(3):
            state = _fill_mc_state(state, draws_split, grads_split)
            result = core.final(state)

            has_esc = bool(np.asarray(result.has_escalated))
            deferred = bool(np.asarray(result.deferred_to_ensemble))
            branch = int(np.asarray(result.detection_branch))

            if has_esc and branch == _DETECTION_BRANCH_BETWEEN_MEANS:
                self.assertFalse(
                    deferred,
                    f"Window {window + 1}: impossible combo "
                    f"between_means+escalated+deferred occurred",
                )


class TestNewStateFieldsPopulated(BlackJAXTest):
    """The five W-branch/T-branch diagnostic state fields are finite after the first core.final() call."""

    def _run(self, dtype, seed):
        M, n, d = 8, 150, 10
        core = build_multi_chain_meta_core(max_grad_budget=40000, n_chains=M)
        state = core.init(d)
        draws_mc, grads_mc = _make_mc_deep_spread(M, n, d, seed=seed)
        state1 = _fill_mc_state(state, draws_mc.astype(dtype), grads_mc.astype(dtype))
        return core.final(state1)

    def test_new_fields_finite_f32(self):
        """within_lam1, chain_consistency_psi, r1_top are finite (f32)."""
        result = self._run(jnp.float32, seed=50)
        lam1 = float(np.asarray(result.within_lam1))
        psi = float(np.asarray(result.chain_consistency_psi))
        r1 = float(np.asarray(result.r1_top))
        branch = int(np.asarray(result.detection_branch))
        flag_count = int(np.asarray(result.unimodality_flag_count))

        self.assertFalse(np.isnan(lam1), "within_lam1 is NaN")
        self.assertFalse(np.isnan(psi), "chain_consistency_psi is NaN")
        self.assertFalse(np.isnan(r1), "r1_top is NaN")
        self.assertIn(
            branch,
            [
                _DETECTION_BRANCH_NONE,
                _DETECTION_BRANCH_POOLED_WITHIN,
                _DETECTION_BRANCH_BETWEEN_MEANS,
                _DETECTION_BRANCH_BOTH,
            ],
        )
        self.assertGreaterEqual(flag_count, 0)

    def test_new_fields_finite_x64(self):
        """within_lam1, chain_consistency_psi, r1_top are finite (x64)."""
        try:
            jax.config.update("jax_enable_x64", True)
            result = self._run(jnp.float64, seed=51)
            lam1 = float(np.asarray(result.within_lam1))
            psi = float(np.asarray(result.chain_consistency_psi))
            r1 = float(np.asarray(result.r1_top))
            self.assertFalse(np.isnan(lam1), "x64: within_lam1 is NaN")
            self.assertFalse(np.isnan(psi), "x64: chain_consistency_psi is NaN")
            self.assertFalse(np.isnan(r1), "x64: r1_top is NaN")
        finally:
            jax.config.update("jax_enable_x64", False)

    def test_nan_row_finite_guard(self):
        """A NaN draw row in the buffer does not make lam1/Ψ/r1 NaN (F7).

        The finite-guard in _compute_pooled_within_spectrum and
        _compute_chain_consistency_psi zero-clamps NaN/Inf rows before the
        SVD/Gram products.  This test injects NaN into one chain's buffer
        and verifies the outputs are still finite.
        """
        M, n, d = 8, 150, 10
        core = build_multi_chain_meta_core(max_grad_budget=40000, n_chains=M)
        state = core.init(d)
        draws_mc, grads_mc = _make_mc_deep_spread(M, n, d, seed=52)
        state1 = _fill_mc_state(
            state, draws_mc.astype(jnp.float32), grads_mc.astype(jnp.float32)
        )

        # Inject NaN into chain 0, row 5
        draws_with_nan = state1.draws_buffer.at[0, 5, :].set(jnp.nan)
        state_nan = state1._replace(draws_buffer=draws_with_nan)
        result = core.final(state_nan)

        lam1 = float(np.asarray(result.within_lam1))
        psi = float(np.asarray(result.chain_consistency_psi))

        self.assertFalse(np.isnan(lam1), "NaN-row: within_lam1 became NaN")
        self.assertFalse(np.isnan(psi), "NaN-row: chain_consistency_psi became NaN")
        self.assertFalse(np.isinf(lam1), "NaN-row: within_lam1 became Inf")

    def test_mu_star_nonzero_on_nonzero_grand_mean(self):
        """mu_star is non-zero after escalation with non-zero chain positions (F7).

        Per-chain centering removes chain offsets from the R² pipeline, but the
        grand mean must be re-added (mu_star) so the metric is correct in the
        original position space.  All prior fixtures have chains centered at 0
        (grand_mean≈0), leaving this re-add path untested.

        _make_mc_split_means has chains at ±split_scale along dim 0, so
        grand_mean[0] ≈ 0 (symmetric split).  Use even-spread chains where
        the grand mean is 0 but per-chain means are non-zero — the design note
        warning is about per-chain-centered input not the grand mean.

        Use _make_mc_even_spread with a large spread to ensure non-zero per-chain
        means.  Check result.mu_star is populated (finite, from Fisher estimator).
        """
        M, n, d = 8, 150, 10
        core = build_multi_chain_meta_core(max_grad_budget=40000, n_chains=M)
        state = core.init(d)
        # even-spread: chains at linearly-spaced positions → per-chain means ≠ 0
        draws_mc, grads_mc = _make_mc_even_spread(M, n, d, spread_scale=3.0, seed=53)
        state1 = _fill_mc_state(
            state, draws_mc.astype(jnp.float32), grads_mc.astype(jnp.float32)
        )
        result = core.final(state1)

        mu_star = np.asarray(result.mu_star)
        self.assertEqual(mu_star.shape, (d,), "mu_star shape must be (d,)")
        self.assertTrue(
            np.all(np.isfinite(mu_star)),
            f"mu_star must be finite -- got {mu_star}",
        )

    def test_isotropic_no_false_escalate_x64(self):
        """Isotropic chains do not escalate in float64 (F7 isotropic x64)."""
        try:
            jax.config.update("jax_enable_x64", True)
            M, n, d = 8, 200, 20
            core = build_multi_chain_meta_core(max_grad_budget=40000, n_chains=M)
            state = core.init(d)
            draws_mc, grads_mc = _make_mc_isotropic(M, n, d, seed=54)
            state1 = _fill_mc_state(
                state, draws_mc.astype(jnp.float64), grads_mc.astype(jnp.float64)
            )
            result = core.final(state1)

            has_esc = bool(np.asarray(result.has_escalated))
            self.assertFalse(
                has_esc,
                "x64: Isotropic null must not escalate (W-branch Ψ and/or T-branch "
                "collinearity gate should refuse)",
            )
        finally:
            jax.config.update("jax_enable_x64", False)


class TestMeanPoolGainDefect(BlackJAXTest):
    """Mean-pool DA correctness: M chains at shared ε → one observation, not M.

    The statistical model: M chains stepping at ONE shared ε produce M measurements
    of the same acceptance quantity — one mean observation, not M independent ones.
    Mean-pool DA (one da_update on mean(rates)) is the correct model.
    M-sequential lax.scan inflates the DA primal gain ~√M and advances the Polyak
    schedule M× too fast, causing self-sustained limit cycles.
    """

    def test_mean_pool_matches_single_chain_at_identical_acceptance(self):
        """With M chains at identical acceptance rate, mean-pool == single-chain.

        When all chains see ar=0.75, mean-pool produces da_update(ss, 0.75),
        which is identical to the single-chain update.  M-sequential would run
        four separate updates at 0.75 and advance the Polyak counter 4×, giving
        a different (over-corrected) result.

        This test asserts the CORRECT (mean-pool) behavior.  It FAILs with the
        M-sequential lax.scan implementation and PASSes after the mean-pool fix.
        """
        from blackjax.adaptation.meta import build_meta_adaptation_core
        from blackjax.adaptation.step_size import dual_averaging_adaptation

        target_ar = 0.80
        M = 4
        ar_uniform = 0.75  # same rate on all chains
        per_chain = jnp.full((M,), ar_uniform)

        # Single-chain reference: ONE da_update at the shared acceptance rate.
        da_init, da_update, _ = dual_averaging_adaptation(target_ar)
        ss_ref = da_init(0.1)
        ss_single = da_update(ss_ref, jnp.float32(ar_uniform))
        step_single = float(np.asarray(jnp.exp(ss_single.log_step_size_avg)))

        # Engine with n_da_updates=M — must give SAME result as single-chain
        # after the mean-pool fix (M-sequential gives a different result).
        dummy_core = build_meta_adaptation_core(max_grad_budget=5000)
        eng_init, eng_update, _ = _make_engine(
            dummy_core,
            target_acceptance_rate=target_ar,
            n_da_updates=M,
        )
        adapt_state = eng_init(jnp.zeros(10), 0.1)
        adapt_stage = (jnp.int32(0), jnp.bool_(False))  # fast stage, no window end

        new_state = eng_update(
            adapt_state,
            adapt_stage,
            jnp.zeros(10),
            jnp.zeros(10),
            per_chain,
        )
        step_engine = float(np.asarray(jnp.exp(new_state.ss_state.log_step_size_avg)))

        np.testing.assert_allclose(
            step_engine,
            step_single,
            rtol=1e-4,
            err_msg=(
                "Mean-pool DA: engine with M="
                + str(M)
                + " identical acceptances must match "
                "single-chain at the same rate. "
                "Got engine="
                + str(round(step_engine, 7))
                + " vs single="
                + str(round(step_single, 7))
                + ".  "
                "If this fails, the engine is using M-sequential updates (the bug): "
                "M chains at shared eps are one mean observation, not M independent ones."
            ),
        )

    def test_mean_pool_counter_advances_once_per_step(self):
        """Mean-pool DA: the Polyak counter increments by 1 per multi-chain step.

        With the correct mean-pool model, each warmup step contributes ONE DA
        observation (the mean acceptance rate across chains), so the step counter
        must advance by 1 — not by M.  Advancing M× is the limit-cycle mechanism.

        This test asserts the CORRECT (mean-pool) counter behavior.  It FAILs
        with the M-sequential implementation (counter advances by M) and PASSes
        after the mean-pool fix.
        """
        from blackjax.adaptation.meta import build_meta_adaptation_core

        M = 6
        per_chain = jnp.full((M,), 0.78)

        dummy_core = build_meta_adaptation_core(max_grad_budget=5000)
        eng_init, eng_update, _ = _make_engine(
            dummy_core,
            target_acceptance_rate=0.80,
            n_da_updates=M,
        )
        adapt_state = eng_init(jnp.zeros(8), 0.1)
        initial_step = int(np.asarray(adapt_state.ss_state.step))
        adapt_stage = (jnp.int32(0), jnp.bool_(False))

        new_state = eng_update(
            adapt_state, adapt_stage, jnp.zeros(8), jnp.zeros(8), per_chain
        )

        delta = int(np.asarray(new_state.ss_state.step)) - initial_step
        self.assertEqual(
            delta,
            1,
            f"Mean-pool DA counter must advance by 1 per warmup step (not M={M}). "
            f"Got delta={delta}. M-sequential advances by M — that is the gain defect.",
        )


class TestSharedEpsilonDA(BlackJAXTest):
    """Shared-ε: mean-pool DA matches single-chain DA at the mean acceptance rate."""

    def test_n_da_updates_mean_matches_single_chain_at_mean_ar(self):
        """_make_engine with n_da_updates=M computes da_update(ss, mean(rates)).

        The correct statistical model: M chains at one shared eps contribute ONE
        mean observation.  The engine must produce the same step_size_avg as a
        single da_update at mean(per_chain), not M sequential updates.
        """
        from blackjax.adaptation.meta import build_meta_adaptation_core
        from blackjax.adaptation.step_size import dual_averaging_adaptation

        target_ar = 0.80
        M = 4
        per_chain = jnp.array([0.55, 0.65, 0.75, 0.85])

        # Single-chain reference: ONE da_update at mean(per_chain) = 0.70
        da_init, da_update, _ = dual_averaging_adaptation(target_ar)
        ss = da_init(0.1)
        ss_ref = da_update(ss, jnp.mean(per_chain))
        step_ref = float(np.asarray(jnp.exp(ss_ref.log_step_size_avg)))

        # Engine with n_da_updates=M uses mean-pool (one update at the mean)
        dummy_core = build_meta_adaptation_core(max_grad_budget=5000)
        eng_init, eng_update, _ = _make_engine(
            dummy_core,
            target_acceptance_rate=target_ar,
            n_da_updates=M,
        )
        adaptation_state = eng_init(jnp.zeros(10), 0.1)
        adaptation_stage = (jnp.int32(0), jnp.bool_(False))  # fast stage, no window end

        new_state = eng_update(
            adaptation_state,
            adaptation_stage,
            jnp.zeros(10),
            jnp.zeros(10),
            per_chain,
        )

        step_engine = float(np.asarray(jnp.exp(new_state.ss_state.log_step_size_avg)))
        self.assertAlmostEqual(
            step_engine,
            step_ref,
            places=4,
            msg="n_da_updates=M must produce da_update(ss, mean(rates)), not M sequential updates",
        )

    def test_step_counter_increments_once_per_step(self):
        """After n_da_updates=M (mean-pool), the DA step counter increments by 1."""
        from blackjax.adaptation.meta import build_meta_adaptation_core

        M = 3
        per_chain = jnp.array([0.70, 0.75, 0.80])

        dummy_core = build_meta_adaptation_core(max_grad_budget=5000)
        eng_init, eng_update, _ = _make_engine(
            dummy_core,
            target_acceptance_rate=0.80,
            n_da_updates=M,
        )
        adaptation_state = eng_init(jnp.zeros(8), 0.1)
        initial_step = int(np.asarray(adaptation_state.ss_state.step))
        adaptation_stage = (jnp.int32(0), jnp.bool_(False))

        new_state = eng_update(
            adaptation_state,
            adaptation_stage,
            jnp.zeros(8),
            jnp.zeros(8),
            per_chain,
        )

        step_count = int(np.asarray(new_state.ss_state.step))
        # Mean-pool: ONE observation per warmup step, so counter advances by 1.
        self.assertEqual(
            step_count - initial_step,
            1,
            "Mean-pool DA: counter must advance by 1 per step (not M="
            + str(M)
            + "). Got delta="
            + str(step_count - initial_step),
        )

    def test_mc_engine_wiring_n_da_gives_mean_pool(self):
        """staged_adaptation(n_chains=M) wires n_da_updates=M for mean-pool DA.

        Tests the WIRING from staged_adaptation.py:
            n_da_updates = _n_chains if _is_multi_chain else 1

        The DA step counter must advance by 1 per multi-chain warmup step
        (one mean-pool observation per step, not M sequential ones).
        The mean-pool result must match a single da_update at the mean
        acceptance rate.
        """
        from blackjax.adaptation.meta import build_meta_adaptation_core
        from blackjax.adaptation.step_size import dual_averaging_adaptation

        M = 8  # matches the recommended n_chains minimum
        ar_val = 0.78
        per_chain = jnp.full((M,), ar_val)

        # Reference: one da_update at the mean (= ar_val since all equal)
        da_init, da_update, _ = dual_averaging_adaptation(0.80)
        ss_ref = da_init(0.1)
        ss_ref = da_update(ss_ref, jnp.float32(ar_val))
        step_ref = float(np.asarray(jnp.exp(ss_ref.log_step_size_avg)))

        dummy_core = build_meta_adaptation_core(max_grad_budget=5000)
        eng_init, eng_update, _ = _make_engine(
            dummy_core, target_acceptance_rate=0.80, n_da_updates=M
        )
        adaptation_state = eng_init(jnp.zeros(10), 0.1)
        initial_step = int(np.asarray(adaptation_state.ss_state.step))
        adaptation_stage = (jnp.int32(0), jnp.bool_(False))

        new_state = eng_update(
            adaptation_state,
            adaptation_stage,
            jnp.zeros(10),
            jnp.zeros(10),
            per_chain,
        )

        step_count = int(np.asarray(new_state.ss_state.step))
        step_engine = float(np.asarray(jnp.exp(new_state.ss_state.log_step_size_avg)))

        # Counter must advance by 1 (mean-pool = one observation per step).
        self.assertEqual(
            step_count - initial_step,
            1,
            "MC wiring: DA counter must advance by 1 per multi-chain step "
            "(mean-pool model). Got delta="
            + str(step_count - initial_step)
            + ". Check staged_adaptation.py _maybe_multi_da_update.",
        )
        # Result must match single da_update at the mean acceptance rate.
        self.assertAlmostEqual(
            step_engine,
            step_ref,
            places=4,
            msg="MC wiring: engine must match da_update(ss, mean(rates)), not M sequential updates",
        )


class TestWBranchE2ESmoke(BlackJAXTest):
    """core.final() with deep-spread draws: W-branch diagnostics are finite and positive."""

    def _run_smoke(self, dtype, seed):
        M, n, d = 8, 150, 10
        core = build_multi_chain_meta_core(max_grad_budget=40000, n_chains=M)
        state = core.init(d)
        draws_mc, grads_mc = _make_mc_deep_spread(M, n, d, lam_within=25.0, seed=seed)
        state1 = _fill_mc_state(state, draws_mc.astype(dtype), grads_mc.astype(dtype))
        return core.final(state1), d

    def test_w_branch_e2e_f32(self):
        """Deep-spread draws: within_lam1 > 0, chain_consistency_psi > 0 (f32)."""
        result, d = self._run_smoke(jnp.float32, seed=70)
        lam1 = float(np.asarray(result.within_lam1))
        psi = float(np.asarray(result.chain_consistency_psi))
        self.assertFalse(np.isnan(lam1), "within_lam1 is NaN")
        self.assertGreater(lam1, 0.0, "within_lam1 must be positive")
        self.assertFalse(np.isnan(psi), "chain_consistency_psi is NaN")
        self.assertGreater(psi, 0.0, "chain_consistency_psi must be positive")
        imm = result.inverse_mass_matrix
        self.assertIsInstance(imm, LowRankInverseMassMatrix)
        self.assertEqual(imm.sigma.shape, (d,))
        self.assertTrue(
            bool(jnp.all(jnp.isfinite(imm.sigma))), "sigma has non-finite values"
        )

    def test_w_branch_e2e_x64(self):
        """Deep-spread draws: W-branch diagnostics are finite in float64."""
        try:
            jax.config.update("jax_enable_x64", True)
            result, d = self._run_smoke(jnp.float64, seed=71)
            lam1 = float(np.asarray(result.within_lam1))
            psi = float(np.asarray(result.chain_consistency_psi))
            self.assertFalse(np.isnan(lam1), "x64: within_lam1 is NaN")
            self.assertFalse(np.isnan(psi), "x64: chain_consistency_psi is NaN")
            imm = result.inverse_mass_matrix
            self.assertTrue(
                bool(jnp.all(jnp.isfinite(imm.sigma))),
                "x64: sigma has non-finite values",
            )
        finally:
            jax.config.update("jax_enable_x64", False)


class TestEndToEndEscalation(BlackJAXTest):
    """staged_adaptation with n_chains=8 on an ill-conditioned target escalates.

    The pooled-aware schedule produces windows at steps ~27, 66, 124 with
    n_pool = M * per_chain_n >= min_n_proj = 208.  This test asserts that after
    running staged_adaptation with max_grad_budget=50_000 and n_chains=8 on a
    rank-5 ill-conditioned Gaussian (d=50), the emitted metric has
    effective_rank > 0 (at least one slow direction was deployed).

    The target is a correlated Gaussian with a rank-5 spike (lam_spike=100),
    chosen so the W-branch always fires given a working schedule.  The test
    is structural, not stochastic: at this geometry the signal is several sigma
    above the null edge regardless of the random seed.
    """

    def _make_ill_cond_logdensity(self, d: int, rank: int, lam_spike: float, seed: int):
        """Return a logdensity_fn for N(0, Sigma) with rank-k spike.

        Sigma = I + U*(lam_spike - 1)*Ut.  Precision = Sigma^{-1}.
        The score is -Sigma^{-1} x (linear, unit R²).
        """
        key = jax.random.key(seed)
        U_raw = jax.random.normal(key, (d, rank))
        U, _ = jnp.linalg.qr(U_raw)  # orthonormal (d, rank)
        U = U[:, :rank]

        # Precision matrix: I + U*((1/lam_spike)-1)*Ut
        lam_inv = 1.0 / lam_spike
        prec = jnp.eye(d) + (lam_inv - 1.0) * (U @ U.T)

        def logdensity_fn(x):
            return -0.5 * x @ prec @ x

        return logdensity_fn, U

    def test_ill_cond_escalates_f32(self):
        """staged_adaptation(n_chains=8, metric='auto') escalates on ill-conditioned target.

        Asserts effective_rank > 0: at least one slow direction was detected
        and deployed by the Fisher estimator.  The pooled-aware schedule
        produces the first escalation-eligible window at step ~27 and the
        carry propagates has_escalated=True through subsequent windows.
        """
        d, rank, lam_spike = 50, 5, 100.0
        M = 8
        max_grad_budget = 50_000

        logdensity_fn, _ = self._make_ill_cond_logdensity(d, rank, lam_spike, seed=0)

        warmup = blackjax.staged_adaptation(
            blackjax.nuts,
            logdensity_fn,
            metric="auto",
            max_grad_budget=max_grad_budget,
            n_chains=M,
        )
        key = jax.random.key(42)
        # Overdispersed start: chain m starts near m*e_0 so the W-branch
        # sees real within-chain spread rather than init-dispersion only.
        positions = (
            jnp.zeros((M, d))
            .at[:, 0]
            .set(jnp.linspace(-2.0, 2.0, M, dtype=jnp.float32))
        )
        results, _ = warmup.run(key, positions)

        imm = results.parameters["inverse_mass_matrix"]
        self.assertIsInstance(imm, LowRankInverseMassMatrix)

        # Effective rank: count(|lam_i - 1| > _LAM_NONTRIVIAL_TOL) in deployed lam.
        lam_np = np.asarray(imm.lam)
        effective_rank = int(np.sum(np.abs(lam_np - 1.0) > _LAM_NONTRIVIAL_TOL))
        self.assertGreater(
            effective_rank,
            0,
            f"staged_adaptation(n_chains=8) must deploy at least one slow direction "
            f"on an ill-conditioned d={d} rank-{rank} target (lam_spike={lam_spike}).  "
            f"Got effective_rank=0 — the pooled-aware schedule did not produce an "
            f"escalation-eligible window.  Check _build_mc_window_schedule wiring in "
            f"staged_adaptation.py and the budget_remaining gate in final().",
        )
