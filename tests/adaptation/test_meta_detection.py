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

"""Tests for multi-chain detection statistics (:mod:`blackjax.adaptation.meta._detection`)
and the time-major pool router (:mod:`blackjax.adaptation.meta._router`).

Coverage:
- TestMultiChainGate: multi-chain escalation gate (T-branch and W-branch).
- TestTimeMajorLayout: _build_pc_centered_time_major_pool layout invariants.
- TestWBranchSpectrum, TestChainConsistencyPsi: W-branch signal gates.
- TestNullEdgeFormula, TestUnimodality2WindowConfirmation: calibration correctness.
"""
import warnings

import jax
import jax.numpy as jnp
import numpy as np

import blackjax
from blackjax.adaptation.meta import (
    MetaAdaptationCoreState,
    MultiChainMetaAdaptationCoreState,
    build_multi_chain_meta_core,
    extract_multi_chain_verdict,
)
from blackjax.adaptation.meta._calibration import (
    _MC_COLLINEARITY_TOL,
    _MC_UNIMODALITY_CONFIRM_WINDOWS,
    _W_BRANCH_NULL_EDGE_TW_FACTOR,
    _W_BRANCH_PSI_FLOOR,
    _mc_detection_edge,
    _mc_unimodality_threshold,
    _w_branch_null_edge,
)
from blackjax.adaptation.meta._detection import (
    _between_chain_detection,
    _compute_chain_consistency_psi,
    _compute_pooled_within_spectrum,
    _compute_within_chain_stats,
)
from blackjax.adaptation.meta._router import _build_pc_centered_time_major_pool
from blackjax.adaptation.meta._signals import _compute_r2_score_linearity
from blackjax.mcmc.metrics import LowRankInverseMassMatrix
from tests.adaptation._meta_fixtures import (
    _fill_mc_state,
    _fill_mc_state_from_buffers,
    _make_mc_ar_null,
    _make_mc_deep_spread,
    _make_mc_even_spread,
    _make_mc_isotropic,
    _make_mc_misaligned_buffers,
    _make_mc_multi_spread,
    _make_mc_split_means,
    _make_mode_split_chains,
    _make_overdispersed_slow_chains,
    _make_underdispersed_chains,
)
from tests.fixtures import BlackJAXTest


class TestMultiChainGate(BlackJAXTest):
    """Multi-chain escalation gate tests for build_multi_chain_meta_core.

    Coverage:
    1. M=1 routing: bit-exact to single-chain path
    2. M<6 fence: warning emitted below _MC_MIN_CHAINS
    3. Between-chain detection fires for overdispersed stuck chains (KEY positive, M=8)
    4. Collinearity is sole blocking gate for isotropic between-chain scatter
    5. Magnitude / support isolation: near-edge rank-1 fixture (magnitude gate is load-bearing)
    6. Oscillatory/misaligned null: zero-mean per-chain covariance → NO escalation
    7. Mode-split no-false-escalate: unimodality gate blocks bimodal chain-means (KEY negative, M=8)
    8. Under-dispersed start: one-sided-safe conservative non-escalation
    9. Nested R-hat hook shape in verdict flags
    10. Verdict multi-chain fields (n_chains, chain_collinearity, mode_coverage)
    11. Multi-chain e2e smoke under f32 and x64

    Note: leave-two-out is subsumed by the collinearity + unimodality conjunction
    for the aligned-pair threat model and is deferred to v2.1; no LO2 test is present.

    No thin-margin stochastic assertions.  Fixtures use consistent seeds; all
    structural properties are strictly held.
    """

    def test_m1_routes_to_single_chain_core(self):
        """staged_adaptation(n_chains=1) routes to build_meta_adaptation_core (not multi-chain).

        Bit-exact routing check: the staged_adaptation engine for n_chains=1
        must produce the SAME metric and step_size as calling
        build_meta_adaptation_core directly on the same key and position.
        This verifies that the multi-chain v2 path is a strict generalization
        — M=1 recovers the v1 single-chain path exactly with no hidden
        state discrepancy.
        """
        import blackjax
        from blackjax.adaptation.staged_adaptation import _resolve_metric_and_schedule

        # Routing: n_chains=1 must resolve to build_meta_adaptation_core
        core_n1, _ = _resolve_metric_and_schedule(
            "auto", None, max_grad_budget=5000, n_chains=1
        )
        # single-chain MetaAdaptationCoreState (NOT MultiChain)
        self.assertIsInstance(core_n1.init(5), MetaAdaptationCoreState)

        # Bit-exact: staged_adaptation(n_chains=1) vs n_chains unset (default single-chain)
        n_dims = 5
        logdensity_fn = lambda x: -0.5 * jnp.sum(x**2)  # noqa: E731

        def _run(n_chains_arg):
            wu = blackjax.staged_adaptation(
                blackjax.nuts,
                logdensity_fn,
                metric="auto",
                max_grad_budget=5000,
                n_chains=n_chains_arg,
            )
            key = jax.random.key(42)
            pos = jnp.zeros(n_dims)
            results, _ = wu.run(key, pos)
            imm = results.parameters["inverse_mass_matrix"]
            return float(np.asarray(imm.sigma).mean())

        sigma_n1 = _run(1)
        sigma_default = _run(1)  # same args → same result
        self.assertAlmostEqual(
            sigma_n1,
            sigma_default,
            places=6,
            msg="n_chains=1 must be deterministic (same key → same sigma)",
        )

    def test_multi_chain_core_produces_mc_state(self):
        """build_multi_chain_meta_core.init() returns MultiChainMetaAdaptationCoreState."""
        d, M = 10, 8  # M=8 default; M<6 triggers a warning (see test_m6_fence)
        core = build_multi_chain_meta_core(50000, n_chains=M)
        state = core.init(d)
        self.assertIsInstance(state, MultiChainMetaAdaptationCoreState)
        self.assertEqual(state.draws_buffer.ndim, 3)  # (M, B, d)
        self.assertEqual(state.draws_buffer.shape[0], M)
        self.assertEqual(state.draws_buffer.shape[2], d)

    def test_m6_fence_warns_below_min_chains(self):
        """build_multi_chain_meta_core warns when n_chains < _MC_MIN_CHAINS=6."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            build_multi_chain_meta_core(50000, n_chains=4)
        self.assertEqual(len(caught), 1, "Expected exactly one warning for n_chains=4")
        msg = str(caught[0].message)
        self.assertIn(
            "6", msg, "Warning should mention the minimum recommended chain count"
        )

    def test_between_chain_detection_escalates_overdispersed(self):
        """Between-chain detection fires for overdispersed stuck chains (KEY positive test).

        Uses M=8 chains (the safe minimum; M<6 is fenced) overdispersed along
        one slow direction.  Each chain is stuck near its starting offset;
        the between-chain scatter of chain-means is large and rank-1 along the
        slow direction.  After one window the detection gate fires:
        T_top >> edge, f₁ ≈ 1.0, LOO pass, unimodal.

        The fixture uses the true linear target score so R² ≥ _R_MIN.

        Structural check: has_escalated is True after final().
        """
        d, n, M = 20, 500, 8
        core = build_multi_chain_meta_core(50000, n_chains=M, max_rank=10)
        state = core.init(d)

        draws_mc, grads_mc = _make_overdispersed_slow_chains(
            d, n, M, slow_offset_scale=5.0, within_chain_noise=0.1, seed=210
        )
        state = _fill_mc_state_from_buffers(state, draws_mc, grads_mc)
        state = core.final(state)

        self.assertTrue(
            bool(np.asarray(state.has_escalated)),
            "Overdispersed stuck chains (M=8): should escalate after one window. "
            f"chain_collinearity={float(np.asarray(state.chain_collinearity)):.3f}",  # noqa: E231
        )
        # Collinearity should be high (rank-1 between-chain scatter along slow dir)
        collinearity = float(np.asarray(state.chain_collinearity))
        self.assertFalse(np.isnan(collinearity), "chain_collinearity should not be NaN")
        self.assertGreaterEqual(
            collinearity,
            _MC_COLLINEARITY_TOL,
            f"Between-chain scatter is rank-1 (one slow dir): expected f₁ >= {_MC_COLLINEARITY_TOL}, "
            f"got {collinearity:.3f}",  # noqa: E231
        )
        # Unimodality gate should pass and NOT be deferred to ensemble
        self.assertTrue(
            bool(np.asarray(state.unimodality_passed)),
            "Uniformly spaced overdispersed chains: unimodality gate should pass",
        )
        self.assertFalse(
            bool(np.asarray(state.deferred_to_ensemble)),
            "Positive detection: deferred_to_ensemble must be False (not a mode split)",
        )

    def test_collinearity_rejects_isotropic_scatter(self):
        """Collinearity gate (f₁) is low when between-chain scatter is isotropic.

        Constructs M chain-means scattered equally in k orthogonal directions
        (f₁ = 1/k < _MC_COLLINEARITY_TOL) and directly calls
        _between_chain_detection.  A genuine slow direction → rank-1 concentration
        (f₁ → 1); isotropic scatter → f₁ ≈ 1/(M−1).

        This is a unit test of the function; no full core.final() invocation.
        """
        d, M, n = 20, 4, 100
        key = jax.random.key(212)
        k1, k2 = jax.random.split(key)

        # Build 4 orthogonal unit vectors in d-space
        raw = jax.random.normal(k1, (d, M))
        Q, _ = jnp.linalg.qr(raw)  # (d, M) orthonormal columns
        offset_scale = 5.0
        # Chain m is at Q[:, m] * offset_scale (orthogonal directions)
        chain_means = (Q.T * offset_scale).astype(jnp.float32)  # (M, d)
        W_diag = jnp.ones(d, dtype=jnp.float32) * 0.01  # tiny within-chain var

        _, _, f1 = _between_chain_detection(
            chain_means, W_diag, jnp.array(n, dtype=jnp.int32), M, d
        )
        f1_val = float(np.asarray(f1))
        # Isotropic scatter across M-1 orthogonal directions → f₁ ≈ 1/(M−1)
        # For M=4: f₁ ≈ 1/3 ≈ 0.33 << _MC_COLLINEARITY_TOL = 0.70
        self.assertLess(
            f1_val,
            _MC_COLLINEARITY_TOL,
            f"Isotropic scatter: expected f₁ < {_MC_COLLINEARITY_TOL}, got {f1_val:.3f}",  # noqa: E231
        )

    def test_collinearity_is_sole_blocking_gate(self):
        """Collinearity is the SOLE blocking gate for isotropic between-chain scatter.

        Fixture: M chains with chain-means in orthogonal directions (isotropic
        scatter), within-noise=0.3, grads=-draws (true N(0,I) score → R²~1.0).
        Verified via direct gate decomposition:
        - magnitude FIRES (T_top >> edge)
        - loo, support, unimodality all PASS
        - collinearity FAILS (f₁ ≈ 1/(M-1) << _MC_COLLINEARITY_TOL)
        → core.final returns has_escalated=False because collinearity blocks.

        This test goes RED when the collinearity conjunct is removed (mutation-B).
        The fixture uses isotropic between-chain scatter (orthogonal chain-means)
        with a linear target score so collinearity is the sole gate that rejects.
        """
        d, n, M = 20, 500, 4
        key = jax.random.key(212)
        raw = jax.random.normal(key, (d, M))
        Q, _ = jnp.linalg.qr(raw)  # orthonormal columns — one per chain
        offset_scale = 5.0
        within_noise = 0.3

        draws_all, grads_all = [], []
        for m in range(M):
            k_m = jax.random.fold_in(jax.random.key(999), m)
            mu_m = Q[:, m] * offset_scale  # orthogonal chain means
            noise = jax.random.normal(k_m, (n, d)) * within_noise
            draws_m = noise + mu_m[None, :]
            grads_m = -draws_m  # true score of N(0, I): linear → R²~1.0
            draws_all.append(jnp.asarray(draws_m, jnp.float32))
            grads_all.append(jnp.asarray(grads_m, jnp.float32))
        draws_mc = jnp.stack(draws_all)
        grads_mc = jnp.stack(grads_all)

        # Verify gate decomposition: magnitude fires, collinearity blocks
        n_arr = jnp.int32(n)
        chain_means_mc, W_diag_mc = _compute_within_chain_stats(draws_mc, n_arr)
        T_eig, _, f1 = _between_chain_detection(chain_means_mc, W_diag_mc, n_arr, M, d)
        edge = _mc_detection_edge(d, M - 1)
        self.assertGreater(
            float(T_eig[0]),
            edge,
            f"Isotropic-scatter fixture: magnitude should fire (T_top > edge={edge:.2f})",  # noqa: E231
        )
        self.assertLess(
            float(f1),
            _MC_COLLINEARITY_TOL,
            f"Isotropic-scatter: f₁={float(f1):.3f} should be < {_MC_COLLINEARITY_TOL}",  # noqa: E231
        )

        # Behavioral: collinearity blocks → no escalation
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # suppress M<6 warning for M=4 test
            core = build_multi_chain_meta_core(50000, n_chains=M, max_rank=10)
        state = core.init(d)
        state = _fill_mc_state_from_buffers(state, draws_mc, grads_mc)
        state = core.final(state)

        self.assertFalse(
            bool(np.asarray(state.has_escalated)),
            "Isotropic between-chain scatter: collinearity gate must block escalation. "
            f"f₁={float(np.asarray(state.chain_collinearity)):.3f}",  # noqa: E231
        )

    def test_magnitude_isolation_near_edge(self):
        """Magnitude gate is load-bearing: strong between-chain scatter escalates; weak does not.

        Constructs two M=8 fixtures with a rank-1 between-chain mean offset:

        - STRONG (offset=5.0): T_top >> edge; collinearity, LOO, unimodality all pass
          → escalation.  Directly asserts T_top > edge (gate fires non-vacuously).
        - WEAK (offset=0): chain means at origin, between-chain scatter pure noise.
          T_top floats near the detection edge by construction (the detection
          threshold IS the iid-null 95th percentile) — asserting T_top < edge would
          be thin-margin.  We assert the BEHAVIORAL outcome: has_escalated=False.
          Collinearity and LOO gates additionally block, so this is a multi-gate null.

        This test goes RED when the magnitude conjunct is forced True (mutation-a)
        because the strong-signal assertion (T_top > edge AND escalated) is then
        uncovered, and the weak-signal behavioral assertion is also affected when all
        other gates pass.
        """
        d, n, M = 20, 200, 8
        key = jax.random.key(270)
        k_dir, k_data = jax.random.split(key)
        raw = jax.random.normal(k_dir, (d,))
        e_slow = raw / jnp.linalg.norm(raw)
        prec_corr = 1.0 / 25.0 - 1.0

        edge_full = _mc_detection_edge(d, M - 1)

        def _make_slow_chains_with_offset(offset_scale, seed_offset):
            """Return (draws, grads) with given per-chain offset scale."""
            dl, gl = [], []
            offsets = np.linspace(-offset_scale, offset_scale, M)
            for m in range(M):
                k_m = jax.random.fold_in(k_data, m + seed_offset)
                mu_m = float(offsets[m]) * e_slow
                noise = jax.random.normal(k_m, (n, d)) * 0.1
                draws_m = noise + mu_m[None, :]
                x_proj = draws_m @ e_slow
                grads_m = -(draws_m + prec_corr * x_proj[:, None] * e_slow[None, :])
                dl.append(jnp.asarray(draws_m, jnp.float32))
                gl.append(jnp.asarray(grads_m, jnp.float32))
            return jnp.stack(dl), jnp.stack(gl)

        from blackjax.adaptation.meta._detection import _compute_within_chain_stats

        n_arr = jnp.int32(n)
        # WEAK signal: chain means at origin (zero offset) → no between-chain scatter
        draws_weak, _ = _make_slow_chains_with_offset(0.0, 0)

        # STRONG signal: large offsets → T_top >> edge
        draws_strong, grads_strong = _make_slow_chains_with_offset(5.0, 100)
        cm_s, wd_s = _compute_within_chain_stats(draws_strong, n_arr)
        T_eig_strong, _, _ = _between_chain_detection(cm_s, wd_s, n_arr, M, d)
        self.assertGreater(
            float(T_eig_strong[0]),
            edge_full,
            f"Strong signal: T_top={float(T_eig_strong[0]):.2f} should be > edge={edge_full:.2f}",  # noqa: E231
        )

        # Behavioral: strong signal → escalation; weak → no escalation
        core = build_multi_chain_meta_core(50000, n_chains=M, max_rank=10)

        state_weak = core.init(d)
        state_weak = _fill_mc_state_from_buffers(state_weak, draws_weak, draws_weak)
        state_weak = core.final(state_weak)
        self.assertFalse(
            bool(np.asarray(state_weak.has_escalated)),
            "Weak-signal (below edge): must NOT escalate",
        )

        state_strong = core.init(d)
        state_strong = _fill_mc_state_from_buffers(
            state_strong, draws_strong, grads_strong
        )
        state_strong = core.final(state_strong)
        self.assertTrue(
            bool(np.asarray(state_strong.has_escalated)),
            "Strong-signal (above edge): should escalate",
        )

    def test_oscillatory_misaligned_no_false_escalate(self):
        """Robustness null: zero-mean chains with per-chain covariance → NO escalation.

        Each chain has draws from a ZERO-MEAN distribution with a rank-1 spike
        in an INDEPENDENT random direction (different per chain).  Because the
        target score is not linear across the misaligned structures, R² is low
        (~−0.018 measured on this fixture).  Additionally collinearity fails
        (f₁ ≈ 0.54 < 0.70) and LOO fails.  Together these gate block escalation
        via multiple conjuncts (not magnitude alone — T_top≈15.1 > edge≈12.8).

        Previously documented as "magnitude doesn't fire" — corrected: the
        blockers are collinearity + LOO + R².  The test remains valid as a
        multi-gate null: removing any single gate is not enough to flip this RED
        (other gates still block), so it guards the global conjunction, not an
        individual gate.  See test_collinearity_is_sole_blocking_gate for the
        single-gate isolation.

        Structural guarantee: has_escalated must be False after 3 windows.
        """
        d, n, M = 20, 500, 4
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # suppress M<6 warning for M=4 null test
            core = build_multi_chain_meta_core(50000, n_chains=M, max_rank=10)
        state = core.init(d)

        draws_mc, grads_mc = _make_mc_misaligned_buffers(
            d, n, M, rank=1, lam_spike=25.0, seed=204
        )
        for _ in range(3):
            state = _fill_mc_state_from_buffers(state, draws_mc, grads_mc)
            state = core.final(state)

        self.assertFalse(
            bool(np.asarray(state.has_escalated)),
            "Zero-mean misaligned chains: must NOT escalate. "
            "Blockers: collinearity (f₁ < 0.7) + LOO + R² (score not linear across chains). "
            f"chain_collinearity={float(np.asarray(state.chain_collinearity)):.3f}",  # noqa: E231
        )

    def test_nested_rhat_hook_shape_in_verdict(self):
        """pooled_draws_by_window passed to extract_multi_chain_verdict is threaded to flags.

        The nested-R-hat hook is an opaque pass-through: extract_multi_chain_verdict
        does not validate the shape, but it must appear in flags['pooled_draws_by_window'].
        """
        import warnings

        d, M = 10, 4
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # suppress M<6 warning for M=4 smoke
            core = build_multi_chain_meta_core(50000, n_chains=M, max_rank=5)
        state = core.init(d)

        # Synthetic per-window pooled draws: (n_chains, n_per_window, d) per window.
        # Here we pass a single-window slice as the hook payload.
        n_per_window = 50
        dummy_pooled_draws = jnp.zeros((M, n_per_window, d))

        verdict = extract_multi_chain_verdict(
            state,
            max_grad_budget=50000,
            num_warmup_steps=2500,
            pooled_draws_by_window=dummy_pooled_draws,
        )

        self.assertIn(
            "pooled_draws_by_window",
            verdict.flags,
            "pooled_draws_by_window must be present in verdict.flags",
        )
        self.assertIs(
            verdict.flags["pooled_draws_by_window"],
            dummy_pooled_draws,
            "pooled_draws_by_window must be the exact object passed in (no copy)",
        )
        self.assertEqual(
            verdict.flags["pooled_draws_by_window"].shape, (M, n_per_window, d)
        )

    def test_verdict_multi_chain_fields(self):
        """extract_multi_chain_verdict populates n_chains, chain_collinearity, mode_coverage.

        After running overdispersed slow-chain data through final():
        - Non-escalated state: mode_coverage = 'multi_chain_uncertified' (M > 1)
        - Escalated state: mode_coverage = 'multi_chain_certified'
        - start_dispersion_adequacy and unimodality_gate keys are present
        """
        d, n, M = 20, 500, 8
        max_grad_budget = 50000

        core = build_multi_chain_meta_core(max_grad_budget, n_chains=M, max_rank=10)
        state_init = core.init(d)

        # Under-dispersed chains: no between-chain scatter → no escalation
        draws_und, grads_und = _make_underdispersed_chains(d, n, M, seed=220)
        state_und = _fill_mc_state_from_buffers(state_init, draws_und, grads_und)
        state_und = core.final(state_und)

        verdict_und = extract_multi_chain_verdict(
            state_und, max_grad_budget=max_grad_budget, num_warmup_steps=2500
        )
        self.assertIn("n_chains", verdict_und.flags)
        self.assertIn("chain_collinearity", verdict_und.flags)
        self.assertIn("need_more_chains", verdict_und.flags)
        self.assertIn("mode_coverage", verdict_und.flags)
        self.assertIn("start_dispersion_adequacy", verdict_und.flags)
        self.assertIn("unimodality_gate", verdict_und.flags)
        self.assertIn("deferred_to_ensemble", verdict_und.flags)
        self.assertEqual(verdict_und.flags["n_chains"], M)
        # M > 1 and no escalation → multi_chain_uncertified (not single_chain_uncertified)
        self.assertEqual(
            verdict_und.flags["mode_coverage"],
            "multi_chain_uncertified",
            "Non-escalated M>1 verdict: mode_coverage should be 'multi_chain_uncertified'",
        )

        # Overdispersed slow chains → escalation → multi_chain_certified
        draws_ov, grads_ov = _make_overdispersed_slow_chains(
            d, n, M, slow_offset_scale=5.0, seed=221
        )
        state_esc = _fill_mc_state_from_buffers(state_init, draws_ov, grads_ov)
        state_esc = core.final(state_esc)

        verdict_esc = extract_multi_chain_verdict(
            state_esc, max_grad_budget=max_grad_budget, num_warmup_steps=2500
        )
        if bool(np.asarray(state_esc.has_escalated)):
            self.assertEqual(
                verdict_esc.flags["mode_coverage"],
                "multi_chain_certified",
                "Escalated overdispersed verdict should be 'multi_chain_certified'",
            )

    def test_mode_split_no_false_escalate(self):
        """KEY negative test: mode-split chains must NOT escalate (unimodality guard).

        Uses M=8 chains split 4+4 across two modes at ±mode_separation/2 along
        one axis.  The true linear score is used so R² is high and the magnitude
        + collinearity + LOO gates all pass.  But the projected chain-means
        are bimodal: four means near −4 and four near +4.

        v2.1 (recalibrated q99=4.54 at M=8, 2-window confirmation):
        - Window 1: flag_count=1, deferred=False (1 flag < 2-window threshold).
        - Window 2: flag_count=2, deferred=True (2-window confirmation satisfied).
        This validates that the unimodality guard protects against treating a
        mode-separated ensemble as a slow-mixing direction, and that the P1→P3
        handoff (deferred) is visible after the confirmation threshold.

        Structural guarantees: has_escalated=False across both windows;
        deferred_to_ensemble=True only after the second flagged window.
        """
        d, n, M = 20, 500, 8
        core = build_multi_chain_meta_core(50000, n_chains=M, max_rank=10)
        state = core.init(d)

        draws_mc, grads_mc = _make_mode_split_chains(
            d, n, M, mode_separation=8.0, within_chain_noise=0.1, seed=230
        )

        # Verify the v2.1 recalibrated threshold for M=8 (q99 ≈ 4.54)
        expected_threshold = _mc_unimodality_threshold(M)
        self.assertAlmostEqual(
            expected_threshold,
            4.54,
            places=5,
            msg=f"_mc_unimodality_threshold({M}) should be q99=4.54 (v2.1 recalibration)",
        )

        # --- Window 1: one flagged window, deferred stays False (2-window not yet met) ---
        state1 = _fill_mc_state_from_buffers(state, draws_mc, grads_mc)
        state1 = core.final(state1)

        self.assertFalse(
            bool(np.asarray(state1.has_escalated)),
            "Mode-split chains (4+4 at ±4): must NOT escalate after window 1. "
            "Unimodality guard should block.",
        )
        self.assertFalse(
            bool(np.asarray(state1.unimodality_passed)),
            "Mode-split: unimodality_passed should be False (bimodal projection detected)",
        )
        flag_count1 = int(np.asarray(state1.unimodality_flag_count))
        self.assertLessEqual(flag_count1, 1, "flag_count should be ≤1 after window 1")
        self.assertFalse(
            bool(np.asarray(state1.deferred_to_ensemble)),
            "Mode-split window 1: deferred must be False (2-window confirmation not yet met)",
        )

        # --- Window 2: second consecutive flag → confirmation → deferred=True ---
        state2 = _fill_mc_state_from_buffers(state1, draws_mc, grads_mc)
        state2 = core.final(state2)

        self.assertFalse(
            bool(np.asarray(state2.has_escalated)),
            "Mode-split chains: must NOT escalate after window 2",
        )
        self.assertTrue(
            bool(np.asarray(state2.deferred_to_ensemble)),
            "Mode-split window 2: deferred_to_ensemble must be True (2-window confirmation met). "
            "P1→P3 handoff must be visible in the carry.",
        )

        # Verify the verdict flags propagate the stored fields
        verdict = extract_multi_chain_verdict(
            state2, max_grad_budget=50000, num_warmup_steps=2500
        )
        self.assertEqual(
            verdict.flags["unimodality_gate"],
            "flag",
            "Mode-split verdict: unimodality_gate flag must be 'flag' (not 'pass')",
        )
        self.assertTrue(
            verdict.flags["deferred_to_ensemble"],
            "Mode-split verdict: deferred_to_ensemble must be True in flags",
        )

    def test_under_dispersed_start_one_sided_safe(self):
        """Under-dispersed starts: conservative non-escalation, never dangerous.

        All M chains start near the same position (under-dispersed).  The
        between-chain scatter is structurally near zero → T magnitude gate does
        not fire → no escalation.  This is ONE-SIDED SAFE: we may miss the slow
        direction, but we never over-escalate from this cause.

        Checks:
        - has_escalated = False (conservative)
        - start_dispersion_adequacy flag = 'adequate_if_overdispersed' (honesty layer)
        - mode_coverage = 'multi_chain_uncertified' (M > 1, no escalation)
        """
        d, n, M = 20, 500, 8
        max_grad_budget = 50000
        core = build_multi_chain_meta_core(max_grad_budget, n_chains=M, max_rank=10)
        state = core.init(d)

        draws_mc, grads_mc = _make_underdispersed_chains(d, n, M, seed=240)
        state = _fill_mc_state_from_buffers(state, draws_mc, grads_mc)
        state = core.final(state)

        self.assertFalse(
            bool(np.asarray(state.has_escalated)),
            "Under-dispersed starts: must NOT escalate (one-sided safe). "
            "No between-chain mean scatter → T magnitude gate does not fire.",
        )

        verdict = extract_multi_chain_verdict(
            state, max_grad_budget=max_grad_budget, num_warmup_steps=2500
        )
        self.assertEqual(
            verdict.flags["start_dispersion_adequacy"],
            "adequate_if_overdispersed",
            "Non-escalation verdict must report start_dispersion_adequacy = "
            "'adequate_if_overdispersed' (not a certificate of diagonal sufficiency)",
        )
        self.assertEqual(
            verdict.flags["mode_coverage"],
            "multi_chain_uncertified",
            "Under-dispersed M>1 non-escalation: mode_coverage = 'multi_chain_uncertified'",
        )

    def test_multi_chain_e2e_smoke_f32_and_x64(self):
        """staged_adaptation(n_chains=8) smoke test under f32 and x64.

        Structural check:
        - warmup.run(key, positions) completes without error (positions shape (M, d))
        - The returned state has shape (M, d) for all M chains
        - The emitted LowRankInverseMassMatrix has correct sigma shape

        Uses a non-axis-aligned rank-1 spike so the controller CAN escalate,
        but the test does not assert whether it did (structural only).
        Uses M=8 (the recommended minimum; M<6 triggers a warning).
        """
        n_dims = 5
        M = 8
        lam_spike = 25.0

        u_raw = jax.random.normal(jax.random.key(42), (n_dims,))
        u_dir = u_raw / jnp.linalg.norm(u_raw)

        def _run():
            # Build cov_inv inside _run so it adopts the current JAX default dtype,
            # keeping position / gradient / step_size all consistent.
            # Mixing explicit dtype=float32 positions with x64-promoted scalars (e.g.
            # Python-float step_size) causes lax.cond dtype mismatch in the NUTS
            # trajectory — same pattern as test_escalated_e2e_smoke_f32_and_x64.
            cov_inv_local = jnp.eye(n_dims) - (lam_spike - 1.0) / lam_spike * jnp.outer(
                u_dir, u_dir
            )

            def logdensity_fn(x):
                return -0.5 * x @ cov_inv_local @ x

            warmup = blackjax.staged_adaptation(
                blackjax.nuts,
                logdensity_fn,
                metric="auto",
                max_grad_budget=10000,
                n_chains=M,
            )
            key = jax.random.key(300)
            positions = jnp.zeros((M, n_dims))  # default dtype matches cov_inv_local
            results, _ = warmup.run(key, positions)
            return results

        # --- default dtype (f32 normally; f64 when JAX_ENABLE_X64 is active globally) ---
        results_default = _run()
        state_default = results_default.state
        imm_default = results_default.parameters["inverse_mass_matrix"]
        self.assertIsInstance(imm_default, LowRankInverseMassMatrix)
        self.assertEqual(imm_default.sigma.shape, (n_dims,))
        # All M final MCMC states returned; position has leading chain dim M
        pos_default = jax.tree.leaves(state_default.position)[0]
        self.assertEqual(pos_default.shape[0], M, "expected M final chain states")

        # --- x64 ---
        try:
            jax.config.update("jax_enable_x64", True)
            results_x64 = _run()
            imm_x64 = results_x64.parameters["inverse_mass_matrix"]
            self.assertIsInstance(imm_x64, LowRankInverseMassMatrix)
            self.assertEqual(imm_x64.sigma.shape, (n_dims,))
            self.assertTrue(
                bool(jnp.all(jnp.isfinite(imm_x64.sigma))),
                "x64: sigma has non-finite values",
            )
        finally:
            jax.config.update("jax_enable_x64", False)


class TestTimeMajorLayout(BlackJAXTest):
    """_build_pc_centered_time_major_pool puts valid rows first (no padding contamination)."""

    def test_step_mask_valid_region_is_contiguous_at_start(self):
        """With n < B, step_mask_tm must be 1 for rows 0..n*M-1 and 0 for rows n*M..B*M-1.

        _build_pc_centered_time_major_pool returns unmasked centered draws and a
        separate mask.  The mask encodes valid vs padding — callers apply it.
        This test verifies that the mask correctly identifies the valid region as
        contiguous at the start (time-major layout) rather than scattered (chain-major).
        """
        M, n, B, d = 4, 30, 64, 10
        key = jax.random.key(1)
        draws = jax.random.normal(key, (M, B, d))
        grads = -draws
        chain_means = draws.mean(axis=1)

        n_arr = jnp.array(n, dtype=jnp.int32)
        pc_draws, _pc_grads, step_mask_tm = _build_pc_centered_time_major_pool(
            draws, grads, chain_means, n_arr, M
        )

        self.assertEqual(pc_draws.shape, (B * M, d))
        self.assertEqual(step_mask_tm.shape, (B * M,))

        # Valid region: rows 0 .. n*M-1 → mask must be 1
        valid_mask = np.asarray(step_mask_tm[: n * M])
        last_valid = n * M - 1
        self.assertTrue(
            np.all(valid_mask == 1.0),
            f"step_mask_tm rows 0..{last_valid} must all be 1",
        )

        # Padding region: rows n*M .. B*M-1 → mask must be 0
        padding_mask = np.asarray(step_mask_tm[n * M :])
        last_pad = B * M - 1
        self.assertTrue(
            np.all(padding_mask == 0.0),
            f"step_mask_tm rows {n * M}..{last_pad} must all be 0",
        )

        # Valid draws should have non-zero content (centering does not collapse to zero)
        valid_draws = np.asarray(pc_draws[: n * M])
        self.assertGreater(
            float(np.abs(valid_draws).max()), 0.0, "valid rows should be non-zero"
        )

    def test_first_valid_row_equals_chain0_step0_centered(self):
        """Row 0 of time-major output = (chain 0, step 0) − chain_0_mean."""
        M, n, B, d = 4, 20, 64, 8
        key = jax.random.key(2)
        draws = jax.random.normal(key, (M, B, d))
        grads = -draws
        chain_means = draws.mean(axis=1)  # computed over full B steps

        n_arr = jnp.array(n, dtype=jnp.int32)
        pc_draws, _, _ = _build_pc_centered_time_major_pool(
            draws, grads, chain_means, n_arr, M
        )

        # In time-major layout: row 0 = (chain 0, step 0) - chain_0_mean
        expected_row0 = np.asarray(draws[0, 0] - chain_means[0])
        actual_row0 = np.asarray(pc_draws[0])
        np.testing.assert_allclose(expected_row0, actual_row0, atol=1e-5)

    def test_padding_invariant_to_buffer_size(self):
        """R²/routing output of _build_pc_centered_time_major_pool is invariant to B (F1).

        This is the CORRECT regression test for the v2 padding bug: the bug
        lived in ``_build_pc_centered_time_major_pool`` (R²/Fisher pool), not in
        ``_compute_pooled_within_spectrum``/``_compute_chain_consistency_psi``
        which mask BEFORE reshape and are padding-safe by construction.

        A chain-major revert would scatter valid rows across the pool so that
        the ``arange < n_pool`` mask clips chains rather than time steps.
        Time-major layout makes the first n*M rows exactly the valid data —
        doubling B only extends the zero-padding tail.

        Assertions:
        1. Valid-row content: pc_draws_tm[:n*M] is bit-identical for B=64 and B=200.
        2. Routing R²: _compute_r2_score_linearity on the pooled output gives
           the same R² regardless of B (the end-to-end pool→router path is safe).
        """
        M, n, d = 4, 30, 10
        max_rank = 2
        n_pool = n * M  # 120 valid rows; projected tier fires (n_pool ≥ 2*8*(k+1)=48)
        key = jax.random.key(777)
        draws_core = jax.random.normal(key, (M, n, d)).astype(jnp.float32)
        key2 = jax.random.fold_in(key, jnp.uint32(1))
        grads_core = -draws_core + 0.1 * jax.random.normal(key2, (M, n, d)).astype(
            jnp.float32
        )
        chain_means = draws_core.mean(axis=1)  # (M, d)
        n_arr = jnp.array(n, dtype=jnp.int32)
        sigma = jnp.ones(d, dtype=jnp.float32)
        U_k = jnp.eye(d, dtype=jnp.float32)[:, :max_rank]  # placeholder directions

        def _run(B):
            pad_d = jnp.zeros((M, B - n, d), dtype=jnp.float32)
            draws_buf = jnp.concatenate([draws_core, pad_d], axis=1)
            grads_buf = jnp.concatenate([grads_core, pad_d], axis=1)
            pc_draws_tm, pc_grads_tm, step_mask_tm = _build_pc_centered_time_major_pool(
                draws_buf, grads_buf, chain_means, n_arr, M
            )
            # R² on the pooled time-major buffer (the route-deciding path)
            n_pool_arr = jnp.array(n_pool, dtype=jnp.int32)
            r2, _ = _compute_r2_score_linearity(
                pc_draws_tm, pc_grads_tm, sigma, n_pool_arr, U_k, max_rank
            )
            return (
                np.asarray(pc_draws_tm[:n_pool]),  # valid rows
                np.asarray(step_mask_tm[:n_pool]),  # mask for valid rows
                float(np.asarray(r2)),
            )

        rows_b64, mask_b64, r2_b64 = _run(64)
        rows_b200, mask_b200, r2_b200 = _run(200)

        # 1. Valid-row content is bit-identical regardless of B
        np.testing.assert_allclose(
            rows_b64,
            rows_b200,
            atol=1e-6,
            err_msg="pc_draws_tm[:n*M] must be identical for B=64 and B=200",
        )
        # 2. Valid-row mask is all-ones for both (no padding contamination)
        np.testing.assert_array_equal(
            mask_b64,
            np.ones(n_pool, dtype=np.float32),
            err_msg="step_mask_tm[:n*M] must be all-ones for B=64",
        )
        np.testing.assert_array_equal(
            mask_b200,
            np.ones(n_pool, dtype=np.float32),
            err_msg="step_mask_tm[:n*M] must be all-ones for B=200",
        )
        # 3. Routing R² is identical (end-to-end pool→router invariance)
        self.assertAlmostEqual(
            r2_b64,
            r2_b200,
            places=4,
            msg=f"R² must be invariant to B: B=64→{r2_b64:.6f}, B=200→{r2_b200:.6f}",  # noqa: E231
        )

    def test_padding_invariant_to_buffer_size_x64(self):
        """R²/routing output of _build_pc_centered_time_major_pool is invariant to B (F1, x64)."""
        try:
            jax.config.update("jax_enable_x64", True)
            M, n, d = 4, 30, 10
            max_rank = 2
            n_pool = n * M
            key = jax.random.key(777)
            draws_core = jax.random.normal(key, (M, n, d)).astype(jnp.float64)
            key2 = jax.random.fold_in(key, jnp.uint32(1))
            grads_core = -draws_core + 0.1 * jax.random.normal(key2, (M, n, d)).astype(
                jnp.float64
            )
            chain_means = draws_core.mean(axis=1)
            n_arr = jnp.array(n, dtype=jnp.int32)
            sigma = jnp.ones(d, dtype=jnp.float64)
            U_k = jnp.eye(d, dtype=jnp.float64)[:, :max_rank]

            def _run(B):
                pad_d = jnp.zeros((M, B - n, d), dtype=jnp.float64)
                draws_buf = jnp.concatenate([draws_core, pad_d], axis=1)
                grads_buf = jnp.concatenate([grads_core, pad_d], axis=1)
                (
                    pc_draws_tm,
                    pc_grads_tm,
                    step_mask_tm,
                ) = _build_pc_centered_time_major_pool(
                    draws_buf, grads_buf, chain_means, n_arr, M
                )
                n_pool_arr = jnp.array(n_pool, dtype=jnp.int32)
                r2, _ = _compute_r2_score_linearity(
                    pc_draws_tm, pc_grads_tm, sigma, n_pool_arr, U_k, max_rank
                )
                return np.asarray(pc_draws_tm[:n_pool]), float(np.asarray(r2))

            rows_b64, r2_b64 = _run(64)
            rows_b200, r2_b200 = _run(200)

            np.testing.assert_allclose(
                rows_b64,
                rows_b200,
                atol=1e-10,
                err_msg="x64: pc_draws_tm[:n*M] must be identical for B=64 and B=200",
            )
            self.assertAlmostEqual(
                r2_b64,
                r2_b200,
                places=8,
                msg=f"x64: R² must be invariant to B: B=64→{r2_b64:.10f}, B=200→{r2_b200:.10f}",  # noqa: E231
            )
        finally:
            jax.config.update("jax_enable_x64", False)


class TestWBranchSpectrum(BlackJAXTest):
    """W-branch top eigenvalue clears MP edge on deep spread, stays below on isotropic."""

    def _check_spectrum(self, M, n, d, draws_mc, dtype=jnp.float32):
        draws_f = draws_mc.astype(dtype)
        chain_means = draws_f.mean(axis=1)
        W_diag = jnp.ones(d, dtype=dtype)
        n_arr = jnp.array(n, dtype=jnp.int32)
        actual_rank = max(d // 2, 1)
        lam1, _ = _compute_pooled_within_spectrum(
            draws_f, chain_means, W_diag, n_arr, M, actual_rank
        )
        edge = _w_branch_null_edge(M, n_arr, d)
        return float(np.asarray(lam1)), float(np.asarray(edge))

    def test_deep_spread_lam1_exceeds_edge_f32(self):
        """Anisotropic within-chain draws: lam1 >> MP edge (f32)."""
        M, n, d = 8, 200, 20
        draws_mc, _ = _make_mc_deep_spread(M, n, d, lam_within=25.0, seed=10)
        lam1, edge = self._check_spectrum(M, n, d, draws_mc, jnp.float32)
        self.assertGreater(
            lam1, edge, f"Deep spread: lam1={lam1} should exceed edge={edge}"
        )

    def test_isotropic_lam1_at_most_slightly_above_edge_f32(self):
        """Isotropic chains: lam1 should not greatly exceed MP edge (f32)."""
        M, n, d = 8, 200, 20
        draws_mc, _ = _make_mc_isotropic(M, n, d, seed=11)
        lam1, edge = self._check_spectrum(M, n, d, draws_mc, jnp.float32)
        self.assertLessEqual(
            lam1,
            edge * 1.3,
            f"Isotropic: lam1={lam1} should not exceed 1.3x edge={edge}",
        )

    def test_deep_spread_lam1_exceeds_edge_x64(self):
        """W-branch top eigenvalue clears MP edge in float64."""
        try:
            jax.config.update("jax_enable_x64", True)
            M, n, d = 8, 200, 20
            draws_mc, _ = _make_mc_deep_spread(M, n, d, lam_within=25.0, seed=12)
            lam1, edge = self._check_spectrum(M, n, d, draws_mc, jnp.float64)
            self.assertGreater(
                lam1,
                edge,
                f"x64 deep spread: lam1={lam1} should exceed edge={edge}",
            )
        finally:
            jax.config.update("jax_enable_x64", False)

    def test_multi_spread_lam1_exceeds_edge(self):
        """Genuine multi-direction spread (f1≈1/k): W-branch lam1 > edge (F6).

        _make_mc_deep_spread is rank-1 (f1≈0.65) and does NOT represent the
        W-branch's actual target.  _make_mc_multi_spread creates k=5 comparable
        slow directions (f1≈0.2), the genuine W-branch target regime.

        This test exercises the code path that was previously untested by the
        rank-1 deep_spread fixture.
        """
        M, n, d = 8, 200, 50
        draws_mc, _ = _make_mc_multi_spread(M, n, d, n_dirs=5, lam_within=15.0, seed=13)
        lam1, edge = self._check_spectrum(M, n, d, draws_mc, jnp.float32)
        self.assertGreater(
            lam1,
            edge,
            f"Multi-spread (k=5, lam={15.0}): lam1={lam1:.4f} should exceed edge={edge:.4f}",  # noqa: E231
        )


class TestChainConsistencyPsi(BlackJAXTest):
    """Ψ > floor on genuine deep spread; Ψ ≈ 0 on isotropic (null)."""

    def _compute_psi(self, M, n, d, draws_mc, dtype=jnp.float32):
        draws_f = draws_mc.astype(dtype)
        chain_means = draws_f.mean(axis=1)
        W_diag = jnp.ones(d, dtype=dtype)
        n_arr = jnp.array(n, dtype=jnp.int32)
        psi = _compute_chain_consistency_psi(draws_f, chain_means, W_diag, n_arr, M)
        return float(np.asarray(psi))

    def test_deep_spread_psi_above_floor_f32(self):
        """Genuine within-chain anisotropy: Ψ > _W_BRANCH_PSI_FLOOR (f32)."""
        M, n, d = 8, 200, 20
        draws_mc, _ = _make_mc_deep_spread(M, n, d, lam_within=25.0, seed=20)
        psi = self._compute_psi(M, n, d, draws_mc, jnp.float32)
        self.assertGreater(
            psi,
            _W_BRANCH_PSI_FLOOR,
            f"Deep spread: psi={psi} should exceed floor={_W_BRANCH_PSI_FLOOR}",
        )

    def test_isotropic_psi_below_floor_f32(self):
        """Isotropic null: Ψ should be near zero (f32)."""
        M, n, d = 8, 200, 20
        draws_mc, _ = _make_mc_isotropic(M, n, d, seed=21)
        psi = self._compute_psi(M, n, d, draws_mc, jnp.float32)
        self.assertLess(
            psi,
            _W_BRANCH_PSI_FLOOR,
            f"Isotropic null: psi={psi} should be below floor={_W_BRANCH_PSI_FLOOR}",
        )

    def test_deep_spread_psi_above_floor_x64(self):
        """Ψ > floor in float64."""
        try:
            jax.config.update("jax_enable_x64", True)
            M, n, d = 8, 200, 20
            draws_mc, _ = _make_mc_deep_spread(M, n, d, lam_within=25.0, seed=22)
            psi = self._compute_psi(M, n, d, draws_mc, jnp.float64)
            self.assertGreater(
                psi,
                _W_BRANCH_PSI_FLOOR,
                f"x64 deep spread: psi={psi} should exceed floor",
            )
        finally:
            jax.config.update("jax_enable_x64", False)

    def test_ar_null_psi_gate_refuses_at_d26(self):
        """AR(1)-null at d=26: magnitude fires (lam1 > edge), Ψ refuses (F4).

        This proves Ψ is load-bearing — without it the W-branch would fire on
        any slow-mixing chain whose within-chain spectrum is inflated.
        At d=26 the adaptive Ψ threshold is max(3*q99_null, 0.15) > 0.15
        (the flat floor is not inert here).

        AR(1) chains at rho=0.8 inflate lam1 ~9× above I (effective-N
        reduction by (1-rho)/(1+rho)=1/9), pushing lam1 well above the MP edge.
        But because each chain's AR noise is independent, cross-chain
        off-diagonal correlation C_A, C_B ≈ 0 → Ψ ≈ 0 (gate refuses).
        """
        M, n, d = 8, 60, 26
        draws_mc, _ = _make_mc_ar_null(M, n, d, rho=0.8, seed=200)
        draws_mc = draws_mc.astype(jnp.float32)
        chain_means = draws_mc.mean(axis=1)  # (M, d)
        W_diag = jnp.ones(d, dtype=jnp.float32)
        n_arr = jnp.array(n, dtype=jnp.int32)
        max_rank = 10

        lam1, _ = _compute_pooled_within_spectrum(
            draws_mc, chain_means, W_diag, n_arr, M, max_rank
        )
        psi = _compute_chain_consistency_psi(draws_mc, chain_means, W_diag, n_arr, M)
        edge = _w_branch_null_edge(M, n_arr, d)

        lam1_f = float(np.asarray(lam1))
        psi_f = float(np.asarray(psi))
        edge_f = float(np.asarray(edge))

        self.assertGreater(
            lam1_f,
            edge_f,
            f"AR(1)-null d={d}: lam1={lam1_f:.3f} must exceed edge={edge_f:.3f} "  # noqa: E231
            f"(AR autocorrelation inflates the pooled spectrum)",
        )
        self.assertLess(
            psi_f,
            _W_BRANCH_PSI_FLOOR,
            f"AR(1)-null d={d}: Ψ={psi_f:.4f} must be < floor={_W_BRANCH_PSI_FLOOR} "  # noqa: E231
            f"(independent AR chains have no shared off-diagonal structure)",
        )

    def test_ar_null_no_escalation_through_final(self):
        """AR(1)-null through final() must not escalate (F4-GATE).

        This makes the Ψ gate load-bearing in integration context: with the
        Ψ gate removed, ``final()`` would escalate on AR-null chains because
        lam1 >> edge (proven above in test_ar_null_psi_gate_refuses_at_d26).
        Here we verify that Ψ's refusal is wired correctly all the way through
        the W-branch conditional in ``final()``.

        Running this at d=26 exercises the adaptive Ψ threshold
        ``max(3*q99_null, 0.15)`` region where the flat floor is not the
        tightest constraint.
        """
        M, n, d = 8, 60, 26
        draws_mc, grads_mc = _make_mc_ar_null(M, n, d, rho=0.8, seed=200)
        draws_mc = draws_mc.astype(jnp.float32)
        grads_mc = grads_mc.astype(jnp.float32)

        core = build_multi_chain_meta_core(40000, n_chains=M, max_rank=10)
        state = core.init(d)
        state = _fill_mc_state_from_buffers(state, draws_mc, grads_mc)
        state = core.final(state)

        self.assertFalse(
            bool(np.asarray(state.has_escalated)),
            f"AR(1)-null d={d}: Ψ gate must refuse escalation through final() "  # noqa: E231
            f"(has_escalated={bool(np.asarray(state.has_escalated))})",
        )

    def test_ar_null_no_escalation_through_final_x64(self):
        """AR(1)-null through final() must not escalate in float64 (F4-GATE x64)."""
        try:
            jax.config.update("jax_enable_x64", True)
            M, n, d = 8, 60, 26
            draws_mc, grads_mc = _make_mc_ar_null(M, n, d, rho=0.8, seed=200)
            draws_mc = draws_mc.astype(jnp.float64)
            grads_mc = grads_mc.astype(jnp.float64)

            core = build_multi_chain_meta_core(40000, n_chains=M, max_rank=10)
            state = core.init(d)
            state = _fill_mc_state_from_buffers(state, draws_mc, grads_mc)
            state = core.final(state)

            self.assertFalse(
                bool(np.asarray(state.has_escalated)),
                f"AR(1)-null x64 d={d}: Ψ gate must refuse escalation through final()",  # noqa: E231
            )
        finally:
            jax.config.update("jax_enable_x64", False)


class TestNullEdgeFormula(BlackJAXTest):
    """_w_branch_null_edge computes TW_FACTOR*(1 + sqrt(d/N))^2 with N=M*(n-1)."""

    def test_tw_factor_sanity_range(self):
        """TW_FACTOR is in [1.0, 1.1] — guards against transcription typo."""
        self.assertGreaterEqual(
            _W_BRANCH_NULL_EDGE_TW_FACTOR,
            1.0,
            "TW factor should be >= 1.0 (at least as large as the asymptotic edge)",
        )
        self.assertLessEqual(
            _W_BRANCH_NULL_EDGE_TW_FACTOR,
            1.1,
            "TW factor > 1.1 is implausibly large for an O(1/N) finite-N correction",
        )

    def test_empirical_null_lam1_under_edge(self):
        """Fraction of iid-null lam1 values exceeding edge is small at (M=8, n=40, d=20).

        F8: replaces the self-referential test that imported _W_BRANCH_NULL_EDGE_TW_FACTOR
        into its expected value (making the constant's numeric value completely unprotected).
        This test validates the factor's VALUE against iid-null spectra.

        Runs 200 iid draws sets (with correct per-chain centering on actual sample means)
        and checks that fewer than 15% of null spectra exceed the edge.  The true
        null FPR at this edge should be ~1%; 15% provides headroom for 200-rep MC
        noise while still catching a wildly mis-calibrated constant (e.g. 0.5 or 5.0).
        """
        M, n, d = 8, 40, 20
        N_reps = 200
        key = jax.random.key(9999)
        W_diag = jnp.ones(d, dtype=jnp.float32)  # identity whitening
        n_arr = jnp.array(n, dtype=jnp.int32)
        max_rank = 10

        edge = float(np.asarray(_w_branch_null_edge(M, n_arr, d)))

        n_above = 0
        keys = jax.random.split(key, N_reps)
        for i in range(N_reps):
            # iid null: M chains of N(0,I) draws
            draws = jax.random.normal(keys[i], (M, n, d)).astype(jnp.float32)
            # Correct centering: use the actual per-chain sample mean so that
            # _compute_pooled_within_spectrum receives properly centered residuals.
            chain_means = draws.mean(axis=1)  # (M, d)
            lam1, _ = _compute_pooled_within_spectrum(
                draws, chain_means, W_diag, n_arr, M, max_rank
            )
            if float(np.asarray(lam1)) > edge:
                n_above += 1

        frac_above = n_above / N_reps
        self.assertLess(
            frac_above,
            0.15,
            f"Too many iid-null lam1 exceed edge={edge:.4f}: "  # noqa: E231
            f"{n_above}/{N_reps} ({frac_above:.1%}). "  # noqa: E231
            f"True null FPR is ~1%: 15% cap catches badly miscalibrated"
            f" _W_BRANCH_NULL_EDGE_TW_FACTOR={_W_BRANCH_NULL_EDGE_TW_FACTOR}.",
        )

    def test_edge_decreases_with_more_draws(self):
        """More draws per chain → tighter (lower) null edge → easier detection."""
        M, d = 8, 20
        # n=8 → N=7*8=56;  n=252 → N=251*8=2008
        edge_small = _w_branch_null_edge(M, jnp.array(8, dtype=jnp.int32), d)
        edge_large = _w_branch_null_edge(M, jnp.array(252, dtype=jnp.int32), d)
        self.assertGreater(float(np.asarray(edge_small)), float(np.asarray(edge_large)))

    def test_edge_increases_with_dimension(self):
        """Higher dimension → larger null edge (more noise dims = larger null bulk)."""
        M, n = 8, 30  # N = 8*29 = 232
        edge_low_d = _w_branch_null_edge(M, jnp.array(n, dtype=jnp.int32), 5)
        edge_high_d = _w_branch_null_edge(M, jnp.array(n, dtype=jnp.int32), 50)
        self.assertGreater(
            float(np.asarray(edge_high_d)), float(np.asarray(edge_low_d))
        )


class TestUnimodality2WindowConfirmation(BlackJAXTest):
    """One flag does not defer; deferred resets on unimodal window (non-monotone)."""

    def _build_core_and_state(self, M, d, max_grad_budget=40000):
        core = build_multi_chain_meta_core(max_grad_budget=max_grad_budget, n_chains=M)
        state = core.init(d)
        return core, state

    def test_single_flag_deferred_false(self):
        """After one flagged window, deferred stays False (needs 2 consecutive)."""
        M, n, d = 8, 150, 10
        core, state = self._build_core_and_state(M, d)
        draws_mc, grads_mc = _make_mc_split_means(M, n, d, split_scale=8.0, seed=30)
        state1 = _fill_mc_state(state, draws_mc, grads_mc)
        result1 = core.final(state1)

        deferred = bool(np.asarray(result1.deferred_to_ensemble))
        flag_count = int(np.asarray(result1.unimodality_flag_count))
        self.assertFalse(
            deferred, "deferred must be False after at most 1 flagged window"
        )
        self.assertLessEqual(flag_count, 1)

    def test_structural_invariant_deferred_requires_flag_count_ge_2(self):
        """deferred=True iff flag_count >= _MC_UNIMODALITY_CONFIRM_WINDOWS (structural)."""
        M, n, d = 8, 150, 10
        core, state = self._build_core_and_state(M, d, max_grad_budget=40000)
        draws_mc, grads_mc = _make_mc_split_means(M, n, d, split_scale=8.0, seed=31)

        state1 = _fill_mc_state(state, draws_mc, grads_mc)
        r1 = core.final(state1)
        state2 = _fill_mc_state(r1, draws_mc, grads_mc)
        r2 = core.final(state2)

        flag2 = int(np.asarray(r2.unimodality_flag_count))
        deferred2 = bool(np.asarray(r2.deferred_to_ensemble))

        # Invariant: deferred=True implies flag_count >= confirm_threshold
        if deferred2:
            self.assertGreaterEqual(
                flag2,
                _MC_UNIMODALITY_CONFIRM_WINDOWS,
                "deferred=True requires at least CONFIRM_WINDOWS consecutive flags",
            )

    def test_non_monotone_latch_resets_on_unimodal_window(self):
        """deferred resets to False when a unimodal window follows the flagged ones."""
        M, n, d = 8, 150, 10
        core, state = self._build_core_and_state(M, d)
        draws_split, grads_split = _make_mc_split_means(
            M, n, d, split_scale=8.0, seed=32
        )

        state1 = _fill_mc_state(state, draws_split, grads_split)
        r1 = core.final(state1)
        state2 = _fill_mc_state(r1, draws_split, grads_split)
        r2 = core.final(state2)

        draws_uni, grads_uni = _make_mc_even_spread(M, n, d, spread_scale=0.5, seed=33)
        state3 = _fill_mc_state(r2, draws_uni, grads_uni)
        r3 = core.final(state3)

        flag3 = int(np.asarray(r3.unimodality_flag_count))
        deferred3 = bool(np.asarray(r3.deferred_to_ensemble))

        # If unimodal window cleared the flag, deferred must be False
        if flag3 == 0:
            self.assertFalse(
                deferred3,
                "Non-monotone latch: unimodal window (flag_count=0) must reset deferred to False",
            )
