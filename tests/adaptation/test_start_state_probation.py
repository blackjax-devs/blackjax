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
"""Tests for the post-warmup per-chain start-state probation mechanism (Issue#1064).

Coverage:
- TestLooStateGate: stage 1 (ensemble-calibrated LOO gate) as a pure function.
- TestRedrawFlaggedChains: stage 2a (redraw near a healthy anchor), including
  a dict/PyTree position case (guards the blind spot fixed in PR #1013).
- TestComputeChainEnergies: the energy OBSERVABLE (amendment A; not a trigger).
- TestStagedAdaptationProbationWiring: opt-in validation, RNG-disable-path
  identity (amendment B), and a dict-position smoke test through the full
  vmapped probation scan (amendment C).
- TestApplyStartStateProbationRedCheck: the red-check promised in the
  ratified design note -- an engineered bad post-warmup start state (i)
  reproduces a divergence-storm transient in miniature when the mechanism is
  disabled, and (ii) is materially rehabilitated when it is enabled.
- TestExtractMultiChainVerdictProbationResult: the verdict/flags fold-in.
"""
import jax
import jax.numpy as jnp
import numpy as np

import blackjax
from blackjax.adaptation.base import return_all_adapt_info
from blackjax.adaptation.meta._calibration import (
    _PROBATION_WINDOW_DEFAULT,
    _STATE_GATE_LOO_Z_THRESHOLD,
    _STATE_GATE_MIN_CHAINS,
)
from blackjax.adaptation.meta._start_state import (
    compute_chain_energies,
    loo_state_gate,
    redraw_flagged_chains,
    select_healthy_anchor,
)
from blackjax.adaptation.meta.verdict import extract_multi_chain_verdict
from blackjax.adaptation.staged_adaptation import (
    StagedAdaptationInfo,
    _apply_start_state_probation,
)
from blackjax.mcmc.metrics import LowRankInverseMassMatrix
from tests.adaptation._meta_fixtures import _make_mc_isotropic
from tests.fixtures import BlackJAXTest

_IDENTITY_METRIC = lambda d: LowRankInverseMassMatrix(  # noqa: E731
    sigma=jnp.ones(d), U=jnp.zeros((d, 1)), lam=jnp.ones(1)
)


# ---------------------------------------------------------------------------
# Stage 1: ensemble-calibrated LOO state gate
# ---------------------------------------------------------------------------


class TestLooStateGate(BlackJAXTest):
    def test_flags_clear_outlier(self):
        """A single far outlier among 7 tightly-clustered chains gets flagged."""
        ld = jnp.array([-10.0, -10.2, -9.8, -10.1, -9.9, -10.05, -9.95, -50.0])
        mask = loo_state_gate(ld, _STATE_GATE_LOO_Z_THRESHOLD, _STATE_GATE_MIN_CHAINS)
        self.assertTrue(bool(mask[7]), "Clear outlier chain must be flagged")
        self.assertFalse(
            bool(jnp.any(mask[:7])), "Tightly-clustered chains must not be flagged"
        )

    def test_no_flag_when_uniform(self):
        """No chain is flagged when the ensemble's logdensities are all comparable."""
        key = jax.random.key(0)
        ld = -10.0 + 0.1 * jax.random.normal(key, (8,))
        mask = loo_state_gate(ld, _STATE_GATE_LOO_Z_THRESHOLD, _STATE_GATE_MIN_CHAINS)
        self.assertFalse(bool(jnp.any(mask)))

    def test_skipped_below_min_chains(self):
        """Below _STATE_GATE_MIN_CHAINS the gate is skipped (no flags), even
        for data that would otherwise clearly flag -- the LOO reference from
        only 1-2 other chains is too noisy to act on."""
        ld = jnp.array([-10.0, -10.0, -1000.0])  # M=3 < _STATE_GATE_MIN_CHAINS=4
        mask = loo_state_gate(ld, _STATE_GATE_LOO_Z_THRESHOLD, _STATE_GATE_MIN_CHAINS)
        self.assertFalse(bool(jnp.any(mask)))

    def test_healthy_anchor_excludes_flagged(self):
        """select_healthy_anchor never returns a flagged chain's index."""
        ld = jnp.array([-10.0, -10.2, -9.8, -10.1, -9.9, -10.05, -9.95, -50.0])
        mask = loo_state_gate(ld, _STATE_GATE_LOO_Z_THRESHOLD, _STATE_GATE_MIN_CHAINS)
        anchor = int(select_healthy_anchor(ld, mask))
        self.assertNotEqual(anchor, 7, "Anchor must not be the flagged outlier")
        self.assertIn(anchor, range(7))


# ---------------------------------------------------------------------------
# Stage 2a: redraw near a healthy anchor
# ---------------------------------------------------------------------------


class TestRedrawFlaggedChains(BlackJAXTest):
    def test_redraw_moves_flagged_near_anchor_flat_array(self):
        M, d = 8, 4
        position = jnp.zeros((M, d)).at[7].set(50.0)
        flagged = jnp.zeros((M,), dtype=bool).at[7].set(True)
        anchor_idx = jnp.array(0)
        key = jax.random.key(1)

        new_pos = redraw_flagged_chains(
            position, flagged, anchor_idx, _IDENTITY_METRIC(d), key, 0.1
        )

        np.testing.assert_array_equal(
            np.asarray(new_pos[:7]),
            np.asarray(position[:7]),
            err_msg="Non-flagged chains must be bit-identical after redraw",
        )
        dist_before = float(jnp.linalg.norm(position[7] - position[0]))
        dist_after = float(jnp.linalg.norm(new_pos[7] - position[0]))
        self.assertGreater(dist_before, 40.0, "sanity: chain 7 was far before redraw")
        self.assertLess(
            dist_after, 2.0, f"Redrawn chain should land near the anchor: {dist_after}"
        )

    def test_redraw_dict_position(self):
        """PyTree (dict) position case -- guards the blind spot fixed in PR #1013."""
        M = 8
        position = {
            "a": jnp.zeros((M, 3)).at[3].set(20.0),
            "b": jnp.zeros((M, 2)).at[3].set(-20.0),
        }
        flagged = jnp.zeros((M,), dtype=bool).at[3].set(True)
        anchor_idx = jnp.array(0)
        key = jax.random.key(2)

        new_pos = redraw_flagged_chains(
            position, flagged, anchor_idx, _IDENTITY_METRIC(5), key, 0.1
        )

        self.assertEqual(set(new_pos.keys()), {"a", "b"})
        np.testing.assert_array_equal(
            np.asarray(new_pos["a"][0]), np.asarray(position["a"][0])
        )
        np.testing.assert_array_equal(
            np.asarray(new_pos["b"][2]), np.asarray(position["b"][2])
        )
        dist_a = float(jnp.linalg.norm(new_pos["a"][3] - position["a"][0]))
        dist_b = float(jnp.linalg.norm(new_pos["b"][3] - position["b"][0]))
        self.assertLess(dist_a, 2.0)
        self.assertLess(dist_b, 2.0)


# ---------------------------------------------------------------------------
# Energy observable (amendment A)
# ---------------------------------------------------------------------------


class TestComputeChainEnergies(BlackJAXTest):
    def test_shape_and_finite(self):
        M, d = 8, 5
        draws_mc, _ = _make_mc_isotropic(M, 10, d, seed=3)
        position = draws_mc[:, 0, :]
        logdensity = -0.5 * jnp.sum(position**2, axis=1)
        key = jax.random.key(4)

        energies = compute_chain_energies(
            position, logdensity, _IDENTITY_METRIC(d), key
        )

        self.assertEqual(energies.shape, (M,))
        self.assertTrue(bool(jnp.all(jnp.isfinite(energies))))


# ---------------------------------------------------------------------------
# staged_adaptation wiring: validation, RNG discipline, dict-position smoke
# ---------------------------------------------------------------------------


class TestStagedAdaptationProbationWiring(BlackJAXTest):
    def _logdensity(self, x):
        return -0.5 * jnp.sum(x**2)

    def test_requires_multichain(self):
        with self.assertRaisesRegex(ValueError, "start_state_probation"):
            blackjax.staged_adaptation(
                blackjax.nuts,
                self._logdensity,
                metric="auto",
                max_grad_budget=5000,
                n_chains=1,
                start_state_probation=True,
            )

    def test_disabled_path_bit_identical_kwarg_omitted_vs_explicit_false(self):
        """RNG discipline (amendment B): the disabled path never touches the
        probation-only RNG split, so omitting the kwarg and passing it
        explicitly as False must give bit-identical results."""
        n_dims, n_chains = 5, 8
        key = jax.random.key(20)
        pos = jnp.zeros((n_chains, n_dims))

        warmup_omitted = blackjax.staged_adaptation(
            blackjax.nuts,
            self._logdensity,
            metric="auto",
            max_grad_budget=8000,
            n_chains=n_chains,
        )
        warmup_explicit = blackjax.staged_adaptation(
            blackjax.nuts,
            self._logdensity,
            metric="auto",
            max_grad_budget=8000,
            n_chains=n_chains,
            start_state_probation=False,
        )

        results_omitted, info_omitted = warmup_omitted.run(key, pos, num_steps=40)
        results_explicit, info_explicit = warmup_explicit.run(key, pos, num_steps=40)

        np.testing.assert_array_equal(
            np.asarray(results_omitted.state.position),
            np.asarray(results_explicit.state.position),
            err_msg="Disabled path: omitted vs explicit False must be bit-identical",
        )
        np.testing.assert_array_equal(
            np.asarray(results_omitted.parameters["step_size"]),
            np.asarray(results_explicit.parameters["step_size"]),
        )
        self.assertNotIn("start_state_probation", results_omitted.parameters)
        self.assertNotIn("start_state_probation", results_explicit.parameters)
        # Same leading (step) dim on both -- no extra steps folded in when disabled.
        self.assertEqual(
            jax.tree.leaves(info_omitted)[0].shape[0],
            jax.tree.leaves(info_explicit)[0].shape[0],
        )

    def test_enabled_dict_position_smoke(self):
        """Dict/PyTree position through the FULL vmapped probation scan
        (amendment C) -- guards the blind spot fixed in PR #1013."""
        n_chains = 8

        def logdensity_fn(x):
            return -0.5 * jnp.sum(x["a"] ** 2) - 0.5 * jnp.sum(x["b"] ** 2)

        position = {
            "a": jnp.zeros((n_chains, 3)),
            "b": jnp.zeros((n_chains, 2)),
        }
        warmup = blackjax.staged_adaptation(
            blackjax.nuts,
            logdensity_fn,
            metric="auto",
            max_grad_budget=8000,
            n_chains=n_chains,
            start_state_probation=True,
            probation_window=10,
        )
        key = jax.random.key(21)
        results, info = warmup.run(key, position, num_steps=40)

        self.assertEqual(set(results.state.position.keys()), {"a", "b"})
        self.assertEqual(results.state.position["a"].shape, (n_chains, 3))
        self.assertEqual(results.state.position["b"].shape, (n_chains, 2))
        # Issue#1090 fix 1: diagnostics never land in `parameters` -- it must
        # stay a pure sampler-kwargs dict.
        self.assertNotIn("start_state_probation", results.parameters)
        self.assertIsInstance(info, StagedAdaptationInfo)
        diag = info.diagnostics
        self.assertTrue(diag["enabled"])
        self.assertEqual(diag["window"], 10)
        self.assertEqual(
            jax.tree.leaves(info.trace)[0].shape[0],
            40 + 10,
            "info.trace must be the main warmup steps plus the probation window",
        )

    def test_dict_position_engineered_outlier_through_stage1_2a_2b(self):
        """Amendment C (v2): a dict/PyTree position with an ENGINEERED outlier
        chain driven through stage 1 + 2a + 2b end-to-end via
        _apply_start_state_probation directly -- confirms stage 2a's jitter
        and redraw are pytree-correct PER KEY (not just structurally
        non-crashing, which test_enabled_dict_position_smoke above already
        covers for the identical-start case where nothing gets flagged)."""
        M = 8

        def logdensity_fn(x):
            return -0.5 * jnp.sum(x["a"] ** 2) - 0.5 * jnp.sum(x["b"] ** 2)

        position = {
            "a": jnp.zeros((M, 3)).at[7].set(30.0),
            "b": jnp.zeros((M, 2)).at[7].set(-30.0),
        }
        states = jax.vmap(lambda pos: blackjax.nuts.init(pos, logdensity_fn))(position)
        mcmc_kernel = blackjax.nuts.build_kernel()
        step_size = jnp.asarray(0.5)
        metric = _IDENTITY_METRIC(5)

        rehab_states, _probation_info, diagnostics = _apply_start_state_probation(
            jax.random.key(30),
            states,
            mcmc_kernel,
            blackjax.nuts.init,
            logdensity_fn,
            step_size,
            metric,
            M,
            _PROBATION_WINDOW_DEFAULT,
            {},
            return_all_adapt_info,
            states,
        )

        self.assertIn(
            7, diagnostics["flagged_chain_idx"], "Engineered outlier must be flagged"
        )
        self.assertTrue(diagnostics["redraw_applied"])
        self.assertEqual(set(rehab_states.position.keys()), {"a", "b"})
        self.assertEqual(rehab_states.position["a"].shape, (M, 3))
        self.assertEqual(rehab_states.position["b"].shape, (M, 2))

        # Amendment A': pre/post logdensity pair is auditable against stage
        # 1's own decision -- the outlier chain's logdensity must move
        # substantially toward the healthy ensemble's range in BOTH dict
        # keys' worth of mass (a single flat number, but only reachable if
        # both "a" and "b" were correctly redrawn -- a pytree bug that only
        # fixed one key would leave the other contributing -450 alone).
        pre_ld = diagnostics["pre_probation_logdensity"]
        post_ld = diagnostics["post_probation_logdensity"]
        self.assertLess(
            pre_ld[7], -500.0, "sanity: engineered chain must be a clear outlier"
        )
        self.assertGreater(
            post_ld[7],
            pre_ld[7] + 400.0,
            "Rehabilitated chain's logdensity must move substantially "
            "toward the ensemble in both pytree keys",
        )


# ---------------------------------------------------------------------------
# Issue#1090: PR #1015 integration defects found by the P1 corpus-rerun
# consumer.  Each class below reproduces one of the two repro patterns +
# the tuple-shaped info_fn regression named in the issue.
# ---------------------------------------------------------------------------


class TestIssue1090ParametersDeployPatternRegression(BlackJAXTest):
    """Fix 1 red-check: PR #1015 injected a diagnostics dict under
    ``parameters["start_state_probation"]``, so the standard sampler-deploy
    pattern ``algorithm(logdensity_fn, **result.parameters)`` -- used
    throughout BlackJAX's own docs and by Paper 1's fixed-suite harness --
    raised ``TypeError: unexpected keyword argument 'start_state_probation'``
    whenever ``start_state_probation=True``.  Reproduces exactly that call."""

    def test_deploy_pattern_does_not_raise(self):
        n_chains = 8

        def logdensity_fn(x):
            return -0.5 * jnp.sum(x**2)

        position = jnp.zeros((n_chains, 4))
        warmup = blackjax.staged_adaptation(
            blackjax.nuts,
            logdensity_fn,
            metric="auto",
            max_grad_budget=8000,
            n_chains=n_chains,
            start_state_probation=True,
            probation_window=10,
        )
        results, info = warmup.run(jax.random.key(40), position, num_steps=40)

        self.assertNotIn("start_state_probation", results.parameters)

        # The documented deploy pattern -- must not raise TypeError.
        sampler = blackjax.nuts(logdensity_fn, **results.parameters)
        keys = jax.random.split(jax.random.key(41), n_chains)
        new_state, sample_info = jax.vmap(sampler.step)(keys, results.state)

        self.assertEqual(new_state.position.shape, (n_chains, 4))


class TestIssue1090ScheduleBoundaryRegression(BlackJAXTest):
    """Fix 2 red-check: the probation window's extra steps were
    concatenated onto the info stream past the caller's declared
    ``num_steps`` with no explicit boundary exposed, so a
    prescribed-schedule-length consumer (e.g. Paper 1's
    ``extract_schedule_events``, which asserts an exact-length trace against
    ``num_warmup_steps``) broke.  Reproduces a boundary-aware trace
    consumer against ``info.diagnostics["warmup_boundary_index"]``."""

    def test_boundary_index_recovers_prescribed_length(self):
        n_chains, num_steps, window = 8, 40, 10

        def logdensity_fn(x):
            return -0.5 * jnp.sum(x**2)

        position = jnp.zeros((n_chains, 4))
        warmup = blackjax.staged_adaptation(
            blackjax.nuts,
            logdensity_fn,
            metric="auto",
            max_grad_budget=8000,
            n_chains=n_chains,
            start_state_probation=True,
            probation_window=window,
        )
        results, info = warmup.run(jax.random.key(42), position, num_steps=num_steps)

        self.assertIsInstance(info, StagedAdaptationInfo)
        boundary = info.diagnostics["warmup_boundary_index"]
        self.assertEqual(boundary, num_steps)

        full_trace_len = jax.tree.leaves(info.trace)[0].shape[0]
        self.assertEqual(full_trace_len, num_steps + window)

        # A prescribed-boundary trace consumer (mirrors Paper 1's
        # extract_schedule_events) slices to the boundary and recovers
        # exactly the caller's declared num_warmup_steps.
        prescribed_trace = jax.tree.map(lambda x: x[:boundary], info.trace)
        self.assertEqual(
            jax.tree.leaves(prescribed_trace)[0].shape[0],
            num_steps,
            "trace sliced at warmup_boundary_index must equal num_warmup_steps",
        )


class TestIssue1090ExtraGradEvalsTupleInfoFnRegression(BlackJAXTest):
    """Fix 3 red-check: ``extra_grad_evals`` fell back to a silent ``-1``
    sentinel whenever the caller's ``adaptation_info_fn`` didn't shape its
    record with a ``.info`` attribute -- e.g. a bare-tuple info_fn, which is
    a valid ``adaptation_info_fn`` per the documented ``(state, info,
    adaptation_state) -> Any`` signature (``return_all_adapt_info``'s
    ``AdaptationInfo`` NamedTuple is only the default, not a requirement)."""

    def test_tuple_shaped_info_fn_does_not_silently_zero_the_ledger(self):
        n_chains = 8

        def logdensity_fn(x):
            return -0.5 * jnp.sum(x**2)

        def tuple_info_fn(state, info, adaptation_state):
            # A bare tuple has no `.info` attribute at all, unlike the
            # default return_all_adapt_info's AdaptationInfo(state, info,
            # adaptation_state) NamedTuple.
            return (state, info)

        position = jnp.zeros((n_chains, 4))
        warmup = blackjax.staged_adaptation(
            blackjax.nuts,
            logdensity_fn,
            metric="auto",
            max_grad_budget=8000,
            n_chains=n_chains,
            adaptation_info_fn=tuple_info_fn,
            start_state_probation=True,
            probation_window=10,
        )
        results, info = warmup.run(jax.random.key(43), position, num_steps=40)

        self.assertIsInstance(info, StagedAdaptationInfo)
        extra_grad_evals = info.diagnostics["extra_grad_evals"]
        self.assertGreater(
            extra_grad_evals,
            0,
            "extra_grad_evals must be a real positive count, not the silent "
            "-1 sentinel, regardless of adaptation_info_fn's return shape",
        )


class TestIssue1090ExtraGradEvalsX64Regression(BlackJAXTest):
    """Regression for a scan-carry dtype mismatch caught by the downstream
    consumer's own repro convention (``JAX_ENABLE_X64=1``, matching the
    paper1 harness).  An earlier version of the fix 3 patch accumulated
    ``extra_grad_evals`` in the probation scan's CARRY
    (``grad_evals_p + jnp.sum(infos_p.num_integration_steps)``), which raised
    ``jax.lax.scan``'s "carry input and carry output must have equal types"
    (``int32[]`` in, ``int64[]`` out) once 64-bit ints were enabled --
    ``jnp.sum``'s integer-promotion result dtype does not match a hardcoded
    ``jnp.int32`` carry dtype under x64.  Exercises the full multi-chain
    probation path inside ``jax.enable_x64()`` (the repo's established x64
    test idiom; see ``StagedAdaptationX64SmokeTest`` in
    ``test_staged_adaptation.py``)."""

    def test_probation_scan_does_not_raise_under_x64(self):
        n_chains = 8

        def logdensity_fn(x):
            return -0.5 * jnp.sum(x**2)

        with jax.enable_x64():
            position = jnp.zeros((n_chains, 4), dtype=jnp.float64)
            warmup = blackjax.staged_adaptation(
                blackjax.nuts,
                logdensity_fn,
                metric="auto",
                max_grad_budget=8000,
                n_chains=n_chains,
                start_state_probation=True,
                probation_window=10,
            )
            results, info = warmup.run(jax.random.key(44), position, num_steps=40)

            self.assertIsInstance(info, StagedAdaptationInfo)
            extra_grad_evals = info.diagnostics["extra_grad_evals"]
            self.assertGreater(
                extra_grad_evals,
                0,
                "the probation scan must complete under x64 and produce a "
                "real positive grad-eval count",
            )


# ---------------------------------------------------------------------------
# Red-check: engineered bad post-warmup start state (ratified design note)
# ---------------------------------------------------------------------------


class TestApplyStartStateProbationRedCheck(BlackJAXTest):
    """One chain is placed deep in a funnel's neck (tight local curvature)
    while the other 7 sit near the mouth -- mirroring "a single chain handed
    a bad post-warmup start state" from Issue#1064.  A step size fine for
    the mouth is engineered to reliably diverge the neck chain.
    """

    _D = 6
    _N_HEALTHY = 7
    _STEP_SIZE = 0.5

    def _funnel_logdensity(self, x):
        x0 = x[0]
        tail = x[1:]
        return (
            -0.5 * (x0 / 3.0) ** 2
            - 0.5 * (self._D - 1) * x0
            - 0.5 * jnp.sum(tail**2) * jnp.exp(-x0)
        )

    def _make_states(self):
        key = jax.random.key(7)
        healthy_key, bad_key = jax.random.split(key)
        healthy_keys = jax.random.split(healthy_key, self._N_HEALTHY)

        def _mk_healthy(k):
            k0, k1 = jax.random.split(k)
            return jnp.concatenate(
                [
                    jax.random.normal(k0, (1,)) * 0.3,
                    jax.random.normal(k1, (self._D - 1,)),
                ]
            )

        healthy_positions = jax.vmap(_mk_healthy)(healthy_keys)
        # Deep in the neck (x0=-5: conditional std exp(-2.5)~=0.08) but with
        # a MOUTH-scale tail -- both a logdensity outlier (stage 1 target)
        # and a locally stiff / divergence-prone point under a
        # mouth-calibrated step size (stage 2b target).
        bad_position = jnp.concatenate(
            [jnp.array([-5.0]), jax.random.normal(bad_key, (self._D - 1,))]
        )
        positions = jnp.concatenate([healthy_positions, bad_position[None]], axis=0)
        states = jax.vmap(lambda pos: blackjax.nuts.init(pos, self._funnel_logdensity))(
            positions
        )
        return states

    def _divergence_rate(self, states, mcmc_kernel, metric, key, n_steps, M):
        keys = jax.random.split(key, n_steps)

        def _step(carry, k):
            chain_keys = jax.random.split(k, M)
            new_states, infos = jax.vmap(
                lambda ck, s: mcmc_kernel(
                    ck, s, self._funnel_logdensity, self._STEP_SIZE, metric
                )
            )(chain_keys, carry)
            return new_states, infos.is_divergent

        _, divergences = jax.lax.scan(_step, states, keys)
        return jnp.mean(divergences, axis=0)

    def test_disabled_transient_reproduces_in_miniature(self):
        """Mechanism disabled: naively sampling from the raw post-warmup
        state at the shared step size reproduces the divergence storm on the
        engineered bad chain, while every healthy chain stays clean."""
        states = self._make_states()
        M = self._N_HEALTHY + 1
        bad_idx = M - 1
        mcmc_kernel = blackjax.nuts.build_kernel()
        metric = _IDENTITY_METRIC(self._D)

        naive_rate = self._divergence_rate(
            states, mcmc_kernel, metric, jax.random.key(11), 80, M
        )

        self.assertGreater(
            float(naive_rate[bad_idx]),
            0.5,
            "Engineered bad chain must show a high naive divergence rate "
            "(mechanism disabled); got "
            f"{float(naive_rate[bad_idx]):.2f}",  # noqa: E231
        )
        self.assertEqual(
            float(jnp.max(naive_rate[:bad_idx])),
            0.0,
            "Healthy chains must stay clean at this step size (isolation check)",
        )

    def test_enabled_rehabilitates_bad_chain(self):
        """Mechanism enabled: stage 1 flags the bad chain, stage 2a/2b
        rehabilitate it, and re-sampling from the rehabilitated state at the
        SAME shared step size no longer diverges."""
        states = self._make_states()
        M = self._N_HEALTHY + 1
        bad_idx = M - 1
        mcmc_kernel = blackjax.nuts.build_kernel()
        metric = _IDENTITY_METRIC(self._D)
        step_size = jnp.asarray(self._STEP_SIZE)

        naive_rate = self._divergence_rate(
            states, mcmc_kernel, metric, jax.random.key(11), 80, M
        )

        rehab_states, _probation_info, diagnostics = _apply_start_state_probation(
            jax.random.key(12),
            states,
            mcmc_kernel,
            blackjax.nuts.init,
            self._funnel_logdensity,
            step_size,
            metric,
            M,
            _PROBATION_WINDOW_DEFAULT,
            {},
            return_all_adapt_info,
            states,
        )

        self.assertIn(
            bad_idx,
            diagnostics["flagged_chain_idx"],
            "Stage 1 must flag the engineered bad chain",
        )
        self.assertTrue(diagnostics["redraw_applied"])

        post_rate = self._divergence_rate(
            rehab_states, mcmc_kernel, metric, jax.random.key(13), 80, M
        )

        self.assertLess(
            float(post_rate[bad_idx]),
            float(naive_rate[bad_idx]),
            "Probation must reduce the previously-bad chain's divergence rate",
        )
        self.assertLessEqual(
            float(post_rate[bad_idx]),
            0.1,
            "Rehabilitated chain should sample cleanly at the shared step size",
        )


# ---------------------------------------------------------------------------
# Verdict fold-in (no silent interventions)
# ---------------------------------------------------------------------------


class TestExtractMultiChainVerdictProbationResult(BlackJAXTest):
    def test_probation_result_none_is_backward_compatible(self):
        """Omitting probation_result leaves flags exactly as before (no new keys)."""
        from blackjax.adaptation.meta.builders import build_multi_chain_meta_core

        core = build_multi_chain_meta_core(max_grad_budget=40000, n_chains=8)
        state = core.init(10)
        verdict = extract_multi_chain_verdict(
            state, max_grad_budget=40000, num_warmup_steps=100
        )
        self.assertFalse(
            any(k.startswith("start_state_probation_") for k in verdict.flags)
        )

    def test_probation_result_folded_into_flags(self):
        from blackjax.adaptation.meta.builders import build_multi_chain_meta_core

        core = build_multi_chain_meta_core(max_grad_budget=40000, n_chains=8)
        state = core.init(10)
        probation_result = {
            "enabled": True,
            "n_chains_flagged_by_state_gate": 1,
            "n_chains_unresolved": 0,
        }
        verdict = extract_multi_chain_verdict(
            state,
            max_grad_budget=40000,
            num_warmup_steps=100,
            probation_result=probation_result,
        )
        self.assertEqual(verdict.flags["start_state_probation_enabled"], True)
        self.assertEqual(
            verdict.flags["start_state_probation_n_chains_flagged_by_state_gate"], 1
        )
        self.assertEqual(verdict.flags["start_state_probation_n_chains_unresolved"], 0)
