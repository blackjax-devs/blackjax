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
"""Tests for opt-in metric-publication telemetry.

Three groups:

* **Default-path preservation.**  With telemetry off, the state types, their
  treedefs, their field lists and every number a warmup produces must be what
  they were before this feature existed -- on both the single- and multi-chain
  paths.
* **Contract.**  What the two masks mean, what the counters count, and what the
  three branch fields say.  These are the places where a plausible-looking
  reading of the record would be wrong.
* **Observation correctness.**  Checked against directly captured pre-``final()``
  inputs, or against branch outcomes the controller genuinely reaches on
  purpose-built buffers.  No controller outcome is forced or monkeypatched: the
  tests read the real gate locals through the record.

The buffer helpers produce fixed pseudorandom blocks -- deterministic states for
a controller call, not draws from an exactly-iid source.
"""
import chex
import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest, parameterized

import blackjax
from blackjax.adaptation.low_rank_adaptation import build_growing_window_schedule
from blackjax.adaptation.meta import (
    build_meta_adaptation_core,
    build_multi_chain_meta_core,
)
from blackjax.adaptation.meta._state import (
    MetaAdaptationCoreState,
    MetaAdaptationTelemetryCoreState,
    MultiChainMetaAdaptationCoreState,
    MultiChainMetaAdaptationTelemetryCoreState,
)
from blackjax.adaptation.meta._telemetry import (
    BRANCH_BOTH,
    BRANCH_NONE,
    BRANCH_T,
    BRANCH_W,
    GATE_BITS,
    ROUTE_DIAGONAL,
    ROUTE_T,
    ROUTE_W,
    SCHEMA_VERSION,
    MetricPublicationRecord,
    decode_gates,
    encode_gates,
    extract_publication_chronology,
    publication_adapt_info_fn,
    record_nbytes,
)
from blackjax.adaptation.staged_adaptation import build_schedule, staged_adaptation

from ._meta_fixtures import (
    _fill_mc_state,
    _fill_state_from_buffer,
    _make_isotropic_buffer,
    _make_mc_both_branches,
    _make_mc_deep_spread,
    _make_mc_even_spread,
    _make_mc_isotropic,
    _make_mc_split_means,
)

_BUDGET = 1600
_NUM_STEPS = 200
_D = 5
_MC_M, _MC_N, _MC_D = 8, 60, 6
_MC_BUDGET = 40_000

#: Frozen: the controller states as they were before telemetry existed.
_EXPECTED_SC_FIELDS = (
    "inverse_mass_matrix",
    "mu_star",
    "draws_buffer",
    "grads_buffer",
    "buffer_idx",
    "background_split",
    "recompute_counter",
    "has_escalated",
    "escalation_rank",
    "s_gap_prev",
    "s_gap_curr",
    "r2_latest",
    "r2_mode",
    "budget_used",
    "converged_at_step",
    "prev_lam",
    "airm_vel_prev",
    "airm_vel_curr",
    "is_slow_mixing",
)


def _logdensity_fn(x):
    return -0.5 * jnp.sum((x / jnp.arange(1.0, _D + 1)) ** 2)


def _run(**kwargs):
    warmup = staged_adaptation(
        blackjax.nuts,
        _logdensity_fn,
        metric="auto",
        max_grad_budget=_BUDGET,
        **kwargs,
    )
    return warmup.run(jax.random.key(0), jnp.zeros(_D), num_steps=_NUM_STEPS)


def _mc_core(**kwargs):
    return build_multi_chain_meta_core(_MC_BUDGET, _MC_M, **kwargs)


def _mc_record(fixture, core=None, **fx_kwargs):
    """Run one real window boundary on purpose-built multi-chain buffers."""
    core = core or _mc_core(telemetry=True)
    draws, grads = fixture(_MC_M, _MC_N, _MC_D, **fx_kwargs)
    state = _fill_mc_state(core.init(_MC_D), draws, grads)
    return core.final(state).publication


class DefaultPathPreservationTest(chex.TestCase):
    """Telemetry off must be indistinguishable from telemetry not existing."""

    def test_single_chain_state_fields_unchanged(self):
        self.assertEqual(MetaAdaptationCoreState._fields, _EXPECTED_SC_FIELDS)

    def test_state_types_are_selected_at_build_time(self):
        self.assertIs(
            type(build_meta_adaptation_core(_BUDGET).init(_D)),
            MetaAdaptationCoreState,
        )
        self.assertIs(
            type(build_meta_adaptation_core(_BUDGET, telemetry=True).init(_D)),
            MetaAdaptationTelemetryCoreState,
        )
        self.assertIs(type(_mc_core().init(_MC_D)), MultiChainMetaAdaptationCoreState)
        self.assertIs(
            type(_mc_core(telemetry=True).init(_MC_D)),
            MultiChainMetaAdaptationTelemetryCoreState,
        )

    @parameterized.named_parameters(
        ("single", MetaAdaptationCoreState, MetaAdaptationTelemetryCoreState),
        (
            "multi",
            MultiChainMetaAdaptationCoreState,
            MultiChainMetaAdaptationTelemetryCoreState,
        ),
    )
    def test_telemetry_state_extends_default_state(self, plain, telemetry):
        """Drift guard: the twins must stay in step, telemetry field last."""
        self.assertEqual(telemetry._fields, plain._fields + ("publication",))

    @parameterized.named_parameters(
        ("single", False),
        ("multi", True),
    )
    def test_default_treedef_and_unpacking_preserved(self, multi):
        """Why telemetry uses a sibling type rather than a defaulted field.

        A trailing ``publication: ... = None`` adds zero *leaves* but still
        changes the treedef (``[*, *]`` vs ``[*, *, None]``), the tuple length
        and exact unpacking.  The default path must show none of that.
        """
        if multi:
            state = _mc_core().init(_MC_D)
            cls = MultiChainMetaAdaptationCoreState
        else:
            state = build_meta_adaptation_core(_BUDGET).init(_D)
            cls = MetaAdaptationCoreState
        self.assertEqual(jax.tree.structure(state), jax.tree.structure(cls(*state)))
        self.assertEqual(tuple(state._asdict()), cls._fields)

    def test_warmup_results_bitwise_identical_off_vs_on(self):
        """Telemetry must not move a single bit of the sampler's arithmetic."""
        (state_off, params_off), _ = _run()
        (state_on, params_on), _ = _run(
            metric_telemetry=True, adaptation_info_fn=publication_adapt_info_fn()
        )
        np.testing.assert_array_equal(
            np.asarray(state_off.position), np.asarray(state_on.position)
        )
        np.testing.assert_array_equal(
            np.asarray(params_off["step_size"]), np.asarray(params_on["step_size"])
        )
        for field in params_off["inverse_mass_matrix"]._fields:
            np.testing.assert_array_equal(
                np.asarray(getattr(params_off["inverse_mass_matrix"], field)),
                np.asarray(getattr(params_on["inverse_mass_matrix"], field)),
                err_msg=f"inverse_mass_matrix.{field} differs with telemetry on",
            )

    def test_core_parity_through_an_actual_escalation(self):
        """Deterministic replay, off vs on, along a path that really escalates.

        A sampler seed search is not needed to cover the escalated branch: the
        same update/final sequence is driven through both cores and every
        controller field is compared bitwise.
        """
        off, on = _mc_core(), _mc_core(telemetry=True)
        draws, grads = _make_mc_deep_spread(_MC_M, _MC_N, _MC_D)
        s_off = _fill_mc_state(off.init(_MC_D), draws, grads)
        s_on = _fill_mc_state(on.init(_MC_D), draws, grads)
        for _ in range(2):  # escalate, then republish
            s_off, s_on = off.final(s_off), on.final(s_on)
            self.assertTrue(bool(s_on.publication.has_escalated))
            s_off = _fill_mc_state(s_off, draws, grads)
            s_on = _fill_mc_state(s_on, draws, grads)
        for field in MultiChainMetaAdaptationCoreState._fields:
            a, b = getattr(s_off, field), getattr(s_on, field)
            for leaf_a, leaf_b in zip(jax.tree.leaves(a), jax.tree.leaves(b)):
                np.testing.assert_array_equal(
                    np.asarray(leaf_a),
                    np.asarray(leaf_b),
                    err_msg=f"{field} diverged with telemetry on",
                )


class MaskContractTest(chex.TestCase):
    """The two masks mean different things and are deliberately not nested."""

    def test_gate_bits_are_branch_scoped_and_disjoint(self):
        self.assertEqual(len(set(GATE_BITS.values())), len(GATE_BITS))
        self.assertEqual(GATE_BITS["deadline"], 0)
        for name in ("sc_r2", "sc_s_gap_magnitude", "sc_s_gap_stability"):
            self.assertIn(name, GATE_BITS)
        for name in ("w_magnitude", "w_psi", "w_r1", "w_r2_raw"):
            self.assertIn(name, GATE_BITS)
        for name in (
            "t_magnitude",
            "t_collinearity",
            "t_loo",
            "t_support",
            "t_unimodality",
            "t_r2_routed",
        ):
            self.assertIn(name, GATE_BITS)

    def test_encode_decode_round_trip_exhaustive(self):
        names = sorted(GATE_BITS)
        for combo in range(1 << len(names)):
            bits = {n: bool((combo >> i) & 1) for i, n in enumerate(names)}
            self.assertEqual(decode_gates(encode_gates(**bits)), bits)

    def test_encode_rejects_unknown_gate(self):
        with self.assertRaisesRegex(KeyError, "unknown gate"):
            encode_gates(no_such_gate=True)

    def test_stability_gate_inapplicable_in_first_window(self):
        """It needs the previous window's S_gap, which window 0 does not have."""
        _, records = _run(
            metric_telemetry=True, adaptation_info_fn=publication_adapt_info_fn()
        )
        chronology = extract_publication_chronology(records)
        first = chronology[0]
        self.assertFalse(first["gates_applicable"]["sc_s_gap_stability"])
        for name in ("sc_r2", "sc_s_gap_magnitude", "deadline"):
            self.assertTrue(first["gates_applicable"][name])
        for later in chronology[1:]:
            self.assertTrue(later["gates_applicable"]["sc_s_gap_stability"])

    def test_raw_predicates_survive_escalation_while_applicability_clears(self):
        """The masks are NOT nested: raw truth persists, applicability does not.

        JAX's ``&`` is eager, so the controller computes every predicate on
        every window including after escalation.  Reporting them as unevaluated
        would be false, and the multi-chain deferral latch -- which is gated on
        ``~escalate_T`` and not on ``has_escalated`` -- needs them.
        """
        core = _mc_core(telemetry=True)
        draws, grads = _make_mc_deep_spread(_MC_M, _MC_N, _MC_D)
        state = _fill_mc_state(core.init(_MC_D), draws, grads)

        first = core.final(state)
        self.assertTrue(bool(first.publication.escalated_now))
        self.assertNotEqual(int(first.publication.escalation_gate_applicable), 0)

        second = core.final(_fill_mc_state(first, draws, grads)).publication
        self.assertTrue(bool(second.has_escalated_before))
        self.assertFalse(bool(second.escalated_now))
        # Applicability is gone ...
        self.assertEqual(int(second.escalation_gate_applicable), 0)
        # ... but the raw predicates are still reported, and still true.
        self.assertNotEqual(int(second.gate_predicate_true), 0)
        self.assertEqual(
            int(second.gate_predicate_true),
            int(first.publication.gate_predicate_true),
        )

    def test_a_false_sibling_does_not_clear_other_predicates(self):
        """T fires while both W magnitude and W psi are false, and vice versa."""
        t_only = _mc_record(_make_mc_even_spread)
        true_bits = decode_gates(t_only.gate_predicate_true)
        self.assertFalse(true_bits["w_magnitude"])
        self.assertTrue(true_bits["t_magnitude"])
        self.assertTrue(true_bits["t_loo"])
        self.assertTrue(true_bits["w_r1"])  # a W sibling still reported true


class UnitsTest(chex.TestCase):
    """Counters are core updates, not warmup steps, and per-chain != pooled."""

    def test_core_update_count_lags_the_scan_index_under_fast_stages(self):
        """The host calls the core's update() only on slow stages.

        Stan's schedule opens with a 75-step fast buffer, so the two counts
        differ by exactly that.  A schedule with no fast prefix would hide this.
        """
        warmup = staged_adaptation(
            blackjax.nuts,
            _logdensity_fn,
            metric="auto",
            max_grad_budget=8000,
            schedule_fn=build_schedule,
            metric_telemetry=True,
            adaptation_info_fn=publication_adapt_info_fn(),
        )
        _, records = warmup.run(jax.random.key(0), jnp.zeros(_D), num_steps=400)
        chronology = extract_publication_chronology(records)
        schedule = np.asarray(build_schedule(400))
        fast_prefix = int(np.argmax(schedule[:, 0] == 1))
        self.assertGreater(fast_prefix, 0)
        for entry in chronology:
            self.assertEqual(
                entry["warmup_step_index"] + 1 - entry["core_update_steps_per_chain"],
                fast_prefix,
            )

    def test_warmup_step_index_matches_the_schedule_boundaries(self):
        _, records = _run(
            metric_telemetry=True, adaptation_info_fn=publication_adapt_info_fn()
        )
        schedule = np.asarray(build_growing_window_schedule(_NUM_STEPS))
        boundaries = np.flatnonzero(schedule[:, 1] == 1).tolist()
        got = [e["warmup_step_index"] for e in extract_publication_chronology(records)]
        self.assertEqual(got, boundaries)

    def test_single_chain_per_chain_and_total_counts_agree(self):
        _, records = _run(
            metric_telemetry=True, adaptation_info_fn=publication_adapt_info_fn()
        )
        for entry in extract_publication_chronology(records):
            self.assertEqual(entry["n_chains"], 1)
            self.assertEqual(
                entry["core_update_steps_per_chain"],
                entry["core_update_chain_steps_total"],
            )
            self.assertEqual(entry["support_per_chain"], entry["support_pooled_rows"])

    def test_multi_chain_units_are_separated(self):
        record = _mc_record(_make_mc_deep_spread)
        self.assertEqual(int(record.n_chains), _MC_M)
        self.assertEqual(
            int(record.core_update_chain_steps_total),
            int(record.core_update_steps_per_chain) * _MC_M,
        )
        # Pooled rows are retained buffer rows -- not an ESS, not independent
        # observations -- and are exactly per-chain support times chain count.
        self.assertEqual(
            int(record.support_pooled_rows),
            int(record.support_per_chain) * _MC_M,
        )


class SupportAccountingTest(chex.TestCase):
    """Capacity reached and draws dropped are different facts."""

    @parameterized.parameters(3, 17, 40)
    def test_support_matches_captured_pre_final_input(self, n_updates):
        core = build_meta_adaptation_core(_BUDGET, telemetry=True)
        state = core.init(_D)
        capacity = state.draws_buffer.shape[0]
        draws, grads = _make_isotropic_buffer(_D, n_updates)
        for i in range(n_updates):
            state = core.update(state, draws[i], grads[i])

        captured = int(state.buffer_idx)
        self.assertEqual(captured, n_updates)

        out = core.final(state)
        record = out.publication
        self.assertEqual(int(record.support_per_chain), min(captured, capacity))
        self.assertEqual(int(record.buffer_capacity), capacity)
        self.assertFalse(bool(record.buffer_capacity_reached))
        self.assertEqual(int(record.dropped_draws), 0)
        # The information the record preserves is gone from the state itself.
        self.assertEqual(int(out.buffer_idx), 0)

    def test_capacity_reached_exactly_drops_nothing(self):
        core = build_meta_adaptation_core(_BUDGET, telemetry=True)
        state = core.init(_D)
        capacity = state.draws_buffer.shape[0]
        draws, grads = _make_isotropic_buffer(_D, capacity)
        for i in range(capacity):
            state = core.update(state, draws[i], grads[i])
        record = core.final(state).publication
        self.assertTrue(bool(record.buffer_capacity_reached))
        self.assertEqual(int(record.dropped_draws), 0)
        self.assertEqual(int(record.support_per_chain), capacity)

    def test_overflow_reports_the_drop_count(self):
        core = build_meta_adaptation_core(_BUDGET, telemetry=True)
        state = core.init(_D)
        capacity = state.draws_buffer.shape[0]
        overflow = 12
        draws, grads = _make_isotropic_buffer(_D, capacity + overflow)
        for i in range(capacity + overflow):
            state = core.update(state, draws[i], grads[i])
        record = core.final(state).publication
        self.assertTrue(bool(record.buffer_capacity_reached))
        self.assertEqual(int(record.dropped_draws), overflow)
        self.assertEqual(int(record.support_per_chain), capacity)


class ChronologyTest(chex.TestCase):
    """One record per boundary, in order, deduplicated by window_index."""

    def setUp(self):
        super().setUp()
        _, self.records = _run(
            metric_telemetry=True, adaptation_info_fn=publication_adapt_info_fn()
        )
        self.chronology = extract_publication_chronology(self.records)

    def test_one_publication_per_scheduled_boundary(self):
        schedule = np.asarray(build_growing_window_schedule(_NUM_STEPS))
        self.assertLen(self.chronology, int((schedule[:, 1] == 1).sum()))

    def test_window_index_is_contiguous_from_zero(self):
        self.assertEqual(
            [e["window_index"] for e in self.chronology],
            list(range(len(self.chronology))),
        )

    def test_window_index_is_minus_one_before_the_first_publication(self):
        schedule = np.asarray(build_growing_window_schedule(_NUM_STEPS))
        first = int(np.flatnonzero(schedule[:, 1] == 1)[0])
        stacked = np.asarray(self.records.window_index)
        self.assertTrue(np.all(stacked[:first] == -1))
        self.assertEqual(int(stacked[first]), 0)

    def test_record_is_held_unchanged_between_boundaries(self):
        schedule = np.asarray(build_growing_window_schedule(_NUM_STEPS))
        boundaries = np.flatnonzero(schedule[:, 1] == 1)
        stacked = np.asarray(self.records.window_index)
        self.assertTrue(np.all(stacked[int(boundaries[0]) : int(boundaries[1])] == 0))


class EpsilonChronologyTest(chex.TestCase):
    """Four distinct epsilons; none collapsed into another."""

    def setUp(self):
        super().setUp()

        def info_fn(state, info, adaptation_state):
            del state, info
            return (adaptation_state.imm_state.publication, adaptation_state.step_size)

        _, (self.records, self.step_sizes) = _run(
            metric_telemetry=True, adaptation_info_fn=info_fn
        )
        schedule = np.asarray(build_growing_window_schedule(_NUM_STEPS))
        self.boundaries = np.flatnonzero(schedule[:, 1] == 1)

    def test_epsilon_in_force_is_the_value_the_kernel_used(self):
        in_force = np.asarray(self.records.epsilon_in_force)
        observed = np.asarray(self.step_sizes)
        for boundary in self.boundaries:
            np.testing.assert_array_equal(in_force[boundary], observed[boundary - 1])

    def test_epsilon_next_window_is_the_published_step_size(self):
        next_eps = np.asarray(self.records.epsilon_next_window)
        observed = np.asarray(self.step_sizes)
        for boundary in self.boundaries:
            np.testing.assert_array_equal(next_eps[boundary], observed[boundary])

    def test_the_four_epsilons_are_captured_separately(self):
        """No equality is asserted between them: they are different quantities.

        ``epsilon_window_average`` and ``epsilon_next_window`` are related by
        ``exp(log(.))``, which is not guaranteed bitwise-identical, so this
        records that each is finite and captured rather than forcing agreement --
        and nothing in the adaptation is touched to make them agree.
        """
        for name in (
            "epsilon_in_force",
            "epsilon_after_window_da",
            "epsilon_window_average",
            "epsilon_next_window",
        ):
            values = np.asarray(getattr(self.records, name))[self.boundaries]
            self.assertTrue(np.all(np.isfinite(values)), msg=f"{name} not finite")
            self.assertTrue(np.all(values > 0.0), msg=f"{name} not positive")

        in_force = np.asarray(self.records.epsilon_in_force)[self.boundaries]
        after_da = np.asarray(self.records.epsilon_after_window_da)[self.boundaries]
        next_eps = np.asarray(self.records.epsilon_next_window)[self.boundaries]
        self.assertTrue(np.any(in_force != next_eps))
        self.assertTrue(np.any(in_force != after_da))

    def test_epsilons_carry_the_dual_averaging_dtype(self):
        """Never a hardcoded float32: that breaks bitwise chronology under x64."""
        expected = np.asarray(self.step_sizes).dtype
        for name in (
            "epsilon_in_force",
            "epsilon_after_window_da",
            "epsilon_window_average",
            "epsilon_next_window",
        ):
            self.assertEqual(np.asarray(getattr(self.records, name)).dtype, expected)


class SingleChainCandidateTest(chex.TestCase):
    """The candidate must be visible in the windows where it is withheld."""

    def _record(self, full_matrices=False):
        core = build_meta_adaptation_core(
            _BUDGET, telemetry=True, full_matrices=full_matrices
        )
        state = core.init(_D)
        draws, grads = _make_isotropic_buffer(_D, 40)
        return core.final(_fill_state_from_buffer(state, draws, grads)).publication

    def test_withheld_candidate_is_still_reported(self):
        record = self._record()
        self.assertFalse(bool(record.escalated_now))
        candidate = record.single_chain.candidate
        for name in ("logdet", "lam_max", "sigma_gm"):
            self.assertTrue(np.isfinite(np.asarray(getattr(candidate, name))), msg=name)
        self.assertGreater(float(candidate.lam_max), 0.0)

    def test_the_three_ranks_are_reported_separately(self):
        record = self._record()
        self.assertGreaterEqual(int(record.single_chain.detection_rank), 0)
        self.assertGreaterEqual(int(record.single_chain.candidate.effective_rank), 0)
        self.assertEqual(int(record.deployed_effective_rank), 0)  # not escalated
        self.assertGreaterEqual(int(record.escalation_rank_stored), 0)

    def test_in_force_and_deployed_metrics_are_both_reported(self):
        record = self._record()
        self.assertTrue(np.isfinite(np.asarray(record.in_force_logdet)))
        self.assertTrue(np.isfinite(np.asarray(record.deployed_logdet)))

    def test_full_matrices_absent_by_default_present_when_asked(self):
        self.assertIsNone(self._record().single_chain.candidate.full)
        self.assertIsNone(self._record().deployed_full)
        full = self._record(full_matrices=True)
        self.assertIsNotNone(full.single_chain.candidate.full)
        self.assertEqual(full.single_chain.candidate.full.sigma.shape, (_D,))

    def test_epsilons_are_nan_when_final_is_called_without_the_host(self):
        """The core cannot see the step size; it must not invent one."""
        record = self._record()
        self.assertTrue(np.isnan(np.asarray(record.epsilon_in_force)))
        self.assertTrue(np.isnan(np.asarray(record.epsilon_next_window)))
        self.assertEqual(int(record.warmup_step_index), -1)


class MultiChainBranchTest(chex.TestCase):
    """Fired / carried / deployed are three questions with three answers."""

    def test_w_only(self):
        record = _mc_record(_make_mc_deep_spread)
        self.assertTrue(bool(record.escalated_now))
        self.assertEqual(int(record.multi_chain.branch_fired_this_window), BRANCH_W)
        self.assertEqual(int(record.multi_chain.deployed_metric_route), ROUTE_W)

    def test_t_only(self):
        record = _mc_record(_make_mc_even_spread)
        self.assertTrue(bool(record.escalated_now))
        self.assertEqual(int(record.multi_chain.branch_fired_this_window), BRANCH_T)
        self.assertEqual(int(record.multi_chain.deployed_metric_route), ROUTE_T)

    def test_both_branches_fire_but_the_w_metric_is_deployed(self):
        """The case the route field exists for: BOTH is not a route."""
        record = _mc_record(_make_mc_both_branches)
        self.assertTrue(bool(record.escalated_now))
        self.assertEqual(int(record.multi_chain.branch_fired_this_window), BRANCH_BOTH)
        self.assertEqual(int(record.multi_chain.deployed_metric_route), ROUTE_W)

    def test_neither_branch_fires(self):
        record = _mc_record(_make_mc_isotropic)
        self.assertFalse(bool(record.escalated_now))
        self.assertEqual(int(record.multi_chain.branch_fired_this_window), BRANCH_NONE)
        self.assertEqual(int(record.multi_chain.deployed_metric_route), ROUTE_DIAGONAL)
        self.assertEqual(int(record.multi_chain.detection_branch_history), BRANCH_NONE)
        self.assertEqual(int(record.multi_chain.branch_first_set_at_window), -1)

    def test_both_candidates_are_reported_before_escalation(self):
        """W and T candidates are built before routing and are not the same."""
        record = _mc_record(_make_mc_isotropic)
        detail = record.multi_chain
        for candidate in (detail.candidate_w, detail.candidate_t):
            self.assertTrue(np.isfinite(np.asarray(candidate.logdet)))
        self.assertNotEqual(
            float(detail.candidate_w.logdet), float(detail.candidate_t.logdet)
        )

    def test_raw_and_routed_r2_are_separate_and_can_disagree(self):
        """The W branch uses the raw R2; the T branch uses the routed one."""
        core = _mc_core(telemetry=True)
        warmup = staged_adaptation(
            blackjax.nuts,
            _logdensity_fn,
            metric="auto",
            max_grad_budget=16_000,
            n_chains=_MC_M,
            metric_telemetry=True,
            adaptation_info_fn=publication_adapt_info_fn(),
        )
        x0 = jax.random.normal(jax.random.key(1), (_MC_M, _D))
        _, records = warmup.run(jax.random.key(0), x0)
        chronology = extract_publication_chronology(records)
        disagreements = [
            e
            for e in chronology
            if np.isnan(e["multi_chain.r2_routed"]) != np.isnan(e["r2_raw"])
        ]
        self.assertNotEmpty(
            disagreements,
            msg="expected at least one window where raw and routed R2 differ",
        )
        del core


class MultiChainEscalationSequenceTest(chex.TestCase):
    """Bounded deterministic update/final sequences starting un-escalated."""

    def test_first_escalation_then_republication(self):
        """After escalation the route republishes a freshly computed candidate.

        "Neither fires so the candidates were discarded" is a statement about
        the pre-escalation regime only.
        """
        core = _mc_core(telemetry=True)
        draws, grads = _make_mc_deep_spread(_MC_M, _MC_N, _MC_D)
        state = _fill_mc_state(core.init(_MC_D), draws, grads)

        first = core.final(state)
        rec1 = first.publication
        self.assertTrue(bool(rec1.escalated_now))
        self.assertFalse(bool(rec1.has_escalated_before))
        self.assertEqual(int(rec1.first_escalation_window_index), 0)
        self.assertEqual(int(rec1.multi_chain.branch_first_set_at_window), 0)

        second = core.final(_fill_mc_state(first, draws, grads))
        rec2 = second.publication
        self.assertFalse(bool(rec2.escalated_now))
        self.assertTrue(bool(rec2.has_escalated_before))
        self.assertEqual(int(rec2.window_index), 1)
        # Stamped on the genuine transition only, and unchanged afterwards.
        self.assertEqual(int(rec2.first_escalation_window_index), 0)
        self.assertEqual(int(rec2.multi_chain.branch_first_set_at_window), 0)
        # The carried history still routes, and a metric is still published.
        self.assertEqual(int(rec2.multi_chain.branch_fired_this_window), BRANCH_NONE)
        self.assertEqual(int(rec2.multi_chain.detection_branch_history), BRANCH_W)
        self.assertEqual(int(rec2.multi_chain.deployed_metric_route), ROUTE_W)
        self.assertTrue(np.isfinite(np.asarray(rec2.deployed_logdet)))

    def test_deferral_needs_two_consecutive_flagged_windows(self):
        """The latch is a counter, and the count is on the record."""
        core = _mc_core(telemetry=True)
        draws, grads = _make_mc_split_means(_MC_M, _MC_N, _MC_D)
        state = _fill_mc_state(core.init(_MC_D), draws, grads)

        first = core.final(state)
        rec1 = first.publication
        self.assertFalse(bool(rec1.multi_chain.deferred_to_ensemble))
        self.assertEqual(int(rec1.multi_chain.unimodality_flag_count), 1)

        second = core.final(_fill_mc_state(first, draws, grads)).publication
        self.assertEqual(int(second.multi_chain.unimodality_flag_count), 2)
        self.assertTrue(bool(second.multi_chain.deferred_to_ensemble))

    def test_deferral_resets_when_the_flag_clears(self):
        core = _mc_core(telemetry=True)
        split_d, split_g = _make_mc_split_means(_MC_M, _MC_N, _MC_D)
        calm_d, calm_g = _make_mc_isotropic(_MC_M, _MC_N, _MC_D)
        state = _fill_mc_state(core.init(_MC_D), split_d, split_g)
        after = core.final(state)
        self.assertEqual(int(after.publication.multi_chain.unimodality_flag_count), 1)
        reset = core.final(_fill_mc_state(after, calm_d, calm_g)).publication
        self.assertEqual(int(reset.multi_chain.unimodality_flag_count), 0)
        self.assertFalse(bool(reset.multi_chain.deferred_to_ensemble))

    def test_deferral_observables_remain_available_after_escalation(self):
        """Deferral is gated on ~escalate_T, not on has_escalated.

        Everything needed to reconstruct it must therefore still be reported in
        a post-escalation window.
        """
        core = _mc_core(telemetry=True)
        draws, grads = _make_mc_deep_spread(_MC_M, _MC_N, _MC_D)
        state = _fill_mc_state(core.init(_MC_D), draws, grads)
        first = core.final(state)
        self.assertTrue(bool(first.publication.escalated_now))
        after = core.final(_fill_mc_state(first, draws, grads)).publication

        self.assertEqual(int(after.escalation_gate_applicable), 0)
        raw = decode_gates(after.gate_predicate_true)
        for name in ("t_magnitude", "t_loo", "t_support", "t_r2_routed"):
            self.assertIn(name, raw)
        detail = after.multi_chain
        for name in (
            "is_unimodal",
            "any_mode_flag",
            "unimodality_flag_count",
            "deferred_to_ensemble",
        ):
            self.assertIsNotNone(getattr(detail, name))
        self.assertTrue(np.isfinite(np.asarray(detail.t_contraction_stat)))

    def test_three_unimodality_observations_and_the_resolved_outcome(self):
        record = _mc_record(_make_mc_split_means)
        detail = record.multi_chain
        resolved = bool(detail.is_converging) or (
            bool(detail.is_unimodal) and not bool(detail.any_mode_flag)
        )
        self.assertEqual(bool(detail.t_unimodality_resolved), resolved)
        self.assertTrue(np.isfinite(np.asarray(detail.unimodality_gap_ratio)))
        self.assertTrue(np.isfinite(np.asarray(detail.t_contraction_stat)))

    def test_stored_escalation_rank_is_the_t_detection_rank_even_when_w_fires(self):
        """Reported, not corrected: the controller stores k_new regardless.

        This is why the record keeps the stored nominal rank, the T detection
        rank and the deployed effective rank as three separate fields.
        """
        record = _mc_record(_make_mc_deep_spread)
        self.assertEqual(int(record.multi_chain.branch_fired_this_window), BRANCH_W)
        self.assertEqual(
            int(record.escalation_rank_stored),
            int(record.multi_chain.t_detection_rank),
        )


class PayloadAndSchemaTest(chex.TestCase):
    """Bounded, and measured from real leaves rather than a nominal constant."""

    def _record(self, full_matrices=False):
        core = build_meta_adaptation_core(
            _BUDGET, telemetry=True, full_matrices=full_matrices
        )
        draws, grads = _make_isotropic_buffer(_D, 40)
        state = _fill_state_from_buffer(core.init(_D), draws, grads)
        return core.final(state).publication

    def test_compact_record_leaves_are_all_scalar(self):
        for leaf in jax.tree.leaves(self._record()):
            self.assertEqual(jnp.asarray(leaf).shape, ())

    def test_compact_record_carries_no_buffers(self):
        record = self._record()
        total = sum(int(jnp.asarray(x).size) for x in jax.tree.leaves(record))
        self.assertEqual(total, len(jax.tree.leaves(record)))

    def test_record_nbytes_reports_actual_leaf_bytes(self):
        record = self._record()
        expected = sum(int(jnp.asarray(x).nbytes) for x in jax.tree.leaves(record))
        self.assertEqual(record_nbytes(record), expected)
        self.assertLess(record_nbytes(record), 1024)

    def test_full_matrices_cost_is_why_they_are_opt_in(self):
        self.assertGreater(
            record_nbytes(self._record(full_matrices=True)),
            record_nbytes(self._record()),
        )

    def test_exactly_one_detail_subrecord_is_populated(self):
        single = self._record()
        self.assertIsNotNone(single.single_chain)
        self.assertIsNone(single.multi_chain)
        multi = _mc_record(_make_mc_isotropic)
        self.assertIsNone(multi.single_chain)
        self.assertIsNotNone(multi.multi_chain)

    def test_chronology_reports_only_the_relevant_controller_gates(self):
        _, records = _run(
            metric_telemetry=True, adaptation_info_fn=publication_adapt_info_fn()
        )
        entry = extract_publication_chronology(records)[0]
        self.assertEqual(entry["schema_version"], SCHEMA_VERSION)
        self.assertIn("sc_r2", entry["gates_true"])
        self.assertNotIn("w_psi", entry["gates_true"])
        self.assertIn("deadline", entry["gates_true"])

    def test_chronology_rejects_unstacked_records(self):
        with self.assertRaisesRegex(ValueError, "leading step axis"):
            extract_publication_chronology(self._record())

    def test_schema_version_is_two(self):
        self.assertEqual(SCHEMA_VERSION, 2)

    def test_record_field_count_matches_the_documented_layout(self):
        """Provisional, not frozen: catches an accidental change only."""
        self.assertLen(MetricPublicationRecord._fields, 32)


class UnsupportedConfigurationTest(chex.TestCase):
    """Contradictory requests must fail loudly, never silently."""

    def test_full_matrices_requires_telemetry(self):
        with self.assertRaisesRegex(ValueError, "requires telemetry=True"):
            build_meta_adaptation_core(_BUDGET, full_matrices=True)
        with self.assertRaisesRegex(ValueError, "requires telemetry=True"):
            build_multi_chain_meta_core(_MC_BUDGET, _MC_M, full_matrices=True)
        with self.assertRaisesRegex(ValueError, "requires"):
            staged_adaptation(
                blackjax.nuts,
                _logdensity_fn,
                metric="auto",
                max_grad_budget=_BUDGET,
                telemetry_full_matrices=True,
            )

    def test_core_without_publication_is_rejected(self):
        warmup = staged_adaptation(
            blackjax.nuts,
            _logdensity_fn,
            metric="welford_diag",
            metric_telemetry=True,
        )
        with self.assertRaisesRegex(ValueError, "no 'publication' field"):
            warmup.run(jax.random.key(0), jnp.zeros(_D), num_steps=60)


if __name__ == "__main__":
    absltest.main()
