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
"""Tests for opt-in metric-publication telemetry (single-chain controller).

Two groups:

* **Default-path preservation.**  With telemetry off, the state type, its
  treedef, its field list and every number the warmup produces must be what
  they were before this feature existed.
* **Observation correctness.**  What the record reports must equal what the
  controller actually consumed and decided — checked against directly captured
  pre-``final()`` inputs rather than inferred from the post-``final()`` state,
  which is exactly the information the record exists to preserve.

The buffer helpers produce fixed pseudorandom blocks: deterministic states for
a controller call, not draws from an exactly-iid source.
"""
import chex
import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest, parameterized

import blackjax
from blackjax.adaptation.low_rank_adaptation import build_growing_window_schedule
from blackjax.adaptation.meta import build_meta_adaptation_core
from blackjax.adaptation.meta._state import (
    MetaAdaptationCoreState,
    MetaAdaptationTelemetryCoreState,
)
from blackjax.adaptation.meta._telemetry import (
    GATE_BITS,
    SCHEMA_VERSION,
    MetricPublicationRecord,
    decode_gates,
    encode_gates,
    extract_publication_chronology,
    publication_adapt_info_fn,
    record_nbytes,
)
from blackjax.adaptation.staged_adaptation import staged_adaptation

from ._meta_fixtures import _fill_state_from_buffer, _make_isotropic_buffer

_BUDGET = 1600
_NUM_STEPS = 200
_D = 5

#: Frozen: the controller state as it was before telemetry existed.  A change
#: here is a real compatibility break and must be deliberate.
_EXPECTED_CORE_FIELDS = (
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


class DefaultPathPreservationTest(chex.TestCase):
    """Telemetry off must be indistinguishable from telemetry not existing."""

    def test_core_state_fields_unchanged(self):
        self.assertEqual(MetaAdaptationCoreState._fields, _EXPECTED_CORE_FIELDS)

    def test_default_core_returns_default_state_type(self):
        state = build_meta_adaptation_core(_BUDGET).init(_D)
        self.assertIs(type(state), MetaAdaptationCoreState)

    def test_telemetry_core_returns_telemetry_state_type(self):
        state = build_meta_adaptation_core(_BUDGET, telemetry=True).init(_D)
        self.assertIs(type(state), MetaAdaptationTelemetryCoreState)

    def test_telemetry_state_extends_default_state(self):
        """Drift guard: the two types must stay in step, telemetry field last."""
        self.assertEqual(
            MetaAdaptationTelemetryCoreState._fields,
            MetaAdaptationCoreState._fields + ("publication",),
        )

    def test_default_treedef_and_unpacking_preserved(self):
        """The reason telemetry uses a separate type rather than a defaulted field.

        A trailing ``publication: ... = None`` would add zero *leaves* but still
        change the treedef (``[*, *]`` vs ``[*, *, None]``), the tuple length and
        exact unpacking.  The default path must show none of that.
        """
        state = build_meta_adaptation_core(_BUDGET).init(_D)
        rebuilt = MetaAdaptationCoreState(*state)
        self.assertEqual(jax.tree.structure(state), jax.tree.structure(rebuilt))
        self.assertLen(state._fields, len(_EXPECTED_CORE_FIELDS))
        self.assertEqual(tuple(state._asdict()), _EXPECTED_CORE_FIELDS)

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
        imm_off = params_off["inverse_mass_matrix"]
        imm_on = params_on["inverse_mass_matrix"]
        for field in imm_off._fields:
            np.testing.assert_array_equal(
                np.asarray(getattr(imm_off, field)),
                np.asarray(getattr(imm_on, field)),
                err_msg=f"inverse_mass_matrix.{field} differs with telemetry on",
            )


class UnsupportedConfigurationTest(chex.TestCase):
    """Contradictory or unsupported requests must fail loudly, never silently."""

    def test_multi_chain_core_rejects_telemetry(self):
        from blackjax.adaptation.meta import build_multi_chain_meta_core

        with self.assertRaisesRegex(NotImplementedError, "single-chain only"):
            build_multi_chain_meta_core(_BUDGET, 8, telemetry=True)

    def test_staged_adaptation_rejects_multi_chain_telemetry(self):
        with self.assertRaisesRegex(NotImplementedError, "single-chain only"):
            staged_adaptation(
                blackjax.nuts,
                _logdensity_fn,
                metric="auto",
                max_grad_budget=_BUDGET,
                n_chains=8,
                metric_telemetry=True,
            )

    def test_full_matrices_requires_telemetry(self):
        with self.assertRaisesRegex(ValueError, "requires telemetry=True"):
            build_meta_adaptation_core(_BUDGET, full_matrices=True)
        with self.assertRaisesRegex(ValueError, "requires"):
            staged_adaptation(
                blackjax.nuts,
                _logdensity_fn,
                metric="auto",
                max_grad_budget=_BUDGET,
                telemetry_full_matrices=True,
            )

    def test_core_without_publication_is_rejected(self):
        """A core that cannot carry a record must not silently produce nothing."""
        warmup = staged_adaptation(
            blackjax.nuts,
            _logdensity_fn,
            metric="welford_diag",
            metric_telemetry=True,
        )
        with self.assertRaisesRegex(ValueError, "no 'publication' field"):
            warmup.run(jax.random.key(0), jnp.zeros(_D), num_steps=60)


class GateMaskTest(chex.TestCase):
    """Two masks, so an unevaluated predicate never reads as a failure."""

    def test_gate_bits_are_frozen(self):
        self.assertEqual(
            GATE_BITS,
            {"r2": 0, "s_gap_magnitude": 1, "s_gap_stability": 2, "deadline": 3},
        )

    def test_encode_decode_round_trip_exhaustive(self):
        names = sorted(GATE_BITS)
        for combo in range(1 << len(names)):
            bits = {n: bool((combo >> i) & 1) for i, n in enumerate(names)}
            self.assertEqual(decode_gates(encode_gates(**bits)), bits)

    def test_encode_rejects_unknown_gate(self):
        with self.assertRaisesRegex(KeyError, "unknown gate"):
            encode_gates(no_such_gate=True)

    def test_stability_gate_unevaluated_in_first_window(self):
        """S_gap stability needs the previous window's S_gap; window 0 has none.

        It must be reported as not-evaluated, not as failed.
        """
        _, records = _run(
            metric_telemetry=True, adaptation_info_fn=publication_adapt_info_fn()
        )
        chronology = extract_publication_chronology(records)
        first = chronology[0]
        self.assertFalse(first["gates_evaluated"]["s_gap_stability"])
        self.assertFalse(first["gates_passed"]["s_gap_stability"])
        for name in ("r2", "s_gap_magnitude", "deadline"):
            self.assertTrue(
                first["gates_evaluated"][name], msg=f"{name} should be evaluated"
            )
        for later in chronology[1:]:
            self.assertTrue(later["gates_evaluated"]["s_gap_stability"])

    def test_no_gate_evaluated_once_escalated(self):
        """After escalation the controller stops consulting the gates entirely."""
        core = build_meta_adaptation_core(_BUDGET, telemetry=True)
        state = core.init(_D)
        draws, grads = _make_isotropic_buffer(_D, 40)
        state = _fill_state_from_buffer(state, draws, grads)
        escalated = state._replace(has_escalated=jnp.array(True))
        record = core.final(escalated).publication
        self.assertEqual(int(record.gate_evaluated), 0)
        self.assertEqual(int(record.gate_passed), 0)
        self.assertTrue(bool(record.has_escalated_before))
        self.assertFalse(bool(record.escalated_now))


class SupportAccountingTest(chex.TestCase):
    """support_n must equal what final() consumed, not what survives it."""

    @parameterized.parameters(3, 17, 40)
    def test_support_matches_captured_pre_final_input(self, n_updates):
        """Compare against the buffer index captured immediately before final().

        The post-final state has already zeroed buffer_idx, which is precisely
        why the record must carry this.
        """
        core = build_meta_adaptation_core(_BUDGET, telemetry=True)
        state = core.init(_D)
        capacity = state.draws_buffer.shape[0]
        draws, grads = _make_isotropic_buffer(_D, n_updates)
        for i in range(n_updates):
            state = core.update(state, draws[i], grads[i])

        captured_buffer_idx = int(state.buffer_idx)
        self.assertEqual(captured_buffer_idx, n_updates)

        out = core.final(state)
        record = out.publication
        self.assertEqual(int(record.support_n), min(captured_buffer_idx, capacity))
        self.assertEqual(int(record.buffer_capacity), capacity)
        self.assertFalse(bool(record.support_saturated))
        # The information the record preserves is gone from the state itself.
        self.assertEqual(int(out.buffer_idx), 0)

    def test_buffer_wrap_is_reported_and_support_clamps_to_capacity(self):
        """A window longer than the buffer drops draws; that must be visible."""
        core = build_meta_adaptation_core(_BUDGET, telemetry=True)
        state = core.init(_D)
        capacity = state.draws_buffer.shape[0]
        n_updates = capacity + 12
        draws, grads = _make_isotropic_buffer(_D, n_updates)
        for i in range(n_updates):
            state = core.update(state, draws[i], grads[i])

        record = core.final(state).publication
        self.assertTrue(bool(record.support_saturated))
        self.assertEqual(int(record.support_n), capacity)
        self.assertLess(int(record.support_n), n_updates)


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
        boundaries = np.flatnonzero(schedule[:, 1] == 1)
        self.assertLen(self.chronology, len(boundaries))

    def test_window_index_is_contiguous_from_zero(self):
        self.assertEqual(
            [e["window_index"] for e in self.chronology],
            list(range(len(self.chronology))),
        )

    def test_window_index_starts_at_minus_one_before_first_publication(self):
        schedule = np.asarray(build_growing_window_schedule(_NUM_STEPS))
        first_boundary = int(np.flatnonzero(schedule[:, 1] == 1)[0])
        stacked = np.asarray(self.records.window_index)
        self.assertTrue(np.all(stacked[:first_boundary] == -1))
        self.assertEqual(int(stacked[first_boundary]), 0)

    def test_record_is_held_unchanged_between_boundaries(self):
        """Consumers dedupe on window_index; that requires the carry to hold."""
        schedule = np.asarray(build_growing_window_schedule(_NUM_STEPS))
        boundaries = np.flatnonzero(schedule[:, 1] == 1)
        stacked = np.asarray(self.records.window_index)
        first, second = int(boundaries[0]), int(boundaries[1])
        self.assertTrue(np.all(stacked[first:second] == 0))

    def test_step_at_publication_is_strictly_increasing(self):
        steps = [e["step_at_publication"] for e in self.chronology]
        self.assertEqual(steps, sorted(steps))
        self.assertLen(set(steps), len(steps))


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
        """Bitwise: the step size carried into the boundary step's transition."""
        in_force = np.asarray(self.records.epsilon_in_force)
        observed = np.asarray(self.step_sizes)
        for boundary in self.boundaries:
            np.testing.assert_array_equal(
                in_force[boundary],
                observed[boundary - 1],
                err_msg=f"epsilon_in_force at boundary {boundary}",
            )

    def test_epsilon_next_window_is_the_published_step_size(self):
        next_eps = np.asarray(self.records.epsilon_next_window)
        observed = np.asarray(self.step_sizes)
        for boundary in self.boundaries:
            np.testing.assert_array_equal(next_eps[boundary], observed[boundary])

    def test_the_four_epsilons_are_captured_separately(self):
        """No equality is asserted between them: they are different quantities.

        ``epsilon_window_average`` and ``epsilon_next_window`` are related by
        ``exp(log(.))``, which is not guaranteed bitwise-identical, so the test
        records that they are each finite and captured — it does not force them
        to agree, and it does not touch the adaptation to make them agree.
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
        # The mismatch the record exists to expose: what drove the completed
        # transition is not what the next window starts from.
        self.assertTrue(np.any(in_force != next_eps))
        self.assertTrue(np.any(in_force != after_da))

    def test_epsilons_carry_the_dual_averaging_dtype(self):
        """Never a hardcoded float32: that would break bitwise chronology under x64."""
        observed_dtype = np.asarray(self.step_sizes).dtype
        for name in (
            "epsilon_in_force",
            "epsilon_after_window_da",
            "epsilon_window_average",
            "epsilon_next_window",
        ):
            self.assertEqual(
                np.asarray(getattr(self.records, name)).dtype,
                observed_dtype,
                msg=f"{name} dtype diverged from the step-size dtype",
            )


class CandidateObservationTest(chex.TestCase):
    """The candidate must be visible even in the windows where it is withheld."""

    def _record_for(self, n_draws=40):
        core = build_meta_adaptation_core(_BUDGET, telemetry=True)
        state = core.init(_D)
        draws, grads = _make_isotropic_buffer(_D, n_draws)
        state = _fill_state_from_buffer(state, draws, grads)
        return core.final(state).publication

    def test_withheld_candidate_is_still_reported(self):
        record = self._record_for()
        self.assertFalse(bool(record.escalated_now))
        # Nothing was deployed, but the candidate the controller computed and
        # discarded is on the record.
        self.assertTrue(np.isfinite(np.asarray(record.candidate_logdet)))
        self.assertTrue(np.isfinite(np.asarray(record.candidate_lam_max)))
        self.assertTrue(np.isfinite(np.asarray(record.candidate_sigma_gm)))
        self.assertGreater(float(record.candidate_lam_max), 0.0)

    def test_detection_rank_is_distinct_from_effective_ranks(self):
        """Three different quantities; the record must not conflate them."""
        record = self._record_for()
        for name in (
            "detection_rank",
            "candidate_effective_rank",
            "deployed_effective_rank",
            "escalation_rank",
        ):
            value = int(getattr(record, name))
            self.assertGreaterEqual(value, 0)
        # Not escalated: nothing low-rank is deployed.
        self.assertEqual(int(record.deployed_effective_rank), 0)

    def test_in_force_metric_is_reported_alongside_the_published_one(self):
        record = self._record_for()
        self.assertTrue(np.isfinite(np.asarray(record.in_force_logdet)))
        self.assertTrue(np.isfinite(np.asarray(record.deployed_logdet)))

    def test_full_matrices_are_absent_by_default_and_present_when_asked(self):
        record = self._record_for()
        self.assertIsNone(record.candidate)
        self.assertIsNone(record.deployed)

        core = build_meta_adaptation_core(_BUDGET, telemetry=True, full_matrices=True)
        state = core.init(_D)
        draws, grads = _make_isotropic_buffer(_D, 40)
        state = _fill_state_from_buffer(state, draws, grads)
        full = core.final(state).publication
        self.assertIsNotNone(full.candidate)
        self.assertIsNotNone(full.deployed)
        self.assertEqual(full.candidate.sigma.shape, (_D,))

    def test_epsilons_are_nan_when_final_is_called_without_the_host(self):
        """The core cannot see the step size; it must not invent one."""
        record = self._record_for()
        self.assertTrue(np.isnan(np.asarray(record.epsilon_in_force)))
        self.assertTrue(np.isnan(np.asarray(record.epsilon_next_window)))


class PayloadShapeTest(chex.TestCase):
    """Bounded, and measured from real leaves rather than a nominal constant."""

    def _record(self, full_matrices=False):
        core = build_meta_adaptation_core(
            _BUDGET, telemetry=True, full_matrices=full_matrices
        )
        state = core.init(_D)
        draws, grads = _make_isotropic_buffer(_D, 40)
        state = _fill_state_from_buffer(state, draws, grads)
        return core.final(state).publication

    def test_default_record_leaves_are_all_scalar(self):
        record = self._record()
        for name in record._fields:
            value = getattr(record, name)
            if value is None:
                continue
            self.assertEqual(
                jnp.asarray(value).shape, (), msg=f"{name} is not a scalar"
            )

    def test_default_record_carries_no_buffers(self):
        """The compact record must never drag the draw/gradient buffers along."""
        record = self._record()
        total = sum(int(jnp.asarray(leaf).size) for leaf in jax.tree.leaves(record))
        self.assertEqual(total, len(jax.tree.leaves(record)))

    def test_record_nbytes_reports_actual_leaf_bytes(self):
        record = self._record()
        expected = sum(
            int(jnp.asarray(leaf).nbytes) for leaf in jax.tree.leaves(record)
        )
        self.assertEqual(record_nbytes(record), expected)
        self.assertLess(record_nbytes(record), 1024)

    def test_full_matrices_cost_is_the_reason_they_are_opt_in(self):
        compact = record_nbytes(self._record())
        full = record_nbytes(self._record(full_matrices=True))
        self.assertGreater(full, compact)

    def test_schema_version_is_exposed_to_consumers(self):
        self.assertIsInstance(SCHEMA_VERSION, int)
        _, records = _run(
            metric_telemetry=True, adaptation_info_fn=publication_adapt_info_fn()
        )
        for entry in extract_publication_chronology(records):
            self.assertEqual(entry["schema_version"], SCHEMA_VERSION)

    def test_chronology_rejects_unstacked_records(self):
        record = self._record()
        with self.assertRaisesRegex(ValueError, "leading step axis"):
            extract_publication_chronology(record)

    def test_record_field_count_matches_the_documented_layout(self):
        """Provisional, not frozen: this catches an accidental change only."""
        self.assertLen(MetricPublicationRecord._fields, 34)


if __name__ == "__main__":
    absltest.main()
