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
"""Tests for verdict extraction (:mod:`blackjax.adaptation.meta.verdict`).

Coverage:
- TestEffectiveRankHonesty: effective_rank vs nominal_rank reporting.
- TestExtractMultiChainVerdictNewFields: v2.1 diagnostic flags in multi-chain verdict.
"""
import jax
import jax.numpy as jnp
import numpy as np

from blackjax.adaptation.meta import (
    build_meta_adaptation_core,
    build_multi_chain_meta_core,
    extract_meta_verdict,
    extract_multi_chain_verdict,
)
from tests.adaptation._meta_fixtures import _fill_mc_state, _make_mc_deep_spread
from tests.fixtures import BlackJAXTest


class TestEffectiveRankHonesty(BlackJAXTest):
    """FIX 2: effective_rank reports the deployed rank; nominal_rank is in flags.

    The fixture constructs a state where escalation_rank (nominal, from
    _choose_rank) is larger than the count of truly active Fisher-metric
    directions (|lam_i - 1| > _LAM_NONTRIVIAL_TOL).  This reproduces the
    over-counting that occurs in high-d finite-sample settings where the
    finite-sample noise floor pushes spurious eigenvalues above the fixed
    cutoff=2.0, inflating the pre-mask rank count beyond the true deployed
    structure.

    All assertions are structural (not stochastic): the state is constructed
    deterministically by patching lam directly.  The fix is a reporting change
    only — no escalation-decision path is altered.

    The suite runs under both f32 (default) and x64 via _run_under_x64.
    """

    def _build_overcount_state(self, d=12, max_rank=6, nominal=6):
        """Construct a post-escalation state where nominal_rank > effective_rank.

        d=12, max_rank=6 → actual_rank = min(6, max(12//2,1)) = 6.
        lam = [3.5, 0.2, 1.0, 1.0, 1.0, 1.0]: only first 2 are non-trivial.
        escalation_rank = nominal (6): simulates _choose_rank over-count.
        effective_rank = 2: only the first two directions are deployed.
        """
        core = build_meta_adaptation_core(50000, max_rank=max_rank)
        state = core.init(d)

        lam_deployed = jnp.array([3.5, 0.2] + [1.0] * (max_rank - 2), dtype=jnp.float32)
        state = state._replace(
            has_escalated=jnp.array(True, dtype=jnp.bool_),
            escalation_rank=jnp.array(nominal, dtype=jnp.int32),
            inverse_mass_matrix=state.inverse_mass_matrix._replace(lam=lam_deployed),
        )
        return state, nominal

    def test_effective_rank_reflects_deployed_count(self):
        """effective_rank = count(|lam_i - 1| > tol) = 2, not nominal_rank = 6."""
        state, nominal = self._build_overcount_state()
        verdict = extract_meta_verdict(
            state, max_grad_budget=50000, num_warmup_steps=2500
        )
        # Deployed: only lam[0]=3.5 and lam[1]=0.2 are non-trivial.
        self.assertEqual(
            verdict.effective_rank,
            2,
            f"effective_rank should be 2 (deployed); got {verdict.effective_rank}",  # noqa: E702
        )

    def test_nominal_rank_in_flags(self):
        """flags['nominal_rank'] preserves the pre-mask escalation_rank."""
        state, nominal = self._build_overcount_state()
        verdict = extract_meta_verdict(
            state, max_grad_budget=50000, num_warmup_steps=2500
        )
        self.assertIn(
            "nominal_rank", verdict.flags, "nominal_rank must be present in flags"
        )
        self.assertEqual(
            verdict.flags["nominal_rank"],
            nominal,
            f"flags['nominal_rank'] should equal escalation_rank={nominal}",
        )

    def test_effective_rank_strictly_less_than_nominal(self):
        """When _choose_rank over-counts, effective_rank < nominal_rank."""
        state, nominal = self._build_overcount_state()
        verdict = extract_meta_verdict(
            state, max_grad_budget=50000, num_warmup_steps=2500
        )
        self.assertLess(
            verdict.effective_rank,
            verdict.flags["nominal_rank"],
            f"effective_rank ({verdict.effective_rank}) must be < "
            f"nominal_rank ({verdict.flags['nominal_rank']}) for over-count fixture",
        )

    def test_effective_rank_zero_before_escalation(self):
        """Before escalation, effective_rank = 0 (lam = ones → all |lam_i - 1| = 0)."""
        core = build_meta_adaptation_core(50000, max_rank=5)
        state = core.init(10)
        verdict = extract_meta_verdict(
            state, max_grad_budget=50000, num_warmup_steps=2500
        )
        self.assertEqual(
            verdict.effective_rank,
            0,
            "Before escalation, all lam = 1 → effective_rank must be 0",
        )
        self.assertEqual(
            verdict.flags["nominal_rank"],
            0,
            "Before escalation, nominal_rank must also be 0",
        )

    def test_effective_rank_no_effect_on_escalation_decision(self):
        """Changing effective_rank reporting does not affect has_escalated in carry.

        Verifies that effective_rank is a pure reporting transformation:
        the underlying has_escalated flag in the state is unchanged after
        calling extract_meta_verdict (it only reads, never writes).
        """
        state, nominal = self._build_overcount_state()
        self.assertTrue(bool(np.asarray(state.has_escalated)))
        verdict = extract_meta_verdict(
            state, max_grad_budget=50000, num_warmup_steps=2500
        )
        # has_escalated is still True in the original state
        self.assertTrue(bool(np.asarray(state.has_escalated)))
        self.assertEqual(verdict.route, "low_rank")

    def test_effective_rank_under_x64(self):
        """effective_rank count is consistent under x64 (lam dtype may widen to f64)."""
        try:
            jax.config.update("jax_enable_x64", True)
            state, nominal = self._build_overcount_state()
            verdict = extract_meta_verdict(
                state, max_grad_budget=50000, num_warmup_steps=2500
            )
            self.assertEqual(
                verdict.effective_rank,
                2,
                "Under x64: effective_rank should still be 2 (2 non-trivial lam entries)",
            )
            self.assertEqual(verdict.flags["nominal_rank"], nominal)
        finally:
            jax.config.update("jax_enable_x64", False)


class TestExtractMultiChainVerdictNewFields(BlackJAXTest):
    """extract_multi_chain_verdict flags dict includes all four v2.1 diagnostic keys."""

    def test_new_flags_present_and_finite(self):
        """Flags dict contains within_lam1, chain_consistency_psi, r1_top, detection_branch."""
        M, n, d = 8, 150, 10
        core = build_multi_chain_meta_core(max_grad_budget=40000, n_chains=M)
        state = core.init(d)
        draws_mc, grads_mc = _make_mc_deep_spread(M, n, d, seed=60)
        state1 = _fill_mc_state(state, draws_mc, grads_mc)
        final_state = core.final(state1)

        verdict = extract_multi_chain_verdict(
            final_state, max_grad_budget=40000, num_warmup_steps=1000
        )
        flags = verdict.flags

        for key in (
            "within_lam1",
            "chain_consistency_psi",
            "r1_top",
            "detection_branch",
        ):
            self.assertIn(key, flags, f"Missing verdict flag: {key}")

        self.assertFalse(np.isnan(flags["within_lam1"]))
        self.assertFalse(np.isnan(flags["chain_consistency_psi"]))
        self.assertFalse(np.isnan(flags["r1_top"]))
        self.assertIn(
            flags["detection_branch"],
            ["none", "pooled_within", "between_means", "both"],
        )

    def test_detection_branch_none_on_init_state(self):
        """Before any final() call, detection_branch flag is 'none'."""
        M, d = 8, 10
        core = build_multi_chain_meta_core(max_grad_budget=40000, n_chains=M)
        state = core.init(d)
        verdict = extract_multi_chain_verdict(
            state, max_grad_budget=40000, num_warmup_steps=1000
        )
        self.assertEqual(verdict.flags["detection_branch"], "none")
