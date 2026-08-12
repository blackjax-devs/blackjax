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
"""Tests for the post-warmup per-chain start-state safety mechanism (Issue#1064).

Coverage (this file grows in a follow-up commit that wires the mechanism into
:mod:`blackjax.adaptation.staged_adaptation`):
- TestLooStateGate: stage 1 (ensemble-calibrated LOO gate) as a pure function.
- TestRedrawFlaggedChains: stage 2a (redraw near a healthy anchor), including
  a dict/PyTree position case (guards the blind spot fixed in PR #1013).
- TestComputeChainEnergies: the energy OBSERVABLE (amendment A; not a trigger).
"""
import jax
import jax.numpy as jnp
import numpy as np

from blackjax.adaptation.meta._calibration import (
    _STATE_GATE_LOO_Z_THRESHOLD,
    _STATE_GATE_MIN_CHAINS,
)
from blackjax.adaptation.meta._start_state import (
    compute_chain_energies,
    loo_state_gate,
    redraw_flagged_chains,
    select_healthy_anchor,
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
