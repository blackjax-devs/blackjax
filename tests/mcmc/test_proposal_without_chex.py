import unittest
from unittest.mock import MagicMock

import numpy as np
from absl.testing import parameterized

from blackjax.mcmc.proposal import (
    Proposal,
    asymmetric_proposal_generator,
    proposal_from_energy_diff,
)


class TestAsymmetricProposalGenerator(unittest.TestCase):
    def test_new(self):
        state = MagicMock()
        new, _ = asymmetric_proposal_generator(None, None)
        assert new(state) == Proposal(state, 0.0, 0.0, -np.inf)

    def test_update(self):
        def transition_energy(prev, next):
            return next - prev

        new_proposal = MagicMock()

        def proposal_factory(prev_energy, new_energy, new_state):
            assert prev_energy == -20
            assert new_energy == 20
            assert new_state == 50
            return new_proposal

        _, update = asymmetric_proposal_generator(transition_energy, proposal_factory)
        proposed = update(30, 50)
        assert proposed == new_proposal


class TestProposalFromEnergyDiff(parameterized.TestCase):
    @parameterized.parameters(
        [
            (5, 10, 2, True),
            (5, 10, 4, True),
            (5, 10, 6, False),
            (5, 10, 5, False),
            (10, 5, 2, True),
            (10, 5, 4, True),
            (10, 5, 6, False),
            (10, 5, 5, False),
        ]
    )
    def test_divergence_threshold(self, before, after, threshold, is_divergent):
        state = MagicMock()
        proposal = proposal_from_energy_diff(5, 10, state)
        divergence = -proposal.weight > threshold
        assert divergence == is_divergent

    def test_sum_log_paccept(self):
        state = MagicMock()
        proposal = proposal_from_energy_diff(5, 10, state)
        np.testing.assert_allclose(proposal.sum_log_p_accept, -5.0)

        proposal = proposal_from_energy_diff(10, 5, state)
        np.testing.assert_allclose(proposal.sum_log_p_accept, 0.0)

    def test_delta_energy_is_nan(self):
        state = MagicMock()
        proposal = proposal_from_energy_diff(np.nan, np.nan, state)
        assert np.isneginf(proposal.weight)

    def test_weight(self):
        state = MagicMock()
        proposal = proposal_from_energy_diff(5, 10, state)

        assert proposal.state == state
        np.testing.assert_allclose(proposal.weight, -5)
        np.testing.assert_allclose(proposal.energy, 10)
