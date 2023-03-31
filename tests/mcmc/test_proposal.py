import chex
import jax
import numpy as np
import pytest
from absl.testing import parameterized
from jax import numpy as jnp

from blackjax.mcmc.proposal import (
    Proposal,
    asymmetric_proposal_generator,
    proposal_from_energy_diff,
)
from blackjax.mcmc.random_walk import normal


class TestAsymmetricProposalGenerator(chex.TestCase):
    @chex.all_variants
    def test_new(self):
        state = (jnp.array([1, 2, 3]),)
        new, _ = asymmetric_proposal_generator(None, None, None)

        assert_proposal_equals(
            self.variant(new)(state), Proposal(state, 0.0, 0.0, -np.inf)
        )

    @chex.all_variants
    def test_update(self):
        def transition_energy(prev, next):
            return next - prev

        new_proposal = Proposal(jnp.array([1, 2]), 0.0, 2.0, 50)

        @chex.chexify
        def proposal_factory(prev_energy, new_energy, divergence_threshold, new_state):
            chex.assert_tree_all_close(prev_energy, -20)
            chex.assert_tree_all_close(new_energy, 20)
            chex.assert_tree_all_close(divergence_threshold, 50)
            chex.assert_tree_all_close(new_state, 50)
            return new_proposal

        _, update = asymmetric_proposal_generator(
            transition_energy, 50, proposal_factory
        )

        proposed = self.variant(update)(30, 50)
        assert_proposal_equals(proposed, new_proposal)


def assert_proposal_equals(left, right):
    np.testing.assert_allclose(left.energy, right.energy)
    np.testing.assert_allclose(left.weight, right.weight)
    np.testing.assert_allclose(left.sum_log_p_accept, right.sum_log_p_accept)
    np.testing.assert_allclose(left.state, right.state)


class TestProposalFromEnergyDiff(chex.TestCase):
    @chex.all_variants
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
        state = None
        proposal, divergence = self.variant(proposal_from_energy_diff)(
            5, 10, threshold, state
        )
        assert divergence == is_divergent

    @chex.all_variants
    def test_sum_log_paccept(self):
        state = None
        proposal, _ = self.variant(proposal_from_energy_diff)(5, 10, 0, state)
        np.testing.assert_allclose(proposal.sum_log_p_accept, -5.0)

        proposal, _ = self.variant(proposal_from_energy_diff)(10, 5, 0, state)
        np.testing.assert_allclose(proposal.sum_log_p_accept, 0.0)

    @chex.all_variants
    def test_delta_energy_is_nan(self):
        state = None
        proposal, _ = self.variant(proposal_from_energy_diff)(np.nan, np.nan, 0, state)
        assert np.isneginf(proposal.weight)

    @chex.all_variants
    def test_weight(self):
        state = jnp.array([1, 2])
        proposal, _ = self.variant(proposal_from_energy_diff)(5, 10, 0, state)

        np.testing.assert_allclose(state, proposal.state)
        np.testing.assert_allclose(proposal.weight, -5)
        np.testing.assert_allclose(proposal.energy, 10)


class TestNormalProposalDistribution(chex.TestCase):
    def setUp(self):
        super().setUp()
        self.key = jax.random.PRNGKey(20220611)

    def test_normal_univariate(self):
        """
        Move samples are generated in the univariate case,
        with std following sigma, and independently of the position.
        """
        proposal = normal(sigma=jnp.array([1.0]))
        samples_from_initial_position = [
            proposal(key, jnp.array([10.0])) for key in jax.random.split(self.key, 100)
        ]
        samples_from_another_position = [
            proposal(key, jnp.array([15000.0]))
            for key in jax.random.split(self.key, 100)
        ]

        for samples in [samples_from_initial_position, samples_from_another_position]:
            np.testing.assert_allclose(0.0, np.mean(samples), rtol=1e-2, atol=1e-1)
            np.testing.assert_allclose(1.0, np.std(samples), rtol=1e-2, atol=1e-1)

    def test_normal_multivariate(self):
        proposal = normal(sigma=jnp.array([1.0, 2.0]))
        samples = [
            proposal(key, jnp.array([10.0, 20.0]))
            for key in jax.random.split(self.key, 1000)
        ]
        self._check_mean_and_std(jnp.array([0.0, 0.0]), jnp.array([1.0, 2.0]), samples)

    def test_normal_multivariate_full_sigma(self):
        proposal = normal(sigma=jnp.array([[1.0, 0.0], [0.0, 2.0]]))
        samples = [
            proposal(key, jnp.array([10.0, 20.0]))
            for key in jax.random.split(self.key, 1000)
        ]
        self._check_mean_and_std(
            expected_mean=jnp.array([0.0, 0.0]),
            expected_std=jnp.array([1.0, 2.0]),
            samples=samples,
        )

    def test_normal_wrong_sigma(self):
        with pytest.raises(ValueError):
            normal(sigma=jnp.array([[[1.0, 2.0]]]))

    @staticmethod
    def _check_mean_and_std(expected_mean, expected_std, samples):
        np.testing.assert_allclose(
            expected_mean, np.mean(samples), rtol=1e-2, atol=1e-1
        )
        np.testing.assert_allclose(
            expected_std,
            np.sqrt(np.diag(np.cov(np.array(samples).T))),
            rtol=1e-2,
            atol=1e-1,
        )
