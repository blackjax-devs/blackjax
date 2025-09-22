import chex
import jax
import numpy as np
import pytest
from absl.testing import parameterized
from jax import numpy as jnp

from blackjax.mcmc.random_walk import normal


class TestNormalProposalDistribution(chex.TestCase):
    def setUp(self):
        super().setUp()
        self.key = jax.random.key(20250120)

    @parameterized.parameters([10.0, 15000.0])
    def test_normal_univariate(self, initial_position):
        """
        Move samples are generated in the univariate case,
        with std following sigma, and independently of the position.
        """
        keys = jax.random.split(self.key, 200)
        proposal = normal(sigma=jnp.array([1.0]))
        samples = [proposal(key, jnp.array([initial_position])) for key in keys]
        self._check_mean_and_std(jnp.array([0.0]), jnp.array([1.0]), samples)

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
            np.sqrt(np.diag(np.atleast_2d(np.cov(np.array(samples).T)))),
            rtol=1e-2,
            atol=1e-1,
        )
