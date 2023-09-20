import chex
import jax
import numpy as np
import pytest
from jax import numpy as jnp

from blackjax.mcmc.random_walk import normal


class TestNormalProposalDistribution(chex.TestCase):
    def setUp(self):
        super().setUp()
        self.key = jax.random.key(20220611)

    def test_normal_univariate(self):
        """
        Move samples are generated in the univariate case,
        with std following sigma, and independently of the position.
        """
        key1, key2 = jax.random.split(self.key)
        proposal = normal(sigma=jnp.array([1.0]))
        samples_from_initial_position = [
            proposal(key, jnp.array([10.0])) for key in jax.random.split(key1, 100)
        ]
        samples_from_another_position = [
            proposal(key, jnp.array([15000.0])) for key in jax.random.split(key2, 100)
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
