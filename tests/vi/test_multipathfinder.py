"""Tests for the multi-path Pathfinder algorithm."""
import chex
import jax
import jax.numpy as jnp
import jax.scipy.stats as stats
import numpy as np
from absl.testing import absltest

import blackjax


class MultipathfinderTest(chex.TestCase):
    def setUp(self):
        super().setUp()
        self.key = jax.random.key(42)

    def test_recover_posterior(self):
        """Multi-path Pathfinder should estimate a Gaussian posterior."""
        ndim = 2
        true_mean = jnp.array([1.0, -1.0])
        true_cov = jnp.array([[1.0, 0.3], [0.3, 2.0]])

        def logdensity_fn(x):
            return stats.multivariate_normal.logpdf(x, true_mean, true_cov)

        n_paths = 4
        key_init, key_run, key_sample = jax.random.split(self.key, 3)
        initial_positions = jax.random.normal(key_init, (n_paths, ndim))

        algo = blackjax.multipathfinder(logdensity_fn)
        state, info = algo.init(key_run, initial_positions, num_samples=100)

        # State should have one PathfinderState per path.
        assert jax.tree.leaves(state.path_states.position)[0].shape[0] == n_paths

        # samples, logp, logq should have shape (n_paths, num_samples, ...).
        assert state.samples.shape == (n_paths, 100, ndim)
        assert state.logp.shape == (n_paths, 100)
        assert state.logq.shape == (n_paths, 100)

        # Draw samples and check that the mean is close to true_mean.
        samples = algo.sample(key_sample, state, 2000)
        np.testing.assert_allclose(samples.mean(0), true_mean, atol=0.2)

    def test_step_is_noop(self):
        """The step function should return the same state unchanged."""
        ndim = 2
        n_paths = 2

        def logdensity_fn(x):
            return -0.5 * jnp.sum(x**2)

        key_init, key_run, key_step = jax.random.split(self.key, 3)
        initial_positions = jax.random.normal(key_init, (n_paths, ndim))

        algo = blackjax.multipathfinder(logdensity_fn)
        state, _ = algo.init(key_run, initial_positions, num_samples=20)
        new_state, info = algo.step(key_step, state)

        # State should be identical after step.
        np.testing.assert_array_equal(state.logp, new_state.logp)
        assert info is None


if __name__ == "__main__":
    absltest.main()
