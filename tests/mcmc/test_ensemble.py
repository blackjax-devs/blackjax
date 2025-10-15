"""Test the ensemble MCMC kernels."""

import chex
import jax
import jax.numpy as jnp
import jax.scipy.stats as stats
from absl.testing import absltest

import blackjax
from blackjax.mcmc.ensemble import EnsembleState, stretch_move


class EnsembleTest(chex.TestCase):
    """Test the ensemble MCMC algorithms."""

    def test_stretch_move(self):
        """Test that stretch_move produces valid proposals."""
        rng_key = jax.random.PRNGKey(0)

        # Simple 2D case
        walker_coords = jnp.array([1.0, 2.0])
        complementary_coords = jnp.array([[0.0, 0.0], [2.0, 4.0], [3.0, 1.0]])

        proposal, log_hastings_ratio = stretch_move(
            rng_key, walker_coords, complementary_coords, a=2.0
        )

        # Check shapes
        self.assertEqual(proposal.shape, walker_coords.shape)
        self.assertEqual(log_hastings_ratio.shape, ())

        # Check that proposal is finite
        self.assertTrue(jnp.isfinite(proposal).all())
        self.assertTrue(jnp.isfinite(log_hastings_ratio))

    def test_stretch_move_pytree(self):
        """Test that stretch_move works with PyTree structures."""
        rng_key = jax.random.PRNGKey(0)

        # PyTree case
        walker_coords = {"a": jnp.array([1.0, 2.0]), "b": jnp.array(3.0)}
        complementary_coords = {
            "a": jnp.array([[0.0, 0.0], [2.0, 4.0], [3.0, 1.0]]),
            "b": jnp.array([1.0, 2.0, 3.0]),
        }

        proposal, log_hastings_ratio = stretch_move(
            rng_key, walker_coords, complementary_coords, a=2.0
        )

        # Check structure
        self.assertEqual(set(proposal.keys()), {"a", "b"})
        self.assertEqual(proposal["a"].shape, walker_coords["a"].shape)
        self.assertEqual(proposal["b"].shape, walker_coords["b"].shape)
        self.assertEqual(log_hastings_ratio.shape, ())

    def test_stretch_algorithm_2d_gaussian(self):
        """Test the stretch algorithm on a 2D Gaussian distribution."""

        # Define a 2D Gaussian target
        mu = jnp.array([1.0, 2.0])
        cov = jnp.array([[1.0, 0.5], [0.5, 2.0]])

        def logdensity_fn(x):
            return stats.multivariate_normal.logpdf(x, mu, cov)

        # Initialize ensemble of 20 walkers
        rng_key = jax.random.PRNGKey(42)
        init_key, sample_key = jax.random.split(rng_key)

        n_walkers = 20
        initial_position = jax.random.normal(init_key, (n_walkers, 2))

        # Create algorithm
        algorithm = blackjax.ensemble(logdensity_fn, a=2.0)
        initial_state = algorithm.init(initial_position)

        # Run a few steps
        def run_step(state, key):
            new_state, info = algorithm.step(key, state)
            return new_state, (new_state, info)

        keys = jax.random.split(sample_key, 100)
        final_state, (states, infos) = jax.lax.scan(run_step, initial_state, keys)

        # Check that we get valid states
        self.assertIsInstance(final_state, EnsembleState)
        self.assertEqual(final_state.coords.shape, (n_walkers, 2))
        self.assertEqual(final_state.log_probs.shape, (n_walkers,))

        # Check that acceptance rate is reasonable
        mean_acceptance = jnp.mean(infos.acceptance_rate)
        self.assertGreater(mean_acceptance, 0.1)  # Should accept some proposals
        self.assertLess(mean_acceptance, 0.9)  # Should reject some proposals

    def test_stretch_algorithm_convergence(self):
        """Test that the stretch algorithm converges to the correct distribution."""

        # Simple 1D Gaussian
        mu = 2.0
        sigma = 1.5

        def logdensity_fn(x):
            return stats.norm.logpdf(x.squeeze(), mu, sigma)

        rng_key = jax.random.PRNGKey(123)
        init_key, sample_key = jax.random.split(rng_key)

        n_walkers = 50
        initial_position = jax.random.normal(init_key, (n_walkers, 1))

        # Run algorithm
        algorithm = blackjax.ensemble(logdensity_fn, a=2.0)
        initial_state = algorithm.init(initial_position)

        def run_step(state, key):
            new_state, info = algorithm.step(key, state)
            return new_state, new_state.coords

        keys = jax.random.split(sample_key, 1000)
        final_state, samples = jax.lax.scan(run_step, initial_state, keys)

        # Take samples from the second half (burn-in)
        samples = samples[500:]  # Shape: (500, n_walkers, 1)
        samples = samples.reshape(-1, 1)  # Flatten to (500 * n_walkers, 1)

        # Check convergence
        sample_mean = jnp.mean(samples)
        sample_std = jnp.std(samples)

        # Allow for some tolerance due to finite sampling
        self.assertAlmostEqual(sample_mean.item(), mu, places=1)
        self.assertAlmostEqual(sample_std.item(), sigma, places=1)


if __name__ == "__main__":
    absltest.main()
