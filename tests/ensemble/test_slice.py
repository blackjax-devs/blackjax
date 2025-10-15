"""Test the ensemble slice sampling kernel."""

import chex
import jax
import jax.numpy as jnp
import jax.scipy.stats as stats
from absl.testing import absltest

from blackjax.ensemble.slice import (
    SliceEnsembleInfo,
    SliceEnsembleState,
    as_top_level_api,
    differential_direction,
    init,
    random_direction,
    slice_along_direction,
)


class EnsembleSliceTest(chex.TestCase):
    """Test the ensemble slice sampling algorithm."""

    def test_differential_direction(self):
        """Test that differential_direction produces valid directions."""
        rng_key = jax.random.PRNGKey(0)

        # Complementary ensemble
        complementary_coords = jnp.array([[0.0, 0.0], [2.0, 4.0], [3.0, 1.0]])
        n_update = 2
        mu = 1.0

        directions, tune_once = differential_direction(
            rng_key, complementary_coords, n_update, mu
        )

        # Check shape and properties
        self.assertEqual(directions.shape, (n_update, 2))
        self.assertTrue(tune_once)
        self.assertTrue(jnp.isfinite(directions).all())

    def test_random_direction(self):
        """Test that random_direction produces valid directions."""
        rng_key = jax.random.PRNGKey(0)

        # Template coordinates
        template_coords = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        n_update = 2
        mu = 1.0

        directions, tune_once = random_direction(rng_key, template_coords, n_update, mu)

        # Check shape and properties
        self.assertEqual(directions.shape, template_coords.shape)
        self.assertTrue(tune_once)
        self.assertTrue(jnp.isfinite(directions).all())

    def test_slice_along_direction_1d(self):
        """Test slice sampling along a direction in 1D."""
        rng_key = jax.random.PRNGKey(42)

        # Simple 1D Gaussian
        def logprob_fn(x):
            return stats.norm.logpdf(x, 0.0, 1.0)

        x0 = 0.0
        logp0 = logprob_fn(x0)
        direction = 1.0

        x1, logp1, nexp, ncon, neval = slice_along_direction(
            rng_key, x0, logp0, direction, logprob_fn, maxsteps=100, maxiter=1000
        )

        # Check that we got valid results
        self.assertTrue(jnp.isfinite(x1))
        self.assertTrue(jnp.isfinite(logp1))
        self.assertGreater(neval, 0)

    def test_init(self):
        """Test initialization of ensemble slice sampler."""

        def logdensity_fn(x):
            return stats.norm.logpdf(x, 0.0, 1.0).sum()

        rng_key = jax.random.PRNGKey(0)
        n_walkers = 10
        initial_position = jax.random.normal(rng_key, (n_walkers, 2))

        state = init(initial_position, logdensity_fn, has_blobs=False, mu=1.0)

        # Check state properties
        self.assertIsInstance(state, SliceEnsembleState)
        self.assertEqual(state.coords.shape, (n_walkers, 2))
        self.assertEqual(state.log_probs.shape, (n_walkers,))
        self.assertIsNone(state.blobs)
        self.assertEqual(state.mu, 1.0)
        self.assertTrue(state.tuning_active)
        self.assertEqual(state.patience_count, 0)

    def test_ensemble_slice_1d_gaussian(self):
        """Test ensemble slice sampling on a 1D Gaussian distribution."""

        # Define 1D Gaussian target
        mu_true = 2.0
        sigma_true = 1.5

        def logdensity_fn(x):
            return stats.norm.logpdf(x.squeeze(), mu_true, sigma_true)

        rng_key = jax.random.PRNGKey(123)
        init_key, sample_key = jax.random.split(rng_key)

        # Initialize with 20 walkers
        n_walkers = 20
        initial_position = jax.random.normal(init_key, (n_walkers, 1))

        # Create algorithm
        algorithm = as_top_level_api(
            logdensity_fn, move="differential", mu=1.0, maxsteps=100, maxiter=1000
        )
        initial_state = algorithm.init(initial_position)

        # Run a few steps
        def run_step(state, key):
            new_state, info = algorithm.step(key, state)
            return new_state, (new_state, info)

        keys = jax.random.split(sample_key, 100)
        final_state, (states, infos) = jax.lax.scan(run_step, initial_state, keys)

        # Check that we get valid states
        self.assertIsInstance(final_state, SliceEnsembleState)
        self.assertEqual(final_state.coords.shape, (n_walkers, 1))
        self.assertEqual(final_state.log_probs.shape, (n_walkers,))

        # Check info
        self.assertIsInstance(infos, SliceEnsembleInfo)
        # Slice sampling always accepts
        self.assertTrue(jnp.all(infos.acceptance_rate == 1.0))
        self.assertTrue(jnp.all(infos.is_accepted))

        # Check that evaluations are happening
        self.assertTrue(jnp.all(infos.nevals > 0))

    def test_ensemble_slice_2d_gaussian(self):
        """Test ensemble slice sampling on a 2D Gaussian distribution."""

        # Define 2D Gaussian target
        mu = jnp.array([1.0, 2.0])
        cov = jnp.array([[1.0, 0.5], [0.5, 2.0]])

        def logdensity_fn(x):
            return stats.multivariate_normal.logpdf(x, mu, cov)

        rng_key = jax.random.PRNGKey(42)
        init_key, sample_key = jax.random.split(rng_key)

        # Initialize ensemble
        n_walkers = 20
        initial_position = jax.random.normal(init_key, (n_walkers, 2))

        # Create algorithm
        algorithm = as_top_level_api(
            logdensity_fn, move="differential", mu=1.0, maxsteps=100, maxiter=1000
        )
        initial_state = algorithm.init(initial_position)

        # Run steps
        def run_step(state, key):
            new_state, info = algorithm.step(key, state)
            return new_state, new_state.coords

        keys = jax.random.split(sample_key, 200)
        final_state, samples = jax.lax.scan(run_step, initial_state, keys)

        # Take second half as samples (burn-in)
        samples = samples[100:]  # Shape: (100, n_walkers, 2)
        samples = samples.reshape(-1, 2)  # Flatten to (100 * n_walkers, 2)

        # Check convergence (loose tolerance for quick test)
        sample_mean = jnp.mean(samples, axis=0)
        self.assertAlmostEqual(sample_mean[0].item(), mu[0], places=0)
        self.assertAlmostEqual(sample_mean[1].item(), mu[1], places=0)

    def test_jit_compilation(self):
        """Test that the algorithm can be JIT compiled."""

        def logdensity_fn(x):
            return stats.norm.logpdf(x.squeeze(), 0.0, 1.0)

        rng_key = jax.random.PRNGKey(0)
        n_walkers = 10
        initial_position = jax.random.normal(rng_key, (n_walkers, 1))

        algorithm = as_top_level_api(logdensity_fn, move="differential")
        initial_state = algorithm.init(initial_position)

        # JIT compile step function
        jitted_step = jax.jit(algorithm.step)

        # Run one step
        key = jax.random.PRNGKey(1)
        new_state, info = jitted_step(key, initial_state)

        # Check results are valid
        self.assertIsInstance(new_state, SliceEnsembleState)
        self.assertIsInstance(info, SliceEnsembleInfo)

    def test_random_move(self):
        """Test ensemble slice sampling with random move."""

        def logdensity_fn(x):
            return stats.norm.logpdf(x, 0.0, 1.0).sum()

        rng_key = jax.random.PRNGKey(99)
        n_walkers = 10
        initial_position = jax.random.normal(rng_key, (n_walkers, 2))

        # Use random move instead of differential
        algorithm = as_top_level_api(
            logdensity_fn, move="random", mu=1.0, maxsteps=100, maxiter=1000
        )
        initial_state = algorithm.init(initial_position)

        # Run a few steps
        keys = jax.random.split(rng_key, 10)

        def run_step(state, key):
            new_state, info = algorithm.step(key, state)
            return new_state, info

        final_state, infos = jax.lax.scan(run_step, initial_state, keys)

        # Check valid results
        self.assertIsInstance(final_state, SliceEnsembleState)
        self.assertTrue(jnp.all(infos.acceptance_rate == 1.0))


if __name__ == "__main__":
    absltest.main()
