"""Test the generic SMC sampler"""
import itertools

import chex
import jax
import jax.numpy as jnp
import jax.scipy.stats as stats
import numpy as np
from absl.testing import absltest, parameterized

import blackjax.hmc as hmc
import blackjax.inference.smc.resampling as resampling
from blackjax.inference.smc.base import _normalize, new_smc_state, smc


def kernel_logprob_fn(position):
    return jnp.sum(stats.norm.logpdf(position))


def log_weights_fn(x, y):
    return jnp.sum(stats.norm.logpdf(y - x))


def _weighted_avg_and_std(values, weights):
    """
    Code is taken from
    https://stackoverflow.com/a/2415343
    """
    average = np.average(values, weights=weights)
    # Fast and numerically precise:
    variance = np.average((values - average) ** 2, weights=weights)
    return average, np.sqrt(variance)


class SMCTest(chex.TestCase):
    def setUp(self):
        super().setUp()
        self.key = jax.random.PRNGKey(42)

    @chex.all_variants(with_pmap=False)
    @parameterized.parameters(itertools.product([500, 1000, 5000], [True, False]))
    def test_smc(self, N, is_waste_free):
        """Simple test for sampling correctness"""
        np.random.seed(0)

        num_iterations = 25
        num_samples = N * num_iterations

        mcmc_factory = lambda logprob_fn: hmc.kernel(
            logprob_fn,
            step_size=1e-2,
            inverse_mass_matrix=jnp.eye(1),
            num_integration_steps=50,
        )

        specialized_log_weights_fn = lambda tree: log_weights_fn(tree, 1.0)

        kernel = smc(
            mcmc_factory,
            hmc.new_state,
            resampling.systematic,
            num_iterations,
            is_waste_free,
        )
        kernel = self.variant(kernel, static_argnums=(2, 3))

        # Don't use exactly the invariant distribution for the MCMC kernel
        init_particles = 0.25 + np.random.randn(num_samples)
        init_state, _ = new_smc_state(init_particles, lambda z: 1.0)

        (weights, particles), _ = kernel(
            self.key,
            init_state,
            kernel_logprob_fn,
            specialized_log_weights_fn,
        )

        expected_mean = 0.5
        expected_std = np.sqrt(0.5)
        actual_mean, actual_std = _weighted_avg_and_std(particles, weights)

        np.testing.assert_allclose(expected_mean, actual_mean, rtol=1e-1, atol=1e-2)
        np.testing.assert_allclose(expected_std, actual_std, rtol=1e-1, atol=1e-2)

        if is_waste_free:
            with self.assertRaises(ValueError):
                n_samples = N * num_iterations + 1
                init_particles = 0.25 + np.random.randn(n_samples)
                init_state, _ = new_smc_state(init_particles, lambda z: 1.0)

                updated_state, _ = kernel(
                    jax.random.PRNGKey(42),
                    init_state,
                    kernel_logprob_fn,
                    specialized_log_weights_fn,
                )

    @chex.all_variants(with_pmap=False)
    def test_normalize(self):
        """Test the function that returns normalized weighs and likelihood increments from raw log-weights."""
        logw = jax.random.normal(self.key, shape=[1234])
        w, loglikelihood_increment = self.variant(_normalize)(logw)

        np.testing.assert_allclose(np.sum(w), 1.0, rtol=1e-6)
        np.testing.assert_allclose(
            np.max(np.log(w) - logw), np.min(np.log(w) - logw), rtol=1e-6
        )
        np.testing.assert_allclose(
            loglikelihood_increment, np.log(np.mean(np.exp(logw))), rtol=1e-6
        )


if __name__ == "__main__":
    absltest.main()
