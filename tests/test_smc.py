"""Test the generic SMC sampler"""
import functools

import chex
import jax
import jax.numpy as jnp
import jax.scipy.stats as stats
import numpy as np
from absl.testing import absltest, parameterized

import blackjax
import blackjax.smc.base as base
import blackjax.smc.resampling as resampling


def kernel_logdensity_fn(position):
    return jnp.sum(stats.norm.logpdf(position))


def log_weights_fn(x, y):
    return jnp.sum(stats.norm.logpdf(y - x))


class SMCTest(chex.TestCase):
    def setUp(self):
        super().setUp()
        self.key = jax.random.PRNGKey(42)

    @chex.all_variants(with_pmap=False)
    @parameterized.parameters([500, 1000, 5000])
    def test_smc(self, N):

        mcmc_factory = lambda logdensity_fn: blackjax.hmc(
            logdensity_fn,
            step_size=1e-2,
            inverse_mass_matrix=jnp.eye(1),
            num_integration_steps=50,
        ).step

        specialized_log_weights_fn = lambda tree: log_weights_fn(tree, 1.0)

        kernel = base.kernel(
            mcmc_factory, blackjax.mcmc.hmc.init, resampling.systematic, 1000
        )

        # Don't use exactly the invariant distribution for the MCMC kernel
        init_particles = 0.25 + np.random.randn(N)

        updated_particles, _ = self.variant(
            functools.partial(
                kernel,
                logdensity_fn=kernel_logdensity_fn,
                log_weight_fn=specialized_log_weights_fn,
            )
        )(self.key, init_particles)

        expected_mean = 0.5
        expected_std = np.sqrt(0.5)

        np.testing.assert_allclose(
            expected_mean, updated_particles.mean(), rtol=1e-2, atol=1e-1
        )
        np.testing.assert_allclose(
            expected_std, updated_particles.std(), rtol=1e-2, atol=1e-1
        )

    @chex.all_variants(with_pmap=False)
    def test_normalize(self):
        logw = jax.random.normal(self.key, shape=[1234])
        w, loglikelihood_increment = self.variant(base._normalize)(logw)

        np.testing.assert_allclose(np.sum(w), 1.0, rtol=1e-6)
        np.testing.assert_allclose(
            np.max(np.log(w) - logw), np.min(np.log(w) - logw), rtol=1e-6
        )
        np.testing.assert_allclose(
            loglikelihood_increment, np.log(np.mean(np.exp(logw))), rtol=1e-6
        )


if __name__ == "__main__":
    absltest.main()
