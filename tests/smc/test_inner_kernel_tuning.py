from unittest.mock import MagicMock

import chex
import jax
import jax.numpy as jnp
import jax.scipy.stats as stats
import numpy as np
from absl.testing import absltest

import blackjax
import blackjax.smc.resampling as resampling
from blackjax.smc.inner_kernel_tuning import inner_kernel_tuning
from blackjax.smc.tuning.from_kernel_info import update_scale_from_acceptance_rate
from blackjax.smc.tuning.from_particles import (
    mass_matrix_from_particles,
    particles_covariance_matrix,
    particles_means,
    particles_stds,
)
from tests.mcmc.test_sampling import irmh_proposal_distribution


class MultivariableParticlesDistribution:
    """
    Builds particles for tests belonging to a posterior with more than one variable.
    sample from P(x,y) x ~ N(mean, cov) y ~ N(mean, cov)
    """

    def __init__(self, n_particles, mean_x=None, mean_y=None, cov_x=None, cov_y=None):
        self.n_particles = n_particles
        self.mean_x = mean_x if mean_x is not None else [10.0, 5.0]
        self.mean_y = mean_y if mean_y is not None else [0.0, 0.0]
        self.cov_x = cov_x if cov_x is not None else [[1.0, 0.0], [0.0, 1.0]]
        self.cov_y = cov_y if cov_y is not None else [[1.0, 0.0], [0.0, 1.0]]

    def get_particles(self):
        return [
            np.random.multivariate_normal(
                mean=self.mean_x, cov=self.cov_x, size=self.n_particles
            ),
            np.random.multivariate_normal(
                mean=self.mean_y, cov=self.cov_y, size=self.n_particles
            ),
        ]


def kernel_logprob_fn(position):
    return jnp.sum(stats.norm.logpdf(position))


def log_weights_fn(x, y):
    return jnp.sum(stats.norm.logpdf(y - x))


class SMCParameterTuningTest(chex.TestCase):
    def setUp(self):
        super().setUp()
        self.key = jax.random.PRNGKey(42)

    def logdensity_fn(self, log_scale, coefs, preds, x):
        """Linear regression"""
        scale = jnp.exp(log_scale)
        y = jnp.dot(x, coefs)
        logpdf = stats.norm.logpdf(preds, y, scale)
        return jnp.sum(logpdf)

    def test_smc_inner_kernel_adaptive_tempered(self):
        self.smc_inner_kernel_tuning_test_case(
            blackjax.adaptive_tempered_smc,
            smc_parameters={"target_ess": 0.5},
            step_parameters={},
        )

    def test_smc_inner_kernel_tempered(self):
        self.smc_inner_kernel_tuning_test_case(
            blackjax.tempered_smc, smc_parameters={}, step_parameters={"lmbda": 0.75}
        )

    def smc_inner_kernel_tuning_test_case(
        self, smc_algorithm, smc_parameters, step_parameters
    ):
        specialized_log_weights_fn = lambda tree: log_weights_fn(tree, 1.0)
        # Don't use exactly the invariant distribution for the MCMC kernel
        init_particles = 0.25 + np.random.randn(1000) * 50

        proposal_factory = MagicMock()
        proposal_factory.return_value = 100

        def mcmc_parameter_update_fn(state, info):
            return 100

        mcmc_factory = MagicMock()
        sampling_algorithm = MagicMock()
        mcmc_factory.return_value = sampling_algorithm
        prior = lambda x: stats.norm.logpdf(x)

        def kernel_factory(proposal_distribution):
            kernel = blackjax.irmh.build_kernel()

            def wrapped_kernel(rng_key, state, logdensity):
                return kernel(rng_key, state, logdensity, proposal_distribution)

            return wrapped_kernel

        kernel = inner_kernel_tuning(
            logprior_fn=prior,
            loglikelihood_fn=specialized_log_weights_fn,
            mcmc_factory=kernel_factory,
            mcmc_init_fn=blackjax.irmh.init,
            resampling_fn=resampling.systematic,
            smc_algorithm=smc_algorithm,
            mcmc_parameters={},
            mcmc_parameter_update_fn=mcmc_parameter_update_fn,
            initial_parameter_value=irmh_proposal_distribution,
            **smc_parameters,
        )

        new_state, new_info = kernel.step(
            self.key, state=kernel.init(init_particles), **step_parameters
        )
        assert new_state.parameter_override == 100


class MeanAndStdFromParticlesTest(chex.TestCase):
    def setUp(self):
        super().setUp()
        self.key = jax.random.PRNGKey(42)

    def test_mean_and_std(self):
        particles = np.array(
            [
                jnp.array([10]) + jax.random.normal(key) * jnp.array([0.5])
                for key in jax.random.split(self.key, 1000)
            ]
        )
        mean = particles_means(particles)
        std = particles_stds(particles)
        cov = particles_covariance_matrix(particles)
        np.testing.assert_allclose(mean, 10.0, rtol=1e-1)
        np.testing.assert_allclose(std, 0.5, rtol=1e-1)
        np.testing.assert_allclose(cov, 0.24, rtol=1e-1)

    def test_mean_and_std_multivariate_particles(self):
        particles = np.array(
            [
                jnp.array([10.0, 15.0]) + jax.random.normal(key) * jnp.array([0.5, 0.7])
                for key in jax.random.split(self.key, 1000)
            ]
        )

        mean = particles_means(particles)
        std = particles_stds(particles)
        cov = particles_covariance_matrix(particles)
        np.testing.assert_allclose(mean, np.array([10.0, 15.0]), rtol=1e-1)
        np.testing.assert_allclose(std, np.array([0.5, 0.7]), rtol=1e-1)
        np.testing.assert_allclose(
            cov, np.array([[0.249529, 0.34934], [0.34934, 0.489076]]), atol=1e-1
        )

    def test_mean_and_std_multivariable_particles(self):
        particles_distribution = MultivariableParticlesDistribution(
            50000,
            mean_x=[10.0, 3.0],
            mean_y=[5.0, 20.0],
            cov_x=[[2.0, 0.0], [0.0, 5.0]],
        )
        particles = particles_distribution.get_particles()
        mean = particles_means(particles)
        std = particles_stds(particles)
        cov = particles_covariance_matrix(particles)
        np.testing.assert_allclose(
            mean[0],
            particles_distribution.mean_x,
            rtol=1e-1,
        )

        np.testing.assert_allclose(
            mean[1],
            particles_distribution.mean_y,
            rtol=1e-1,
        )

        np.testing.assert_allclose(
            std[0],
            np.sqrt(np.diag(particles_distribution.cov_x)),
            rtol=1e-1,
        )

        np.testing.assert_allclose(
            std[1],
            np.sqrt(np.diag(particles_distribution.cov_y)),
            rtol=1e-1,
        )
        np.testing.assert_allclose(cov[0], particles_distribution.cov_x, atol=1e-1)

        np.testing.assert_allclose(cov[1], particles_distribution.cov_y, atol=1e-1)


class InverseMassMatrixFromParticles(chex.TestCase):
    def setUp(self):
        super().setUp()
        self.key = jax.random.PRNGKey(42)

    def test_inverse_mass_matrix_from_particles(self):
        inverse_mass_matrix = mass_matrix_from_particles(
            np.array([np.array(10.0), np.array(3.0)])
        )
        np.testing.assert_allclose(
            inverse_mass_matrix, np.diag(np.array([0.08163])), rtol=1e-4
        )

    def test_inverse_mass_matrix_from_multivariate_particles(self):
        inverse_mass_matrix = mass_matrix_from_particles(
            np.array([jnp.array([10.0, 15.0]), jnp.array([3.0, 4.0])])
        )
        np.testing.assert_allclose(
            inverse_mass_matrix, np.diag(np.array([0.081633, 0.033058])), rtol=1e-4
        )


class ScaleCovarianceFromAcceptanceRates(chex.TestCase):
    def test_scale_when_aceptance_below_optimal(self):
        """
        Given that the acceptance rate is below optimal,
        the scale gets reduced.
        """
        np.testing.assert_allclose(
            update_scale_from_acceptance_rate(
                scales=jnp.array([0.5]), acceptance_rates=jnp.array([0.2])
            ),
            jnp.array([0.483286]),
            rtol=1e-4,
        )

    def test_scale_when_aceptance_above_optimal(self):
        """
        Given that the acceptance rate is above optimal
        the scale increases
        -------
        """
        np.testing.assert_allclose(
            update_scale_from_acceptance_rate(
                scales=jnp.array([0.5]), acceptance_rates=jnp.array([0.3])
            ),
            jnp.array([0.534113]),
            rtol=1e-4,
        )

    def test_scale_mean_smoothes(self):
        """
        The end result depends on the mean acceptance rate,
        smoothing the results
        """
        np.testing.assert_allclose(
            update_scale_from_acceptance_rate(
                scales=jnp.array([0.5, 0.5]), acceptance_rates=jnp.array([0.3, 0.2])
            ),
            jnp.array([0.521406, 0.495993]),
            rtol=1e-4,
        )


if __name__ == "__main__":
    absltest.main()
