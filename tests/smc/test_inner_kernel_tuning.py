import functools
import unittest
from unittest.mock import MagicMock

import chex
import jax
import jax.numpy as jnp
import jax.scipy.stats as stats
import numpy as np
from absl.testing import absltest

import blackjax
import blackjax.smc.resampling as resampling
from blackjax import adaptive_tempered_smc, tempered_smc
from blackjax.smc.inner_kernel_tuning import inner_kernel_tuning
from blackjax.smc.tuning.from_kernel_info import update_scale_from_acceptance_rate
from blackjax.smc.tuning.from_particles import (
    mass_matrix_from_particles,
    particles_as_rows,
    particles_covariance_matrix,
    particles_means,
    particles_stds,
)
from tests.mcmc.test_sampling import irmh_proposal_distribution
from tests.smc import SMCLinearRegressionTestCase


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
        self.key = jax.random.key(42)

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
        self.key = jax.random.key(42)

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
        var1 = np.array([jnp.array([10.0, 15.0]), jnp.array([3.0, 4.0])])
        var2 = np.array([jnp.array([10.0]), jnp.array([3.0])])
        particles = {"var1": var1, "var2": var2}
        mean = particles_means(particles)
        std = particles_stds(particles)
        cov = particles_covariance_matrix(particles)
        np.testing.assert_allclose(mean, np.array([6.5, 9.5, 6.5]))
        np.testing.assert_allclose(std, np.array([3.5, 5.5, 3.5]))
        np.testing.assert_allclose(
            cov,
            np.array(
                [[12.25, 19.25, 12.25], [19.25, 30.25, 19.25], [12.25, 19.25, 12.25]]
            ),
        )


class InverseMassMatrixFromParticles(chex.TestCase):
    def setUp(self):
        super().setUp()
        self.key = jax.random.key(42)

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

    def test_inverse_mass_matrix_from_multivariable_particles(self):
        var1 = np.array([jnp.array([10.0, 15.0]), jnp.array([3.0, 4.0])])
        var2 = np.array([jnp.array([10.0]), jnp.array([3.0])])
        init_particles = {"var1": var1, "var2": var2}
        mass_matrix = mass_matrix_from_particles(init_particles)
        assert mass_matrix.shape == (3, 3)
        np.testing.assert_allclose(
            np.diag(mass_matrix),
            np.array([0.081633, 0.033058, 0.081633], dtype="float32"),
            rtol=1e-4,
        )

    def test_inverse_mass_matrix_from_multivariable_univariate_particles(self):
        var1 = np.array([3.0, 2.0])
        var2 = np.array([10.0, 3.0])
        init_particles = {"var1": var1, "var2": var2}
        mass_matrix = mass_matrix_from_particles(init_particles)
        assert mass_matrix.shape == (2, 2)
        np.testing.assert_allclose(
            np.diag(mass_matrix), np.array([4, 0.081633], dtype="float32"), rtol=1e-4
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


class InnerKernelTuningJitTest(SMCLinearRegressionTestCase):
    def setUp(self):
        super().setUp()
        self.key = jax.random.key(42)

    def mcmc_factory(self, mass_matrix):
        return functools.partial(
            blackjax.hmc.build_kernel(),
            inverse_mass_matrix=mass_matrix,
            step_size=10e-2,
            num_integration_steps=50,
        )

    @chex.all_variants(with_pmap=False)
    def test_with_adaptive_tempered(self):
        (
            init_particles,
            logprior_fn,
            loglikelihood_fn,
        ) = self.particles_prior_loglikelihood()

        init, step = blackjax.inner_kernel_tuning(
            adaptive_tempered_smc,
            logprior_fn,
            loglikelihood_fn,
            self.mcmc_factory,
            blackjax.hmc.init,
            {},
            resampling.systematic,
            mcmc_parameter_update_fn=lambda state, info: mass_matrix_from_particles(
                state.particles
            ),
            initial_parameter_value=jnp.eye(2),
            num_mcmc_steps=10,
            target_ess=0.5,
        )
        init_state = init(init_particles)
        smc_kernel = self.variant(step)

        def inference_loop(kernel, rng_key, initial_state):
            def cond(carry):
                state, key = carry
                return state.sampler_state.lmbda < 1

            def body(carry):
                state, op_key = carry
                op_key, subkey = jax.random.split(op_key, 2)
                state, _ = kernel(subkey, state)
                return state, op_key

            return jax.lax.while_loop(cond, body, (initial_state, rng_key))

        state, _ = inference_loop(smc_kernel, self.key, init_state)

        assert state.parameter_override.shape == (2, 2)
        self.assert_linear_regression_test_case(state.sampler_state)

    @chex.all_variants(with_pmap=False)
    def test_with_tempered_smc(self):
        num_tempering_steps = 10
        (
            init_particles,
            logprior_fn,
            loglikelihood_fn,
        ) = self.particles_prior_loglikelihood()

        init, step = blackjax.inner_kernel_tuning(
            tempered_smc,
            logprior_fn,
            loglikelihood_fn,
            self.mcmc_factory,
            blackjax.hmc.init,
            {},
            resampling.systematic,
            mcmc_parameter_update_fn=lambda state, info: mass_matrix_from_particles(
                state.particles
            ),
            initial_parameter_value=jnp.eye(2),
            num_mcmc_steps=10,
        )

        init_state = init(init_particles)
        smc_kernel = self.variant(step)

        lambda_schedule = np.logspace(-5, 0, num_tempering_steps)

        def body_fn(carry, lmbda):
            rng_key, state = carry
            rng_key, subkey = jax.random.split(rng_key)
            new_state, info = smc_kernel(subkey, state, lmbda=lmbda)
            return (rng_key, new_state), (new_state, info)

        (_, result), _ = jax.lax.scan(body_fn, (self.key, init_state), lambda_schedule)
        self.assert_linear_regression_test_case(result.sampler_state)


class ParticlesAsRowsTest(unittest.TestCase):
    def test_particles_as_rows(self):
        n_particles = 1000
        test_particles = {
            "a": np.zeros(n_particles),
            "b": np.ones([n_particles, 1]),
            "c": np.repeat(
                (np.arange(3 * 5) + 2).reshape(3, 5)[None, ...], n_particles, axis=0
            ),
        }
        flatten_particles = particles_as_rows(test_particles)
        assert flatten_particles.shape == (n_particles, 3 * 5 + 2)
        np.testing.assert_array_equal(np.arange(3 * 5 + 2), flatten_particles[0])


if __name__ == "__main__":
    absltest.main()
