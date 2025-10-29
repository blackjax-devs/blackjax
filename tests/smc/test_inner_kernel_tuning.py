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
from blackjax.mcmc.random_walk import build_irmh
from blackjax.smc import extend_params
from blackjax.smc.inner_kernel_tuning import as_top_level_api as inner_kernel_tuning
from blackjax.smc.pretuning import build_pretune
from blackjax.smc.tuning.from_kernel_info import update_scale_from_acceptance_rate
from blackjax.smc.tuning.from_particles import (
    inverse_mass_matrix_from_particles,
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
            blackjax.tempered_smc,
            smc_parameters={},
            step_parameters={"tempering_param": 0.75},
        )

    def smc_inner_kernel_tuning_test_case(
        self, smc_algorithm, smc_parameters, step_parameters
    ):
        specialized_log_weights_fn = lambda tree: log_weights_fn(tree, 1.0)
        # Don't use exactly the invariant distribution for the MCMC kernel
        init_particles = 0.25 + np.random.randn(1000) * 50

        proposal_factory = MagicMock()
        proposal_factory.return_value = 100

        def mcmc_parameter_update_fn(key, state, info):
            return extend_params({"mean": 100})

        prior = lambda x: stats.norm.logpdf(x)

        def wrapped_kernel(rng_key, state, logdensity, mean):
            return build_irmh()(
                rng_key,
                state,
                logdensity,
                functools.partial(irmh_proposal_distribution, mean=mean),
            )

        kernel = inner_kernel_tuning(
            logprior_fn=prior,
            loglikelihood_fn=specialized_log_weights_fn,
            mcmc_step_fn=wrapped_kernel,
            mcmc_init_fn=blackjax.irmh.init,
            resampling_fn=resampling.systematic,
            smc_algorithm=smc_algorithm,
            mcmc_parameter_update_fn=mcmc_parameter_update_fn,
            initial_parameter_value=extend_params({"mean": 1.0}),
            **smc_parameters,
        )

        new_state, new_info = kernel.step(
            self.key, state=kernel.init(init_particles), **step_parameters
        )
        assert set(new_state.parameter_override.keys()) == {
            "mean",
        }
        np.testing.assert_allclose(new_state.parameter_override["mean"], 100)


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
        inverse_mass_matrix = inverse_mass_matrix_from_particles(
            np.array([np.array(10.0), np.array(3.0)])
        )
        np.testing.assert_allclose(
            inverse_mass_matrix, np.diag(np.array([12.25])), rtol=1e-4
        )

    def test_inverse_mass_matrix_from_multivariate_particles(self):
        inverse_mass_matrix = inverse_mass_matrix_from_particles(
            np.array([jnp.array([10.0, 15.0]), jnp.array([3.0, 4.0])])
        )
        np.testing.assert_allclose(
            inverse_mass_matrix, np.diag(np.array([12.25, 30.25])), rtol=1e-4
        )

    def test_inverse_mass_matrix_from_multivariable_particles(self):
        var1 = np.array([jnp.array([10.0, 15.0]), jnp.array([3.0, 4.0])])
        var2 = np.array([jnp.array([10.0]), jnp.array([3.0])])
        init_particles = {"var1": var1, "var2": var2}
        mass_matrix = inverse_mass_matrix_from_particles(init_particles)
        assert mass_matrix.shape == (3, 3)
        np.testing.assert_allclose(
            np.diag(mass_matrix),
            np.array([12.25, 30.25, 12.25], dtype="float32"),
            rtol=1e-4,
        )

    def test_inverse_mass_matrix_from_multivariable_univariate_particles(self):
        var1 = np.array([3.0, 2.0])
        var2 = np.array([10.0, 3.0])
        init_particles = {"var1": var1, "var2": var2}
        mass_matrix = inverse_mass_matrix_from_particles(init_particles)
        assert mass_matrix.shape == (2, 2)
        np.testing.assert_allclose(
            np.diag(mass_matrix), np.array([0.25, 12.25], dtype="float32"), rtol=1e-4
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

    @chex.all_variants(with_pmap=False)
    def test_with_adaptive_tempered(self):
        (
            init_particles,
            logprior_fn,
            loglikelihood_fn,
        ) = self.particles_prior_loglikelihood()

        def parameter_update(key, state, info):
            return extend_params(
                {
                    "inverse_mass_matrix": inverse_mass_matrix_from_particles(
                        state.particles
                    ),
                    "step_size": 10e-2,
                    "num_integration_steps": 50,
                },
            )

        init, step = blackjax.inner_kernel_tuning(
            adaptive_tempered_smc,
            logprior_fn,
            loglikelihood_fn,
            blackjax.hmc.build_kernel(),
            blackjax.hmc.init,
            resampling.systematic,
            mcmc_parameter_update_fn=parameter_update,
            initial_parameter_value=extend_params(
                dict(
                    inverse_mass_matrix=jnp.eye(2),
                    step_size=10e-2,
                    num_integration_steps=50,
                ),
            ),
            num_mcmc_steps=10,
            target_ess=0.5,
        )
        init_state = init(init_particles)
        smc_kernel = self.variant(step)
        _, state = adaptive_tempered_loop(smc_kernel, self.key, init_state)

        assert state.parameter_override["inverse_mass_matrix"].shape == (1, 2, 2)
        self.assert_linear_regression_test_case(state.sampler_state)

    @chex.all_variants(with_pmap=False)
    def test_with_tempered_smc(self):
        num_tempering_steps = 10
        (
            init_particles,
            logprior_fn,
            loglikelihood_fn,
        ) = self.particles_prior_loglikelihood()

        def parameter_update(key, state, info):
            return extend_params(
                {
                    "inverse_mass_matrix": inverse_mass_matrix_from_particles(
                        state.particles
                    ),
                    "step_size": 10e-2,
                    "num_integration_steps": 50,
                },
            )

        init, step = blackjax.inner_kernel_tuning(
            tempered_smc,
            logprior_fn,
            loglikelihood_fn,
            blackjax.hmc.build_kernel(),
            blackjax.hmc.init,
            resampling.systematic,
            mcmc_parameter_update_fn=parameter_update,
            initial_parameter_value=extend_params(
                dict(
                    inverse_mass_matrix=jnp.eye(2),
                    step_size=10e-2,
                    num_integration_steps=50,
                ),
            ),
            num_mcmc_steps=10,
        )

        init_state = init(init_particles)
        smc_kernel = self.variant(step)

        lambda_schedule = np.logspace(-5, 0, num_tempering_steps)

        def body_fn(carry, tempering_param):
            i, state = carry
            subkey = jax.random.fold_in(self.key, i)
            new_state, info = smc_kernel(subkey, state, tempering_param=tempering_param)
            return (i + 1, new_state), (new_state, info)

        (_, result), _ = jax.lax.scan(body_fn, (0, init_state), lambda_schedule)
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


def adaptive_tempered_loop(kernel, rng_key, initial_state):
    def cond(carry):
        _, state = carry
        return state.sampler_state.tempering_param < 1

    def body(carry):
        i, state = carry
        subkey = jax.random.fold_in(rng_key, i)
        state, _ = kernel(subkey, state)
        return i + 1, state

    return jax.lax.while_loop(cond, body, (0, initial_state))


class MultipleTuningTest(SMCLinearRegressionTestCase):
    def setUp(self):
        super().setUp()
        self.key = jax.random.key(42)

    @chex.all_variants(with_pmap=False)
    def test_tuning_pretuning(self):
        """
        Tests that we can apply tuning on some parameters
        and pretuning in some others at the same time.
        """

        (
            init_particles,
            logprior_fn,
            loglikelihood_fn,
        ) = self.particles_prior_loglikelihood()

        n_particles = 100
        dimentions = 2

        step_size_key, integration_steps_key = jax.random.split(self.key, 2)

        # Set initial samples for integration steps and step sizes.
        integration_steps_distribution = jnp.round(
            jax.random.uniform(
                integration_steps_key, (n_particles,), minval=1, maxval=50
            )
        ).astype(int)

        step_sizes_distribution = jax.random.uniform(
            step_size_key, (n_particles,), minval=1e-1 / 2, maxval=1e-1 * 2
        )

        # Fixes inverse_mass_matrix and distribution for the other two parameters.
        initial_parameters = dict(
            inverse_mass_matrix=extend_params(jnp.eye(dimentions)),
            step_size=step_sizes_distribution,
            num_integration_steps=integration_steps_distribution,
        )

        pretune = build_pretune(
            blackjax.hmc.init,
            blackjax.hmc.build_kernel(),
            alpha=2,
            n_particles=n_particles,
            sigma_parameters={
                "step_size": jnp.array(0.1),
                "num_integration_steps": jnp.array(2.0),
            },
            natural_parameters=["num_integration_steps"],
            positive_parameters=["step_size"],
        )

        def pretuning_factory(
            logprior_fn,
            loglikelihood_fn,
            mcmc_step_fn,
            mcmc_init_fn,
            mcmc_parameters,
            resampling_fn,
            num_mcmc_steps,
            initial_parameter_value,
            target_ess,
        ):
            # we need to wrap the pretuning into a factory, which is what
            # the inner_kernel_tuning expects
            return blackjax.pretuning(
                blackjax.adaptive_tempered_smc,
                logprior_fn,
                loglikelihood_fn,
                mcmc_step_fn,
                mcmc_init_fn,
                resampling_fn,
                num_mcmc_steps,
                initial_parameter_value,
                pretune,
                target_ess=target_ess,
            )

        def mcmc_parameter_update_fn(key, state, info):
            imm = inverse_mass_matrix_from_particles(state.sampler_state.particles)
            return {"inverse_mass_matrix": extend_params(imm)}

        step = blackjax.smc.inner_kernel_tuning.build_kernel(
            pretuning_factory,
            logprior_fn,
            loglikelihood_fn,
            blackjax.hmc.build_kernel(),
            blackjax.hmc.init,
            resampling.systematic,
            mcmc_parameter_update_fn=mcmc_parameter_update_fn,
            initial_parameter_value=initial_parameters,
            num_mcmc_steps=10,
            target_ess=0.5,
            smc_returns_state_with_parameter_override=True,
        )

        def init(position):
            return blackjax.smc.inner_kernel_tuning.init(
                blackjax.adaptive_tempered_smc.init, position, initial_parameters
            )

        init_state = init(init_particles)
        smc_kernel = self.variant(step)
        _, state = adaptive_tempered_loop(smc_kernel, self.key, init_state)
        self.assert_linear_regression_test_case(state.sampler_state)


if __name__ == "__main__":
    absltest.main()
