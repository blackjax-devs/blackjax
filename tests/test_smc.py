"""Test the generic SMC sampler"""
import functools
from collections import Callable
from unittest.mock import MagicMock, create_autospec

import chex
import jax
import jax.numpy as jnp
import jax.scipy.stats as stats
import numpy as np
from absl.testing import absltest, parameterized

import blackjax
import blackjax.smc.base as base
import blackjax.smc.resampling as resampling
from blackjax.base import SamplingAlgorithm
from blackjax.smc.parameter_tuning import (
    no_tuning,
    normal_proposal_from_particles,
    proposal_distribution_tuning,
)
from blackjax.types import PyTree
from tests.smc_test_utils import MultivariableParticlesDistribution


def kernel_logprob_fn(position):
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
        mcmc_factory = lambda logprob_fn, particles: blackjax.hmc(
            logprob_fn,
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
                logprob_fn=kernel_logprob_fn,
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


def step_fn(rng_key, state):
    raise ValueError("Not suposed to be called")


def mcmc_with_proposal_distribution(
    logprob_fn: Callable,
    proposal_distribution: Callable,
):
    def init_fn(position: PyTree):
        return None

    return SamplingAlgorithm(init_fn, step_fn)


class ParameterTunningStrategiesTest(chex.TestCase):
    def test_no_tunning(self):
        """
        When no tunning is used, only
        the relevant mcmc_parameters are passed in,
        exactly as they were before function call,
        so particles are not used.
        """
        to_return = MagicMock()
        step = MagicMock()
        to_return.step = step
        mock_mcmc_algorithm = create_autospec(
            mcmc_with_proposal_distribution, return_value=to_return
        )
        mock_proposal_distribution = MagicMock()
        factory = no_tuning(
            mock_mcmc_algorithm, {"proposal_distribution": mock_proposal_distribution}
        )
        tuned_mcmc = factory(kernel_logprob_fn, 0.25 + np.random.randn(50))
        assert tuned_mcmc == step
        mock_mcmc_algorithm.assert_called_once_with(
            kernel_logprob_fn, proposal_distribution=mock_proposal_distribution
        )

    def test_proposal_distribution_tunning(self):
        """
        The proposal distribution tunning strategy
        builds a factory that delegates particles and logprob_fn into
        the proposal_distribution factory.
        """
        particles = 0.25 + np.random.randn(50)
        logprob_fn = kernel_logprob_fn

        proposal_factory = MagicMock()
        proposal_distribution = MagicMock()
        proposal_factory.return_value = proposal_distribution

        mcmc_factory = MagicMock()
        sampling_algorithm = MagicMock()
        mcmc_factory.return_value = sampling_algorithm
        sampling_algorithm.step = step_fn

        tunned_mcmc_factory = proposal_distribution_tuning(
            mcmc_factory, {"proposal_distribution_factory": proposal_factory}
        )
        tuned_mcmc = tunned_mcmc_factory(logprob_fn, particles)

        proposal_factory.assert_called_once_with(logprob_fn, particles)
        mcmc_factory.assert_called_once_with(
            logprob_fn, proposal_distribution=proposal_distribution
        )
        assert tuned_mcmc == step_fn

    def test_proposal_distribution_tunning_no_param(self):
        with self.assertRaises(ValueError) as context:
            tuned_factory = proposal_distribution_tuning(MagicMock(), {})
            tuned_factory(kernel_logprob_fn, None)
        self.assertTrue(
            "you need to include a 'proposal_distribution_factory' parameter "
            in str(context.exception)
        )


class NormalProposalOnParticlesTest(chex.TestCase):
    def setUp(self):
        super().setUp()
        self.key = jax.random.PRNGKey(42)

    def test_normal_on_particles(self):
        particles = np.array(
            [
                jnp.array(10) + jax.random.normal(key) * jnp.array(0.5)
                for key in jax.random.split(self.key, 1000)
            ]
        )
        particles = np.expand_dims(particles, axis=1)
        proposal_distribution = normal_proposal_from_particles(
            kernel_logprob_fn, particles
        )
        samples = np.array(
            [proposal_distribution(key) for key in jax.random.split(self.key, 1000)]
        )
        np.testing.assert_allclose(np.mean(samples), 10.0, rtol=1e-1)
        np.testing.assert_allclose(np.std(samples), 0.5, rtol=1e-1)

    def test_normal_on_multivariate_particles(self):
        particles = np.array(
            [
                jnp.array([10.0, 15.0]) + jax.random.normal(key) * jnp.array([0.5, 0.7])
                for key in jax.random.split(self.key, 1000)
            ]
        )
        proposal_distribution = normal_proposal_from_particles(
            kernel_logprob_fn, particles
        )
        samples = np.array(
            [proposal_distribution(key) for key in jax.random.split(self.key, 2000)]
        )
        np.testing.assert_allclose(
            np.mean(samples, axis=0), np.array([10.0, 15.0]), rtol=1e-1
        )
        np.testing.assert_allclose(
            np.std(samples, axis=0), np.array([0.5, 0.7]), rtol=1e-1
        )

    def test_normal_on_multivariable_posterior_particles(self):
        particles_distribution = MultivariableParticlesDistribution(
            50000,
            mean_x=[10.0, 3.0],
            mean_y=[5.0, 20.0],
            cov_x=[[2.0, 0.0], [0.0, 5.0]],
        )

        proposal_distribution = normal_proposal_from_particles(
            kernel_logprob_fn, particles_distribution.get_particles()
        )

        samples = np.array(
            [proposal_distribution(key) for key in jax.random.split(self.key, 4000)]
        )

        np.testing.assert_allclose(
            np.mean([sample[0] for sample in samples.tolist()], axis=0),
            particles_distribution.mean_x,
            rtol=1e-1,
        )

        np.testing.assert_allclose(
            np.mean([sample[1] for sample in samples.tolist()], axis=0),
            particles_distribution.mean_y,
            rtol=1e-1,
        )

        np.testing.assert_allclose(
            np.std([sample[0] for sample in samples.tolist()], axis=0),
            np.sqrt(np.diag(particles_distribution.cov_x)),
            rtol=1e-1,
        )

        np.testing.assert_allclose(
            np.std([sample[1] for sample in samples.tolist()], axis=0),
            np.sqrt(np.diag(particles_distribution.cov_y)),
            rtol=1e-1,
        )


class IRMHProposalTunningTest(chex.TestCase):
    """
    An integration test to verify
    that the Independent RMH can be used
    with proposal distribution tunning.
    """

    def setUp(self):
        super().setUp()
        self.key = jax.random.PRNGKey(42)

    @chex.all_variants(with_pmap=False)
    def test_tune_distribution(self):
        """
        When tunning the proposal distribution using
        particles mean and std, then the end particles
        mean converges to the target mean, and the std
        converges to zero.
        """
        mcmc_factory = proposal_distribution_tuning(
            blackjax.irmh,
            mcmc_parameters={
                "proposal_distribution_factory": normal_proposal_from_particles
            },
        )

        specialized_log_weights_fn = lambda tree: log_weights_fn(tree, 1.0)

        kernel = base.kernel(
            mcmc_factory, blackjax.irmh.init, resampling.systematic, 50
        )

        # Don't use exactly the invariant distribution for the MCMC kernel
        init_particles = 0.25 + np.random.randn(1000) * 50

        def one_step(current_particles, key):
            updated_particles, _ = self.variant(
                functools.partial(
                    kernel,
                    logprob_fn=kernel_logprob_fn,
                    log_weight_fn=specialized_log_weights_fn,
                )
            )(key, current_particles)
            return updated_particles, updated_particles

        num_steps = 50
        keys = jax.random.split(self.key, num_steps)
        carry, states = jax.lax.scan(one_step, init_particles, keys)

        expected_mean = 0.5

        np.testing.assert_allclose(
            expected_mean, np.mean(states[-1]), rtol=1e-2, atol=1e-1
        )
        np.testing.assert_allclose(0, np.std(states[-1]), rtol=1e-2, atol=1e-1)


if __name__ == "__main__":
    absltest.main()
