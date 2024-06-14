import unittest

import jax
from jax import numpy as jnp
from jax.scipy.stats import multivariate_normal

import blackjax
from blackjax import adaptive_tempered_smc
from blackjax.mcmc.random_walk import normal
from blackjax.smc import extend_params


class SMCAndMCMCIntegrationTest(unittest.TestCase):
    """
    An integration test that verifies which MCMC can be used as
    SMC mutation step kernels.
    """

    def setUp(self):
        super().setUp()
        self.key = jax.random.key(42)
        self.n_particles = 3
        self.initial_particles = jax.random.multivariate_normal(
            self.key, jnp.zeros(2), jnp.eye(2), (self.n_particles,)
        )

    def check_compatible(self, mcmc_step_fn, mcmc_init_fn, mcmc_parameters):
        """
        Runs one SMC step
        """
        init, kernel = adaptive_tempered_smc(
            self.prior_log_prob,
            self.loglikelihood,
            mcmc_step_fn,
            mcmc_init_fn,
            mcmc_parameters=mcmc_parameters,
            resampling_fn=self.resampling_fn,
            target_ess=0.5,
            root_solver=self.root_solver,
            num_mcmc_steps=1,
        )
        kernel(self.key, init(self.initial_particles))

    def test_compatible_with_rwm(self):
        rwm = blackjax.additive_step_random_walk.build_kernel()

        def kernel(rng_key, state, logdensity_fn, proposal_mean):
            return rwm(rng_key, state, logdensity_fn, normal(proposal_mean))

        self.check_compatible(
            kernel,
            blackjax.additive_step_random_walk.init,
            extend_params({"proposal_mean": 1.0}),
        )

    def test_compatible_with_rmh(self):
        rmh = blackjax.rmh.build_kernel()

        def kernel(
            rng_key, state, logdensity_fn, proposal_mean, proposal_logdensity_fn=None
        ):
            return rmh(
                rng_key,
                state,
                logdensity_fn,
                lambda a, b: blackjax.mcmc.random_walk.normal(proposal_mean)(a, b),
                proposal_logdensity_fn,
            )

        self.check_compatible(
            kernel,
            blackjax.rmh.init,
            extend_params({"proposal_mean": 1.0}),
        )

    def test_compatible_with_hmc(self):
        self.check_compatible(
            blackjax.hmc.build_kernel(),
            blackjax.hmc.init,
            extend_params(
                {
                    "step_size": 0.3,
                    "inverse_mass_matrix": jnp.array([1.0]),
                    "num_integration_steps": 1,
                },
            ),
        )

    def test_compatible_with_irmh(self):
        def kernel(rng_key, state, logdensity_fn, mean, proposal_logdensity_fn=None):
            return blackjax.irmh.build_kernel()(
                rng_key,
                state,
                logdensity_fn,
                lambda key: mean + jax.random.normal(key),
                proposal_logdensity_fn,
            )

        self.check_compatible(
            kernel,
            blackjax.irmh.init,
            extend_params({"mean": jnp.array([1.0, 1.0])}),
        )

    def test_compatible_with_nuts(self):
        self.check_compatible(
            blackjax.nuts.build_kernel(),
            blackjax.nuts.init,
            extend_params(
                {"step_size": 1e-10, "inverse_mass_matrix": jnp.eye(2)},
            ),
        )

    def test_compatible_with_mala(self):
        self.check_compatible(
            blackjax.mala.build_kernel(),
            blackjax.mala.init,
            extend_params({"step_size": 1e-10}),
        )

    @staticmethod
    def prior_log_prob(x):
        d = x.shape[0]
        return multivariate_normal.logpdf(x, jnp.zeros((d,)), jnp.eye(d))

    @staticmethod
    def loglikelihood(x):
        return -5 * jnp.sum(jnp.square(x**2 - 1))

    @staticmethod
    def root_solver(fun, min_delta, max_delta, eps=1e-4, max_iter=100):
        return 0.8

    @staticmethod
    def resampling_fn(rng_key, weights: jax.Array, num_samples: int):
        return jnp.array([0, 1, 2])
