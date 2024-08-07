"""Test the tempered SMC steps and routine"""
import functools

import chex
import jax
import jax.numpy as jnp
import jax.scipy.stats as stats
import numpy as np
from absl.testing import absltest

import blackjax
import blackjax.smc.resampling as resampling
import blackjax.smc.solver as solver
from blackjax import adaptive_tempered_smc, tempered_smc
from blackjax.smc import extend_params
from blackjax.smc.waste_free import update_waste_free, waste_free_smc
from tests.smc import SMCLinearRegressionTestCase


class WasteFreeSMCTest(SMCLinearRegressionTestCase):
    """Test posterior mean estimate."""

    def setUp(self):
        super().setUp()
        self.key = jax.random.key(42)
    @chex.variants(with_jit=True)
    def test_fixed_schedule_tempered_smc(self):
        (
            init_particles,
            logprior_fn,
            loglikelihood_fn,
        ) = self.particles_prior_loglikelihood()

        num_tempering_steps = 10

        lambda_schedule = np.logspace(-5, 0, num_tempering_steps)
        hmc_init = blackjax.hmc.init
        hmc_kernel = blackjax.hmc.build_kernel()
        hmc_parameters = extend_params(
            {
                "step_size": 10e-2,
                "inverse_mass_matrix": jnp.eye(2),
                "num_integration_steps": 50,
            },
        )

        tempering = tempered_smc(
            logprior_fn,
            loglikelihood_fn,
            hmc_kernel,
            hmc_init,
            hmc_parameters,
            resampling.systematic,
            None,
            waste_free_smc(100,4)
        )
        init_state = tempering.init(init_particles)
        smc_kernel = self.variant(tempering.step)

        def body_fn(carry, lmbda):
            i, state = carry
            subkey = jax.random.fold_in(self.key, i)
            new_state, info = smc_kernel(subkey, state, lmbda)
            return (i + 1, new_state), (new_state, info)

        (_, result), _ = jax.lax.scan(body_fn, (0, init_state), lambda_schedule)
        self.assert_linear_regression_test_case(result)





if __name__ == "__main__":
    absltest.main()
