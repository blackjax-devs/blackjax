"""Test the tempered SMC steps and routine"""

import functools

import chex
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from absl.testing import absltest

import blackjax
import blackjax.smc.resampling as resampling
from blackjax import adaptive_tempered_smc, tempered_smc
from blackjax.smc import extend_params
from blackjax.smc.waste_free import update_waste_free, waste_free_smc
from tests.smc import SMCLinearRegressionTestCase
from tests.smc.test_tempered_smc import inference_loop


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
            waste_free_smc(100, 4),
        )
        init_state = tempering.init(init_particles)
        smc_kernel = self.variant(tempering.step)

        def body_fn(carry, tempering_param):
            i, state = carry
            subkey = jax.random.fold_in(self.key, i)
            new_state, info = smc_kernel(subkey, state, tempering_param)
            return (i + 1, new_state), (new_state, info)

        (_, result), _ = jax.lax.scan(body_fn, (0, init_state), lambda_schedule)
        self.assert_linear_regression_test_case(result)

    @chex.variants(with_jit=True)
    def test_adaptive_tempered_smc(self):
        (
            init_particles,
            logprior_fn,
            loglikelihood_fn,
        ) = self.particles_prior_loglikelihood()

        hmc_init = blackjax.hmc.init
        hmc_kernel = blackjax.hmc.build_kernel()
        hmc_parameters = extend_params(
            {
                "step_size": 10e-2,
                "inverse_mass_matrix": jnp.eye(2),
                "num_integration_steps": 50,
            },
        )

        tempering = adaptive_tempered_smc(
            logprior_fn,
            loglikelihood_fn,
            hmc_kernel,
            hmc_init,
            hmc_parameters,
            resampling.systematic,
            0.5,
            update_strategy=waste_free_smc(100, 4),
            num_mcmc_steps=None,
        )
        init_state = tempering.init(init_particles)

        n_iter, result, log_likelihood = self.variant(
            functools.partial(inference_loop, tempering.step)
        )(self.key, init_state)

        self.assert_linear_regression_test_case(result)


class Update_waste_free_multivariate_particles(chex.TestCase):
    @chex.variants(with_jit=True)
    def test_update_waste_free_multivariate_particles(self):
        """
        Given resampled multivariate particles,
        when updating with waste free, they are joined
        by the result of iterating the MCMC chain to
        get a bigger set of particles.
        """
        resampled_particles = np.ones((50, 3))
        n_particles = 100

        def normal_logdensity(x):
            return jnp.log(
                jax.scipy.stats.multivariate_normal.pdf(
                    x, mean=np.zeros(3), cov=np.diag(np.ones(3))
                )
            )

        def rmh_proposal_distribution(rng_key, position):
            return position + jax.random.normal(rng_key, (3,)) * 25.0

        kernel = functools.partial(
            blackjax.rmh.build_kernel(), transition_generator=rmh_proposal_distribution
        )
        init = blackjax.rmh.init
        update, _ = waste_free_smc(n_particles, 2)(
            init, normal_logdensity, kernel, n_particles
        )

        updated_particles, infos = self.variant(update)(
            jax.random.split(jax.random.PRNGKey(10), 50), resampled_particles, {}
        )

        assert updated_particles.shape == (n_particles, 3)


def test_waste_free_set_num_mcmc_steps():
    with pytest.raises(ValueError) as exc_info:
        update_waste_free(
            lambda x: x, lambda x: 1, lambda x: 1, 100, 10, 3, num_mcmc_steps=50
        )
        assert str(exc_info.value).startswith(
            "Can't use waste free SMC with a num_mcmc_steps parameter"
        )


def test_waste_free_p_non_divier():
    with pytest.raises(ValueError) as exc_info:
        waste_free_smc(100, 3)
        assert str(exc_info.value).startswith("p must be a divider")


if __name__ == "__main__":
    absltest.main()
