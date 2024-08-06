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
from tests.smc import SMCLinearRegressionTestCase


def inference_loop(kernel, rng_key, initial_state):
    def cond(carry):
        _, state, *_ = carry
        return state.lmbda < 1

    def body(carry):
        i, state, curr_loglikelihood = carry
        subkey = jax.random.fold_in(rng_key, i)
        state, info = kernel(subkey, state)
        return i + 1, state, curr_loglikelihood + info.log_likelihood_increment

    total_iter, final_state, log_likelihood = jax.lax.while_loop(
        cond, body, (0, initial_state, 0.0)
    )

    return total_iter, final_state, log_likelihood


class TemperedSMCTest(SMCLinearRegressionTestCase):
    """Test posterior mean estimate."""

    def setUp(self):
        super().setUp()
        self.key = jax.random.key(42)

    @chex.variants(with_jit=True)
    def test_adaptive_tempered_smc(self):
        num_particles = 100

        x_data = np.random.normal(0, 1, size=(1000, 1))
        y_data = 3 * x_data + np.random.normal(size=x_data.shape)
        observations = {"x": x_data, "preds": y_data}

        def logprior_fn(x):
            return (
                stats.expon.logpdf(jnp.exp(x[0]), 0, 1) + x[0] + stats.norm.logpdf(x[1])
            )

        loglikelihood_fn = lambda x: self.logdensity_fn(*x, **observations)

        log_scale_init = np.log(np.random.exponential(1, num_particles))
        coeffs_init = 3 + 2 * np.random.randn(num_particles)
        smc_state_init = [log_scale_init, coeffs_init]

        iterates = []
        results = []

        hmc_kernel = blackjax.hmc.build_kernel()
        hmc_init = blackjax.hmc.init

        base_params = extend_params(
            {
                "step_size": 10e-2,
                "inverse_mass_matrix": jnp.eye(2),
                "num_integration_steps": 50,
            }
        )

        # verify results are equivalent with all shared, all unshared, and mixed params
        hmc_parameters_list = [
            base_params,
            jax.tree.map(lambda x: jnp.repeat(x, num_particles, axis=0), base_params),
            jax.tree_util.tree_map_with_path(
                lambda path, x: jnp.repeat(x, num_particles, axis=0)
                if path[0].key == "step_size"
                else x,
                base_params,
            ),
        ]

        for target_ess, hmc_parameters in zip([0.5, 0.5, 0.75], hmc_parameters_list):
            tempering = adaptive_tempered_smc(
                logprior_fn,
                loglikelihood_fn,
                hmc_kernel,
                hmc_init,
                hmc_parameters,
                resampling.systematic,
                target_ess,
                solver.dichotomy,
                5,
            )
            init_state = tempering.init(smc_state_init)

            n_iter, result, log_likelihood = self.variant(
                functools.partial(inference_loop, tempering.step)
            )(self.key, init_state)
            iterates.append(n_iter)
            results.append(result)

            np.testing.assert_allclose(
                np.mean(np.exp(result.particles[0])), 1.0, rtol=1e-1
            )
            np.testing.assert_allclose(np.mean(result.particles[1]), 3.0, rtol=1e-1)

        assert iterates[1] >= iterates[0]

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
            10,
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


def normal_logdensity_fn(x, chol_cov):
    """minus log-density of a centered multivariate normal distribution"""
    dim = chol_cov.shape[0]
    y = jax.scipy.linalg.solve_triangular(chol_cov, x, lower=True)
    normalizing_constant = (
        np.sum(np.log(np.abs(np.diag(chol_cov)))) + dim * np.log(2 * np.pi) / 2.0
    )
    norm_y = jnp.sum(y * y, -1)
    return -(0.5 * norm_y + normalizing_constant)


class NormalizingConstantTest(chex.TestCase):
    """Test normalizing constant estimate."""

    @chex.variants(with_jit=True)
    def test_normalizing_constant(self):
        num_particles = 200
        num_dim = 2

        rng_key = jax.random.key(2356)
        rng_key, cov_key = jax.random.split(rng_key, 2)
        chol_cov = jax.random.uniform(cov_key, shape=(num_dim, num_dim))
        iu = np.triu_indices(num_dim, 1)
        chol_cov = chol_cov.at[iu].set(0.0)
        cov = chol_cov @ chol_cov.T

        logprior_fn = lambda x: stats.multivariate_normal.logpdf(
            x, jnp.zeros((num_dim,)), jnp.eye(num_dim)
        )
        loglikelihood_fn = lambda x: normal_logdensity_fn(x, chol_cov)

        rng_key, init_key = jax.random.split(rng_key, 2)
        x_init = jax.random.normal(init_key, shape=(num_particles, num_dim))

        hmc_init = blackjax.hmc.init
        hmc_kernel = blackjax.hmc.build_kernel()
        hmc_parameters = extend_params(
            {
                "step_size": 10e-2,
                "inverse_mass_matrix": jnp.eye(num_dim),
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
            0.9,
            solver.dichotomy,
            10,
        )
        tempered_smc_state_init = tempering.init(x_init)
        n_iter, result, log_likelihood = self.variant(
            functools.partial(inference_loop, tempering.step)
        )(rng_key, tempered_smc_state_init)

        expected_log_likelihood = -0.5 * np.linalg.slogdet(np.eye(num_dim) + cov)[
            1
        ] - num_dim / 2 * np.log(2 * np.pi)

        np.testing.assert_allclose(log_likelihood, expected_log_likelihood, rtol=1e-1)


if __name__ == "__main__":
    absltest.main()
