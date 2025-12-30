import unittest

import chex
import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest

import blackjax
from blackjax.smc import extend_params, resampling
from blackjax.smc.pretuning import (
    build_pretune,
    esjd,
    init,
    update_parameter_distribution,
)
from tests.smc import SMCLinearRegressionTestCase


class TestMeasureOfChainMixing(unittest.TestCase):
    previous_position = np.array([jnp.array([10.0, 15.0]), jnp.array([3.0, 4.0])])

    next_position = np.array([jnp.array([20.0, 30.0]), jnp.array([9.0, 12.0])])

    def test_measure_of_chain_mixing_identity(self):
        """
        Given identity matrix and 1. acceptance probability
        then the mixing is the square of norm 2.
        """
        m = np.eye(2)

        acceptance_probabilities = np.array([1.0, 1.0])
        chain_mixing = esjd(m)(
            self.previous_position, self.next_position, acceptance_probabilities
        )
        np.testing.assert_allclose(chain_mixing[0], 325)
        np.testing.assert_allclose(chain_mixing[1], 100)

    def test_measure_of_chain_mixing_with_non_1_acceptance_rate(self):
        """
        Given identity matrix
        then the mixing is the square of norm 2. multiplied by the acceptance rate
        """
        m = np.eye(2)

        acceptance_probabilities = np.array([0.5, 0.2])
        chain_mixing = esjd(m)(
            self.previous_position, self.next_position, acceptance_probabilities
        )
        np.testing.assert_allclose(chain_mixing[0], 162.5)
        np.testing.assert_allclose(chain_mixing[1], 20)

    def test_measure_of_chain_mixing(self):
        m = np.array([[3, 0], [0, 5]])

        previous_position = np.array([jnp.array([10.0, 15.0]), jnp.array([3.0, 4.0])])

        next_position = np.array([jnp.array([20.0, 30.0]), jnp.array([9.0, 12.0])])

        acceptance_probabilities = np.array([1.0, 1.0])

        chain_mixing = esjd(m)(
            previous_position, next_position, acceptance_probabilities
        )

        assert chain_mixing.shape == (2,)
        np.testing.assert_allclose(chain_mixing[0], 10 * 10 * 3 + 15 * 15 * 5)
        np.testing.assert_allclose(chain_mixing[1], 6 * 6 * 3 + 8 * 8 * 5)


class TestUpdateParameterDistribution(chex.TestCase):
    def setUp(self):
        super().setUp()
        self.key = jax.random.key(42)
        self.previous_position = np.array(
            [jnp.array([10.0, 15.0]), jnp.array([10.0, 15.0]), jnp.array([3.0, 4.0])]
        )
        self.next_position = np.array(
            [jnp.array([20.0, 30.0]), jnp.array([10.0, 15.0]), jnp.array([9.0, 12.0])]
        )

    def test_update_param_distribution(self):
        """
        Given an extremely good mixing on one chain,
        and that the alpha parameter is 0, then the parameters
        of that chain with a slight mutation due to noise are reused.
        """

        (
            new_parameter_distribution,
            chain_mixing_measurement,
        ) = update_parameter_distribution(
            self.key,
            jnp.array([1.0, 2.0, 3.0]),
            self.previous_position,
            self.next_position,
            measure_of_chain_mixing=lambda x, y, z: jnp.array([1.0, 0.0, 0.0]),
            alpha=0,
            sigma_parameters=0.0001,
            acceptance_probability=None,
        )

        np.testing.assert_allclose(
            new_parameter_distribution,
            np.array([1, 1, 1], dtype="float32"),
            rtol=1e-3,
        )
        np.testing.assert_allclose(
            chain_mixing_measurement,
            np.array([1, 0, 0], dtype="float32"),
            rtol=1e-6,
        )

    def test_update_multi_sigmas(self):
        """
        When we have multiple parameters, the performance is attached to its combination
        so sampling must work accordingly.
        """
        (
            new_parameter_distribution,
            chain_mixing_measurement,
        ) = update_parameter_distribution(
            self.key,
            {
                "param_a": jnp.array([1.0, 2.0, 3.0]),
                "param_b": jnp.array([[5.0, 6.0], [6.0, 7.0], [4.0, 5.0]]),
            },
            self.previous_position,
            self.next_position,
            measure_of_chain_mixing=lambda x, y, z: jnp.array([1.0, 0.0, 0.0]),
            alpha=0,
            sigma_parameters={"param_a": 0.0001, "param_b": 0.00001},
            acceptance_probability=None,
        )
        print(chain_mixing_measurement)
        np.testing.assert_allclose(chain_mixing_measurement, np.array([1.0, 0, 0]))

        np.testing.assert_allclose(
            new_parameter_distribution["param_a"], jnp.array([1.0, 1.0, 1.0]), atol=0.1
        )
        np.testing.assert_allclose(
            new_parameter_distribution["param_b"],
            jnp.array([[5.0, 6.0], [5.0, 6.0], [5.0, 6.0]]),
            atol=0.1,
        )


def tuned_adaptive_tempered_inference_loop(kernel, rng_key, initial_state):
    def cond(carry):
        _, state, *_ = carry
        return state.sampler_state.tempering_param < 1

    def body(carry):
        i, state, curr_loglikelihood = carry
        subkey = jax.random.fold_in(rng_key, i)
        state, info = kernel(subkey, state)
        return i + 1, state, curr_loglikelihood + info.log_likelihood_increment

    total_iter, final_state, log_likelihood = jax.lax.while_loop(
        cond, body, (0, initial_state, 0.0)
    )
    return final_state


class PretuningSMCTest(SMCLinearRegressionTestCase):
    def setUp(self):
        super().setUp()
        self.key = jax.random.key(42)

    @chex.variants(with_jit=True)
    def test_tempered(self):
        step_provider = lambda logprior_fn, loglikelihood_fn, pretune: blackjax.smc.pretuning.build_kernel(
            blackjax.tempered_smc,
            logprior_fn,
            loglikelihood_fn,
            blackjax.hmc.build_kernel(),
            blackjax.hmc.init,
            resampling.systematic,
            num_mcmc_steps=10,
            pretune_fn=pretune,
        )

        def loop(smc_kernel, init_particles, initial_parameters):
            initial_state = init(
                blackjax.tempered_smc.init, init_particles, initial_parameters
            )

            def body_fn(carry, tempering_param):
                i, state = carry
                subkey = jax.random.fold_in(self.key, i)
                new_state, info = smc_kernel(
                    subkey, state, tempering_param=tempering_param
                )
                return (i + 1, new_state), (new_state, info)

            num_tempering_steps = 10
            lambda_schedule = np.logspace(-5, 0, num_tempering_steps)

            (_, result), _ = jax.lax.scan(body_fn, (0, initial_state), lambda_schedule)
            return result

        self.linear_regression_test_case(step_provider, loop)

    @chex.variants(with_jit=True)
    def test_adaptive_tempered(self):
        step_provider = lambda logprior_fn, loglikelihood_fn, pretune: blackjax.smc.pretuning.build_kernel(
            blackjax.adaptive_tempered_smc,
            logprior_fn,
            loglikelihood_fn,
            blackjax.hmc.build_kernel(),
            blackjax.hmc.init,
            resampling.systematic,
            num_mcmc_steps=10,
            pretune_fn=pretune,
            target_ess=0.5,
        )

        def loop(smc_kernel, init_particles, initial_parameters):
            initial_state = init(
                blackjax.tempered_smc.init, init_particles, initial_parameters
            )
            return tuned_adaptive_tempered_inference_loop(
                smc_kernel, self.key, initial_state
            )

        self.linear_regression_test_case(step_provider, loop)

    def linear_regression_test_case(self, step_provider, loop):
        (
            init_particles,
            logprior_fn,
            loglikelihood_fn,
        ) = self.particles_prior_loglikelihood()

        num_particles = 100
        sampling_key, step_size_key, integration_steps_key = jax.random.split(
            self.key, 3
        )
        integration_steps_distribution = jnp.round(
            jax.random.uniform(
                integration_steps_key, (num_particles,), minval=1, maxval=100
            )
        ).astype(int)

        step_sizes_distribution = jax.random.uniform(
            step_size_key, (num_particles,), minval=0, maxval=0.1
        )

        # Fixes inverse_mass_matrix and distribution for the other two parameters.
        initial_parameters = dict(
            inverse_mass_matrix=extend_params(jnp.eye(2)),
            step_size=step_sizes_distribution,
            num_integration_steps=integration_steps_distribution,
        )
        assert initial_parameters["step_size"].shape == (num_particles,)
        assert initial_parameters["num_integration_steps"].shape == (num_particles,)

        pretune = build_pretune(
            blackjax.hmc.init,
            blackjax.hmc.build_kernel(),
            alpha=1,
            n_particles=num_particles,
            sigma_parameters={"step_size": 0.01, "num_integration_steps": 2},
            natural_parameters=["num_integration_steps"],
            positive_parameters=["step_size"],
        )

        step = step_provider(logprior_fn, loglikelihood_fn, pretune)

        smc_kernel = self.variant(step)

        result = loop(smc_kernel, init_particles, initial_parameters)
        self.assert_linear_regression_test_case(result.sampler_state)
        assert set(result.parameter_override.keys()) == {
            "step_size",
            "num_integration_steps",
            "inverse_mass_matrix",
        }
        assert result.parameter_override["step_size"].shape == (num_particles,)
        assert result.parameter_override["num_integration_steps"].shape == (
            num_particles,
        )
        assert all(result.parameter_override["step_size"] > 0)
        assert all(result.parameter_override["num_integration_steps"] > 0)


if __name__ == "__main__":
    absltest.main()
