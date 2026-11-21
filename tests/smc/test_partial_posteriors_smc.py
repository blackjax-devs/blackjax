import chex
import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest, parameterized

import blackjax
import blackjax.smc.resampling as resampling
from blackjax.smc import extend_params
from tests.smc import SMCLinearRegressionTestCase


class PartialPosteriorsSMCTest(SMCLinearRegressionTestCase):
    """Test posterior mean estimate."""

    @chex.variants(with_jit=True)
    def test_partial_posteriors(self):
        (
            init_particles,
            logprior_fn,
            observations,
        ) = self.partial_posterior_test_case()

        hmc_init = blackjax.hmc.init
        hmc_kernel = blackjax.hmc.build_kernel()

        hmc_parameters = extend_params(
            {
                "step_size": 10e-3,
                "inverse_mass_matrix": jnp.eye(2),
                "num_integration_steps": 10,
            },
        )

        dataset_size = 1000

        def partial_logposterior_factory(data_mask):
            def partial_logposterior(x):
                lp = logprior_fn(x)
                return lp + jnp.sum(
                    self.logdensity_by_observation(**x, **observations)
                    * data_mask.reshape(-1, 1)
                )

            return jax.jit(partial_logposterior)

        init, kernel = blackjax.partial_posteriors_smc(
            hmc_kernel,
            hmc_init,
            hmc_parameters,
            resampling.systematic,
            50,
            partial_logposterior_factory=partial_logposterior_factory,
        )

        init_state = init(init_particles, 1000)
        smc_kernel = self.variant(kernel)

        data_masks = jnp.array(
            [
                jnp.concat(
                    [
                        jnp.ones(datapoints_chosen),
                        jnp.zeros(dataset_size - datapoints_chosen),
                    ]
                )
                for datapoints_chosen in np.arange(200, 1001, 200)
            ]
        )

        def body_fn(carry, data_mask):
            i, state = carry
            subkey = jax.random.fold_in(self.next_key(), i)
            new_state, info = smc_kernel(subkey, state, data_mask)
            return (i + 1, new_state), (new_state, info)

        (steps, result), it = jax.lax.scan(body_fn, (0, init_state), data_masks)
        assert steps == 5

        self.assert_linear_regression_test_case(result)


class PartialPosteriorsMutationTest(chex.TestCase):
    """Check the mutation target without relying on sampling convergence."""

    @chex.variants(with_jit=True, without_jit=True)
    @parameterized.parameters(0, 3)
    def test_mutation_preserves_previous_posterior(self, batch_size):
        observations = jnp.array([2.0, -1.0])

        def factory(mask):
            def logposterior(x):
                return -0.5 * jnp.sum(x**2) - 0.5 * jnp.sum(
                    mask * (x[0] - observations) ** 2
                )

            return logposterior

        hmc_step = blackjax.hmc.build_kernel()
        parameters = {
            "step_size": 0.4,
            "inverse_mass_matrix": jnp.ones(1),
            "num_integration_steps": 3,
        }
        sampler = blackjax.partial_posteriors_smc(
            hmc_step,
            blackjax.hmc.init,
            extend_params(parameters),
            resampling.systematic,
            1,
            factory,
            batch_size=batch_size,
        )
        state = sampler.init(jnp.linspace(-1.5, 1.5, 8).reshape(-1, 1), 2)
        step = self.variant(sampler.step)
        # Include unchanged masks, then two successive data additions.
        for i, mask in enumerate(([0, 0], [1, 0], [1, 1], [1, 1])):
            mask = jnp.array(mask)
            key = jax.random.key(i)
            result, info = step(key, state, mask)
            updating_key = jax.random.split(key)[0]
            chain_keys = jax.random.split(updating_key, len(state.weights))
            previous_logposterior = factory(state.data_mask)

            def expected_mutation(chain_key, position):
                mcmc_state = blackjax.hmc.init(position, previous_logposterior)
                mcmc_key = jax.random.split(chain_key, 1)[0]
                return hmc_step(
                    mcmc_key, mcmc_state, previous_logposterior, **parameters
                )[0].position

            expected_particles = jax.vmap(expected_mutation)(
                chain_keys, state.particles[info.ancestors]
            )
            log_weights = np.asarray(
                jax.vmap(factory(mask))(expected_particles)
                - jax.vmap(previous_logposterior)(expected_particles)
            )
            unnormalized_weights = np.exp(log_weights - log_weights.max())
            expected_weights = unnormalized_weights / unnormalized_weights.sum()
            expected_log_increment = log_weights.max() + np.log(
                unnormalized_weights.mean()
            )
            np.testing.assert_allclose(
                result.particles, expected_particles, rtol=2e-6, atol=2e-6
            )
            np.testing.assert_allclose(
                result.weights, expected_weights, rtol=2e-6, atol=2e-6
            )
            np.testing.assert_allclose(
                info.log_likelihood_increment,
                expected_log_increment,
                rtol=2e-6,
                atol=2e-6,
            )
            np.testing.assert_array_equal(result.data_mask, mask)
            state = result


if __name__ == "__main__":
    absltest.main()
