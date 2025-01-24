import chex
import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest

import blackjax
import blackjax.smc.resampling as resampling
from blackjax.smc import extend_params
from blackjax.smc.builder_api import SMCSamplerBuilder
from tests.smc import SMCLinearRegressionTestCase


class PartialPosteriorsSMCTest(SMCLinearRegressionTestCase):
    """Test posterior mean estimate."""

    def setUp(self):
        super().setUp()
        self.key = jax.random.key(42)

    @chex.variants(with_jit=True)
    def test_partial_posteriors_top_level_api(self):
        def sampler_provider(
            kernel, init, parameters, steps, partial_logposterior_factory
        ):
            return blackjax.partial_posteriors_smc(
                kernel,
                init,
                parameters,
                resampling.systematic,
                steps,
                partial_logposterior_factory,
            )

        self.partial_posteriors_test_case(sampler_provider)

    @chex.variants(with_jit=True)
    def test_partial_posteriors_builder_api(self):
        def sampler_provider(
            kernel, init, parameters, steps, partial_logposterior_factory
        ):
            return (
                SMCSamplerBuilder()
                .partial_posteriors(partial_logposterior_factory)
                .inner_kernel(init, kernel, parameters)
                .mutate_and_take_last(steps)
                .build()
            )

        self.partial_posteriors_test_case(sampler_provider)

    def partial_posteriors_test_case(self, sampler_provider):
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
                "num_integration_steps": 50,
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

        init, kernel = sampler_provider(
            hmc_kernel,
            hmc_init,
            hmc_parameters,
            50,
            partial_logposterior_factory,
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
                for datapoints_chosen in np.arange(100, 1001, 50)
            ]
        )

        def body_fn(carry, data_mask):
            i, state = carry
            subkey = jax.random.fold_in(self.key, i)
            new_state, info = smc_kernel(subkey, state, data_mask)
            return (i + 1, new_state), (new_state, info)

        (steps, result), it = jax.lax.scan(body_fn, (0, init_state), data_masks)
        assert steps == 19

        self.assert_linear_regression_test_case(result)


if __name__ == "__main__":
    absltest.main()
