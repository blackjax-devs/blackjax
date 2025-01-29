from blackjax.smc.builder_api import SMCSamplerBuilder
from tests.smc import SMCLinearRegressionTestCase
import jax

class PretuningWithTuningSMCTest(SMCLinearRegressionTestCase):
    def setUp(self):
        super().setUp()
        self.key = jax.random.key(42)

    def loop(self, init, smc_kernel, init_particles):
        initial_state = init(init_particles)

        def body_fn(carry, lmbda):
            i, state = carry
            subkey = jax.random.fold_in(self.key, i)
            new_state, info = smc_kernel(subkey, state, lmbda=lmbda)
            return (i + 1, new_state), (new_state, info)

        num_tempering_steps = 10
        lambda_schedule = np.logspace(-5, 0, num_tempering_steps)

        (_, result), _ = jax.lax.scan(body_fn, (0, initial_state), lambda_schedule)
        return result
    @chex.variants(with_jit=True)
    def test_tempered_top_level_api(self):
        def step_provider(
            logprior_fn, loglikelihood_fn, pretune, initial_parameter_value
        ):
            return blackjax.pretuning(
                blackjax.tempered_smc,
                logprior_fn,
                loglikelihood_fn,
                blackjax.hmc.build_kernel(),
                blackjax.hmc.init,
                resampling.systematic,
                num_mcmc_steps=10,
                initial_parameter_value=initial_parameter_value,
                pretune_fn=pretune,
            )

        self.linear_regression_test_case(step_provider, self.loop)

    @chex.variants(with_jit=True)
    def test_adaptive_tempered_builder_api(self):
        step_provider = (
            lambda logprior_fn, loglikelihood_fn, pretune, initial_parameters: (
                SMCSamplerBuilder()
                .adaptive_tempering_sequence(0.5, logprior_fn, loglikelihood_fn)
                .inner_kernel(
                    blackjax.hmc.init, blackjax.hmc.build_kernel(), initial_parameters
                )
                .with_pretuning(pretune)
                .mutate_and_take_last(10)
                .build()
            )
        )

        def loop(init, smc_kernel, init_particles):
            initial_state = init(init_particles)
            return tuned_adaptive_tempered_inference_loop(
                smc_kernel, self.key, initial_state
            )

        self.linear_regression_test_case(step_provider, loop)

    @chex.variants(with_jit=True)
    def test_adaptive_tempered_top_level_api(self):
        step_provider = lambda logprior_fn, loglikelihood_fn, pretune, initial_parameters: blackjax.pretuning(
            blackjax.adaptive_tempered_smc,
            logprior_fn,
            loglikelihood_fn,
            blackjax.hmc.build_kernel(),
            blackjax.hmc.init,
            resampling.systematic,
            num_mcmc_steps=10,
            pretune_fn=pretune,
            initial_parameter_value=initial_parameters,
            target_ess=0.5,
        )

        def loop(init, smc_kernel, init_particles):
            initial_state = init(init_particles)
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

        init, step = step_provider(
            logprior_fn, loglikelihood_fn, pretune, initial_parameters
        )

        smc_kernel = self.variant(step)

        result = loop(init, smc_kernel, init_particles)
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
