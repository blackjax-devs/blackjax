import chex
import jax
import jax.numpy as jnp
from absl.testing import absltest, parameterized

import blackjax
from blackjax.util import run_inference_algorithm, store_only_expectation_values


class RunInferenceAlgorithmTest(chex.TestCase):
    def setUp(self):
        super().setUp()
        self.key = jax.random.key(42)
        self.algorithm = blackjax.hmc(
            logdensity_fn=self.logdensity_fn,
            inverse_mass_matrix=jnp.eye(2),
            step_size=1.0,
            num_integration_steps=1000,
        )
        self.num_steps = 10

    def check_compatible(self, initial_state, progress_bar):
        """
        Runs 10 steps with `run_inference_algorithm` starting with
        `initial_state` and potentially a progress bar.
        """
        _ = run_inference_algorithm(
            rng_key=self.key,
            initial_state=initial_state,
            inference_algorithm=self.algorithm,
            num_steps=self.num_steps,
            progress_bar=progress_bar,
            transform=lambda state, info: state.position,
        )

    def test_streaming(self):
        def logdensity_fn(x):
            return -0.5 * jnp.sum(jnp.square(x))

        initial_position = jnp.ones(
            10,
        )

        init_key, state_key, run_key = jax.random.split(self.key, 3)
        initial_state = blackjax.mcmc.mclmc.init(
            position=initial_position, logdensity_fn=logdensity_fn, rng_key=state_key
        )
        L = 1.0
        step_size = 0.1
        num_steps = 4

        sampling_alg = blackjax.mclmc(
            logdensity_fn,
            L=L,
            step_size=step_size,
        )

        state_transform = lambda x: x.position

        _, samples = run_inference_algorithm(
            rng_key=run_key,
            initial_state=initial_state,
            inference_algorithm=sampling_alg,
            num_steps=num_steps,
            transform=lambda state, info: state_transform(state),
            progress_bar=True,
        )

        print("average of steps (slow way):", samples.mean(axis=0))

        memory_efficient_sampling_alg, transform = store_only_expectation_values(
            sampling_algorithm=sampling_alg, state_transform=state_transform
        )

        initial_state = memory_efficient_sampling_alg.init(initial_state)

        final_state, trace_at_every_step = run_inference_algorithm(
            rng_key=run_key,
            initial_state=initial_state,
            inference_algorithm=memory_efficient_sampling_alg,
            num_steps=num_steps,
            transform=transform,
            progress_bar=True,
        )

        assert jnp.allclose(trace_at_every_step[0][-1], samples.mean(axis=0))

    @parameterized.parameters([True, False])
    def test_compatible_with_initial_pos(self, progress_bar):
        _ = run_inference_algorithm(
            rng_key=self.key,
            initial_position=jnp.array([1.0, 1.0]),
            inference_algorithm=self.algorithm,
            num_steps=self.num_steps,
            progress_bar=progress_bar,
            transform=lambda state, info: state.position,
        )

    @parameterized.parameters([True, False])
    def test_compatible_with_initial_state(self, progress_bar):
        state = self.algorithm.init(jnp.array([1.0, 1.0]))
        self.check_compatible(state, progress_bar)

    @staticmethod
    def logdensity_fn(x):
        return -0.5 * jnp.sum(jnp.square(x))


if __name__ == "__main__":
    absltest.main()
