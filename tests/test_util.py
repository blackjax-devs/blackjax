import chex
import jax
import jax.numpy as jnp
from absl.testing import absltest, parameterized

import blackjax
from blackjax.util import run_inference_algorithm


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
            transform=lambda x: x.position,
        )

    def test_streaming(self):
        def logdensity_fn(x):
            return -0.5 * jnp.sum(jnp.square(x))

        initial_position = jnp.ones(
            10,
        )

        init_key, run_key = jax.random.split(self.key, 2)

        initial_state = blackjax.mcmc.mclmc.init(
            position=initial_position, logdensity_fn=logdensity_fn, rng_key=init_key
        )

        alg = blackjax.mclmc(logdensity_fn=logdensity_fn, L=0.5, step_size=0.1)

        _, states, info = run_inference_algorithm(
            rng_key=run_key,
            initial_state=initial_state,
            inference_algorithm=alg,
            num_steps=50,
            progress_bar=False,
            expectation=lambda x: x,
            transform=lambda x: x.position,
            return_state_history=True,
        )

        average, _ = run_inference_algorithm(
            rng_key=run_key,
            initial_state=initial_state,
            inference_algorithm=alg,
            num_steps=50,
            progress_bar=False,
            expectation=lambda x: x,
            transform=lambda x: x.position,
            return_state_history=False,
        )

        assert jnp.allclose(states.mean(axis=0), average)

    @parameterized.parameters([True, False])
    def test_compatible_with_initial_pos(self, progress_bar):
        _ = run_inference_algorithm(
            rng_key=self.key,
            initial_position=jnp.array([1.0, 1.0]),
            inference_algorithm=self.algorithm,
            num_steps=self.num_steps,
            progress_bar=progress_bar,
            transform=lambda x: x.position,
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
