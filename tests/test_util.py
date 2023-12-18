import chex
import jax
import jax.numpy as jnp
from absl.testing import absltest, parameterized

from blackjax.mcmc.hmc import hmc
from blackjax.util import run_inference_algorithm


class RunInferenceAlgorithmTest(chex.TestCase):
    def setUp(self):
        super().setUp()
        self.key = jax.random.key(42)
        self.algorithm = hmc(
            logdensity_fn=self.logdensity_fn,
            inverse_mass_matrix=jnp.eye(2),
            step_size=1.0,
            num_integration_steps=1000,
        )
        self.num_steps = 10

    def check_compatible(self, initial_state_or_position, progress_bar):
        """
        Runs 10 steps with `run_inference_algorithm` starting with
        `initial_state_or_position` and potentially a progress bar.
        """
        _ = run_inference_algorithm(
            self.key,
            initial_state_or_position,
            self.algorithm,
            self.num_steps,
            progress_bar,
            transform=lambda x: x.position,
        )

    @parameterized.parameters([True, False])
    def test_compatible_with_initial_pos(self, progress_bar):
        self.check_compatible(jnp.array([1.0, 1.0]), progress_bar)

    @parameterized.parameters([True, False])
    def test_compatible_with_initial_state(self, progress_bar):
        state = self.algorithm.init(jnp.array([1.0, 1.0]))
        self.check_compatible(state, progress_bar)

    @staticmethod
    def logdensity_fn(x):
        return -0.5 * jnp.sum(jnp.square(x))


if __name__ == "__main__":
    absltest.main()
