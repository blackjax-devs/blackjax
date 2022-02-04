import functools

import chex
import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest

import blackjax.mcmc.hmc as hmc
from blackjax.adaptation.step_size import find_reasonable_step_size


class StepSizeTest(chex.TestCase):
    @chex.all_variants(with_pmap=False)
    def test_reasonable_step_size(self):
        def logprob_fn(x):
            return -jnp.sum(0.5 * x)

        rng_key = jax.random.PRNGKey(0)
        run_key0, run_key1 = jax.random.split(rng_key, 2)

        init_position = jnp.array([3.0])
        reference_state = hmc.init(init_position, logprob_fn)

        inv_mass_matrix = jnp.array([1.0])

        kernel = hmc.kernel()

        def kernel_generator(step_size: float):
            return functools.partial(
                kernel,
                logprob_fn=logprob_fn,
                step_size=step_size,
                inverse_mass_matrix=inv_mass_matrix,
                num_integration_steps=10,
            )

        # Test that the algorithm actually does something
        _find_step_size = self.variant(
            functools.partial(
                find_reasonable_step_size, kernel_generator=kernel_generator
            )
        )
        epsilon_1 = _find_step_size(
            run_key0,
            reference_state=reference_state,
            initial_step_size=0.01,
            target_accept=0.95,
        )
        assert not epsilon_1 == 1.0
        assert not epsilon_1 == np.inf

        # Different target acceptance rate
        epsilon_2 = _find_step_size(
            run_key1,
            reference_state=reference_state,
            initial_step_size=1.0,
            target_accept=0.05,
        )
        assert not epsilon_2.item == epsilon_1


if __name__ == "__main__":
    absltest.main()
