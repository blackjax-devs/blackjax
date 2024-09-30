"""Test the iterative u-turn criterion."""
import chex
import jax.numpy as jnp
from absl.testing import absltest, parameterized

from blackjax.mcmc.metrics import gaussian_euclidean
from blackjax.mcmc.termination import IterativeUTurnState, iterative_uturn_numpyro


class UTurnTest(chex.TestCase):
    @chex.all_variants(without_device=False, with_pmap=False)
    @parameterized.parameters(
        [
            ((3, 2), False),
            ((3, 3), True),
            ((0, 0), False),
            ((0, 1), True),
            ((1, 3), True),
        ],
    )
    def test_is_iterative_turning(self, checkpoint_idxs, expected_turning):
        inverse_mass_matrix = jnp.ones(1)
        _, _, is_turning, _ = gaussian_euclidean(inverse_mass_matrix)
        _, _, is_iterative_turning = iterative_uturn_numpyro(is_turning)

        momentum = 1.0
        momentum_sum = 3.0

        idx_min, idx_max = checkpoint_idxs
        momentum_ckpts = jnp.array([1.0, 2.0, 3.0, -2.0])
        momentum_sum_ckpts = jnp.array([2.0, 4.0, 4.0, -1.0])
        checkpoints = IterativeUTurnState(
            momentum_ckpts,
            momentum_sum_ckpts,
            idx_min,
            idx_max,
        )

        actual_turning = self.variant(is_iterative_turning)(
            checkpoints, momentum_sum, momentum
        )

        assert expected_turning == actual_turning


if __name__ == "__main__":
    absltest.main()
