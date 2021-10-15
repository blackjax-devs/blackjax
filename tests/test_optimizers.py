import chex
import jax
import jax.numpy as jnp
from absl.testing import absltest

import blackjax.optimizers as optimizers


class DualAveragingTest(chex.TestCase):
    @chex.all_variants(with_pmap=False)
    def test_dual_averaging(self):
        """We test the dual averaging algorithm by searching for the point that
        minimizes the gradient of a simple function.
        """

        # we need to wrap the gradient in a namedtuple as we optimize for a target
        # acceptance probability in the context of HMC.
        f = lambda x: (x - 1) ** 2
        grad_f = jax.jit(jax.grad(f))

        # Our target gradient is 0. we increase the rate of convergence by
        # increasing the value of gamma (see documentation of the algorithm).
        init, update, final = optimizers.dual_averaging(gamma=0.3)
        unpdate_fn = self.variant(update)

        da_state = init(3)
        for _ in range(100):
            x = jnp.exp(da_state.log_x)
            g = grad_f(x)
            da_state = unpdate_fn(da_state, g)

        self.assertAlmostEqual(final(da_state), 1.0, delta=1e-1)


if __name__ == "__main__":
    absltest.main()
