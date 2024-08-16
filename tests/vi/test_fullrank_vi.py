import chex
import jax
import jax.numpy as jnp
import jax.scipy.stats as stats
import optax
from absl.testing import absltest

import blackjax


class FullRankVITest(chex.TestCase):
    def setUp(self):
        super().setUp()
        self.key = jax.random.PRNGKey(42)

    @chex.variants(with_jit=True, without_jit=True)
    def test_recover_posterior(self):
        ground_truth = [
            # loc, scale
            (2, 4),
            (3, 5),
        ]

        def logdensity_fn(x):
            logpdf = stats.norm.logpdf(x["x_1"], *ground_truth[0]) + stats.norm.logpdf(
                x["x_2"], *ground_truth[1]
            )
            return jnp.sum(logpdf)

        initial_position = {"x_1": 0.0, "x_2": 0.0}

        num_steps = 50_000
        num_samples = 500

        optimizer = optax.sgd(1e-2)
        frvi = blackjax.fullrank_vi(logdensity_fn, optimizer, num_samples)
        state = frvi.init(initial_position)

        rng_key = self.key
        for i in range(num_steps):
            subkey = jax.random.split(rng_key, i)
            state, _ = self.variant(frvi.step)(subkey, state)

        loc_1, loc_2 = state.mu["x_1"], state.mu["x_2"]
        self.assertAlmostEqual(loc_1, ground_truth[0][0], delta=0.01)
        self.assertAlmostEqual(loc_2, ground_truth[1][0], delta=0.01)


if __name__ == "__main__":
    absltest.main()
