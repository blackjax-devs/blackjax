import chex
import jax
import jax.numpy as jnp
import jax.scipy.stats as stats
import optax
from absl.testing import absltest

import blackjax


class MFVITest(chex.TestCase):
    def setUp(self):
        super().setUp()
        self.key = jax.random.PRNGKey(42)

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
        mfvi = blackjax.meanfield_vi(logdensity_fn, optimizer, num_samples)
        state = mfvi.init(initial_position)

        rng_key = self.key
        for _ in range(num_steps):
            rng_key, _ = jax.random.split(rng_key)
            state, _ = jax.jit(mfvi.step)(rng_key, state)

        loc_1, loc_2 = state.mu["x_1"], state.mu["x_2"]
        scale = jax.tree_map(jnp.exp, state.rho)
        scale_1, scale_2 = scale["x_1"], scale["x_2"]
        self.assertAlmostEqual(loc_1, ground_truth[0][0], delta=0.01)
        self.assertAlmostEqual(scale_1, ground_truth[0][1], delta=0.01)
        self.assertAlmostEqual(loc_2, ground_truth[1][0], delta=0.01)
        self.assertAlmostEqual(scale_2, ground_truth[1][1], delta=0.01)


if __name__ == "__main__":
    absltest.main()
