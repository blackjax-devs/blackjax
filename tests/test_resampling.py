"""Test the resampling functions for SMC."""
import chex
import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest, parameterized

import blackjax.smc.resampling as resampling

resampling_methods = {
    "systematic": resampling.systematic,
    "stratified": resampling.stratified,
    "multinomial": resampling.multinomial,
    "residual": resampling.residual,
}


def _weighted_avg_and_std(values, weights):
    average = np.average(values, weights=weights)
    variance = np.average((values - average) ** 2, weights=weights)
    return average, np.sqrt(variance)


def integrand(x):
    return np.cos(x)


class ResamplingTest(chex.TestCase):
    @chex.all_variants(with_pmap=False)
    @parameterized.parameters(resampling_methods.keys())
    def test_resampling_methods(self, method_name):
        N = 10_000

        np.random.seed(42)
        batch_size = 100
        w = jnp.array(np.random.rand(N), dtype="float32")
        x = jnp.array(np.random.randn(N), dtype="float32")
        w = w / w.sum()

        resampling_keys = jax.random.split(jax.random.PRNGKey(42), batch_size)

        resampling_idx = self.variant(
            jax.vmap(resampling_methods[method_name], in_axes=[None, 0])
        )(w, resampling_keys)
        resampling_idx = np.asarray(resampling_idx)
        batch_x = np.repeat(x.reshape(1, -1), batch_size, axis=0)
        batch_resampled_x = np.take_along_axis(batch_x, resampling_idx, axis=1)
        batch_integrand = integrand(batch_resampled_x)
        batch_mean_res = batch_integrand.mean(1)
        batch_std_res = batch_integrand.std(1)

        mean_res = batch_mean_res.mean()
        std_res = batch_std_res.mean()
        expected_mean, expected_std = _weighted_avg_and_std(integrand(x), w)

        np.testing.assert_allclose(mean_res, expected_mean, atol=1e-2, rtol=1e-2)
        np.testing.assert_allclose(std_res, expected_std, atol=1e-2, rtol=1e-2)


if __name__ == "__main__":
    absltest.main()
