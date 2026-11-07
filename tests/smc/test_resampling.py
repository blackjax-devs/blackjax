"""Test the resampling functions for SMC."""

import itertools

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
    @chex.variants(with_jit=True, without_jit=True)
    @parameterized.parameters(itertools.product([False, True], [False, True]))
    def test_zero_weight_boundaries(self, is_systematic, upper_boundary):
        if upper_boundary:
            weights = jax.nn.softmax(
                jnp.array([0.0, -1.0, -2.0, -3.0, -4.0, -jnp.inf], dtype=jnp.float32)
            )
            self.assertLess(float(jnp.cumsum(weights)[-1]), 1.0)
            offset = jnp.array(1 - 2**-23, dtype=jnp.float32)
        else:
            weights = jnp.array([0.0, 0.5, 0.0, 0.5, 0.0], dtype=jnp.float32)
            offset = jnp.array(0.0, dtype=jnp.float32)
        num_samples = weights.shape[0] if upper_boundary else 4
        u = offset if is_systematic else jnp.full((num_samples,), offset)
        indices = self.variant(resampling._inverse_cdf_indices, static_argnums=(2,))(
            weights, u, num_samples
        )
        self.assertTrue(np.all(np.asarray(weights[indices]) > 0))
        counts = np.bincount(np.asarray(indices), minlength=weights.shape[0])
        expected_counts = num_samples * np.asarray(weights, dtype=np.float64)
        self.assertTrue(np.all(counts >= np.floor(expected_counts)))
        self.assertTrue(np.all(counts <= np.ceil(expected_counts)))

    @chex.variants(with_jit=True, without_jit=True)
    @parameterized.parameters(
        itertools.product([100, 1000, 2000], resampling_methods.keys())
    )
    def test_resampling_methods(self, num_samples, method_name):
        N = 10_000

        np.random.seed(42)
        batch_size = 100
        w = jnp.array(np.random.rand(N), dtype="float32")
        x = jnp.array(np.random.randn(N), dtype="float32")
        w = w / w.sum()

        resampling_keys = jax.random.split(jax.random.key(42), batch_size)

        resampling_idx = jax.vmap(
            self.variant(resampling_methods[method_name], static_argnums=(2,)),
            in_axes=[0, None, None],
        )(resampling_keys, w, num_samples)

        self.assertEqual(resampling_idx.shape[-1], num_samples)

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

    @chex.variants(with_jit=True, without_jit=True)
    @parameterized.parameters(["first", "middle", "last"])
    def test_sorted_uniforms_finite_with_zero_draw(self, pos):
        """Verify _sorted_uniforms_from handles 0.0 in input gracefully."""
        # Create a short uniform vector with an exact 0.0 at specified position
        us = jnp.array([0.5, 0.3, 0.1, 0.2], dtype=jnp.float32)
        if pos == "first":
            us = us.at[0].set(0.0)
        elif pos == "middle":
            us = us.at[1].set(0.0)
        elif pos == "last":
            us = us.at[-1].set(0.0)

        result = self.variant(resampling._sorted_uniforms_from)(us)

        # All outputs should be finite
        self.assertTrue(jnp.all(jnp.isfinite(result)))
        # Should be non-decreasing
        diffs = jnp.diff(result)
        self.assertTrue(jnp.all(diffs >= 0.0))
        # Should be in [0, 1]
        self.assertTrue(jnp.all(result >= 0.0))
        self.assertTrue(jnp.all(result <= 1.0))
        # Verify against float64 NumPy reference to catch pathological cases
        # (e.g., broken log where output collapses to [0,0,0])
        ref = np.cumsum(-np.log1p(-np.asarray(us, np.float64)))
        ref = ref[:-1] / ref[-1]
        np.testing.assert_allclose(
            np.asarray(result), ref, rtol=1e-6, err_msg=f"Mismatch at position {pos}"
        )

    @chex.variants(with_jit=True, without_jit=True)
    def test_inverse_cdf_explicit_zeros(self):
        """Verify _inverse_cdf with explicit zero weights."""
        weights = jnp.array([0.0, 0.5, 0.0, 0.5, 0.0], dtype=jnp.float32)
        queries = jnp.array([0.0, 0.25, 0.5, 0.75], dtype=jnp.float32)
        idx = self.variant(resampling._inverse_cdf)(weights, queries)
        expected = jnp.array([1, 1, 3, 3], dtype=idx.dtype)
        np.testing.assert_array_equal(idx, expected)
        self.assertTrue(jnp.all(weights[idx] > 0.0))

    @chex.variants(with_jit=True, without_jit=True)
    def test_inverse_cdf_boundary_queries(self):
        """Verify _inverse_cdf with boundary queries on float32-rounded weights."""
        # Use 6 elements so float32 softmax cumsum[-1] < 1.0 (triggers the bug)
        logits = jnp.array([0.0, -1.0, -2.0, -3.0, -4.0, -jnp.inf], dtype=jnp.float32)
        weights = jax.nn.softmax(logits)
        cumsum = jnp.cumsum(weights)
        # Precondition: cumsum[-1] < 1.0 due to float32 precision
        self.assertLess(float(cumsum[-1]), 1.0)
        queries = jnp.array(
            [
                0.0,
                0.5,
                cumsum[-1],  # exact total mass
                jnp.nextafter(1.0, -jnp.inf),  # just below 1.0
                1.0,  # above total mass
            ],
            dtype=jnp.float32,
        )
        idx = self.variant(resampling._inverse_cdf)(weights, queries)
        # All selected weights must be positive
        self.assertTrue(jnp.all(weights[idx] > 0.0))
        # High queries should map to last positive-weight particle (index 4)
        self.assertEqual(int(idx[2]), 4)
        self.assertEqual(int(idx[3]), 4)
        self.assertEqual(int(idx[4]), 4)


if __name__ == "__main__":
    absltest.main()
