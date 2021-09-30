"""Test the resampling functions"""

import jax
import numpy as np
import pytest

import blackjax.inference.smc.resampling as resampling

resampling_methods_to_test = [
    resampling.systematic,
    resampling.stratified,
    resampling.multinomial,
    resampling.residual,
]


def _weighted_avg_and_std(values, weights):
    average = np.average(values, weights=weights)
    variance = np.average((values - average) ** 2, weights=weights)
    return average, np.sqrt(variance)


def integrand(x):
    return np.cos(x)


@pytest.mark.parametrize("N", [100, 500, 1_000, 100_000])
@pytest.mark.parametrize("resampling_method", resampling_methods_to_test)
def test_resampling_methods(N, resampling_method):
    np.random.seed(42)
    batch_size = 100
    w = np.random.rand(N)
    x = np.random.randn(N)
    w = w / w.sum()

    resampling_keys = jax.random.split(jax.random.PRNGKey(42), batch_size)

    resampling_idx = jax.vmap(jax.jit(resampling_method), in_axes=[None, 0])(
        w, resampling_keys
    )
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
