"""Test the accuracy of the HMC kernel"""
import functools as ft

import jax
import jax.numpy as jnp
import jax.scipy.stats as stats
import numpy as np
import pytest

import blackjax.hmc as hmc
import blackjax.nuts as nuts


def potential_fn(scale, coefs, preds, x):
    """Linear regression"""
    logpdf = 0
    logpdf += stats.expon.logpdf(scale, 1, 1)
    logpdf += stats.norm.logpdf(coefs, 3 * jnp.ones(x.shape[-1]), 2)
    y = jnp.dot(x, coefs)
    logpdf += stats.norm.logpdf(preds, y, scale)
    return -jnp.sum(logpdf)


def inference_loop(rng_key, kernel, initial_state, num_samples):
    def one_step(state, rng_key):
        state, _ = kernel(rng_key, state)
        return state, state

    keys = jax.random.split(rng_key, num_samples)
    _, states = jax.lax.scan(one_step, initial_state, keys)

    return states


inv_mass_matrices = [np.array([1.0, 1.0]), np.array([[1.0, 0.0], [0.0, 1.0]])]


@pytest.mark.parametrize("inv_mass_matrix", inv_mass_matrices)
def test_hmc(inv_mass_matrix):
    """Test the HMC kernel.

    This is a very simple sanity-check.
    """
    x_data = np.random.normal(0, 1, size=(1000, 1))
    y_data = 3 * x_data + np.random.normal(size=x_data.shape)
    observations = {"x": x_data, "preds": y_data}

    conditioned_potential = ft.partial(potential_fn, **observations)
    potential = lambda x: conditioned_potential(**x)

    initial_position = {"scale": 1.0, "coefs": 2.0}
    initial_state = hmc.new_state(initial_position, potential)

    params = hmc.HMCParameters(
        num_integration_steps=90, step_size=1e-3, inv_mass_matrix=inv_mass_matrix
    )
    kernel = hmc.kernel(potential, params)

    rng_key = jax.random.PRNGKey(19)
    states = inference_loop(rng_key, kernel, initial_state, 20_000)

    coefs_samples = states.position["coefs"][5000:]
    scale_samples = states.position["scale"][5000:]

    assert np.mean(scale_samples) == pytest.approx(1, 1e-1)
    assert np.mean(coefs_samples) == pytest.approx(3, 1e-1)


@pytest.mark.parametrize("inv_mass_matrix", inv_mass_matrices)
def test_nuts(inv_mass_matrix):
    """Test the HMC kernel.

    This is a very simple sanity-check.
    """
    x_data = np.random.normal(0, 1, size=(1000, 1))
    y_data = 3 * x_data + np.random.normal(size=x_data.shape)
    observations = {"x": x_data, "preds": y_data}

    conditioned_potential = ft.partial(potential_fn, **observations)
    potential = lambda x: conditioned_potential(**x)

    initial_position = {"scale": 1.0, "coefs": 2.0}
    initial_state = hmc.new_state(initial_position, potential)

    params = nuts.NUTSParameters(
        step_size=1e-3, max_tree_depth=10, inv_mass_matrix=inv_mass_matrix
    )
    kernel = nuts.kernel(potential, params)

    rng_key = jax.random.PRNGKey(19)
    states = inference_loop(rng_key, kernel, initial_state, 20_000)

    coefs_samples = states.position["coefs"][5000:]
    scale_samples = states.position["scale"][5000:]

    print(scale_samples, coefs_samples)
    assert np.mean(scale_samples) == pytest.approx(1, 1e-1)
    assert np.mean(coefs_samples) == pytest.approx(3, 1e-1)
