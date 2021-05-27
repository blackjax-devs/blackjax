"""Test the accuracy of the HMC kernel"""
import functools as ft

import jax
import jax.numpy as jnp
import jax.scipy.stats as stats
import numpy as np
import pytest

import blackjax.hmc as hmc
import blackjax.nuts as nuts
import blackjax.stan_warmup as stan_warmup


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


@pytest.mark.parametrize("is_mass_matrix_diagonal", [True, False])
def test_hmc(is_mass_matrix_diagonal):
    """Test the HMC kernel and the Stan warmup."""
    x_data = np.random.normal(0, 1, size=(1000, 1))
    y_data = 3 * x_data + np.random.normal(size=x_data.shape)
    observations = {"x": x_data, "preds": y_data}

    conditioned_potential = ft.partial(potential_fn, **observations)
    potential = lambda x: conditioned_potential(**x)

    rng_key = jax.random.PRNGKey(19)
    initial_position = {"scale": 1.0, "coefs": 2.0}
    initial_state = hmc.new_state(initial_position, potential)

    kernel_factory = lambda step_size, inverse_mass_matrix: hmc.kernel(
        potential, step_size, inverse_mass_matrix, 90
    )

    state, (step_size, inverse_mass_matrix), _ = stan_warmup.run(
        rng_key,
        kernel_factory,
        initial_state,
        3_000,
        is_mass_matrix_diagonal=is_mass_matrix_diagonal,
    )

    if is_mass_matrix_diagonal:
        assert inverse_mass_matrix.ndim == 1
    else:
        assert inverse_mass_matrix.ndim == 2

    kernel = kernel_factory(step_size, inverse_mass_matrix)
    states = inference_loop(rng_key, kernel, initial_state, 2_000)

    coefs_samples = states.position["coefs"]
    scale_samples = states.position["scale"]

    assert np.mean(scale_samples) == pytest.approx(1, 1e-1)
    assert np.mean(coefs_samples) == pytest.approx(3, 1e-1)


@pytest.mark.parametrize("is_mass_matrix_diagonal", [True, False])
def test_nuts(is_mass_matrix_diagonal):
    """Test the NUTS kernel and the Stan warmup."""
    rng_key = jax.random.PRNGKey(19)
    x_data = np.random.normal(0, 1, size=(1000, 1))
    y_data = 3 * x_data + np.random.normal(size=x_data.shape)
    observations = {"x": x_data, "preds": y_data}

    conditioned_potential = ft.partial(potential_fn, **observations)
    potential = lambda x: conditioned_potential(**x)

    initial_position = {"scale": 1.0, "coefs": 2.0}
    initial_state = hmc.new_state(initial_position, potential)

    kernel_factory = lambda step_size, inverse_mass_matrix: nuts.kernel(
        potential, step_size, inverse_mass_matrix
    )

    state, (step_size, inverse_mass_matrix), _ = stan_warmup.run(
        rng_key,
        kernel_factory,
        initial_state,
        1_000,
        is_mass_matrix_diagonal=is_mass_matrix_diagonal,
    )

    if is_mass_matrix_diagonal:
        assert inverse_mass_matrix.ndim == 1
    else:
        assert inverse_mass_matrix.ndim == 2

    kernel = nuts.kernel(potential, step_size, inverse_mass_matrix)
    states = inference_loop(rng_key, kernel, initial_state, 500)

    coefs_samples = states.position["coefs"]
    scale_samples = states.position["scale"]

    assert np.mean(scale_samples) == pytest.approx(1, 1e-1)
    assert np.mean(coefs_samples) == pytest.approx(3, 1e-1)
