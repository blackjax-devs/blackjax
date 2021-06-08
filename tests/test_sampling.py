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


def inference_loop(rng_key, kernel, initial_state, num_samples):
    def one_step(state, rng_key):
        state, _ = kernel(rng_key, state)
        return state, state

    keys = jax.random.split(rng_key, num_samples)
    _, states = jax.lax.scan(one_step, initial_state, keys)

    return states


# -------------------------------------------------------------------
#                        LINEAR REGRESSION
# -------------------------------------------------------------------


def regression_potential_fn(scale, coefs, preds, x):
    """Linear regression"""
    logpdf = 0
    logpdf += stats.expon.logpdf(scale, 1, 1)
    logpdf += stats.norm.logpdf(coefs, 3 * jnp.ones(x.shape[-1]), 2)
    y = jnp.dot(x, coefs)
    logpdf += stats.norm.logpdf(preds, y, scale)
    return -jnp.sum(logpdf)


regresion_test_cases = [
    {
        "algorithm": hmc,
        "initial_position": {"scale": 1.0, "coefs": 2.0},
        "parameters": {"num_integration_steps": 90},
        "num_warmup_steps": 3_000,
        "num_sampling_steps": 2_000,
    },
    {
        "algorithm": nuts,
        "initial_position": {"scale": 1.0, "coefs": 2.0},
        "parameters": {},
        "num_warmup_steps": 1_000,
        "num_sampling_steps": 500,
    },
]


@pytest.mark.parametrize("case", regresion_test_cases)
@pytest.mark.parametrize("is_mass_matrix_diagonal", [True, False])
def test_linear_regression(case, is_mass_matrix_diagonal):
    """Test the HMC kernel and the Stan warmup."""
    x_data = np.random.normal(0, 1, size=(1000, 1))
    y_data = 3 * x_data + np.random.normal(size=x_data.shape)
    observations = {"x": x_data, "preds": y_data}

    conditioned_potential = ft.partial(regression_potential_fn, **observations)
    potential = lambda x: conditioned_potential(**x)

    rng_key = jax.random.PRNGKey(19)
    initial_position = case["initial_position"]
    initial_state = case["algorithm"].new_state(initial_position, potential)

    kernel_factory = lambda step_size, inverse_mass_matrix: case["algorithm"].kernel(
        potential, step_size, inverse_mass_matrix, **case["parameters"]
    )

    state, (step_size, inverse_mass_matrix), _ = stan_warmup.run(
        rng_key,
        kernel_factory,
        initial_state,
        case["num_warmup_steps"],
        is_mass_matrix_diagonal=is_mass_matrix_diagonal,
    )

    if is_mass_matrix_diagonal:
        assert inverse_mass_matrix.ndim == 1
    else:
        assert inverse_mass_matrix.ndim == 2

    kernel = kernel_factory(step_size, inverse_mass_matrix)
    states = inference_loop(rng_key, kernel, initial_state, case["num_sampling_steps"])

    coefs_samples = states.position["coefs"]
    scale_samples = states.position["scale"]

    assert np.mean(scale_samples) == pytest.approx(1, 1e-1)
    assert np.mean(coefs_samples) == pytest.approx(3, 1e-1)


# -------------------------------------------------------------------
#                UNIVARIATE NORMAL DISTRIBUTION
# -------------------------------------------------------------------


def normal_potential_fn(x):
    return -stats.norm.logpdf(x, loc=1.0, scale=2.0).squeeze()


normal_test_cases = [
    {
        "algorithm": hmc,
        "initial_position": {"x": 1.0},
        "parameters": {
            "step_size": 1e-2,
            "inverse_mass_matrix": jnp.array([0.1]),
            "num_integration_steps": 100,
        },
        "num_sampling_steps": 50_000,
    },
    {
        "algorithm": nuts,
        "initial_position": {"x": 1.0},
        "parameters": {"step_size": 1e-2, "inverse_mass_matrix": jnp.array([0.1])},
        "num_sampling_steps": 50_000,
    },
]


@pytest.mark.parametrize("case", normal_test_cases)
@pytest.mark.parametrize("is_mass_matrix_diagonal", [True, False])
def test_univariate_normal(case, is_mass_matrix_diagonal):
    rng_key = jax.random.PRNGKey(19)
    potential = lambda x: normal_potential_fn(**x)
    initial_position = case["initial_position"]
    initial_state = case["algorithm"].new_state(initial_position, potential)

    kernel = case["algorithm"].kernel(potential, **case["parameters"])
    states = inference_loop(rng_key, kernel, initial_state, case["num_sampling_steps"])

    samples = states.position["x"]

    assert np.var(samples).item() == pytest.approx(4.0, 1e-1)
    assert np.mean(samples).item() == pytest.approx(1.0, 1e-1)
