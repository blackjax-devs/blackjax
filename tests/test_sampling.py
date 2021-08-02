"""Test the accuracy of the HMC kernel"""
import functools as ft
from operator import add

import jax
import jax.numpy as jnp
import jax.scipy.stats as stats
import numpy as np
import pytest

import blackjax.diagnostics as diagnostics
import blackjax.hmc as hmc
import blackjax.mh as mh
import blackjax.nuts as nuts
import blackjax.stan_warmup as stan_warmup


def inference_loop(rng_key, kernel, initial_state, num_samples):
    def one_step(state, rng_key):
        state, _ = kernel(rng_key, state)
        return state, state

    keys = jax.random.split(rng_key, num_samples)
    _, states = jax.lax.scan(one_step, initial_state, keys)

    return states


def inference_loop_multiple_chains(
        rng_key, kernel, initial_state, num_samples, num_chains
):
    def one_step(states, rng_key):
        keys = jax.random.split(rng_key, num_chains)
        states, info = jax.vmap(kernel)(keys, states)
        return states, (states, info)

    keys = jax.random.split(rng_key, num_samples)
    _, (states, info) = jax.lax.scan(one_step, initial_state, keys)

    return states, info


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


def get_rw_proposal():
    from blackjax.inference.metrics import gaussian_euclidean
    inverse_mass_matrix = jnp.array([0.01])
    gaussian_momentum, *_ = gaussian_euclidean(inverse_mass_matrix)

    def proposal_generator(key, position):
        from jax import tree_multimap
        return tree_multimap(add, position, gaussian_momentum(key, position))

    proposal_loglikelihood_fn = lambda *_: 0.  # symmetric proposal
    return proposal_generator, proposal_loglikelihood_fn


normal_test_cases = [
    {
        "algorithm": hmc,
        "initial_position": {"x": jnp.array(100.0)},
        "parameters": {
            "step_size": 0.1,
            "inverse_mass_matrix": jnp.array([0.1]),
            "num_integration_steps": 100,
        },
        "num_sampling_steps": 6000,
        "burnin": 5_000,
    },
    {
        "algorithm": nuts,
        "initial_position": {"x": jnp.array(100.0)},
        "parameters": {"step_size": 0.1, "inverse_mass_matrix": jnp.array([0.1])},
        "num_sampling_steps": 6000,
        "burnin": 5_000,
    },
    {
        "algorithm": mh,
        "initial_position": {"x": 1.0},
        "parameters": {"get_proposal_generator": get_rw_proposal},
        "num_sampling_steps": 10_000,
        "burnin": 5_000,
    },
]


@pytest.mark.parametrize("case", normal_test_cases)
def test_univariate_normal(case):
    rng_key = jax.random.PRNGKey(19)
    potential = lambda x: normal_potential_fn(**x)
    initial_position = case["initial_position"]
    initial_state = case["algorithm"].new_state(initial_position, potential)

    kernel = case["algorithm"].kernel(potential, **case["parameters"])
    states = inference_loop(rng_key, kernel, initial_state, case["num_sampling_steps"])

    samples = states.position["x"][case["burnin"]:]

    assert np.var(samples) == pytest.approx(4.0, 1e-1)
    assert np.mean(samples) == pytest.approx(1.0, 1e-1)


# -------------------------------------------------------------------
#                MULTIVARIATE NORMAL DISTRIBUTION
# -------------------------------------------------------------------


def generate_multivariate_target(rng=None):
    if rng is None:
        loc = jnp.array([0.0, 3])
        scale = jnp.array([1.0, 2.0])
        rho = jnp.array(0.75)
    else:
        rng, loc_rng, scale_rng, rho_rng = jax.random.split(rng, 4)
        loc = jax.random.normal(loc_rng, [2]) * 10.0
        scale = jnp.abs(jax.random.normal(scale_rng, [2])) * 2.5
        rho = jax.random.uniform(rho_rng, [], minval=-1.0, maxval=1.0)

    cov = jnp.diag(scale ** 2)
    cov = cov.at[0, 1].set(rho * scale[0] * scale[1])
    cov = cov.at[1, 0].set(rho * scale[0] * scale[1])

    def potential_fn(x):
        return -stats.multivariate_normal.logpdf(x, loc, cov).sum()

    return potential_fn, loc, scale, rho


def mcse_test(samples, true_param, p_val=0.01):
    posterior_mean = jnp.mean(samples, axis=[0, 1])
    ess = diagnostics.effective_sample_size(samples, chain_axis=1, sample_axis=0)
    posterior_sd = jnp.std(samples, axis=0, ddof=1)
    avg_monte_carlo_standard_error = jnp.mean(posterior_sd, axis=0) / jnp.sqrt(ess)
    scaled_error = jnp.abs(posterior_mean - true_param) / avg_monte_carlo_standard_error
    np.testing.assert_array_less(scaled_error, stats.norm.ppf(1 - p_val))
    return scaled_error


mcse_test_cases = [
    {
        "algorithm": hmc,
        "parameters": {
            "step_size": 0.1,
            "num_integration_steps": 32,
        },
    },
    {
        "algorithm": nuts,
        "parameters": {"step_size": 0.07},
    },
]


@pytest.mark.parametrize("case", mcse_test_cases)
def test_mcse(case):
    """Test convergence using Monte Carlo central limit theorem."""
    rng_key = jax.random.PRNGKey(2351235)
    rng_key, init_fn_key, pos_init_key, sample_key = jax.random.split(rng_key, 4)
    potential_fn, true_loc, true_scale, true_rho = generate_multivariate_target(
        init_fn_key
    )
    num_chains = 10
    initial_positions = jax.random.normal(pos_init_key, [num_chains, 2])
    kernel = jax.jit(
        case["algorithm"].kernel(
            potential_fn, inverse_mass_matrix=true_scale, **case["parameters"]
        )
    )
    initial_states = jax.vmap(case["algorithm"].new_state, in_axes=(0, None))(
        initial_positions, potential_fn
    )
    states, _ = inference_loop_multiple_chains(
        sample_key, kernel, initial_states, 2_000, num_chains=num_chains
    )

    posterior_samples = states.position[-1000:]
    posterior_delta = posterior_samples - true_loc
    posterior_variance = posterior_delta ** 2.0
    posterior_correlation = jnp.prod(posterior_delta, axis=-1, keepdims=True) / (
            true_scale[0] * true_scale[1]
    )

    _ = jax.tree_multimap(
        mcse_test,
        [posterior_samples, posterior_variance, posterior_correlation],
        [true_loc, true_scale ** 2, true_rho],
    )
