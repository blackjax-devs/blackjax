"""Test the tempered SMC steps and routine"""
from typing import List

import jax
import jax.numpy as jnp
import jax.scipy.stats as stats
import numpy as np
import pytest

import blackjax.hmc as hmc
from blackjax.inference.smc.resampling import systematic
from blackjax.inference.smc.solver import dichotomy_solver
from blackjax.tempered_smc import TemperedSMCState, adaptive_tempered_smc, tempered_smc


def inference_loop(rng_key, kernel, initial_state):
    def cond(carry):
        _, state, *_ = carry
        return state.lmbda < 1

    def body(carry):
        i, state, op_key, curr_loglikelihood = carry
        op_key, subkey = jax.random.split(op_key, 2)
        state, info = kernel(subkey, state)
        return i + 1, state, op_key, curr_loglikelihood + info.log_likelihood_increment

    total_iter, final_state, _, log_likelihood = jax.lax.while_loop(
        cond, body, (0, initial_state, rng_key, 0.0)
    )

    return total_iter, final_state, log_likelihood


################################
# Test posterior mean estimate #
################################


def potential_fn(scale, coefs, preds, x):
    """Linear regression"""
    y = jnp.dot(x, coefs)
    logpdf = stats.norm.logpdf(preds, y, scale)
    return -jnp.sum(logpdf)


@pytest.mark.parametrize("N", [100, 1000, 5000])
@pytest.mark.parametrize("use_log", [True, False])
def test_adaptive_tempered_smc(N, use_log):
    x_data = np.random.normal(0, 1, size=(1000, 1))
    y_data = 3 * x_data + np.random.normal(size=x_data.shape)
    observations = {"x": x_data, "preds": y_data}

    conditioned_potential = lambda x: potential_fn(*x, **observations)

    prior = lambda x: stats.expon.logpdf(x[0], 1, 1) + stats.norm.logpdf(x[1])
    scale_init = 1 + np.random.exponential(1, N)
    coeffs_init = 3 + 2 * np.random.randn(N)
    smc_state_init = [scale_init, coeffs_init]

    iterates = []
    results = []  # type: List[TemperedSMCState]
    mcmc_kernel_factory = lambda pot: hmc.kernel(pot, 10e-2, jnp.eye(2), 50)

    for target_ess in [0.5, 0.75]:
        tempering_kernel = adaptive_tempered_smc(
            prior,
            conditioned_potential,
            mcmc_kernel_factory,
            hmc.new_state,
            systematic,
            target_ess,
            dichotomy_solver,
            use_log,
            5,
        )
        tempered_smc_state_init = TemperedSMCState(smc_state_init, 0.0)
        n_iter, result, log_likelihood = inference_loop(
            jax.random.PRNGKey(42), tempering_kernel, tempered_smc_state_init
        )
        iterates.append(n_iter)
        results.append(result)

        assert np.mean(result.particles[0]) == pytest.approx(1, 1e-1)
        assert np.mean(result.particles[1]) == pytest.approx(3, 1e-1)

    assert iterates[1] >= iterates[0]


@pytest.mark.parametrize("N", [100, 1000])
@pytest.mark.parametrize("n_schedule", [10, 100])
def test_fixed_schedule_tempered_smc(N, n_schedule):
    x_data = np.random.normal(0, 1, size=(1000, 1))
    y_data = 3 * x_data + np.random.normal(size=x_data.shape)
    observations = {"x": x_data, "preds": y_data}

    conditioned_potential = lambda x: potential_fn(*x, **observations)
    prior = lambda x: stats.norm.logpdf(jnp.log(x[0])) + stats.norm.logpdf(x[1])
    scale_init = np.exp(np.random.randn(N))
    coeffs_init = np.random.randn(N)
    smc_state_init = [scale_init, coeffs_init]

    lambda_schedule = np.logspace(-5, 0, n_schedule)
    mcmc_kernel_factory = lambda pot: hmc.kernel(pot, 10e-2, jnp.eye(2), 50)

    tempering_kernel = tempered_smc(
        prior,
        conditioned_potential,
        mcmc_kernel_factory,
        hmc.new_state,
        systematic,
        10,
    )
    tempered_smc_state_init = TemperedSMCState(smc_state_init, 0.0)

    def body_fn(carry, lmbda):
        rng_key, state = carry
        _, rng_key = jax.random.split(rng_key)
        new_state, info = tempering_kernel(rng_key, state, lmbda)
        return (rng_key, new_state), (new_state, info)

    (_, result), _ = jax.lax.scan(
        body_fn, (jax.random.PRNGKey(42), tempered_smc_state_init), lambda_schedule
    )
    assert np.mean(result.particles[0]) == pytest.approx(1, 1e-1)
    assert np.mean(result.particles[1]) == pytest.approx(3, 1e-1)


######################################
# Test normalizing constant estimate #
######################################


def normal_potential_fn(x, chol_cov):
    """multivariate normal without the normalizing constant"""
    dim = chol_cov.shape[0]
    y = jax.scipy.linalg.solve_triangular(chol_cov, x, lower=True)
    normalizing_constant = (
        np.sum(np.log(np.abs(np.diag(chol_cov)))) + dim * np.log(2 * np.pi) / 2.0
    )
    norm_y = jnp.sum(y * y, -1)
    return 0.5 * norm_y + normalizing_constant


@pytest.mark.parametrize("N", [500, 1_000])
@pytest.mark.parametrize("dim", [2, 5, 10])
def test_normalizing_constant(N, dim):
    np.random.seed(42)
    chol_cov = np.random.rand(dim, dim)
    iu = np.triu_indices(dim, 1)
    chol_cov[iu] = 0.0
    cov = chol_cov @ chol_cov.T
    conditioned_potential = lambda x: normal_potential_fn(x, chol_cov)

    prior = lambda x: stats.multivariate_normal.logpdf(
        x, jnp.zeros((dim,)), jnp.eye(dim)
    )

    x_init = np.random.randn(N, dim)

    mcmc_kernel_factory = lambda pot: hmc.kernel(pot, 1e-2, jnp.eye(dim), 50)

    tempering_kernel = adaptive_tempered_smc(
        prior,
        conditioned_potential,
        mcmc_kernel_factory,
        hmc.new_state,
        systematic,
        0.9,
        dichotomy_solver,
        True,
        10,
    )
    tempered_smc_state_init = TemperedSMCState(x_init, 0.0)
    n_iter, result, log_likelihood = inference_loop(
        jax.random.PRNGKey(42), tempering_kernel, tempered_smc_state_init
    )
    expected_log_likelihood = -0.5 * np.linalg.slogdet(np.eye(dim) + cov)[
        1
    ] - dim / 2 * np.log(2 * np.pi)
    assert log_likelihood == pytest.approx(expected_log_likelihood, rel=5e-2, abs=1e-1)
