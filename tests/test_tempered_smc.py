"""Test the tempered SMC steps and routine"""
import functools as ft
from typing import List

import jax
import jax.numpy as jnp
import jax.scipy.stats as stats
import numpy as np
import pytest

import blackjax.hmc as hmc
from blackjax.inference.smc.resampling import systematic
from blackjax.inference.smc.smc import SMCState
from blackjax.inference.smc.solver import dichotomy_solver
from blackjax.tempered_smc import (
    TemperedSMCState,
    adaptive_tempered_smc,
    fixed_schedule_tempered_smc,
)


def potential_fn(scale, coefs, preds, x):
    """Linear regression"""
    y = jnp.dot(x, coefs)
    logpdf = stats.norm.logpdf(preds, y, scale)
    return -jnp.sum(logpdf)


def inference_loop(rng_key, kernel, initial_state):
    def cond(carry):
        _, state, _ = carry
        return state.lmbda < 1

    def body(carry):
        i, state, op_key = carry
        op_key, subkey = jax.random.split(op_key, 2)
        state, _ = kernel(subkey, state)
        return i + 1, state, op_key

    total_iter, final_state, _ = jax.lax.while_loop(
        cond, body, (0, initial_state, rng_key)
    )

    return total_iter, final_state


@pytest.mark.parametrize("N", [100, 1000])
@pytest.mark.parametrize("use_log", [True, False])
def test_adaptive_tempered_smc(N, use_log):
    x_data = np.random.normal(0, 1, size=(1000, 1))
    y_data = 3 * x_data + np.random.normal(size=x_data.shape)
    observations = {"x": x_data, "preds": y_data}

    conditioned_potential = ft.partial(potential_fn, **observations)
    potential = lambda x: conditioned_potential(*x)

    prior = lambda x: stats.expon.logpdf(x[0], 1, 1) + stats.norm.logpdf(x[1])
    scale_init = 1 + np.random.exponential(1, N)
    coeffs_init = 3 + 2 * np.random.randn(N)
    smc_state_init = SMCState([scale_init, coeffs_init])

    iterates = []
    results = []  # type: List[TemperedSMCState]
    mcmc_kernel_factory = lambda pot: hmc.kernel(
        pot, hmc.HMCParameters(inv_mass_matrix=jnp.eye(2))
    )

    for target_ess in [0.5, 0.75]:
        tempering_kernel = adaptive_tempered_smc(
            prior,
            potential,
            mcmc_kernel_factory,
            hmc.new_state,
            systematic,
            target_ess,
            dichotomy_solver,
            use_log,
            5,
        )
        tempered_smc_state_init = TemperedSMCState(0, smc_state_init, 0.0)
        n_iter, result = inference_loop(
            jax.random.PRNGKey(42), tempering_kernel, tempered_smc_state_init
        )
        iterates.append(n_iter)
        results.append(result)
        assert np.mean(result.smc_state.particles[0]) == pytest.approx(1, 1e-1)
        assert np.mean(result.smc_state.particles[1]) == pytest.approx(3, 1e-1)

    assert iterates[1] >= iterates[0]


@pytest.mark.parametrize("N", [100, 1000])
@pytest.mark.parametrize("n_schedule", [10, 100])
def test_fixed_schedule_tempered_smc(N, n_schedule):
    x_data = np.random.normal(0, 1, size=(1000, 1))
    y_data = 3 * x_data + np.random.normal(size=x_data.shape)
    observations = {"x": x_data, "preds": y_data}

    conditioned_potential = ft.partial(potential_fn, **observations)
    potential = lambda x: conditioned_potential(*x)
    prior = lambda x: stats.norm.logpdf(jnp.log(x[0])) + stats.norm.logpdf(x[1])
    scale_init = np.exp(np.random.randn(N))
    coeffs_init = np.random.randn(N)
    smc_state_init = SMCState([scale_init, coeffs_init])

    lambda_schedule = np.logspace(-5, 0, n_schedule)
    mcmc_kernel_factory = lambda pot: hmc.kernel(
        pot, hmc.HMCParameters(inv_mass_matrix=jnp.eye(2))
    )

    tempering_kernel = fixed_schedule_tempered_smc(
        prior,
        potential,
        mcmc_kernel_factory,
        hmc.new_state,
        systematic,
        lambda_schedule,
        10,
    )
    tempered_smc_state_init = TemperedSMCState(0, smc_state_init, 0.0)
    n_iter, result = inference_loop(
        jax.random.PRNGKey(42), tempering_kernel, tempered_smc_state_init
    )

    assert np.mean(result.smc_state.particles[0]) == pytest.approx(1, 1e-1)
    assert np.mean(result.smc_state.particles[1]) == pytest.approx(3, 1e-1)
