"""Test the ess function"""
import jax
import numpy as np
import pytest
from jax.scipy.stats.norm import logpdf

import blackjax.inference.smc.ess as ess
from blackjax.inference.smc.solver import dichotomy_solver


@pytest.mark.parametrize("N", [100, 1000, 5000])
def test_ess(N):
    w = np.random.rand(N)
    log_w = np.log(w)

    normalized_w = w / w.sum()
    log_normalized_w = np.log(normalized_w)

    log_ess_val = ess.ess(log_w, log=True)
    ess_val = ess.ess(log_w, log=False)

    log_normalized_ess_val = ess.ess(log_normalized_w, log=True)
    normalized_ess_val = ess.ess(log_normalized_w, log=False)

    np.testing.assert_almost_equal(log_ess_val, log_normalized_ess_val, decimal=3)
    np.testing.assert_almost_equal(ess_val, normalized_ess_val, decimal=3)
    np.testing.assert_almost_equal(np.log(ess_val), log_ess_val, decimal=3)
    np.testing.assert_almost_equal(ess_val, 1 / np.sum(normalized_w ** 2), decimal=3)


@pytest.mark.parametrize("target_ess", [0.25, 0.5])
@pytest.mark.parametrize("N", [100, 1000, 5000])
def test_ess_solver(target_ess, N):
    x_data = np.random.normal(0, 1, size=(N, 1))

    potential_fn = lambda pytree: -logpdf(pytree, scale=0.1)

    potential = jax.vmap(lambda x: potential_fn(*x), in_axes=[0])

    particles = x_data
    delta = ess.ess_solver(
        potential, particles, target_ess, 1.0, dichotomy_solver, use_log_ess=True
    )
    delta_log = ess.ess_solver(
        potential, particles, target_ess, 1.0, dichotomy_solver, use_log_ess=False
    )
    assert delta > 0
    np.testing.assert_allclose(delta_log, delta, atol=1e-3, rtol=1e-3)
    log_ess = ess.ess(-delta * potential(particles), log=True)
    np.testing.assert_allclose(np.exp(log_ess), target_ess * N, atol=1e-1, rtol=1e-2)
