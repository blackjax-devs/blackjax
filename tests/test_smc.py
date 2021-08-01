"""Test the generic SMC sampler"""

import jax
import jax.numpy as jnp
import jax.scipy.stats as stats
import numpy as np
import pytest

import blackjax.hmc as hmc
import blackjax.inference.smc.resampling as resampling
from blackjax.inference.smc.base import smc


def kernel_potential_fn(position):
    return -jnp.sum(stats.norm.logpdf(position))


def log_weights_fn(x, y):
    return jnp.sum(stats.norm.logpdf(y - x))


@pytest.mark.parametrize("N", [500, 1000, 5000])
def test_smc(N):
    mcmc_factory = lambda potential_function: hmc.kernel(
        potential_function,
        step_size=1e-2,
        inverse_mass_matrix=jnp.eye(1),
        num_integration_steps=50,
    )

    specialized_log_weights_fn = lambda tree: log_weights_fn(tree, 1.0)

    kernel = smc(mcmc_factory, hmc.new_state, resampling.systematic, 1000)

    # Don't use exactly the invariant distribution for the MCMC kernel
    init_particles = 0.25 + np.random.randn(N)

    updated_particles, _ = kernel(
        jax.random.PRNGKey(42),
        init_particles,
        kernel_potential_fn,
        specialized_log_weights_fn,
    )

    expected_mean = 0.5
    expected_std = np.sqrt(0.5)

    np.testing.assert_allclose(
        expected_mean, updated_particles.mean(), rtol=1e-2, atol=1e-1
    )
    np.testing.assert_allclose(
        expected_std, updated_particles.std(), rtol=1e-2, atol=1e-1
    )
