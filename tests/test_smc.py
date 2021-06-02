"""Test the generic SMC sampler"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.scipy.stats.norm import logpdf

import blackjax.hmc as hmc
import blackjax.inference.smc.resampling as resampling
import blackjax.inference.smc.smc as smc


def kernel_potential_fn(position):
    return -logpdf(position[0])


def log_weights_fn(x, y):
    return logpdf(y - x)


@pytest.mark.parametrize("N", [500, 1000, 5000])
def test_smc(N):
    mcmc_factory = lambda potential_function: hmc.kernel(
        potential_function,
        hmc.HMCParameters(
            inv_mass_matrix=jnp.eye(1), step_size=1e-2, num_integration_steps=50
        ),
    )

    specialized_log_weights_fn = lambda tree: log_weights_fn(tree[0], 1.0)

    smc_kernel = smc.kernel(mcmc_factory, hmc.new_state, resampling.systematic, 100)

    # Don't use exactly the invariant distribution for the MCMC kernel
    init_state = smc.SMCState([0.25 + np.random.randn(N)])

    updated_state, _ = smc_kernel(
        jax.random.PRNGKey(42),
        init_state,
        kernel_potential_fn,
        specialized_log_weights_fn,
    )

    expected_mean = 0.5
    expected_std = np.sqrt(0.5)

    np.testing.assert_allclose(
        expected_mean, updated_state.particles[0].mean(), rtol=1e-2, atol=1e-1
    )
    np.testing.assert_allclose(
        expected_std, updated_state.particles[0].std(), rtol=1e-2, atol=1e-1
    )
