import jax
import jax.numpy as jnp
import scipy.optimize
import pytest

import blackjax
from blackjax.mcmc.slingshot import build_kernel, init_adaptation, dual_averaging_step

# --- Model Definitions (Linear Regression, Logit, etc., kept as provided) ---
def make_linear_regression():
    key = jax.random.PRNGKey(42)
    N, D = 500, 3
    X = jax.random.normal(key, (N, D))
    true_beta = jnp.array([1.5, -2.0, 0.8])
    true_sigma = 0.5
    y = X @ true_beta + true_sigma * jax.random.normal(jax.random.PRNGKey(43), (N,))

    def logdensity(theta):
        beta = theta[:3]
        log_sigma = theta[3]
        sigma = jnp.exp(log_sigma)
        mu = X @ beta
        log_lik = jnp.sum(-0.5 * jnp.log(2 * jnp.pi * sigma**2) - 0.5 * ((y - mu) / sigma) ** 2)
        log_prior_beta = jnp.sum(-0.5 * beta**2)
        log_prior_sigma = -0.5 * log_sigma**2
        return log_lik + log_prior_beta + log_prior_sigma

    initial_positions = jnp.zeros((16, 4))
    true_params = jnp.concatenate([true_beta, jnp.array([jnp.log(true_sigma)])])
    return "1. Linear Regression", logdensity, initial_positions, true_params

# ... (Include other make_* functions here) ...

def run_benchmark_logic(logdensity_fn, initial_positions, dim):
    num_chains = 16
    num_proposals = 1000
    target_rate = 0.65

    def init_chain(pos):
        algo = build_kernel(logdensity_fn, step_size=1.0, num_proposals=num_proposals)
        return algo.init(pos)

    # Simplified MAP/Initialization Logic for Benchmarking
    jitter = jax.random.normal(jax.random.PRNGKey(999), initial_positions.shape) * 0.1
    warm_start_positions = initial_positions + jitter
    states = jax.vmap(init_chain)(warm_start_positions)

    init_adapt_vmap = jax.vmap(lambda ss: init_adaptation(ss, dim))
    da_states = init_adapt_vmap(jnp.ones(16) * 0.1)

    return states, da_states

@pytest.mark.benchmark
@pytest.mark.parametrize("model_name, logdensity_fn, initial_positions, true_params", [
    make_linear_regression(),
    # ... Add other model calls ...
])
def test_slingshot_performance(benchmark, model_name, logdensity_fn, initial_positions, true_params):
    """Benchmark performance using pytest-benchmark fixture."""
    dim = initial_positions.shape[-1]

    def run():
        states, da_states = run_benchmark_logic(logdensity_fn, initial_positions, dim)
        # Execute a minimal sampling run for the benchmark
        return states

    benchmark(run)