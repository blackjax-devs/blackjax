import jax
import jax.numpy as jnp
import pytest

import blackjax
from blackjax.adaptation.window_adaptation import window_adaptation


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
        log_lik = jnp.sum(
            -0.5 * jnp.log(2 * jnp.pi * sigma**2) - 0.5 * ((y - mu) / sigma) ** 2
        )
        log_prior_beta = jnp.sum(-0.5 * beta**2)
        log_prior_sigma = -0.5 * log_sigma**2
        return log_lik + log_prior_beta + log_prior_sigma

    initial_positions = jnp.zeros((16, 4))
    true_params = jnp.concatenate([true_beta, jnp.array([jnp.log(true_sigma)])])
    return "1. Linear Regression", logdensity, initial_positions, true_params


def run_benchmark_logic(logdensity_fn, initial_positions, dim):
    num_chains = initial_positions.shape[0]
    num_proposals = 1000

    warmup = window_adaptation(
        blackjax.slingshot,
        logdensity_fn,
        is_mass_matrix_diagonal=False,
        num_proposals=num_proposals,
    )

    # Initialize the warmup state across chains
    warmup_init_vmap = jax.vmap(warmup.init)
    state, adapt_state, info = warmup_init_vmap(initial_positions)

    # Run the warmup using jax.lax.scan over warmup.step
    warmup_step_vmap = jax.vmap(warmup.step)

    def scan_step(carry, step_key):
        carry_state, carry_adapt_state, carry_info = carry
        keys = jax.random.split(step_key, num_chains)
        next_state, next_adapt_state, next_info = warmup_step_vmap(
            keys, carry_state, carry_adapt_state, carry_info
        )
        return (next_state, next_adapt_state, next_info), None

    # Run warmup steps
    num_warmup_steps = 1000
    warmup_keys = jax.random.split(jax.random.PRNGKey(43), num_warmup_steps)
    (final_state, final_adapt_state, final_info), _ = jax.lax.scan(
        scan_step, (state, adapt_state, info), warmup_keys
    )

    # Pass those tuned parameters into the production sampling step
    step_size = final_adapt_state.step_size
    inverse_mass_matrix = final_adapt_state.inverse_mass_matrix

    def create_and_step(key, current_state, current_step_size, current_imm):
        alg = blackjax.slingshot(
            logdensity_fn,
            step_size=current_step_size,
            num_proposals=num_proposals,
            inverse_mass_matrix=current_imm,
        )
        return alg.step(key, current_state)

    step_vmap = jax.vmap(create_and_step)

    def prod_step(carry_state, step_key):
        keys = jax.random.split(step_key, num_chains)
        next_state, info_out = step_vmap(
            keys, carry_state, step_size, inverse_mass_matrix
        )
        return next_state, info_out

    prod_keys = jax.random.split(
        jax.random.PRNGKey(44), 10
    )  # 10 steps for benchmark mock
    final_states, _ = jax.lax.scan(prod_step, final_state, prod_keys)

    return final_states


@pytest.mark.benchmark
@pytest.mark.parametrize(
    "model_name, logdensity_fn, initial_positions, true_params",
    [make_linear_regression()],
)
def test_slingshot_performance(
    benchmark, model_name, logdensity_fn, initial_positions, true_params
):
    """Benchmark performance using pytest-benchmark fixture."""
    dim = initial_positions.shape[-1]

    def run():
        return run_benchmark_logic(logdensity_fn, initial_positions, dim)

    benchmark(run)
