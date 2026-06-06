import jax
import jax.numpy as jnp
import pytest

import blackjax
from blackjax.adaptation.window_adaptation import window_adaptation


def make_linear_regression():
    key = jax.random.PRNGKey(42)
    N, D = 50, 3
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
    num_proposals = 100

    warmup = window_adaptation(
        blackjax.slingshot,
        logdensity_fn,
        is_mass_matrix_diagonal=False,
        num_proposals=num_proposals,
    )

    # 1. Use the native .run() method and vmap it across our chains
    def do_warmup(key, pos):
        # .run() natively handles the 100 step loop
        # It returns ((state, parameters), info)
        return warmup.run(key, pos, num_steps=100)

    warmup_keys = jax.random.split(jax.random.PRNGKey(43), num_chains)
    (final_state, tuned_params), _ = jax.vmap(do_warmup)(warmup_keys, initial_positions)

    # 2. Safely extract tuned parameters (handling both NamedTuple and Dict returns)
    step_size = (
        tuned_params.step_size
        if hasattr(tuned_params, "step_size")
        else tuned_params["step_size"]
    )
    inverse_mass_matrix = (
        tuned_params.inverse_mass_matrix
        if hasattr(tuned_params, "inverse_mass_matrix")
        else tuned_params["inverse_mass_matrix"]
    )

    # 3. Production Sampling
    def create_and_step(key, current_state, current_step_size, current_imm):
        alg = blackjax.slingshot(
            logdensity_fn,
            step_size=current_step_size,
            inverse_mass_matrix=current_imm,
            num_proposals=num_proposals,
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
        jax.random.PRNGKey(44), 5
    )  # 5 steps for benchmark mock
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
