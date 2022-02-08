"""Simple benchmarks to track potential performance regressions.

(TODO) This is only a first draft. We should add the ESS per gradient evaluation, ESS /
second and other metrics to make sure that the results are "correct", and
obviously more models. It should also be run in CI.

"""
import functools

import jax
import jax.numpy as jnp
import jax.scipy.stats as stats
import pytest

import blackjax


def regression_logprob(scale, coefs, preds, x):
    """Linear regression"""
    logpdf = 0
    logpdf += stats.expon.logpdf(scale, 1, 1)
    logpdf += stats.norm.logpdf(coefs, 3 * jnp.ones(x.shape[-1]), 2)
    y = jnp.dot(x, coefs)
    logpdf += stats.norm.logpdf(preds, y, scale)
    return jnp.sum(logpdf)


def inference_loop(kernel, num_samples, rng_key, initial_state):
    def one_step(state, rng_key):
        state, _ = kernel(rng_key, state)
        return state, state

    keys = jax.random.split(rng_key, num_samples)
    _, states = jax.lax.scan(one_step, initial_state, keys)

    return states


def run_regression(algorithm, **parameters):
    key = jax.random.PRNGKey(0)
    rng_key, init_key0, init_key1 = jax.random.split(key, 3)
    x_data = jax.random.normal(init_key0, shape=(100_000, 1))
    y_data = 3 * x_data + jax.random.normal(init_key1, shape=x_data.shape)

    logposterior_fn_ = functools.partial(regression_logprob, x=x_data, preds=y_data)
    logposterior_fn = lambda x: logposterior_fn_(**x)

    warmup_key, inference_key = jax.random.split(rng_key, 2)

    warmup = blackjax.window_adaptation(
        algorithm,
        logposterior_fn,
        1000,
        False,
        **parameters,
    )
    state, kernel, _ = warmup.run(warmup_key, {"scale": 1.0, "coefs": 2.0})

    states = inference_loop(kernel, 10_000, inference_key, state)

    return states


@pytest.mark.benchmark
def test_regression_nuts(benchmark):
    benchmark.extra_info["algorithm"] = "NUTS"
    benchmark.extra_info["num_warmup_steps"] = "1000"
    benchmark.extra_info["num_samples"] = "10_000"
    benchmark(run_regression, blackjax.nuts)


@pytest.mark.benchmark
def test_regression_hmc(benchmark):
    benchmark.extra_info["algorithm"] = "HMC"
    benchmark.extra_info["num_integration_steps"] = "10"
    benchmark.extra_info["num_warmup_steps"] = "1000"
    benchmark.extra_info["num_samples"] = "10_000"
    benchmark(run_regression, blackjax.hmc, num_integration_steps=10)
