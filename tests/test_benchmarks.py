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
from blackjax.util import run_inference_algorithm


def regression_logprob(log_scale, coefs, preds, x):
    """Linear regression"""
    scale = jnp.exp(log_scale)
    scale_prior = stats.expon.logpdf(scale, 0, 1) + log_scale
    coefs_prior = stats.norm.logpdf(coefs, 0, 5)
    y = jnp.dot(x, coefs)
    logpdf = stats.norm.logpdf(preds, y, scale)
    return sum(x.sum() for x in [scale_prior, coefs_prior, logpdf])


def run_regression(algorithm, **parameters):
    key = jax.random.key(0)
    rng_key, init_key0, init_key1 = jax.random.split(key, 3)
    x_data = jax.random.normal(init_key0, shape=(100_000, 1))
    y_data = 3 * x_data + jax.random.normal(init_key1, shape=x_data.shape)

    logdensity_fn_ = functools.partial(regression_logprob, x=x_data, preds=y_data)
    logdensity_fn = lambda x: logdensity_fn_(**x)

    warmup_key, inference_key = jax.random.split(rng_key, 2)

    warmup = blackjax.window_adaptation(
        algorithm,
        logdensity_fn,
        is_mass_matrix_diagonal=False,
        **parameters,
    )
    (state, parameters), _ = warmup.run(
        warmup_key, {"log_scale": 0.0, "coefs": 2.0}, 1000
    )
    inference_algorithm = algorithm(logdensity_fn, **parameters)

    _, states, _ = run_inference_algorithm(
        rng_key=inference_key,
        initial_state=state,
        inference_algorithm=inference_algorithm,
        num_steps=10_000,
    )

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
