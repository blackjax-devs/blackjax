"""Simple benchmarks to track potential performance regressions.

(TODO) This is only a first draft. We should add the ESS per gradient evaluation, ESS /
second and other metrics to make sure that the results are "correct", and
obviously more models. It should also be run in CI.

"""
import datetime
import functools
import time

import jax
import jax.numpy as jnp
import jax.scipy.stats as stats
import numpy as np
import pytest

import blackjax
from blackjax.diagnostics import effective_sample_size, potential_scale_reduction
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

    _, (states, _) = run_inference_algorithm(
        rng_key=inference_key,
        initial_state=state,
        inference_algorithm=inference_algorithm,
        num_steps=10_000,
    )

    return states


def make_horseshoe_logdensity(
    N=100, M=200, m0=10, slab_scale=3.0, slab_df=25.0, seed=42
):
    """Finnish (regularised) horseshoe sparse linear regression.

    Piironen & Vehtari (2017).  Pure JAX implementation; no TFP required.

    Parameters are sampled in unconstrained space.  Positive parameters
    (sigma, tau_tilde, c2_tilde, lambda_) use a log transform; the
    log-Jacobian correction is included in the log-density.

    Returns
    -------
    logdensity_flat : callable
        Log-density as a function of a flat 1-D array (see speed-up guide §3).
    init_flat : jnp.ndarray
        Flat initial point of shape (2 + 2*M,).
    logdensity_dict : callable
        Same log-density but accepts a dict of named arrays (6 leaves).
        Useful for demonstrating the pytree-carry overhead in lax.scan.
    init_dict : dict
        Dict initial point (unconstrained); all zeros at the constrained init
        (alpha=0, sigma=1, tau_tilde=1, c2_tilde=1, lambda_=1, beta_tilde=0).
    """
    rng = np.random.default_rng(seed)
    X = jnp.array(rng.standard_normal((N, M)), dtype=jnp.float32)
    beta0 = np.zeros(M, dtype=np.float32)
    active = rng.binomial(1, 0.05, M).astype(bool)
    beta0[active] = (rng.standard_normal(active.sum()) + 10).astype(np.float32)
    y = jnp.array(rng.normal(np.array(X) @ beta0, 1.0), dtype=jnp.float32)

    half_slab_df = float(0.5 * slab_df)
    slab_scale2 = float(slab_scale**2)
    tau0_coef = float(m0 / (M - m0)) / float(N) ** 0.5  # tau0 = tau0_coef * sigma

    def logdensity_dict(params):
        # Unconstrained inputs; positive params stored as log(x).
        alpha = params["alpha"]  # scalar, R
        log_sigma = params["sigma"]  # log(sigma > 0)
        log_tau = params["tau_tilde"]  # log(tau_tilde > 0)
        log_c2 = params["c2_tilde"]  # log(c2_tilde > 0)
        log_lam = params["lambda_"]  # log(lambda_ > 0), shape (M,)
        beta_tilde = params["beta_tilde"]  # shape (M,), R

        sigma = jnp.exp(log_sigma)
        tau_tilde = jnp.exp(log_tau)
        c2_tilde = jnp.exp(log_c2)
        lambda_ = jnp.exp(log_lam)

        tau = tau0_coef * sigma * tau_tilde
        c2 = slab_scale2 * c2_tilde
        lam_tilde = jnp.sqrt(
            c2 * jnp.square(lambda_) / (c2 + jnp.square(tau) * jnp.square(lambda_))
        )
        beta = tau * lam_tilde * beta_tilde
        mu = X @ beta + alpha

        # Log-priors (constrained space)
        lp = stats.norm.logpdf(alpha, 0.0, 2.0)
        lp += jnp.log(2.0) + stats.norm.logpdf(sigma, 0.0, 2.0)  # HalfNormal(2)
        lp += (
            jnp.log(2.0) - jnp.log(jnp.pi) - jnp.log1p(tau_tilde**2)
        )  # HalfCauchy(1)
        lp += (
            half_slab_df * jnp.log(half_slab_df)  # InvGamma(a,a)
            - jax.scipy.special.gammaln(half_slab_df)
            - (half_slab_df + 1.0) * jnp.log(c2_tilde)
            - half_slab_df / c2_tilde
        )
        lp += jnp.sum(
            jnp.log(2.0) - jnp.log(jnp.pi) - jnp.log1p(lambda_**2)
        )  # HalfCauchy(1)
        lp += jnp.sum(stats.norm.logpdf(beta_tilde, 0.0, 1.0))

        # Likelihood
        lp += jnp.sum(stats.norm.logpdf(y, mu, sigma))

        # Log-Jacobians for the log transforms (d log x / dx = 1/x → |dx/d log_x| = x)
        lp += log_sigma + log_tau + log_c2 + jnp.sum(log_lam)

        return lp

    # Unconstrained init: log(1)=0 for positive params, 0 for real params
    init_dict = {
        "alpha": jnp.array(0.0),
        "sigma": jnp.array(0.0),
        "tau_tilde": jnp.array(0.0),
        "c2_tilde": jnp.array(0.0),
        "lambda_": jnp.zeros(M),
        "beta_tilde": jnp.zeros(M),
    }

    init_flat, unflatten = jax.flatten_util.ravel_pytree(init_dict)

    def logdensity_flat(flat):
        return logdensity_dict(unflatten(flat))

    return logdensity_flat, init_flat, logdensity_dict, init_dict


def _today_key():
    seed = int(datetime.date.today().strftime("%Y%m%d"))
    return jax.random.key(seed)


def _positions_2d(states):
    """Flatten sampled pytree positions to (num_samples, n_params)."""
    leaves = jax.tree.leaves(states.position)
    n = leaves[0].shape[0]
    return jnp.concatenate([v.reshape(n, -1) for v in leaves], axis=-1)


def _split_rhat(pos_2d):
    """Split-R̂: treat the two halves of a single chain as separate chains."""
    half = pos_2d.shape[0] // 2
    return potential_scale_reduction(
        jnp.stack([pos_2d[:half], pos_2d[half : half * 2]])
    )


@pytest.mark.benchmark
def test_horseshoe_nuts_flat_vs_dict(benchmark):
    """Compare flat-array vs dict parameterisation: timing, ESS/s, and per-parameter-group
    split-R̂.  Warmup runs once on the flat logdensity; adapted parameters are shared so
    timing differences reflect only pytree-carry overhead.

    Uses benchmark.pedantic(..., iterations=1, rounds=1) so the expensive sampling
    loop runs exactly once regardless of pytest-benchmark calibration settings.
    The benchmark timer records the flat-array sampling time for regression tracking.
    """
    logdensity_flat, init_flat, logdensity_dict, init_dict = make_horseshoe_logdensity()

    param_groups = ["alpha", "sigma", "tau_tilde", "c2_tilde", "lambda_", "beta_tilde"]
    param_sizes = [1, 1, 1, 1, 200, 200]
    param_ends = list(np.cumsum(param_sizes))
    param_starts = [0] + param_ends[:-1]

    # Warmup once outside the benchmark; share parameters across both sampling runs
    warmup_key, inference_key = jax.random.split(_today_key())
    (warmup_state, parameters), _ = blackjax.window_adaptation(
        blackjax.nuts, logdensity_flat
    ).run(warmup_key, init_flat, 1000)
    jax.block_until_ready(parameters)
    print(
        f"\n  Warmup:  step_size={float(parameters['step_size']):.5f}"  # noqa: E241,E231
        f"  IMM diag mean={float(jnp.mean(jnp.diag(parameters['inverse_mass_matrix']))):.4f}\n"  # noqa: E231
    )

    ip_flat = warmup_state.position
    ip_dict = jax.flatten_util.ravel_pytree(init_dict)[1](ip_flat)

    def _sample(logdensity_fn, init_pos):
        algo = blackjax.nuts(logdensity_fn, **parameters)
        _, (states, _) = run_inference_algorithm(
            rng_key=inference_key,
            initial_state=algo.init(init_pos),
            inference_algorithm=algo,
            num_steps=1000,
        )
        jax.block_until_ready(states)
        return states

    # benchmark records the flat sampling time; runs exactly once
    t0 = time.perf_counter()
    flat_states = benchmark.pedantic(
        _sample, args=(logdensity_flat, ip_flat), iterations=1, rounds=1
    )
    t_flat = (
        benchmark.stats["mean"]
        if benchmark.stats is not None
        else time.perf_counter() - t0
    )
    t0 = time.perf_counter()
    dict_states = _sample(logdensity_dict, ip_dict)
    t_dict = time.perf_counter() - t0

    results = {}
    for label, states, t_sample in [
        ("flat (1 leaf)", flat_states, t_flat),
        ("dict (6 leaves)", dict_states, t_dict),
    ]:
        pos_2d = _positions_2d(states)
        ess = effective_sample_size(pos_2d[None])
        rhat = _split_rhat(pos_2d)
        group_stats = {
            name: dict(
                min_ess=float(jnp.min(ess[s:e])),
                max_rhat=float(jnp.max(rhat[s:e])),
            )
            for name, s, e in zip(param_groups, param_starts, param_ends)
        }
        min_ess = min(v["min_ess"] for v in group_stats.values())
        results[label] = dict(
            t_sample=t_sample,
            min_ess=min_ess,
            ess_per_s=min_ess / t_sample if t_sample else float("nan"),
            group_stats=group_stats,
        )

    r_flat = results["flat (1 leaf)"]
    r_dict = results["dict (6 leaves)"]

    print(
        "  Model: Finnish horseshoe  N=100 M=200  1 chain  1000 samples  (shared warmup)"
    )
    print()
    hdr = (
        f"  {'Metric':<28} {'flat (1 leaf)':>16} {'dict (6 leaves)':>16}"  # noqa: E231
    )
    print(hdr)
    print("  " + "-" * 62)
    for key, label, fmt in [
        ("t_sample", "sample time (s)", ".2f"),
        ("min_ess", "min ESS", ".1f"),
        ("ess_per_s", "min ESS/s", ".1f"),
    ]:
        row = (
            f"  {label:<28} {r_flat[key]:>16{fmt}} {r_dict[key]:>16{fmt}}"  # noqa: E231
        )
        print(row)
    speedup = r_dict["t_sample"] / r_flat["t_sample"]
    print("  " + "-" * 62)
    print(f"  {'sample speedup (dict/flat)':<28} {speedup:>16.2f}x")  # noqa: E231

    print()
    print(
        f"  {'Parameter':<14} {'size':>5}"  # noqa: E231
        f"  {'ESS flat':>9} {'ESS dict':>9}"  # noqa: E231
        f"  {'Rhat flat':>10} {'Rhat dict':>10}"  # noqa: E231
    )
    print("  " + "-" * 64)
    for name, size in zip(param_groups, param_sizes):
        sf = r_flat["group_stats"][name]
        sd = r_dict["group_stats"][name]
        print(
            f"  {name:<14} {size:>5}"  # noqa: E231
            f"  {sf['min_ess']:>9.1f} {sd['min_ess']:>9.1f}"  # noqa: E231
            f"  {sf['max_rhat']:>10.3f} {sd['max_rhat']:>10.3f}"  # noqa: E231
        )
    print()

    assert (
        r_flat["min_ess"] > 10
    ), f"flat min ESS suspiciously low: {r_flat['min_ess']:.1f}"  # noqa: E231
    assert (
        r_dict["min_ess"] > 10
    ), f"dict min ESS suspiciously low: {r_dict['min_ess']:.1f}"  # noqa: E231


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
