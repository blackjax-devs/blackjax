"""Independent references and failure contracts for rejection sampling."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from blackjax import rejection_sampling


@pytest.fixture
def enable_x64():
    previous = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", previous)


@pytest.mark.parametrize("typed_key", [False, True])
@pytest.mark.parametrize("compiled", [False, True])
def test_identical_target_accepts_first_proposal(typed_key, compiled):
    key = jax.random.key(12) if typed_key else jax.random.PRNGKey(12)
    sampler = lambda key: {"x": jax.random.normal(key, (2,)), "label": jnp.array(3)}
    kernel = rejection_sampling.build_kernel(
        lambda x: -jnp.sum(x["x"] ** 2) / 2,
        sampler,
        lambda x: -jnp.sum(x["x"] ** 2) / 2,
        0.0,
        max_steps=1,
    )
    if compiled:
        kernel = jax.jit(kernel)
    sample, info = kernel(key)
    expected = sampler(jax.random.split(key, 3)[1])
    np.testing.assert_array_equal(sample["x"], expected["x"])
    assert sample["label"] == 3
    assert info.is_accepted
    assert info.is_bound_valid
    assert info.num_proposals == 1


@pytest.mark.parametrize("max_steps", [1, 3])
def test_exhaustion_reports_last_proposal(max_steps):
    kernel = rejection_sampling.build_kernel(
        lambda x: -jnp.inf,
        jax.random.normal,
        lambda x: 0.0,
        0.0,
        max_steps=max_steps,
    )
    key = jax.random.key(1)
    sample, info = jax.jit(kernel)(key)
    for _ in range(max_steps):
        key, proposal_key, _ = jax.random.split(key, 3)
        expected = jax.random.normal(proposal_key)
    np.testing.assert_array_equal(sample, expected)
    assert not info.is_accepted
    assert info.is_bound_valid
    assert info.num_proposals == max_steps


@pytest.mark.parametrize(
    "log_f,log_q,log_bound",
    [(1.0, 0.0, 0.0), (jnp.nan, 0.0, 0.0), (0.0, -jnp.inf, 0.0), (0.0, 0.0, jnp.inf)],
)
def test_invalid_density_ratio_stops(log_f, log_q, log_bound):
    kernel = rejection_sampling.build_kernel(
        lambda x: log_f,
        lambda key: jnp.array(1.0),
        lambda x: log_q,
        log_bound,
        max_steps=4,
    )
    _, info = jax.jit(kernel)(jax.random.key(0))
    assert not info.is_accepted
    assert not info.is_bound_valid
    assert info.num_proposals == 1


def test_zero_uniform_does_not_accept_zero_target(monkeypatch):
    monkeypatch.setattr(
        jax.random, "uniform", lambda key, dtype: jnp.array(0.0, dtype=dtype)
    )
    kernel = rejection_sampling.build_kernel(
        lambda x: -jnp.inf,
        lambda key: jnp.array(1.0),
        lambda x: 0.0,
        0.0,
        max_steps=2,
    )
    _, info = kernel(jax.random.key(0))
    assert not info.is_accepted
    assert info.num_proposals == 2


@pytest.mark.parametrize("max_steps", [0, -1])
def test_nonpositive_limit(max_steps):
    with pytest.raises(ValueError, match="max_steps must be positive"):
        rejection_sampling.build_kernel(
            lambda x: 0.0, jax.random.normal, lambda x: 0.0, 0.0, max_steps=max_steps
        )


def test_proposal_budget_cannot_overflow_counter():
    with pytest.raises(ValueError, match="max_steps must not exceed"):
        rejection_sampling.build_kernel(
            lambda x: 0.0,
            jax.random.normal,
            lambda x: 0.0,
            0.0,
            max_steps=2**31,
        )


def test_scalar_density_contract():
    with pytest.raises(ValueError, match="log_bound must be a scalar"):
        rejection_sampling.build_kernel(
            lambda x: 0.0, jax.random.normal, lambda x: 0.0, jnp.ones(2)
        )
    kernel = rejection_sampling.build_kernel(
        lambda x: jnp.ones(2), jax.random.normal, lambda x: 0.0, 0.0
    )
    with pytest.raises(ValueError, match="log densities must be scalars"):
        kernel(jax.random.key(0))


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_beta_target_against_analytic_cdf(dtype, enable_x64):
    # f(x)=2x, q(x)=1 on [0,1], with M=2. The target CDF is x**2.
    kernel = rejection_sampling.build_kernel(
        lambda x: jnp.log(2 * x),
        lambda key: jax.random.uniform(key, dtype=dtype),
        lambda x: jnp.array(0.0, dtype=dtype),
        jnp.log(jnp.array(2.0, dtype=dtype)),
        max_steps=128,
    )
    samples, info = jax.jit(jax.vmap(kernel))(jax.random.split(jax.random.key(0), 8192))
    assert jnp.all(info.is_accepted)
    assert jnp.all(info.is_bound_valid)
    assert samples.dtype == dtype
    for x in (0.25, 0.5, 0.75):
        expected = x**2
        std_error = np.sqrt(expected * (1 - expected) / samples.size)
        assert abs(np.mean(np.asarray(samples) <= x) - expected) < 8 * std_error
    assert abs(np.mean(samples) - 2 / 3) < 0.02
    assert abs(np.mean(info.num_proposals) - 2) < 0.1


def test_normal_target_against_analytic_moments():
    # N(0,1) target, N(0,2) proposal (standard deviation 2), M=2.
    log_norm = 0.5 * jnp.log(2 * jnp.pi)
    kernel = rejection_sampling.build_kernel(
        lambda x: -0.5 * x**2 - log_norm,
        lambda key: 2 * jax.random.normal(key),
        lambda x: -0.125 * x**2 - log_norm - jnp.log(2.0),
        jnp.log(2.0),
        max_steps=128,
    )
    samples, info = jax.jit(jax.vmap(kernel))(jax.random.split(jax.random.key(4), 8192))
    assert jnp.all(info.is_accepted)
    assert jnp.all(info.is_bound_valid)
    assert abs(np.mean(samples)) < 0.08
    assert abs(np.var(samples) - 1) < 0.13


def test_draw_matches_independent_scalar_rejection_loop():
    key = jax.random.key(23)
    kernel = rejection_sampling.build_kernel(
        lambda x: jnp.log(2 * x), jax.random.uniform, lambda x: 0.0, jnp.log(2.0)
    )
    sample, info = kernel(key)
    # Use probability arithmetic instead of the kernel's log comparison.
    for count in range(1, 1001):
        key, proposal_key, accept_key = jax.random.split(key, 3)
        candidate = jax.random.uniform(proposal_key)
        uniform = jax.random.uniform(accept_key)
        if uniform < candidate:
            break
    np.testing.assert_array_equal(sample, candidate)
    assert info.num_proposals == count
    assert info.is_accepted


def test_batched_failure_is_reported_per_draw():
    kernel = rejection_sampling.build_kernel(
        lambda x: jnp.where(x > 0, -0.5 * x**2, -jnp.inf),
        jax.random.normal,
        lambda x: -0.5 * x**2,
        0.0,
        max_steps=1,
    )
    samples, info = jax.jit(jax.vmap(kernel))(jax.random.split(jax.random.key(32), 64))
    np.testing.assert_array_equal(info.is_accepted, samples > 0)
    assert jnp.any(info.is_accepted)
    assert jnp.any(~info.is_accepted)
    assert jnp.all(info.is_bound_valid)
    assert jnp.all(info.num_proposals == 1)
