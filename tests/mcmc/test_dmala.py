"""Finite-state references for the binary Metropolis-adjusted Langevin kernel."""

import itertools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import blackjax
from blackjax.mcmc import dmala


@pytest.fixture
def enable_x64():
    previous = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", previous)


def _logdensity(x):
    return (
        jnp.array([0.4, -0.7, 0.2]) @ x
        + 0.8 * x[0] * x[1]
        - 0.55 * x[1] * x[2]
        + 0.35 * x[0] * x[1] * x[2]
        + 0.2 * jnp.sin(x[0] + 2 * x[2])
    )


def _reference(alpha):
    """Enumerate equation (1), using NumPy and analytic derivatives."""
    states = np.array(list(itertools.product([0.0, 1.0], repeat=3)))
    x0, x1, x2 = states.T
    values = (
        states @ np.array([0.4, -0.7, 0.2])
        + 0.8 * x0 * x1
        - 0.55 * x1 * x2
        + 0.35 * x0 * x1 * x2
        + 0.2 * np.sin(x0 + 2 * x2)
    )
    gradients = np.column_stack(
        [
            0.4 + 0.8 * x1 + 0.35 * x1 * x2 + 0.2 * np.cos(x0 + 2 * x2),
            -0.7 + 0.8 * x0 - 0.55 * x2 + 0.35 * x0 * x2,
            0.2 - 0.55 * x1 + 0.35 * x0 * x1 + 0.4 * np.cos(x0 + 2 * x2),
        ]
    )
    # Normalise the Gaussian-form weights over all eight discrete states;
    # do not reuse the implementation's Bernoulli/log-sigmoid arithmetic.
    means = states + alpha * gradients / 2
    weights = -np.sum((states[None, :, :] - means[:, None, :]) ** 2, axis=-1) / (
        2 * alpha
    )
    q = np.exp(weights - weights.max(axis=1, keepdims=True))
    q /= q.sum(axis=1, keepdims=True)
    target = np.exp(values - values.max())
    target /= target.sum()
    acceptance = np.minimum(1, target[None, :] * q.T / (target[:, None] * q))
    transition = q * acceptance
    transition[np.diag_indices(8)] += 1 - transition.sum(axis=1)
    return states, target, q, acceptance, transition


@pytest.mark.parametrize("alpha", [0.2, 0.7, 4.0])
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_proposal_matches_enumerated_gaussian_weights(alpha, dtype, enable_x64):
    positions, target, expected, _, _ = _reference(alpha)
    states = jax.vmap(lambda x: dmala.init(x, _logdensity))(
        jnp.asarray(positions, dtype=dtype)
    )
    actual = jnp.exp(
        jax.jit(
            jax.vmap(
                jax.vmap(lambda x, y: dmala._proposal_logprob(x, y, alpha), (None, 0)),
                (0, None),
            )
        )(states, states)
    )
    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=1e-7)
    np.testing.assert_allclose(actual.sum(axis=1), 1, atol=1e-6)
    # Exact transition using these independently validated proposal weights.
    q = np.asarray(actual)
    p = q * np.minimum(1, target[None, :] * q.T / (target[:, None] * q))
    p[np.diag_indices(8)] += 1 - p.sum(axis=1)
    np.testing.assert_allclose(target[:, None] * p, (target[:, None] * p).T, atol=1e-8)
    np.testing.assert_allclose(target @ p, target, atol=1e-8)


@pytest.mark.parametrize("alpha", [0.2, 0.7, 4.0])
def test_actual_kernel_matches_finite_transition_matrix(alpha):
    positions, _, _, acceptance, transition = _reference(alpha)
    algorithm = blackjax.dmala(_logdensity, alpha)
    count = 4096
    keys = jax.random.split(jax.random.key(19), 8 * count).reshape((8, count))
    initial = jax.vmap(algorithm.init)(jnp.asarray(positions))
    samples, info = jax.jit(jax.vmap(jax.vmap(algorithm.step, (0, None)), (0, 0)))(
        keys, initial
    )
    encode = lambda x: (x @ jnp.array([4, 2, 1])).astype(jnp.int32)
    proposed_indices = np.asarray(encode(info.proposal.position))
    expected_acceptance = acceptance[np.arange(8)[:, None], proposed_indices]
    np.testing.assert_allclose(
        info.acceptance_rate, expected_acceptance, rtol=2e-6, atol=1e-7
    )
    indices = np.asarray(encode(samples.position))
    observed = np.stack([np.bincount(row, minlength=8) / count for row in indices])
    standard_error = np.sqrt(transition * (1 - transition) / count)
    assert np.all(np.abs(observed - transition) < 7 * standard_error + 0.003)
    assert jnp.any(info.is_accepted)
    assert jnp.any(~info.is_accepted)
    np.testing.assert_allclose(
        samples.logdensity, jax.vmap(jax.vmap(_logdensity))(samples.position), atol=1e-6
    )
    np.testing.assert_allclose(
        samples.logdensity_grad,
        jax.vmap(jax.vmap(jax.grad(_logdensity)))(samples.position),
        atol=1e-6,
    )


@pytest.mark.parametrize("typed_key", [False, True])
@pytest.mark.parametrize("compiled", [False, True])
def test_pytree_binary_state_and_key_api(typed_key, compiled):
    fn = lambda x: 0.3 * jnp.sum(x["vector"]) - 0.8 * x["scalar"]
    algorithm = blackjax.dmala(fn, 0.6)
    initial = {"vector": jnp.array([0, 1], dtype=jnp.int32), "scalar": False}
    initialise = jax.jit(algorithm.init) if compiled else algorithm.init
    state = initialise(initial)
    step = jax.jit(algorithm.step) if compiled else algorithm.step
    key = jax.random.key(4) if typed_key else jax.random.PRNGKey(4)
    new_state, info = step(key, state)
    for before, after in zip(
        jax.tree.leaves(state.position), jax.tree.leaves(new_state.position)
    ):
        assert after.shape == before.shape
        assert after.dtype == before.dtype == jnp.float32
        assert jnp.all((after == 0) | (after == 1))
    assert jnp.isfinite(info.acceptance_rate)
    assert 0 <= info.acceptance_rate <= 1
    assert info.proposal.position.keys() == initial.keys()


def test_extreme_finite_logits_keep_transition_information_finite():
    fn = lambda x: 1000 * jnp.sum(x)
    algorithm = blackjax.dmala(fn, 1.0)
    state = algorithm.init(jnp.zeros(3))
    new_state, info = jax.jit(algorithm.step)(jax.random.key(2), state)
    assert jnp.isfinite(info.acceptance_rate)
    assert jnp.all(jnp.isfinite(new_state.position))
    assert jnp.isfinite(new_state.logdensity)
    assert jnp.all(jnp.isfinite(new_state.logdensity_grad))


def test_low_level_kernel_and_traced_step_size_match_top_level():
    key = jax.random.key(1)
    algorithm = blackjax.dmala(_logdensity, 0.7)
    state = algorithm.init(jnp.zeros(3))
    kernel = blackjax.dmala.build_kernel()
    direct = jax.jit(lambda alpha: kernel(key, state, _logdensity, alpha))(0.7)
    expected = algorithm.step(key, state)
    for actual, reference in zip(jax.tree.leaves(direct), jax.tree.leaves(expected)):
        np.testing.assert_allclose(actual, reference, rtol=2e-6, atol=1e-7)
