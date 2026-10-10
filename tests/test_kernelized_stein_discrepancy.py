"""Analytic reference checks for the squared KSD V-statistic."""

import functools
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from blackjax.diagnostics import imq_kernel, kernelized_stein_discrepancy
from blackjax.vi.svgd import rbf_kernel


def rbf_reference(samples, scores, length_scale):
    # Closed-form derivatives of exp(-||x-y||²/length_scale), independently
    # evaluated in NumPy rather than through the implementation's autodiff.
    result = 0.0
    d = samples.shape[1]
    for x, score_x in zip(samples, scores):
        for y, score_y in zip(samples, scores):
            delta = x - y
            distance_sq = np.dot(delta, delta)
            value = np.exp(-distance_sq / length_scale)
            result += value * (
                np.dot(score_x, score_y)
                + 2 * np.dot(score_x - score_y, delta) / length_scale
                + 2 * d / length_scale
                - 4 * distance_sq / length_scale**2
            )
    return result / len(samples) ** 2


@pytest.mark.parametrize("length_scale", [0.5, 2.0])
def test_matches_closed_form_rbf(length_scale):
    samples = np.array([[-1.0, 0.5], [0.25, -0.75], [1.5, 2.0]])
    mean = np.array([0.5, -0.25])
    precision = np.array([[2.0, 0.25], [0.25, 1.0]])
    score = lambda x: -jnp.asarray(precision) @ (x - mean)
    kernel = functools.partial(rbf_kernel, length_scale=length_scale)
    expected = rbf_reference(samples, -(samples - mean) @ precision, length_scale)
    actual = jax.jit(
        functools.partial(
            kernelized_stein_discrepancy, grad_logdensity_fn=score, kernel=kernel
        )
    )(samples)
    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-6)


def test_diagonal_is_included_and_result_is_squared():
    samples = jnp.zeros((1, 3))
    result = kernelized_stein_discrepancy(
        samples, lambda x: -x, functools.partial(rbf_kernel, length_scale=2.0)
    )
    # At the Gaussian mode, the only contribution is trace(d_x d_y k)=2d/L.
    np.testing.assert_allclose(result, 3.0)


def test_permutation_and_normalization_constant():
    samples = jnp.array([[-1.0], [0.5], [2.0]])
    logdensity = lambda x: -jnp.sum(x**2) / 2
    score = jax.grad(logdensity)
    result = kernelized_stein_discrepancy(samples, score, rbf_kernel)
    other = kernelized_stein_discrepancy(
        samples[::-1], jax.grad(lambda x: logdensity(x) + 13.0), rbf_kernel
    )
    np.testing.assert_allclose(result, other, rtol=2e-6)


def test_sample_gradient_matches_finite_difference():
    samples = np.array([[-0.7], [0.4], [1.1]])
    actual = jax.grad(
        lambda x: kernelized_stein_discrepancy(x, lambda y: -y, rbf_kernel)
    )(jnp.asarray(samples))
    eps = 1e-3
    expected = np.empty_like(samples)
    for index in np.ndindex(samples.shape):
        plus, minus = samples.copy(), samples.copy()
        plus[index] += eps
        minus[index] -= eps
        expected[index] = (
            rbf_reference(plus, -plus, 1.0) - rbf_reference(minus, -minus, 1.0)
        ) / (2 * eps)
    np.testing.assert_allclose(actual, expected, rtol=2e-4, atol=2e-4)


def test_pytree_samples_match_flat_array():
    """dict/tuple/NamedTuple samples give the same KSD as the equivalent (n, d) array."""

    class Position(NamedTuple):
        a: jnp.ndarray
        b: jnp.ndarray

    samples = np.array([[-1.0, 0.5], [0.25, -0.75], [1.5, 2.0]])
    score = lambda x: -x
    tree_score = lambda t: jax.tree.map(lambda v: -v, t)
    a, b = jnp.asarray(samples[:, 0]), jnp.asarray(samples[:, 1])
    pytree_variants = ({"a": a, "b": b}, (a, b), Position(a, b))

    for statistic in ("v", "u"):
        expected = kernelized_stein_discrepancy(
            samples, score, rbf_kernel, statistic=statistic
        )
        for pytree_samples in pytree_variants:
            actual = kernelized_stein_discrepancy(
                pytree_samples, tree_score, rbf_kernel, statistic=statistic
            )
            np.testing.assert_allclose(actual, expected, rtol=2e-6)


def test_1d_samples_match_column_vector():
    samples_1d = jnp.array([-1.0, 0.5, 2.0])
    samples_2d = samples_1d[:, None]
    score = lambda x: -x

    for statistic in ("v", "u"):
        result_1d = kernelized_stein_discrepancy(
            samples_1d, score, rbf_kernel, statistic=statistic
        )
        result_2d = kernelized_stein_discrepancy(
            samples_2d, score, rbf_kernel, statistic=statistic
        )
        np.testing.assert_allclose(result_1d, result_2d, rtol=2e-6)


def test_rejects_invalid_statistic():
    with pytest.raises(ValueError, match="statistic"):
        kernelized_stein_discrepancy(
            jnp.ones((2, 2)), lambda x: -x, rbf_kernel, statistic="invalid"
        )


def test_linear_kernel_matches_stein_feature_norm():
    samples = np.array([[-1.0, 0.5], [0.25, -0.75], [1.5, 2.0]])
    scores = -samples
    # For k(x,y)=1+x^T y the Stein feature is [score, score*x^T+I].
    features = np.array(
        [np.outer(score, x) + np.eye(2) for x, score in zip(samples, scores)]
    )
    expected = np.sum(scores.mean(axis=0) ** 2) + np.sum(features.mean(axis=0) ** 2)
    result = kernelized_stein_discrepancy(
        jnp.asarray(samples), lambda x: -x, lambda x, y: 1 + jnp.dot(x, y)
    )
    np.testing.assert_allclose(result, expected, rtol=2e-6)


def test_u_statistic_matches_off_diagonal_reference():
    samples = np.array([[-1.0, 0.5], [0.25, -0.75], [1.5, 2.0]])
    n, d = samples.shape
    total = rbf_reference(samples, -samples, 1.0) * n**2
    diagonal = np.sum(samples**2) + 2 * d * n
    expected = (total - diagonal) / (n * (n - 1))
    result = kernelized_stein_discrepancy(
        samples, lambda x: -x, rbf_kernel, statistic="u"
    )
    np.testing.assert_allclose(result, expected, rtol=2e-6, atol=2e-6)


def test_imq_statistical_behavior():
    # Fixed common draws isolate location/scale changes rather than comparing
    # independent Monte Carlo noise. This is not a general convergence test.
    samples = jax.random.normal(jax.random.key(20261003), (400, 4))
    evaluate = jax.jit(
        lambda x: kernelized_stein_discrepancy(
            x, lambda y: -y, imq_kernel, statistic="u"
        )
    )
    exact = evaluate(samples)
    assert abs(exact) < 0.02
    assert evaluate(samples + 1.5) > exact + 0.1
    assert evaluate(samples * 0.5) > exact + 0.1


def test_sequential_rows_and_imq_formula():
    x, y = jnp.array([1.0, 2.0]), jnp.array([-1.0, 1.0])
    np.testing.assert_allclose(imq_kernel(x, y), 6**-0.5)
    samples = jnp.zeros((3, 2))
    jaxpr = jax.make_jaxpr(
        lambda x: kernelized_stein_discrepancy(x, lambda y: -y, imq_kernel)
    )(samples)
    assert any(e.primitive.name == "scan" for e in jaxpr.jaxpr.eqns)
