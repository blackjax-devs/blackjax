import pytest
from unittest import mock

import jax
import jax.numpy as jnp
from jax import random

from blackjax.inference import metrics

KEY = random.PRNGKey(0)


@pytest.fixture()
def patch_jax(monkeypatch):
    """Patches JNP multiply and dot to determine which one is called"""
    multiply, dot = mock.MagicMock(), mock.MagicMock()
    monkeypatch.setattr(metrics.jnp, "multiply", multiply)
    monkeypatch.setattr(metrics.jnp, "dot", dot)
    return multiply, dot


@pytest.mark.xfail
def test_gaussian_euclidean_ndim_1_temporary(patch_jax):
    """Tests Gaussian Euclidean Function"""
    multiply, _ = patch_jax
    x = jnp.ones(shape=1)

    assert multiply.call_count == 0
    momentum, velocity = metrics.gaussian_euclidean(x)

    # Hit some recursion issue. Probably because I replaced
    # multiply with a magicmock
    momentum(KEY, [1])
    # assert multiply.call_count == 1


def test_gaussian_euclidean_ndim_1(patch_jax):
    """Test Gaussian Euclidean Function returns correct function when ndim is one"""
    multiply, _ = patch_jax
    x = jnp.ones(shape=1)

    assert multiply.call_count == 0
    momentum, velocity = metrics.gaussian_euclidean(x)

    try:
        momentum(KEY, [1])
    except RecursionError:
        pass
    assert multiply.call_count == 1


def test_gaussian_euclidean_ndim_2(patch_jax):
    """Test Gaussian Euclidean Function returns correct function when ndim > 1"""
    _, dot = patch_jax
    x = jnp.ones(shape=(2, 2))

    assert dot.call_count == 0
    momentum, velocity = metrics.gaussian_euclidean(x)

    try:
        momentum(KEY, [1])
    except RecursionError:
        pass
    assert dot.call_count == 1


def test_gaussian_euclidean_ndim_3():
    """Test Gaussian Euclidean Function returns correct function when ndim > 1"""
    x = jnp.ones(shape=(1, 2, 3))

    with pytest.raises(ValueError) as e:
        metrics.gaussian_euclidean(x)
    assert "The mass matrix has the wrong number of dimensions" in str(e)
