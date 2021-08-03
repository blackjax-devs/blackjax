"""Test the solving functions"""

import jax.numpy as jnp
import numpy as np
import pytest

import blackjax.inference.smc.solver as solver

solving_methods_to_test = [solver.dichotomy]


def positive_function(x):
    return jnp.square(x) + 0.5


def negative_function(x):
    return jnp.square(x) - 1.5


def increasing_function(x):
    return x - 0.5


def decreasing_function(x, y):
    return jnp.cos(x) - y


functions_to_test = [
    (positive_function, 1.0),
    (negative_function, np.nan),
    (increasing_function, 1.0),
    (lambda x: decreasing_function(x, 0.75), np.arccos(0.75)),  # in range
    (lambda x: decreasing_function(x, 0.5), 1.0),  # out of range
]


@pytest.mark.parametrize("fun_to_test,expected_res", functions_to_test)
@pytest.mark.parametrize("solving_method", solving_methods_to_test)
def test_resampling_methods(fun_to_test, expected_res, solving_method):
    res = solving_method(fun_to_test, 0.5, 0.0, 1.0)
    np.testing.assert_allclose(res, expected_res, atol=1e-3, equal_nan=True)
