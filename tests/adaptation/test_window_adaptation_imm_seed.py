# Copyright 2024- The Blackjax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Surface/validation tests for window_adaptation's initial_inverse_mass_matrix kwarg.

Behavioral/numeric tests have been migrated to target staged_adaptation instead.
This file retains only the surface-level validation tests (error raising, kwarg
validation) that exercise the frozen shim surface of window_adaptation.
"""
import jax.numpy as jnp
import pytest

import blackjax
from tests.fixtures import std_normal_logdensity

# Anisotropic 3-D Gaussian target used by validation tests.
DIM = 3
TARGET_STD = jnp.array([0.1, 1.0, 10.0])


def logdensity_fn(x):
    return std_normal_logdensity(x, scale=TARGET_STD)


# ---------------------------------------------------------------------------
# Surface/validation tests: error raising, kwarg validation
# ---------------------------------------------------------------------------


def test_shape_mismatch_2d_with_diagonal():
    """Providing a 2-D array with is_mass_matrix_diagonal=True raises ValueError."""
    with pytest.raises(ValueError, match="ndim == 1"):
        blackjax.window_adaptation(
            blackjax.nuts,
            logdensity_fn,
            is_mass_matrix_diagonal=True,
            initial_inverse_mass_matrix=jnp.eye(DIM),
        )


def test_shape_mismatch_1d_with_dense():
    """Providing a 1-D array with is_mass_matrix_diagonal=False raises ValueError."""
    with pytest.raises(ValueError, match="2-D square"):
        blackjax.window_adaptation(
            blackjax.nuts,
            logdensity_fn,
            is_mass_matrix_diagonal=False,
            initial_inverse_mass_matrix=jnp.ones(DIM),
        )


def test_shape_mismatch_non_square_with_dense():
    """Providing a non-square 2-D array with is_mass_matrix_diagonal=False raises ValueError."""
    with pytest.raises(ValueError, match="2-D square"):
        blackjax.window_adaptation(
            blackjax.nuts,
            logdensity_fn,
            is_mass_matrix_diagonal=False,
            initial_inverse_mass_matrix=jnp.ones((DIM, DIM + 1)),
        )


def test_imm_shrinkage_negative_raises():
    """Negative imm_shrinkage_to_previous raises ValueError at construction time."""
    with pytest.raises(ValueError, match="imm_shrinkage_to_previous must be >= 0.0"):
        blackjax.window_adaptation(
            blackjax.nuts,
            logdensity_fn,
            is_mass_matrix_diagonal=True,
            imm_shrinkage_to_previous=-1.0,
        )
