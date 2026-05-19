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
"""Tests for window_adaptation's initial_inverse_mass_matrix kwarg (P0)."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import blackjax
from tests.fixtures import std_normal_logdensity

# Anisotropic 3-D Gaussian target — wide per-dim variance range makes the
# IMM-seed effect on early-warmup step-size adaptation measurable.
DIM = 3
TARGET_STD = jnp.array([0.1, 1.0, 10.0])


def logdensity_fn(x):
    return std_normal_logdensity(x, scale=TARGET_STD)


def _run_warmup(rng_key, imm=None, dense=False, num_steps=200):
    """Helper: run window_adaptation and return (step_size, inverse_mass_matrix)."""
    warmup = blackjax.window_adaptation(
        blackjax.nuts,
        logdensity_fn,
        is_mass_matrix_diagonal=not dense,
        initial_inverse_mass_matrix=imm,
    )
    init_pos = jnp.zeros(DIM)
    (state, params), _ = warmup.run(rng_key, init_pos, num_steps=num_steps)
    return params["step_size"], params["inverse_mass_matrix"]


# ---------------------------------------------------------------------------
# 1. Backward compatibility: no initial_inverse_mass_matrix
# ---------------------------------------------------------------------------


def test_backward_compat_no_imm():
    """window_adaptation with no initial_inverse_mass_matrix runs without error."""
    rng_key = jax.random.key(0)
    step_size, imm = _run_warmup(rng_key)
    assert step_size > 0
    assert imm.shape == (DIM,)
    assert jnp.all(imm > 0)


# ---------------------------------------------------------------------------
# 2. Diagonal seed: the warmup runs and the seed IMM is accepted
# ---------------------------------------------------------------------------


def test_diagonal_seed_runs():
    """Diagonal seed IMM does not crash and returns well-shaped outputs."""
    rng_key = jax.random.key(1)
    seed_imm = jnp.array([0.1, 1.0, 10.0])  # matches true covariance diagonal
    step_size, imm = _run_warmup(rng_key, imm=seed_imm)
    assert step_size > 0
    assert imm.shape == (DIM,)


def test_diagonal_seed_differs_from_default():
    """First-window step-size adaptation differs when seeded vs default IMM.

    The seed IMM is very different from identity, so the adapted step sizes
    after a short warmup should differ between the two conditions.
    """
    rng_key = jax.random.key(42)
    # Use a very short warmup so the seed has more influence
    step_default, _ = _run_warmup(rng_key, imm=None, num_steps=100)
    # Extreme seed that strongly scales the geometry
    extreme_seed = jnp.array([100.0, 100.0, 100.0])
    step_seeded, _ = _run_warmup(rng_key, imm=extreme_seed, num_steps=100)
    # They should differ — the seed changes the step size adaptation
    assert not jnp.allclose(step_default, step_seeded, atol=1e-6)


# ---------------------------------------------------------------------------
# 3. Dense seed
# ---------------------------------------------------------------------------


def test_dense_seed_runs():
    """Dense seed IMM (is_mass_matrix_diagonal=False) runs without error."""
    rng_key = jax.random.key(2)
    # Use a diagonal PD matrix as the dense seed
    seed_imm = jnp.diag(jnp.array([0.1, 1.0, 10.0]))
    step_size, imm = _run_warmup(rng_key, imm=seed_imm, dense=True)
    assert step_size > 0
    assert imm.shape == (DIM, DIM)


# ---------------------------------------------------------------------------
# 4. Shape mismatch raises ValueError (validated BEFORE JIT trace)
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


# ---------------------------------------------------------------------------
# 5. Welford convergence: seed doesn't poison final IMM
# ---------------------------------------------------------------------------


def test_welford_convergence_seed_does_not_poison():
    """Final adapted IMM should be close regardless of seed when warmup is long.

    With enough steps, Welford's algorithm overwrites the seed.  We verify that
    both the default-seeded and the truth-seeded adaptations end up with similar
    final IMMs on a 3-D Gaussian where the true diagonal is known.
    """
    rng_key = jax.random.key(99)
    _, imm_default = _run_warmup(rng_key, imm=None, num_steps=1000)
    # Seed with the true covariance diagonal (variance = std^2)
    seed_imm = TARGET_STD**2
    _, imm_seeded = _run_warmup(rng_key, imm=seed_imm, num_steps=1000)

    # Both should be positive
    assert jnp.all(imm_default > 0)
    assert jnp.all(imm_seeded > 0)

    # With 1000 steps the Welford estimator dominates; the two IMMs should be
    # in the same ballpark (within 50% of each other)
    ratio = imm_seeded / imm_default
    np.testing.assert_allclose(ratio, jnp.ones_like(ratio), atol=0.5)
