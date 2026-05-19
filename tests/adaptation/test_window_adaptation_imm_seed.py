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


def _run_warmup(
    rng_key, imm=None, dense=False, num_steps=200, imm_shrinkage_to_previous=0.0
):
    """Helper: run window_adaptation and return (step_size, inverse_mass_matrix)."""
    warmup = blackjax.window_adaptation(
        blackjax.nuts,
        logdensity_fn,
        is_mass_matrix_diagonal=not dense,
        initial_inverse_mass_matrix=imm,
        imm_shrinkage_to_previous=imm_shrinkage_to_previous,
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


# ---------------------------------------------------------------------------
# Phase 2 (P2): imm_shrinkage_to_previous tests (NEW)
# ---------------------------------------------------------------------------


def test_imm_shrinkage_backward_compat_default_zero():
    """Default imm_shrinkage_to_previous=0.0 produces Stan-identical output.

    With the new kwarg defaulting to 0.0, the behavior must be identical to
    the pre-P2 code. We verify that calling with explicit 0.0 and implicit
    default produce the same warmup result.
    """
    rng_key = jax.random.key(100)
    # Explicit 0.0
    _, imm_explicit = _run_warmup(
        rng_key, imm=None, num_steps=300, imm_shrinkage_to_previous=0.0
    )
    # Implicit default (no kwarg)
    _, imm_default = _run_warmup(rng_key, imm=None, num_steps=300)

    # Should be bit-identical (or at least very close due to JAX numerical reproducibility)
    np.testing.assert_allclose(imm_explicit, imm_default, rtol=1e-6)


def test_imm_shrinkage_seed_influence_persists_diagonal():
    """With non-zero pseudo-count, seed IMM influence persists longer.

    Compare two diagonal cases: one with imm_shrinkage_to_previous=0.0 (seed
    loses influence quickly) and one with a large pseudo-count (seed sticky).
    With a deliberately-wrong seed, the sticky version's final IMM should be
    closer to the seed than the non-sticky version's.
    """
    rng_key = jax.random.key(101)
    # Seed that is 100x larger than optimal — will bias the result
    wrong_seed = jnp.array([100.0, 100.0, 100.0])

    # No shrinkage: seed is quickly overwritten by Welford
    _, imm_no_shrink = _run_warmup(
        rng_key,
        imm=wrong_seed,
        num_steps=300,
        imm_shrinkage_to_previous=0.0,
    )

    # Large shrinkage: seed's influence persists
    _, imm_with_shrink = _run_warmup(
        rng_key,
        imm=wrong_seed,
        num_steps=300,
        imm_shrinkage_to_previous=20.0,
    )

    # With shrinkage, the final IMM should be closer to the (wrong) seed
    # than the no-shrinkage case.
    dist_no_shrink = jnp.mean((imm_no_shrink - wrong_seed) ** 2)
    dist_with_shrink = jnp.mean((imm_with_shrink - wrong_seed) ** 2)
    error_msg = (
        f"Expected shrinkage to keep IMM closer to seed: "
        f"got dist_no_shrink={dist_no_shrink: .6f}, "
        f"dist_with_shrink={dist_with_shrink: .6f}"
    )
    assert dist_with_shrink < dist_no_shrink, error_msg


def test_imm_shrinkage_negative_raises():
    """Negative imm_shrinkage_to_previous raises ValueError at construction time."""
    with pytest.raises(ValueError, match="imm_shrinkage_to_previous must be >= 0.0"):
        blackjax.window_adaptation(
            blackjax.nuts,
            logdensity_fn,
            is_mass_matrix_diagonal=True,
            imm_shrinkage_to_previous=-1.0,
        )


def test_imm_shrinkage_dense_matrix_mirrors_diagonal():
    """Dense case with shrinkage applies the formula symmetrically.

    Test that imm_shrinkage_to_previous works correctly for dense matrices
    (is_mass_matrix_diagonal=False), confirming the shrinkage term is applied
    to the full matrix, not just the diagonal.
    """
    rng_key = jax.random.key(102)
    # Use a diagonal PD matrix as the dense seed (same as test 3 in P0)
    wrong_seed = jnp.diag(jnp.array([100.0, 100.0, 100.0]))

    # No shrinkage: dense case
    _, imm_dense_no_shrink = _run_warmup(
        rng_key,
        imm=wrong_seed,
        dense=True,
        num_steps=300,
        imm_shrinkage_to_previous=0.0,
    )

    # With shrinkage: dense case
    _, imm_dense_with_shrink = _run_warmup(
        rng_key,
        imm=wrong_seed,
        dense=True,
        num_steps=300,
        imm_shrinkage_to_previous=20.0,
    )

    # Same logic as the diagonal test: shrinkage should keep the final IMM
    # closer to the (wrong) seed.
    dist_no_shrink = jnp.mean((imm_dense_no_shrink - wrong_seed) ** 2)
    dist_with_shrink = jnp.mean((imm_dense_with_shrink - wrong_seed) ** 2)
    error_msg = (
        f"Expected dense shrinkage to keep IMM closer to seed: "
        f"got dist_no_shrink={dist_no_shrink: .6f}, "
        f"dist_with_shrink={dist_with_shrink: .6f}"
    )
    assert dist_with_shrink < dist_no_shrink, error_msg
