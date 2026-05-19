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
"""Tests for pathfinder_adaptation's num_chains + n_paths kwargs (P1)."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import blackjax


# ---------------------------------------------------------------------------
# Shared fixture: 3-D isotropic Gaussian
# ---------------------------------------------------------------------------

DIM = 3
TARGET_MEAN = jnp.zeros(DIM)
TARGET_STD = jnp.ones(DIM)


def logdensity_fn(x):
    return jax.scipy.stats.norm.logpdf(x, loc=TARGET_MEAN, scale=TARGET_STD).sum()


# ---------------------------------------------------------------------------
# 1. Backward compatibility: no num_chains / n_paths kwargs
# ---------------------------------------------------------------------------


def test_backward_compat_single_chain():
    """pathfinder_adaptation with no new kwargs is identical to original API.

    The original single-chain single-path API returns a dense (d, d) IMM
    from lbfgs_inverse_hessian_formula_1.  Scalar step_size.
    """
    rng_key = jax.random.key(0)
    warmup = blackjax.pathfinder_adaptation(
        blackjax.nuts,
        logdensity_fn,
    )
    init_pos = jnp.zeros(DIM)
    (state, params), _ = warmup.run(rng_key, init_pos, num_steps=200)
    step_size = params["step_size"]
    imm = params["inverse_mass_matrix"]
    assert step_size.shape == ()  # scalar
    # Original returns dense (d, d) IMM from the L-BFGS inverse Hessian
    assert imm.shape == (DIM, DIM)
    assert float(step_size) > 0
    # Pareto-k should NOT be present in single-path single-chain
    assert "_pathfinder_psis_pareto_k" not in params


# ---------------------------------------------------------------------------
# 2. Multi-chain single-path: num_chains > 1, n_paths defaults to 1
# ---------------------------------------------------------------------------


def test_multichain_single_path_shapes():
    """num_chains=4 returns batched step_size (4,) and shared IMM (d,)."""
    rng_key = jax.random.key(1)
    warmup = blackjax.pathfinder_adaptation(
        blackjax.nuts,
        logdensity_fn,
        num_chains=4,
        n_paths=1,
    )
    init_pos = jnp.zeros(DIM)
    (state, params), _ = warmup.run(rng_key, init_pos, num_steps=100)
    assert params["step_size"].shape == (4,)
    assert params["inverse_mass_matrix"].shape == (DIM,)
    assert jnp.all(params["step_size"] > 0)
    assert jnp.all(params["inverse_mass_matrix"] > 0)
    # state.position should have leading dim 4
    assert state.position.shape == (4, DIM)


def test_multichain_single_path_default_n_paths():
    """num_chains=4 with n_paths=None defaults to n_paths=num_chains=4 (multipathfinder)."""
    rng_key = jax.random.key(7)
    warmup = blackjax.pathfinder_adaptation(
        blackjax.nuts,
        logdensity_fn,
        num_chains=4,
        # n_paths defaults to num_chains=4 → triggers multipathfinder path
    )
    init_pos = jnp.zeros(DIM)
    (state, params), _ = warmup.run(rng_key, init_pos, num_steps=100)
    # n_paths = num_chains = 4 → multi-path path
    assert params["step_size"].shape == (4,)
    assert params["inverse_mass_matrix"].shape == (DIM,)
    assert "_pathfinder_psis_pareto_k" in params


# ---------------------------------------------------------------------------
# 3. Single-chain multi-path: num_chains=1, n_paths=4
# ---------------------------------------------------------------------------


def test_single_chain_multipathfinder():
    """num_chains=1, n_paths=4 runs multipathfinder, returns scalar step_size."""
    rng_key = jax.random.key(2)
    warmup = blackjax.pathfinder_adaptation(
        blackjax.nuts,
        logdensity_fn,
        num_chains=1,
        n_paths=4,
        num_samples_per_path=50,
        psis_imm_n_samples=200,
    )
    init_pos = jnp.zeros(DIM)
    (state, params), _ = warmup.run(rng_key, init_pos, num_steps=100)
    assert params["step_size"].shape == ()  # scalar
    assert params["inverse_mass_matrix"].shape == (DIM,)
    assert float(params["step_size"]) > 0
    assert "_pathfinder_psis_pareto_k" in params
    pareto_k = params["_pathfinder_psis_pareto_k"]
    assert pareto_k.shape == ()
    # pareto_k is a scalar diagnostic (can be NaN on degenerate samples, but
    # should always be a finite-or-nan float, not an array)
    assert pareto_k.ndim == 0


# ---------------------------------------------------------------------------
# 4. Paper-canonical: num_chains=4, n_paths=4
# ---------------------------------------------------------------------------


def test_paper_canonical_multichain_multipathfinder():
    """num_chains=4, n_paths=4: multipathfinder → PSIS init → vmap DA."""
    rng_key = jax.random.key(3)
    warmup = blackjax.pathfinder_adaptation(
        blackjax.nuts,
        logdensity_fn,
        num_chains=4,
        n_paths=4,
        num_samples_per_path=50,
        psis_imm_n_samples=200,
    )
    init_pos = jnp.zeros(DIM)
    (state, params), _ = warmup.run(rng_key, init_pos, num_steps=100)
    # Step sizes: one per chain
    assert params["step_size"].shape == (4,)
    assert jnp.all(params["step_size"] > 0)
    # IMM: shared single diagonal
    assert params["inverse_mass_matrix"].shape == (DIM,)
    assert jnp.all(params["inverse_mass_matrix"] > 0)
    # PSIS diagnostic included
    assert "_pathfinder_psis_pareto_k" in params
    # state: batched over 4 chains
    assert state.position.shape == (4, DIM)
    # Empirical means should be near zero on this isotropic Gaussian
    mean = jnp.mean(state.position, axis=0)
    np.testing.assert_allclose(mean, jnp.zeros(DIM), atol=2.0)


# ---------------------------------------------------------------------------
# 5. Edge cases: invalid num_chains / n_paths raise ValueError
# ---------------------------------------------------------------------------


def test_num_chains_zero_raises():
    """num_chains=0 raises ValueError at construction time."""
    with pytest.raises(ValueError, match="num_chains"):
        blackjax.pathfinder_adaptation(
            blackjax.nuts,
            logdensity_fn,
            num_chains=0,
        )


def test_num_chains_negative_raises():
    """num_chains=-1 raises ValueError at construction time."""
    with pytest.raises(ValueError, match="num_chains"):
        blackjax.pathfinder_adaptation(
            blackjax.nuts,
            logdensity_fn,
            num_chains=-1,
        )


def test_n_paths_zero_raises():
    """n_paths=0 raises ValueError at construction time."""
    with pytest.raises(ValueError, match="n_paths"):
        blackjax.pathfinder_adaptation(
            blackjax.nuts,
            logdensity_fn,
            n_paths=0,
        )


def test_n_paths_negative_raises():
    """n_paths=-2 raises ValueError at construction time."""
    with pytest.raises(ValueError, match="n_paths"):
        blackjax.pathfinder_adaptation(
            blackjax.nuts,
            logdensity_fn,
            n_paths=-2,
        )
