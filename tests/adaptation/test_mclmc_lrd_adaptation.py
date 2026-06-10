# Copyright 2020- The Blackjax Authors.
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
"""Tests for mclmc_lrd_adaptation (Scheme A, pilot-free LRD warmup)."""

import warnings

import jax
import jax.numpy as jnp
import pytest

import blackjax
from blackjax.adaptation.mclmc_lrd_adaptation import (
    MCLMCLRDAdaptationState,
    _extract_lrd_from_samples,
    mclmc_lrd_adaptation,
)
from blackjax.mcmc.metrics import LowRankInverseMassMatrix

# ---------------------------------------------------------------------------
# Shared fixture: isotropic Gaussian (d=6)
# ---------------------------------------------------------------------------

D = 6
K = 3  # rank for most tests


def logdensity_fn(x):
    return -0.5 * jnp.sum(x**2)


# ---------------------------------------------------------------------------
# Unit tests: _extract_lrd_from_samples
# ---------------------------------------------------------------------------


class TestExtractLrd:
    """Shape and contract tests for the internal SVD extraction helper."""

    def test_output_shapes(self):
        """sigma (d,), U (d,k), lam (k,) for any valid n>k."""
        rng = jax.random.key(0)
        d, n, k = 8, 50, 3
        samples = jax.random.normal(rng, (n, d))
        sigma, U, lam = _extract_lrd_from_samples(samples, k=k)
        assert sigma.shape == (d,)
        assert U.shape == (d, k)
        assert lam.shape == (k,)

    def test_sigma_positive(self):
        """All diagonal scales must be strictly positive."""
        rng = jax.random.key(1)
        samples = jax.random.normal(rng, (30, 5))
        sigma, _, _ = _extract_lrd_from_samples(samples, k=2)
        assert jnp.all(sigma > 0)

    def test_zero_variance_dim_gets_unit_sigma(self):
        """A dimension with zero variance must not produce NaN/Inf sigma."""
        rng = jax.random.key(2)
        samples = jax.random.normal(rng, (30, 4))
        # Force dim-0 to constant (zero variance)
        samples = samples.at[:, 0].set(0.0)
        sigma, _, _ = _extract_lrd_from_samples(samples, k=2)
        assert jnp.all(jnp.isfinite(sigma))
        assert sigma[0] == pytest.approx(1.0)  # clamped to 1 for zero-var dim

    def test_columns_of_U_are_unit_vectors(self):
        """Each column of U should be (approximately) unit-norm."""
        rng = jax.random.key(3)
        samples = jax.random.normal(rng, (100, 10))
        _, U, _ = _extract_lrd_from_samples(samples, k=4)
        col_norms = jnp.linalg.norm(U, axis=0)
        assert jnp.allclose(col_norms, jnp.ones(4), atol=1e-5)

    def test_isotropic_lam_near_one(self):
        """For an isotropic Gaussian, all eigenvalues should be close to 1."""
        rng = jax.random.key(4)
        n, d, k = 5000, 6, 3
        samples = jax.random.normal(rng, (n, d))
        _, _, lam = _extract_lrd_from_samples(samples, k=k)
        assert jnp.allclose(lam, jnp.ones(k), atol=0.15)


# ---------------------------------------------------------------------------
# Unit tests: rank guard
# ---------------------------------------------------------------------------


class TestRankGuard:
    """Rank-guard clamping and warning behaviour."""

    def test_clamp_k_when_pilot_under_mixed(self):
        """When k > n_eff/2, k_used must equal k_safe and a warning must fire."""
        # Tiny pilot that cannot give high ESS: n=10 draws, request k=20
        rng = jax.random.key(5)
        pos = jnp.zeros(D)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = mclmc_lrd_adaptation(
                logdensity_fn,
                pos,
                rng,
                k=20,
                pilot_num_warmup=50,
                pilot_num_samples=10,  # very few draws → low ESS
                lrd_num_steps=50,
            )

        assert isinstance(result, MCLMCLRDAdaptationState)
        diag = result.diagnostics
        # k_used must be ≤ k_safe
        assert (
            diag["k_used"] <= diag["k_safe"]
        ), f"k_used={diag['k_used']} > k_safe={diag['k_safe']}"
        # Should have emitted a UserWarning about clamping
        assert any(
            issubclass(w.category, UserWarning) for w in caught
        ), "Expected a UserWarning about rank clamping but none was emitted"

    def test_no_warning_when_k_within_bound(self):
        """When k ≤ n_eff/2, no clamping warning should be emitted."""
        rng = jax.random.key(6)
        pos = jnp.zeros(D)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = mclmc_lrd_adaptation(
                logdensity_fn,
                pos,
                rng,
                k=1,  # very low rank — should always be within n_eff/2
                pilot_num_warmup=100,
                pilot_num_samples=500,
                lrd_num_steps=50,
            )

        clamp_warnings = [
            w
            for w in caught
            if issubclass(w.category, UserWarning)
            and "rank-safety" in str(w.message).lower()
        ]
        assert (
            len(clamp_warnings) == 0
        ), f"Unexpected rank-clamp warning for k=1: {clamp_warnings}"
        assert result.diagnostics["k_used"] == 1

    def test_k_safe_at_least_one(self):
        """k_used must be at least 1 even when the pilot is tiny and n_eff rounds to 0."""
        rng = jax.random.key(7)
        pos = jnp.zeros(D)
        # Very small pilot (2 draws) → n_eff ≤ 2 → k_safe likely 0 or 1 → clamped to ≥1
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            result = mclmc_lrd_adaptation(
                logdensity_fn,
                pos,
                rng,
                k=5,
                pilot_num_warmup=30,
                pilot_num_samples=2,
                lrd_num_steps=30,
            )
        assert result.diagnostics["k_used"] >= 1


# ---------------------------------------------------------------------------
# Integration smoke test: isotropic Gaussian (CI-friendly, no heavy compute)
# ---------------------------------------------------------------------------


class TestMCLMCLRDAdaptationSmoke:
    """End-to-end shape + contract checks on a cheap Gaussian target."""

    def test_return_type_and_shapes(self):
        """Result is MCLMCLRDAdaptationState with correct field types and shapes."""
        rng = jax.random.key(42)
        pos = jnp.zeros(D)

        result = mclmc_lrd_adaptation(
            logdensity_fn,
            pos,
            rng,
            k=K,
            pilot_num_warmup=200,
            pilot_num_samples=400,
            lrd_num_steps=200,
        )

        assert isinstance(result, MCLMCLRDAdaptationState)
        assert jnp.isfinite(result.L) and result.L > 0
        assert jnp.isfinite(result.step_size) and result.step_size > 0
        assert isinstance(result.inverse_mass_matrix, LowRankInverseMassMatrix)

        imm = result.inverse_mass_matrix
        assert imm.sigma.shape == (D,)
        assert imm.U.shape == (D, K)
        assert imm.lam.shape == (K,)
        assert jnp.all(jnp.isfinite(imm.sigma))
        assert jnp.all(jnp.isfinite(imm.U))
        assert jnp.all(jnp.isfinite(imm.lam))

    def test_diagnostics_keys_and_types(self):
        """diagnostics dict must contain n_eff, k_safe, k_used, pilot_num_grad_evals."""
        rng = jax.random.key(43)
        pos = jnp.zeros(D)

        result = mclmc_lrd_adaptation(
            logdensity_fn,
            pos,
            rng,
            k=K,
            pilot_num_warmup=100,
            pilot_num_samples=200,
            lrd_num_steps=100,
        )
        diag = result.diagnostics
        for key in ("n_eff", "k_safe", "k_used", "pilot_num_grad_evals"):
            assert key in diag, f"Missing diagnostics key: {key}"

        assert diag["k_used"] <= K
        assert diag["k_used"] >= 1
        assert diag["k_safe"] >= 0
        assert diag["n_eff"] > 0
        assert diag["pilot_num_grad_evals"] == (100 + 200) * 2

    def test_top_level_api_exposed(self):
        """blackjax.mclmc_lrd_adaptation must resolve to the correct function."""
        assert blackjax.mclmc_lrd_adaptation is mclmc_lrd_adaptation

    def test_lrd_imm_usable_with_mclmc_kernel(self):
        """The returned LRD IMM must plug into the mclmc base kernel without error."""
        import blackjax.mcmc.mclmc as mclmc_mod

        rng = jax.random.key(44)
        pos = jnp.zeros(D)

        result = mclmc_lrd_adaptation(
            logdensity_fn,
            pos,
            rng,
            k=K,
            pilot_num_warmup=100,
            pilot_num_samples=200,
            lrd_num_steps=100,
        )

        lrd_imm = result.inverse_mass_matrix
        base_kernel = mclmc_mod.build_kernel()
        init_state = mclmc_mod.init(pos, logdensity_fn, jax.random.key(99))

        next_state, info = base_kernel(
            rng_key=jax.random.key(100),
            state=init_state,
            logdensity_fn=logdensity_fn,
            inverse_mass_matrix=lrd_imm,
            L=result.L,
            step_size=result.step_size,
        )
        assert jnp.all(
            jnp.isfinite(jax.flatten_util.ravel_pytree(next_state.position)[0])
        )

    def test_pytree_position(self):
        """mclmc_lrd_adaptation must work when position is a dict pytree."""
        rng = jax.random.key(45)
        pos = {"a": jnp.zeros(3), "b": jnp.zeros(2)}

        def logp_dict(x):
            return -0.5 * (jnp.sum(x["a"] ** 2) + jnp.sum(x["b"] ** 2))

        result = mclmc_lrd_adaptation(
            logp_dict,
            pos,
            rng,
            k=2,
            pilot_num_warmup=100,
            pilot_num_samples=200,
            lrd_num_steps=100,
        )

        assert isinstance(result, MCLMCLRDAdaptationState)
        # d = 3 + 2 = 5
        assert result.inverse_mass_matrix.sigma.shape == (5,)
        assert result.inverse_mass_matrix.U.shape == (5, 2)
