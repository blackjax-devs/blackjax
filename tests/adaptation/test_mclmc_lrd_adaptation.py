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
"""Tests for mclmc_lrd_warmup (Scheme A, pilot-free LRD warmup)."""

import warnings

import jax
import jax.numpy as jnp
import pytest

import blackjax
from blackjax.adaptation.mclmc_lrd_adaptation import (
    MCLMCLRDAdaptationState,
    _check_da_ceiling_warning,
    _extract_lrd_from_samples,
    _kappa_eff_pilot,
    mclmc_lrd_warmup,
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
        """sigma (d,), U (d,k), lam (k,), lam_all (min(n,d),) for any valid n>k."""
        rng = jax.random.key(0)
        d, n, k = 8, 50, 3
        samples = jax.random.normal(rng, (n, d))
        sigma, U, lam, lam_all = _extract_lrd_from_samples(samples, k=k)
        assert sigma.shape == (d,)
        assert U.shape == (d, k)
        assert lam.shape == (k,)
        # lam_all has shape (min(n,d),) = (d,) since n > d here
        assert lam_all.shape == (d,)
        # lam_all[:k] must equal lam (top-k are first in sorted order)
        assert jnp.allclose(lam_all[:k], lam)

    def test_sigma_positive(self):
        """All diagonal scales must be strictly positive."""
        rng = jax.random.key(1)
        samples = jax.random.normal(rng, (30, 5))
        sigma, _, _, _ = _extract_lrd_from_samples(samples, k=2)
        assert jnp.all(sigma > 0)

    def test_zero_variance_dim_gets_unit_sigma(self):
        """A dimension with zero variance must not produce NaN/Inf sigma."""
        rng = jax.random.key(2)
        samples = jax.random.normal(rng, (30, 4))
        # Force dim-0 to constant (zero variance)
        samples = samples.at[:, 0].set(0.0)
        sigma, _, _, _ = _extract_lrd_from_samples(samples, k=2)
        assert jnp.all(jnp.isfinite(sigma))
        assert sigma[0] == pytest.approx(1.0)  # clamped to 1 for zero-var dim

    def test_columns_of_U_are_unit_vectors(self):
        """Each column of U should be (approximately) unit-norm."""
        rng = jax.random.key(3)
        samples = jax.random.normal(rng, (100, 10))
        _, U, _, _ = _extract_lrd_from_samples(samples, k=4)
        col_norms = jnp.linalg.norm(U, axis=0)
        assert jnp.allclose(col_norms, jnp.ones(4), atol=1e-5)

    def test_isotropic_lam_near_one(self):
        """For an isotropic Gaussian, all eigenvalues should be close to 1."""
        rng = jax.random.key(4)
        n, d, k = 5000, 6, 3
        samples = jax.random.normal(rng, (n, d))
        _, _, lam, _ = _extract_lrd_from_samples(samples, k=k)
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
            result = mclmc_lrd_warmup(
                logdensity_fn,
                pos,
                rng,
                k=20,
                pilot_num_warmup=50,
                pilot_num_samples=10,  # very few draws → low ESS
                lrd_num_steps=50,
                num_chains=2,
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
            result = mclmc_lrd_warmup(
                logdensity_fn,
                pos,
                rng,
                k=1,  # very low rank — should always be within n_eff/2
                pilot_num_warmup=100,
                pilot_num_samples=500,
                lrd_num_steps=50,
                num_chains=2,
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
        """k_used must be at least 1 even when the pilot is tiny."""
        rng = jax.random.key(7)
        pos = jnp.zeros(D)
        # Very small pilot (2 draws) → n_eff ≤ 2 → k_safe likely 0 or 1 → clamped to ≥1
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            result = mclmc_lrd_warmup(
                logdensity_fn,
                pos,
                rng,
                k=5,
                pilot_num_warmup=30,
                pilot_num_samples=2,
                lrd_num_steps=30,
                num_chains=2,
            )
        assert result.diagnostics["k_used"] >= 1


# ---------------------------------------------------------------------------
# Integration smoke tests: isotropic Gaussian (CI-friendly, no heavy compute)
# ---------------------------------------------------------------------------


class TestMCLMCLRDWarmupSmoke:
    """End-to-end shape + contract checks on a cheap Gaussian target."""

    def test_return_type_and_shapes(self):
        """Result is MCLMCLRDAdaptationState with correct field types and shapes."""
        rng = jax.random.key(42)
        pos = jnp.zeros(D)

        result = mclmc_lrd_warmup(
            logdensity_fn,
            pos,
            rng,
            k=K,
            pilot_num_warmup=200,
            pilot_num_samples=400,
            lrd_num_steps=200,
            num_chains=2,
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

    def test_L_and_step_size_are_scalars(self):
        """Phase-3 multi-chain averaging must produce scalar L and step_size."""
        rng = jax.random.key(48)
        pos = jnp.zeros(D)

        result = mclmc_lrd_warmup(
            logdensity_fn,
            pos,
            rng,
            k=K,
            pilot_num_warmup=100,
            pilot_num_samples=200,
            lrd_num_steps=100,
            num_chains=4,
        )

        assert (
            jnp.ndim(result.L) == 0
        ), f"Expected scalar L after multi-chain mean, got shape {jnp.shape(result.L)}"
        assert (
            jnp.ndim(result.step_size) == 0
        ), f"Expected scalar step_size, got shape {jnp.shape(result.step_size)}"

    def test_diagnostics_keys_and_types(self):
        """diagnostics dict must contain n_eff, k_safe, k_used, pilot_num_grad_evals."""
        rng = jax.random.key(43)
        pos = jnp.zeros(D)

        result = mclmc_lrd_warmup(
            logdensity_fn,
            pos,
            rng,
            k=K,
            pilot_num_warmup=100,
            pilot_num_samples=200,
            lrd_num_steps=100,
            num_chains=2,
        )
        diag = result.diagnostics
        for key in ("n_eff", "k_safe", "k_used", "pilot_num_grad_evals"):
            assert key in diag, f"Missing diagnostics key: {key}"

        assert diag["k_used"] <= K
        assert diag["k_used"] >= 1
        assert diag["k_safe"] >= 0
        assert diag["n_eff"] > 0
        assert diag["pilot_num_grad_evals"] == (100 + 200) * 2

    def test_diagnostics_provenance_keys(self):
        """diagnostics must contain pilot_L, pilot_step_size, lrd_L, lrd_step_size."""
        rng = jax.random.key(47)
        pos = jnp.zeros(D)

        result = mclmc_lrd_warmup(
            logdensity_fn,
            pos,
            rng,
            k=K,
            pilot_num_warmup=100,
            pilot_num_samples=200,
            lrd_num_steps=100,
            num_chains=2,
        )
        diag = result.diagnostics
        for key in ("pilot_L", "pilot_step_size", "lrd_L", "lrd_step_size"):
            assert key in diag, f"Missing diagnostics provenance key: {key}"
        assert diag["pilot_L"] > 0
        assert diag["pilot_step_size"] > 0
        assert diag["lrd_L"] > 0
        assert diag["lrd_step_size"] > 0

    def test_top_level_api_exposed(self):
        """blackjax.mclmc_lrd_warmup must resolve to the correct function."""
        assert blackjax.mclmc_lrd_warmup is mclmc_lrd_warmup

    def test_lrd_imm_usable_with_mclmc_kernel(self):
        """The returned LRD IMM must plug into the mclmc base kernel without error."""
        import blackjax.mcmc.mclmc as mclmc_mod

        rng = jax.random.key(44)
        pos = jnp.zeros(D)

        result = mclmc_lrd_warmup(
            logdensity_fn,
            pos,
            rng,
            k=K,
            pilot_num_warmup=100,
            pilot_num_samples=200,
            lrd_num_steps=100,
            num_chains=2,
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
        """mclmc_lrd_warmup must work when position is a dict pytree."""
        rng = jax.random.key(45)
        pos = {"a": jnp.zeros(3), "b": jnp.zeros(2)}

        def logp_dict(x):
            return -0.5 * (jnp.sum(x["a"] ** 2) + jnp.sum(x["b"] ** 2))

        result = mclmc_lrd_warmup(
            logp_dict,
            pos,
            rng,
            k=2,
            pilot_num_warmup=100,
            pilot_num_samples=200,
            lrd_num_steps=100,
            num_chains=2,
        )

        assert isinstance(result, MCLMCLRDAdaptationState)
        # d = 3 + 2 = 5
        assert result.inverse_mass_matrix.sigma.shape == (5,)
        assert result.inverse_mass_matrix.U.shape == (5, 2)

    def test_diagnostics_inner_kernel_field(self):
        """diagnostics must include inner_kernel field matching the argument."""
        rng = jax.random.key(46)
        pos = jnp.zeros(D)
        result = mclmc_lrd_warmup(
            logdensity_fn,
            pos,
            rng,
            k=K,
            pilot_num_warmup=100,
            pilot_num_samples=200,
            lrd_num_steps=100,
            num_chains=2,
            inner_kernel="mclmc",
        )
        assert result.diagnostics["inner_kernel"] == "mclmc"

    def test_invalid_inner_kernel_raises(self):
        """Passing an unknown inner_kernel must raise ValueError immediately."""
        with pytest.raises(ValueError, match="inner_kernel"):
            mclmc_lrd_warmup(
                logdensity_fn,
                jnp.zeros(D),
                jax.random.key(0),
                inner_kernel="nuts",
            )


# ---------------------------------------------------------------------------
# Unit test: DA-ceiling warning helper
# ---------------------------------------------------------------------------


class TestDACeilingWarning:
    """Tests for the DA-ceiling warning logic in _check_da_ceiling_warning."""

    def test_warning_fires_when_at_ceiling(self):
        """Warning must fire when step_size / (L_init/1.1) >= 0.999."""
        L_init = 10.0
        da_clamp = L_init / 1.1  # 9.0909...
        # Set step_size to exactly the clamp (ratio = 1.0)
        step_size = da_clamp

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _check_da_ceiling_warning(step_size, L_init, floor_factor=1.15)

        ceiling_warnings = [w for w in caught if issubclass(w.category, UserWarning)]
        assert (
            len(ceiling_warnings) > 0
        ), "Expected DA-ceiling UserWarning but none was emitted"
        assert "ceiling" in str(ceiling_warnings[0].message).lower()

    def test_warning_fires_near_ceiling(self):
        """Warning must fire when ratio >= 0.999 (just below ceiling)."""
        L_init = 10.0
        da_clamp = L_init / 1.1
        step_size = da_clamp * 0.9995  # ratio = 0.9995 >= 0.999

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _check_da_ceiling_warning(step_size, L_init, floor_factor=1.15)

        assert any(
            issubclass(w.category, UserWarning) for w in caught
        ), "Expected warning at ratio=0.9995 but none fired"

    def test_no_warning_when_well_below_ceiling(self):
        """No warning when step_size is well below DA ceiling."""
        L_init = 10.0
        da_clamp = L_init / 1.1
        step_size = da_clamp * 0.8  # ratio = 0.8 << 0.999

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _check_da_ceiling_warning(step_size, L_init, floor_factor=1.15)

        ceiling_warnings = [w for w in caught if issubclass(w.category, UserWarning)]
        assert (
            len(ceiling_warnings) == 0
        ), f"Unexpected DA-ceiling warning at ratio=0.8: {ceiling_warnings}"

    def test_warning_mentions_floor_factor(self):
        """Warning message must reference floor_factor for actionable guidance."""
        L_init = 5.0
        step_size = L_init / 1.1  # at ceiling

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _check_da_ceiling_warning(step_size, L_init, floor_factor=1.15)

        assert len(caught) > 0
        msg = str(caught[0].message).lower()
        assert (
            "floor_factor" in msg
        ), "Warning message should mention floor_factor for actionable guidance"


# ---------------------------------------------------------------------------
# Smoke tests for adjusted path
# ---------------------------------------------------------------------------


class TestMCLMCLRDAdjustedSmoke:
    """Smoke tests for the adjusted_mclmc inner-kernel path (experimental)."""

    def test_adjusted_path_returns_valid_state(self):
        """inner_kernel='adjusted_mclmc' must return a valid MCLMCLRDAdaptationState."""
        rng = jax.random.key(50)
        pos = jnp.zeros(D)

        # DA-ceiling warning may fire for this cheap isotropic Gaussian (where
        # floor_factor * step_mclmc ≈ L_mclmc, so L_init/1.1 ≈ step_size).
        # That is expected/correct behaviour — suppress it for this smoke test.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            result = mclmc_lrd_warmup(
                logdensity_fn,
                pos,
                rng,
                k=K,
                pilot_num_warmup=200,
                pilot_num_samples=400,
                lrd_num_steps=200,
                num_chains=2,
                inner_kernel="adjusted_mclmc",
            )

        assert isinstance(result, MCLMCLRDAdaptationState)
        assert jnp.isfinite(result.L) and result.L > 0
        assert jnp.isfinite(result.step_size) and result.step_size > 0
        assert isinstance(result.inverse_mass_matrix, LowRankInverseMassMatrix)
        assert result.diagnostics["inner_kernel"] == "adjusted_mclmc"
        assert result.inverse_mass_matrix.sigma.shape == (D,)
        assert result.inverse_mass_matrix.U.shape == (D, K)
        assert result.inverse_mass_matrix.lam.shape == (K,)

    def test_adjusted_path_lrd_imm_usable_with_adjusted_kernel(self):
        """LRD IMM from adjusted path must plug into adjusted_mclmc kernel."""
        import blackjax.mcmc.adjusted_mclmc as adj_mod

        rng = jax.random.key(51)
        pos = jnp.zeros(D)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            result = mclmc_lrd_warmup(
                logdensity_fn,
                pos,
                rng,
                k=K,
                pilot_num_warmup=100,
                pilot_num_samples=200,
                lrd_num_steps=100,
                num_chains=2,
                inner_kernel="adjusted_mclmc",
            )

        lrd_imm = result.inverse_mass_matrix
        adj_kernel = adj_mod.build_kernel()
        init_state = adj_mod.init(pos, logdensity_fn)

        next_state, info = adj_kernel(
            rng_key=jax.random.key(99),
            state=init_state,
            logdensity_fn=logdensity_fn,
            step_size=result.step_size,
            inverse_mass_matrix=lrd_imm,
        )
        assert jnp.all(
            jnp.isfinite(jax.flatten_util.ravel_pytree(next_state.position)[0])
        )

    def test_adjusted_path_provenance_keys_present(self):
        """Adjusted path must also populate pilot_L/step_size and lrd_L/step_size."""
        rng = jax.random.key(52)
        pos = jnp.zeros(D)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            result = mclmc_lrd_warmup(
                logdensity_fn,
                pos,
                rng,
                k=K,
                pilot_num_warmup=100,
                pilot_num_samples=200,
                lrd_num_steps=100,
                num_chains=2,
                inner_kernel="adjusted_mclmc",
            )
        for key in ("pilot_L", "pilot_step_size", "lrd_L", "lrd_step_size"):
            assert (
                key in result.diagnostics
            ), f"Missing provenance key {key!r} in adjusted path diagnostics"

    def test_adjusted_path_l_init_and_floor_active_in_diagnostics(self):
        """Adjusted path must expose L_init and floor_active in diagnostics."""
        rng = jax.random.key(53)
        pos = jnp.zeros(D)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            result = mclmc_lrd_warmup(
                logdensity_fn,
                pos,
                rng,
                k=K,
                pilot_num_warmup=100,
                pilot_num_samples=200,
                lrd_num_steps=100,
                num_chains=2,
                inner_kernel="adjusted_mclmc",
            )
        diag = result.diagnostics
        assert "L_init" in diag, "Missing L_init in adjusted diagnostics"
        assert "floor_active" in diag, "Missing floor_active in adjusted diagnostics"
        assert isinstance(diag["floor_active"], bool)
        assert diag["L_init"] > 0

        # L_init must be >= lrd_L (floor invariant)
        assert diag["L_init"] >= diag["lrd_L"] - 1e-9

    def test_adjusted_path_n_sample_in_diagnostics(self):
        """Adjusted path must expose N_sample = round(L_init / final_step_size)."""
        rng = jax.random.key(57)
        pos = jnp.zeros(D)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            result = mclmc_lrd_warmup(
                logdensity_fn,
                pos,
                rng,
                k=K,
                pilot_num_warmup=100,
                pilot_num_samples=200,
                lrd_num_steps=100,
                num_chains=2,
                inner_kernel="adjusted_mclmc",
            )
        diag = result.diagnostics
        assert "N_sample" in diag, "Missing N_sample in adjusted diagnostics"
        assert isinstance(diag["N_sample"], (int, float)), "N_sample must be numeric"
        assert diag["N_sample"] > 0, "N_sample must be positive"
        # Verify the value matches round(L_init / final_step_size) within rounding.
        expected = float(diag["L_init"]) / float(result.step_size)
        assert abs(diag["N_sample"] - expected) < 1.0  # within one step

    def test_unadjusted_path_has_no_l_init_floor_active(self):
        """Unadjusted path must NOT have L_init or floor_active in diagnostics."""
        rng = jax.random.key(54)
        pos = jnp.zeros(D)
        result = mclmc_lrd_warmup(
            logdensity_fn,
            pos,
            rng,
            k=K,
            pilot_num_warmup=100,
            pilot_num_samples=200,
            lrd_num_steps=100,
            num_chains=2,
            inner_kernel="mclmc",
        )
        assert "L_init" not in result.diagnostics
        assert "floor_active" not in result.diagnostics

    def test_floor_active_true_when_step_dominates(self):
        """floor_active must be True when floor_factor*step_size > lrd_L."""
        # Use a very large floor_factor to force the floor to activate.
        rng = jax.random.key(55)
        pos = jnp.zeros(D)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            result = mclmc_lrd_warmup(
                logdensity_fn,
                pos,
                rng,
                k=K,
                pilot_num_warmup=100,
                pilot_num_samples=200,
                lrd_num_steps=100,
                num_chains=2,
                inner_kernel="adjusted_mclmc",
                floor_factor=1000.0,  # guarantee floor triggers
            )
        diag = result.diagnostics
        assert (
            diag["floor_active"] is True
        ), "Expected floor_active=True with factor=1000"
        # L_init should equal floor_factor * lrd_step_size (within float precision)
        expected = 1000.0 * diag["lrd_step_size"]
        assert abs(diag["L_init"] - expected) < 1e-5 * expected

    def test_floor_inactive_when_l_already_large(self):
        """floor_active must be False when lrd_L > floor_factor*lrd_step_size."""
        # floor_factor=0.0 ensures L_init = lrd_L (floor never active).
        rng = jax.random.key(56)
        pos = jnp.zeros(D)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            result = mclmc_lrd_warmup(
                logdensity_fn,
                pos,
                rng,
                k=K,
                pilot_num_warmup=100,
                pilot_num_samples=200,
                lrd_num_steps=100,
                num_chains=2,
                inner_kernel="adjusted_mclmc",
                floor_factor=0.0,  # floor never triggers
            )
        diag = result.diagnostics
        assert (
            diag["floor_active"] is False
        ), "Expected floor_active=False with factor=0"
        # L_init should equal lrd_L exactly (no floor applied)
        assert abs(diag["L_init"] - diag["lrd_L"]) < 1e-9


# ---------------------------------------------------------------------------
# frac_tune2=0 invariant test via monkeypatch
# ---------------------------------------------------------------------------


class TestAdjustedFracTune2Invariant:
    """Verify that frac_tune2 is always 0.0 in the adjusted tuning call."""

    def test_frac_tune2_is_zero_in_adjusted_call(self, monkeypatch):
        """Monkeypatch adjusted_mclmc_find_L_and_step_size to capture kwargs."""
        import blackjax.adaptation.mclmc_lrd_adaptation as _mod

        captured_kwargs = {}
        original_fn = _mod.adjusted_mclmc_find_L_and_step_size

        def capturing_fn(*args, **kwargs):  # noqa: F841 (original_fn used in body)
            captured_kwargs.update(kwargs)
            return original_fn(*args, **kwargs)

        monkeypatch.setattr(_mod, "adjusted_mclmc_find_L_and_step_size", capturing_fn)

        rng = jax.random.key(60)
        pos = jnp.zeros(D)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            mclmc_lrd_warmup(
                logdensity_fn,
                pos,
                rng,
                k=K,
                pilot_num_warmup=100,
                pilot_num_samples=200,
                lrd_num_steps=100,
                num_chains=2,
                inner_kernel="adjusted_mclmc",
            )

        assert "frac_tune2" in captured_kwargs, "frac_tune2 not passed as kwarg"
        assert (
            captured_kwargs["frac_tune2"] == 0.0
        ), f"Expected frac_tune2=0.0, got {captured_kwargs['frac_tune2']}"
        assert (
            captured_kwargs.get("diagonal_preconditioning") is False
        ), "diagonal_preconditioning must be False to preserve LRD IMM"


# ---------------------------------------------------------------------------
# Unit tests: _kappa_eff_pilot helper
# ---------------------------------------------------------------------------


class TestKappaEffPilot:
    """Unit tests for the corrected κ_eff computation.

    _kappa_eff_pilot(lam_all_sorted, k) takes the FULL eigenspectrum ordered
    by |λ-1| descending, and the rank k.  The top-k directions are whitened
    (eigenvalue 1 in M⁻¹Σ⁻¹); residuals contribute 1/μ_i.
    """

    def test_full_rank_gives_kappa_one(self):
        """k == len(lam_all_sorted) → all directions whitened → κ_eff = 1."""
        # All 4 directions captured at k=4
        lam_all = jnp.array([5.0, 0.2, 1.5, 0.8])  # some spread
        kappa = _kappa_eff_pilot(lam_all_sorted=lam_all, k=4)
        assert abs(kappa - 1.0) < 1e-6

    def test_zero_rank_gives_full_correlation_kappa(self):
        """k=0 → no whitening → κ_eff = max(1/μ) / min(1/μ) = max(μ)/min(μ)."""
        # lam_all_sorted = [5.0, 0.2] → residuals = both → 1/5, 1/0.2=5 → κ=25
        lam_all = jnp.array([5.0, 0.2])
        kappa = _kappa_eff_pilot(lam_all_sorted=lam_all, k=0)
        assert abs(kappa - 25.0) < 1e-6

    def test_residual_spread_drives_kappa(self):
        """κ_eff decreases as k increases (more directions whitened)."""
        # lam_all_sorted (by |λ-1| desc): [10., 0.1, 0.9]
        # k=0: residuals=[10,0.1,0.9] → 1/eigs=[0.1,10,1.11] → κ=100
        # k=1: residuals=[0.1,0.9]   → 1/eigs=[10, 1.11]     → κ=10
        # k=2: residuals=[0.9]       → 1/eigs=[1.11]          → κ=max(1.11,1)/min(1.11,1)=1.11
        # k=3: all whitened          → κ=1
        lam_all = jnp.array([10.0, 0.1, 0.9])
        kappa_0 = _kappa_eff_pilot(lam_all_sorted=lam_all, k=0)
        kappa_1 = _kappa_eff_pilot(lam_all_sorted=lam_all, k=1)
        kappa_2 = _kappa_eff_pilot(lam_all_sorted=lam_all, k=2)
        kappa_3 = _kappa_eff_pilot(lam_all_sorted=lam_all, k=3)
        assert kappa_0 > kappa_1 > kappa_2 > kappa_3
        assert abs(kappa_3 - 1.0) < 1e-6

    def test_kappa_is_positive_finite_float(self):
        """κ_eff must always be a positive finite float."""
        lam_all = jnp.array([2.0, 0.8, 1.1, 0.5])
        kappa = _kappa_eff_pilot(lam_all_sorted=lam_all, k=2)
        assert isinstance(kappa, float)
        assert kappa > 0
        assert jnp.isfinite(kappa)

    def test_ill_conditioned_full_rank_imm_gives_kappa_near_one(self):
        """On an ill-cond target, k=d IMM (full-rank SVD) → κ_eff ≈ 1.

        This is the key regression test: the wrong formula gave κ≈863 at k=d;
        the correct formula must give 1.0 (all directions whitened).
        """
        # Build a d=8 correlated Gaussian with kappa=100
        import numpy as np

        d_test = 8
        rng_np = np.random.default_rng(7)
        eigs_test = np.array([1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 0.5])
        Q_test, _ = np.linalg.qr(rng_np.standard_normal((d_test, d_test)))
        Sigma_test = Q_test @ np.diag(eigs_test) @ Q_test.T
        L_test = np.linalg.cholesky(Sigma_test)
        draws_test = (L_test @ rng_np.standard_normal((d_test, 5000))).T
        flat_test = jnp.asarray(draws_test)

        # k = d (full rank): all directions captured → κ_eff should be ≈ 1
        _, _, _, lam_all_full = _extract_lrd_from_samples(flat_test, k=d_test)
        kappa_full = _kappa_eff_pilot(lam_all_sorted=lam_all_full, k=d_test)
        assert (
            kappa_full < 2.0
        ), "Full-rank IMM on ill-cond target should give kappa_eff < 2, got " + str(
            round(kappa_full, 3)
        )

        # k = 1 (rank-1): only one direction captured → κ_eff should be large
        _, _, _, lam_all_k1 = _extract_lrd_from_samples(flat_test, k=1)
        kappa_k1 = _kappa_eff_pilot(lam_all_sorted=lam_all_k1, k=1)
        assert (
            kappa_k1 > 5.0
        ), "Rank-1 IMM on ill-cond target should give kappa_eff > 5, got " + str(
            round(kappa_k1, 3)
        )


# ---------------------------------------------------------------------------
# Tests: E1 (√d warm-start) gate behaviour
# ---------------------------------------------------------------------------


class TestE1WarmStart:
    """Tests for warmup_step_init="law" / "default" and the κ_eff gate."""

    def test_well_conditioned_target_e1_fires(self):
        """isotropic N(0,I): k_eff ≈ 1 → e1_fired=True, step ≈ 1.22√d (within 15%)."""
        # d=6, sqrt(6) ≈ 2.449, target step ≈ 1.22*2.449 ≈ 2.99
        rng = jax.random.key(200)
        pos = jnp.zeros(D)

        result = mclmc_lrd_warmup(
            logdensity_fn,
            pos,
            rng,
            k=K,
            pilot_num_warmup=300,
            pilot_num_samples=1000,
            lrd_num_steps=500,
            num_chains=2,
            warmup_step_init="law",
        )
        diag = result.diagnostics
        kappa_str = str(round(diag["kappa_eff_pilot"], 3))
        assert diag["e1_fired"] is True, (
            "Expected e1_fired=True on isotropic target, got "
            + repr(diag["e1_fired"])
            + "; kappa_eff_pilot="
            + kappa_str
        )
        # Adapted step should be within 15% of 1.22*sqrt(d)
        import math

        target_step = 1.22 * math.sqrt(D)
        lrd_step = result.diagnostics["lrd_step_size"]
        step_ratio = lrd_step / target_step
        lrd_step_s = str(round(lrd_step, 3))
        target_step_s = str(round(target_step, 3))
        step_ratio_s = str(round(step_ratio, 3))
        assert 0.85 <= step_ratio <= 1.15, (
            "Adapted step "
            + lrd_step_s
            + " deviates >15% from target 1.22*sqrt(d)="
            + target_step_s
            + " (ratio="
            + step_ratio_s
            + ")"
        )

    def test_under_preconditioned_e1_does_not_fire(self):
        """Off-diagonal-correlated target at low rank → κ_eff >> 5 → e1_fired=False.

        The LRD IMM's DIAGONAL part (sigma) fully whitens any axis-aligned
        per-coordinate scaling, so an *axis-aligned* ill-conditioned Σ gives
        κ_eff ≈ 1 and WOULD fire E1.  κ_eff measures residual *off-diagonal
        correlation* conditioning, which only the low-rank part can remove.  We
        therefore use an equicorrelation target R = (1-ρ)I + ρ·11ᵀ (ρ=0.97, unit
        diagonal → nothing for the diagonal sigma to absorb): κ(R) ≈ 195.  With
        k=1 the LRD captures only the single large 1+(d-1)ρ eigendirection,
        leaving (d-1) residual eigenvalues of (1-ρ), so κ_eff ≈ 1/(1-ρ) ≈ 33 in
        the noiseless limit — well above the gate of 5 → e1_fired=False.

        The high-ρ target mixes slowly under the diagonal pilot, so a larger
        pilot (800) is used to keep the κ_eff estimate clean (300 draws straddle
        the gate). When e1_fired=False, "law" and "default" produce bit-identical
        output (same rng_key → same PRNG trajectory → same adapted (L, step)).
        """
        import numpy as np

        d_ill = 6
        rho = 0.97
        R = (1.0 - rho) * np.eye(d_ill) + rho * np.ones((d_ill, d_ill))
        Sigma_inv_ill = jnp.asarray(np.linalg.inv(R))

        def logdensity_ill(x):
            return -0.5 * jnp.dot(x, Sigma_inv_ill @ x)

        rng = jax.random.key(201)
        pos_ill = jnp.zeros(d_ill)

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            result_law = mclmc_lrd_warmup(
                logdensity_ill,
                pos_ill,
                rng,
                k=1,
                pilot_num_warmup=100,
                pilot_num_samples=800,
                lrd_num_steps=100,
                num_chains=2,
                warmup_step_init="law",
            )
            result_def = mclmc_lrd_warmup(
                logdensity_ill,
                pos_ill,
                rng,
                k=1,
                pilot_num_warmup=100,
                pilot_num_samples=800,
                lrd_num_steps=100,
                num_chains=2,
                warmup_step_init="default",
            )

        # κ_eff should be well above 5 (the top singular value >> 1).
        kappa_pilot = result_law.diagnostics["kappa_eff_pilot"]
        kappa_str = str(round(kappa_pilot, 3))
        assert result_law.diagnostics["e1_fired"] is False, (
            "Expected e1_fired=False on ill-conditioned target (kappa_eff_pilot="
            + kappa_str
            + " should be >5)"
        )
        # kappa_eff should be well above the gate threshold of 5.
        assert kappa_pilot > 5.0, (
            "Expected kappa_eff_pilot > 5 for ill-conditioned target with k=1, got "
            + kappa_str
        )
        # When e1_fired=False, "law" and "default" use the same DA init
        # → same rng_key → bit-identical adapted (L, step_size).
        law_L = float(result_law.L)
        def_L = float(result_def.L)
        assert abs(law_L - def_L) < 1e-6, (
            "law L=" + str(round(law_L, 6)) + " != default L=" + str(round(def_L, 6))
        )
        law_step = float(result_law.step_size)
        def_step = float(result_def.step_size)
        assert abs(law_step - def_step) < 1e-6, (
            "law step="
            + str(round(law_step, 6))
            + " != default step="
            + str(round(def_step, 6))
        )

    def test_large_budget_law_and_default_converge(self):
        """At large lrd_num_steps, "law" and "default" produce same (step,L) within tol."""
        rng = jax.random.key(202)
        pos = jnp.zeros(D)

        result_law = mclmc_lrd_warmup(
            logdensity_fn,
            pos,
            rng,
            k=K,
            pilot_num_warmup=300,
            pilot_num_samples=1000,
            lrd_num_steps=2000,
            num_chains=2,
            warmup_step_init="law",
        )
        result_def = mclmc_lrd_warmup(
            logdensity_fn,
            pos,
            rng,
            k=K,
            pilot_num_warmup=300,
            pilot_num_samples=1000,
            lrd_num_steps=2000,
            num_chains=2,
            warmup_step_init="default",
        )

        # At large budget, both should converge to approximately the same answer.
        # Tolerance is generous (10%) because they start from different inits and
        # use independent PRNG trajectories; we're testing convergence not identity.
        law_step = float(result_law.step_size)
        def_step = float(result_def.step_size)
        step_ratio = law_step / def_step
        law_L = float(result_law.L)
        def_L = float(result_def.L)
        L_ratio = law_L / def_L
        step_msg = (
            "law step="
            + str(round(law_step, 4))
            + " vs default step="
            + str(round(def_step, 4))
            + ": ratio="
            + str(round(step_ratio, 3))
            + " outside [0.90, 1.10] at large budget"
        )
        assert 0.90 <= step_ratio <= 1.10, step_msg
        L_msg = (
            "law L="
            + str(round(law_L, 4))
            + " vs default L="
            + str(round(def_L, 4))
            + ": ratio="
            + str(round(L_ratio, 3))
            + " outside [0.90, 1.10] at large budget"
        )
        assert 0.90 <= L_ratio <= 1.10, L_msg

    def test_e1_fired_and_kappa_eff_pilot_in_diagnostics(self):
        """e1_fired and kappa_eff_pilot must be present in diagnostics for both modes."""
        rng = jax.random.key(203)
        pos = jnp.zeros(D)

        for mode in ("law", "default"):
            result = mclmc_lrd_warmup(
                logdensity_fn,
                pos,
                rng,
                k=K,
                pilot_num_warmup=100,
                pilot_num_samples=200,
                lrd_num_steps=100,
                num_chains=2,
                warmup_step_init=mode,
            )
            diag = result.diagnostics
            assert (
                "e1_fired" in diag
            ), f"Missing 'e1_fired' key in diagnostics for warmup_step_init={mode!r}"
            assert (
                "kappa_eff_pilot" in diag
            ), f"Missing 'kappa_eff_pilot' key in diagnostics for warmup_step_init={mode!r}"
            assert isinstance(
                diag["e1_fired"], bool
            ), f"e1_fired must be bool, got {type(diag['e1_fired'])}"
            assert isinstance(
                diag["kappa_eff_pilot"], float
            ), f"kappa_eff_pilot must be float, got {type(diag['kappa_eff_pilot'])}"
            assert (
                diag["kappa_eff_pilot"] >= 1.0
            ), f"kappa_eff_pilot={diag['kappa_eff_pilot']} must be >= 1.0 (condition number)"
            # "default" mode must never fire E1
            if mode == "default":
                assert (
                    diag["e1_fired"] is False
                ), "warmup_step_init='default' must have e1_fired=False, got True"

    def test_invalid_warmup_step_init_raises(self):
        """Passing an unknown warmup_step_init must raise ValueError immediately."""
        with pytest.raises(ValueError, match="warmup_step_init"):
            mclmc_lrd_warmup(
                logdensity_fn,
                jnp.zeros(D),
                jax.random.key(0),
                warmup_step_init="scaling",
            )
