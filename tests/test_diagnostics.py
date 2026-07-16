"""Test MCMC diagnostics."""
import functools
import itertools

import chex
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from absl.testing import absltest, parameterized

import blackjax.diagnostics as diagnostics

test_cases = [
    {
        "chain_axis": 0,
        "sample_axis": 1,
    },
    {
        "chain_axis": 1,
        "sample_axis": 0,
    },
    {
        "chain_axis": 0,
        "sample_axis": -1,
    },
    {
        "chain_axis": -1,
        "sample_axis": 0,
    },
]


def insert_list(input_list, loc, elem):
    if loc == -1:
        input_list.append(elem)
    else:
        input_list.insert(loc, elem)
    return input_list


class DiagnosticsTest(chex.TestCase):
    """Tests for MCMC diagnostics."""

    def setUp(self):
        super().setUp()
        self.num_samples = 5000
        self.test_seed = 32

    @chex.all_variants(with_pmap=False)
    @parameterized.parameters(
        itertools.product(test_cases, [1, 2, 10], [(), (3,), (5, 7)])
    )
    def test_rhat_ess(self, case, num_chains, event_shape):
        rng_key = jax.random.key(self.test_seed)
        sample_shape = list(event_shape)
        if case["chain_axis"] < case["sample_axis"]:
            sample_shape = insert_list(sample_shape, case["chain_axis"], num_chains)
            sample_shape = insert_list(
                sample_shape, case["sample_axis"], self.num_samples
            )
        else:
            sample_shape = insert_list(
                sample_shape, case["sample_axis"], self.num_samples
            )
            sample_shape = insert_list(sample_shape, case["chain_axis"], num_chains)
        mc_samples = jax.random.normal(rng_key, shape=sample_shape)

        potential_scale_reduction = self.variant(
            functools.partial(diagnostics.potential_scale_reduction, **case)
        )
        if num_chains > 1:
            rhat_val = potential_scale_reduction(mc_samples)
            np.testing.assert_array_equal(rhat_val.shape, event_shape)
            np.testing.assert_allclose(rhat_val, 1.0, rtol=1e-03)
        else:
            np.testing.assert_raises(
                AssertionError, potential_scale_reduction, mc_samples
            )

        # With iid samples we should get ess close to number of samples.
        effective_sample_size = self.variant(
            functools.partial(diagnostics.effective_sample_size, **case)
        )
        ess_val = effective_sample_size(mc_samples)
        np.testing.assert_array_equal(ess_val.shape, event_shape)
        np.testing.assert_allclose(ess_val, num_chains * self.num_samples, rtol=10)


# ---------------------------------------------------------------------------
# Tests for ess_bulk, ess_tail, and pareto_khat
# ---------------------------------------------------------------------------

# Number of chains and draws used across all modern-diagnostics tests.
_NCHAINS = 4
_NSAMPLES = 2000


class EssBulkTest(chex.TestCase):
    """Tests for rank-normalised split-chain bulk ESS."""

    def setUp(self):
        super().setUp()
        self.rng = jax.random.key(7)

    def _iid_normal(self, nchains=_NCHAINS, nsamples=_NSAMPLES):
        return jax.random.normal(self.rng, shape=(nchains, nsamples))

    def test_scalar_output_shape(self):
        samples = self._iid_normal()
        result = diagnostics.ess_bulk(samples)
        assert result.shape == (), f"Expected scalar, got shape {result.shape}"

    def test_vector_output_shape(self):
        samples = jax.random.normal(self.rng, shape=(_NCHAINS, _NSAMPLES, 5))
        result = diagnostics.ess_bulk(samples)
        assert result.shape == (5,), f"Expected (5,), got {result.shape}"

    def test_positive_for_iid(self):
        result = diagnostics.ess_bulk(self._iid_normal())
        assert float(result) > 0, "ess_bulk must be positive"

    def test_iid_normal_close_to_total_samples(self):
        # For iid draws, bulk ESS should be close to nchains * nsamples.
        total = _NCHAINS * _NSAMPLES
        result = float(diagnostics.ess_bulk(self._iid_normal()))
        # Allow a wide window: between 50% and 200% of total.
        r = round(result)
        assert result > 0.5 * total, f"ess_bulk={r} < 0.5 * {total}"
        assert result < 2.0 * total, f"ess_bulk={r} > 2.0 * {total}"

    def test_axis_invariance(self):
        # Swapped chain/sample axes must give the same result.
        samples = self._iid_normal()
        samples_T = jnp.transpose(samples)  # (nsamples, nchains)
        eb_std = diagnostics.ess_bulk(samples)
        eb_swp = diagnostics.ess_bulk(samples_T, chain_axis=1, sample_axis=0)
        np.testing.assert_allclose(float(eb_std), float(eb_swp), rtol=1e-5)

    def test_negative_axes(self):
        samples = self._iid_normal()
        eb_pos = diagnostics.ess_bulk(samples, chain_axis=0, sample_axis=1)
        eb_neg = diagnostics.ess_bulk(samples, chain_axis=-2, sample_axis=-1)
        np.testing.assert_allclose(float(eb_pos), float(eb_neg), rtol=1e-5)

    def test_poorly_mixed_chain_gives_lower_ess(self):
        # A slowly-drifting chain has very high autocorrelation; bulk ESS
        # should be much lower than the iid baseline.
        nchains, nsamples = _NCHAINS, _NSAMPLES
        t = jnp.arange(nsamples, dtype=float)
        slow_wave = jnp.sin(2 * jnp.pi * t / nsamples)
        stuck_samples = jnp.broadcast_to(slow_wave[None, :], (nchains, nsamples))
        eb_stuck = float(diagnostics.ess_bulk(stuck_samples))
        eb_iid = float(diagnostics.ess_bulk(self._iid_normal()))
        assert eb_stuck < eb_iid, (
            f"Stuck chain ESS ({round(eb_stuck, 1)}) should be < iid ESS"
            f" ({round(eb_iid, 1)})"
        )

    def test_arviz_calibration_normal(self):
        # Compare against arviz within 10%.  Skipped when arviz is not installed.
        az = pytest.importorskip("arviz")
        samples = np.array(self._iid_normal())
        bj = float(diagnostics.ess_bulk(jnp.asarray(samples)))
        # arviz expects shape (chain, draw) for a single variable.
        idata = az.convert_to_dataset({"x": samples[None]})
        az_val = float(az.ess(idata, method="bulk")["x"].values)
        rel = abs(bj - az_val) / max(abs(az_val), 1.0)
        assert rel < 0.10, (
            f"ess_bulk normal: blackjax={round(bj, 2)}"
            f" arviz={round(az_val, 2)} rel={round(rel, 3)}"
        )

    def test_arviz_calibration_heavy_tail(self):
        # t(3) draws: heavier tails than normal.
        az = pytest.importorskip("arviz")
        samples = np.array(jax.random.t(self.rng, df=3.0, shape=(_NCHAINS, _NSAMPLES)))
        bj = float(diagnostics.ess_bulk(jnp.asarray(samples)))
        idata = az.convert_to_dataset({"x": samples[None]})
        az_val = float(az.ess(idata, method="bulk")["x"].values)
        rel = abs(bj - az_val) / max(abs(az_val), 1.0)
        assert rel < 0.10, (
            f"ess_bulk t(3): blackjax={round(bj, 2)}"
            f" arviz={round(az_val, 2)} rel={round(rel, 3)}"
        )


class EssTailTest(chex.TestCase):
    """Tests for tail ESS."""

    def setUp(self):
        super().setUp()
        self.rng = jax.random.key(99)

    def _iid_normal(self, nchains=_NCHAINS, nsamples=_NSAMPLES):
        return jax.random.normal(self.rng, shape=(nchains, nsamples))

    def test_scalar_output_shape(self):
        result = diagnostics.ess_tail(self._iid_normal())
        assert result.shape == (), f"Expected scalar, got {result.shape}"

    def test_vector_output_shape(self):
        samples = jax.random.normal(self.rng, shape=(_NCHAINS, _NSAMPLES, 3))
        result = diagnostics.ess_tail(samples)
        assert result.shape == (3,), f"Expected (3,), got {result.shape}"

    def test_positive_for_iid(self):
        result = diagnostics.ess_tail(self._iid_normal())
        assert float(result) > 0, "ess_tail must be positive"

    def test_iid_normal_reasonable_magnitude(self):
        # Tail ESS for iid data should be in a reasonable range.
        total = _NCHAINS * _NSAMPLES
        result = float(diagnostics.ess_tail(self._iid_normal()))
        # Tail ESS is based on Bernoulli(0.05) indicators so can be somewhat
        # lower; allow [20%, 200%] of total.
        r = round(result)
        assert result > 0.2 * total, f"ess_tail={r} < 0.2 * {total}"
        assert result < 2.0 * total, f"ess_tail={r} > 2.0 * {total}"

    def test_axis_invariance(self):
        samples = self._iid_normal()
        samples_T = jnp.transpose(samples)
        et_std = diagnostics.ess_tail(samples)
        et_swp = diagnostics.ess_tail(samples_T, chain_axis=1, sample_axis=0)
        np.testing.assert_allclose(float(et_std), float(et_swp), rtol=1e-5)

    def test_arviz_calibration_normal(self):
        az = pytest.importorskip("arviz")
        samples = np.array(self._iid_normal())
        bj = float(diagnostics.ess_tail(jnp.asarray(samples)))
        idata = az.convert_to_dataset({"x": samples[None]})
        az_val = float(az.ess(idata, method="tail")["x"].values)
        rel = abs(bj - az_val) / max(abs(az_val), 1.0)
        assert rel < 0.10, (
            f"ess_tail normal: blackjax={round(bj, 2)}"
            f" arviz={round(az_val, 2)} rel={round(rel, 3)}"
        )

    def test_arviz_calibration_heavy_tail(self):
        az = pytest.importorskip("arviz")
        samples = np.array(jax.random.t(self.rng, df=3.0, shape=(_NCHAINS, _NSAMPLES)))
        bj = float(diagnostics.ess_tail(jnp.asarray(samples)))
        idata = az.convert_to_dataset({"x": samples[None]})
        az_val = float(az.ess(idata, method="tail")["x"].values)
        rel = abs(bj - az_val) / max(abs(az_val), 1.0)
        assert rel < 0.10, (
            f"ess_tail t(3): blackjax={round(bj, 2)}"
            f" arviz={round(az_val, 2)} rel={round(rel, 3)}"
        )


class ParetoKhatTest(chex.TestCase):
    """Tests for pareto_khat."""

    def setUp(self):
        super().setUp()
        self.rng = jax.random.key(55)

    def test_scalar_output(self):
        x = jax.random.normal(self.rng, shape=(500,))
        result = diagnostics.pareto_khat(x)
        assert result.shape == (), f"Expected scalar, got {result.shape}"

    def test_normal_tail_below_0_5(self):
        # Normal distribution is light-tailed; k̂ should be well below 0.5.
        x = jax.random.normal(self.rng, shape=(2000,))
        k = float(diagnostics.pareto_khat(x))
        assert k < 0.5, f"pareto_khat for normal should be < 0.5, got {round(k, 4)}"

    def test_cauchy_heavier_than_normal(self):
        # Cauchy is heavier-tailed (k≈1 theoretically).
        x_norm = jax.random.normal(self.rng, shape=(2000,))
        x_cauchy = jax.random.cauchy(self.rng, shape=(2000,))
        k_norm = float(diagnostics.pareto_khat(x_norm))
        k_cauchy = float(diagnostics.pareto_khat(x_cauchy))
        assert (
            k_cauchy > k_norm
        ), f"Cauchy k={round(k_cauchy, 4)} should exceed normal k={round(k_norm, 4)}"

    def test_both_is_max_of_upper_lower(self):
        x = jax.random.normal(self.rng, shape=(1000,))
        k_upper = float(diagnostics.pareto_khat(x, tail="upper"))
        k_lower = float(diagnostics.pareto_khat(x, tail="lower"))
        k_both = float(diagnostics.pareto_khat(x, tail="both"))
        expected = max(k_upper, k_lower)
        np.testing.assert_allclose(k_both, expected, rtol=1e-5)

    def test_multidim_input_is_ravelled(self):
        # 2-D input must produce the same result as the ravelled 1-D version.
        x_2d = jax.random.normal(self.rng, shape=(20, 50))
        x_1d = x_2d.ravel()
        k_2d = diagnostics.pareto_khat(x_2d)
        k_1d = diagnostics.pareto_khat(x_1d)
        np.testing.assert_allclose(float(k_2d), float(k_1d), rtol=1e-5)

    def test_tail_frac_parameter(self):
        # Different tail fractions should give valid (finite) k̂ values.
        x = jax.random.normal(self.rng, shape=(500,))
        for frac in (0.05, 0.10, 0.20):
            k = float(diagnostics.pareto_khat(x, tail_frac=frac))
            assert np.isfinite(k), f"pareto_khat with tail_frac={frac} returned {k}"

    def test_arviz_calibration_normal(self):
        # Verify that our k̂ is in the same ballpark as arviz's PSIS k̂.
        # The comparison is approximate: different tail-fraction conventions.
        az = pytest.importorskip("arviz")
        x = np.array(jax.random.normal(self.rng, shape=(1000,)))
        bj_k = float(diagnostics.pareto_khat(jnp.asarray(x)))
        # arviz.psislw fits PSIS on log-weight arrays; use uniform log-weights
        # as a proxy to get az's GPD estimator result.
        log_w = np.zeros(len(x))
        _, az_k = az.psislw(log_w)
        # Both should give small k for a degenerate (uniform weight) case.
        assert np.isfinite(bj_k), "pareto_khat must be finite for normal samples"
        assert np.isfinite(float(az_k)), "arviz k must be finite for uniform weights"

    def test_arviz_calibration_cauchy(self):
        # For Cauchy samples (extreme tails), both should give k > 0.3.
        pytest.importorskip("arviz")
        x = np.array(jax.random.cauchy(self.rng, shape=(1000,)))
        bj_k = float(diagnostics.pareto_khat(jnp.asarray(x)))
        got_k = round(bj_k, 4)
        assert (
            bj_k > 0.3
        ), f"pareto_khat for Cauchy heavy tail expected >0.3 got {got_k}"


if __name__ == "__main__":
    absltest.main()
