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


# ---------------------------------------------------------------------------
# Tests for rhat (rank-normalized split-R̂, Vehtari et al. 2021)
# ---------------------------------------------------------------------------


class RhatTest(chex.TestCase):
    """Tests for rank-normalized split-R̂."""

    def setUp(self):
        super().setUp()
        self.rng = jax.random.key(13)

    def _iid_normal(self, nchains=_NCHAINS, nsamples=_NSAMPLES):
        return jax.random.normal(self.rng, shape=(nchains, nsamples))

    def test_scalar_output_shape(self):
        result = diagnostics.rhat(self._iid_normal())
        assert result.shape == (), f"Expected scalar, got shape {result.shape}"

    def test_vector_output_shape(self):
        samples = jax.random.normal(self.rng, shape=(_NCHAINS, _NSAMPLES, 5))
        result = diagnostics.rhat(samples)
        assert result.shape == (5,), f"Expected (5,), got {result.shape}"

    def test_converged_chains_near_one(self):
        # IID samples → R̂ should be very close to 1.
        result = float(diagnostics.rhat(self._iid_normal()))
        assert (
            abs(result - 1.0) < 0.05
        ), f"rhat for iid samples should be ≈1, got {round(result, 4)}"

    def test_non_converged_chains_above_one(self):
        # Chains with distinct means → R̂ >> 1.
        key1, key2, key3, key4 = jax.random.split(self.rng, 4)
        means = jnp.array([0.0, 5.0, -5.0, 10.0])
        chains = jnp.stack(
            [
                jax.random.normal(k, shape=(_NSAMPLES,)) + m
                for k, m in zip([key1, key2, key3, key4], means)
            ]
        )
        result = float(diagnostics.rhat(chains))
        assert (
            result > 1.1
        ), f"rhat for non-converged chains should be > 1.1, got {round(result, 4)}"

    def test_scale_nonconvergence_detected(self):
        # Chains with same mean but very different variances (scale non-convergence).
        # The folded component catches this; plain split-R̂ on the raw draws may miss it.
        key1, key2 = jax.random.split(self.rng)
        chain_narrow = jax.random.normal(key1, shape=(2, _NSAMPLES)) * 0.1
        chain_wide = jax.random.normal(key2, shape=(2, _NSAMPLES)) * 10.0
        chains = jnp.concatenate([chain_narrow, chain_wide], axis=0)
        result = float(diagnostics.rhat(chains))
        # Scale non-convergence → R̂ should be clearly above 1.
        assert (
            result > 1.05
        ), f"rhat should detect scale non-convergence (> 1.05), got {round(result, 4)}"

    def test_axis_invariance(self):
        # Swapped chain/sample axes must give the same result.
        samples = self._iid_normal()
        samples_T = jnp.transpose(samples)  # (nsamples, nchains)
        rh_std = diagnostics.rhat(samples)
        rh_swp = diagnostics.rhat(samples_T, chain_axis=1, sample_axis=0)
        np.testing.assert_allclose(float(rh_std), float(rh_swp), rtol=1e-5)

    def test_negative_axes(self):
        samples = self._iid_normal()
        rh_pos = diagnostics.rhat(samples, chain_axis=0, sample_axis=1)
        rh_neg = diagnostics.rhat(samples, chain_axis=-2, sample_axis=-1)
        np.testing.assert_allclose(float(rh_pos), float(rh_neg), rtol=1e-5)

    def test_top_level_api(self):
        # blackjax.rhat must be the rank-normalized version, not the classic one.
        import blackjax

        samples = self._iid_normal()
        bj = float(blackjax.rhat(samples))
        direct = float(diagnostics.rhat(samples))
        np.testing.assert_allclose(bj, direct, rtol=1e-6)

    def test_arviz_calibration_converged(self):
        # IID normal: both should be ≈1; agree within 1%.
        az = pytest.importorskip("arviz")
        samples = np.array(self._iid_normal())
        bj = float(diagnostics.rhat(jnp.asarray(samples)))
        idata = az.convert_to_dataset({"x": samples})
        az_val = float(np.asarray(az.rhat(idata)["x"]).ravel()[0])
        rel = abs(bj - az_val) / max(abs(az_val), 1e-6)
        assert rel < 0.01, (
            f"rhat converged: blackjax={round(bj, 6)}"
            f" arviz={round(az_val, 6)} rel={round(rel, 6)}"
        )

    def test_arviz_calibration_nonconverged(self):
        # Chains with distinct means: both should detect non-convergence; agree within 5%.
        az = pytest.importorskip("arviz")
        key1, key2, key3, key4 = jax.random.split(self.rng, 4)
        means = jnp.array([0.0, 5.0, -5.0, 10.0])
        chains = np.array(
            jnp.stack(
                [
                    jax.random.normal(k, shape=(_NSAMPLES,)) + m
                    for k, m in zip([key1, key2, key3, key4], means)
                ]
            )
        )
        bj = float(diagnostics.rhat(jnp.asarray(chains)))
        idata = az.convert_to_dataset({"x": chains})
        az_val = float(np.asarray(az.rhat(idata)["x"]).ravel()[0])
        rel = abs(bj - az_val) / max(abs(az_val), 1e-6)
        assert rel < 0.05, (
            f"rhat non-converged: blackjax={round(bj, 4)}"
            f" arviz={round(az_val, 4)} rel={round(rel, 4)}"
        )

    def test_arviz_calibration_heavy_tail(self):
        # t(3) draws: heavier tails; agree within 1%.
        az = pytest.importorskip("arviz")
        samples = np.array(jax.random.t(self.rng, df=3.0, shape=(_NCHAINS, _NSAMPLES)))
        bj = float(diagnostics.rhat(jnp.asarray(samples)))
        idata = az.convert_to_dataset({"x": samples})
        az_val = float(np.asarray(az.rhat(idata)["x"]).ravel()[0])
        rel = abs(bj - az_val) / max(abs(az_val), 1e-6)
        assert rel < 0.01, (
            f"rhat t(3): blackjax={round(bj, 6)}"
            f" arviz={round(az_val, 6)} rel={round(rel, 6)}"
        )


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
        # arviz.convert_to_dataset expects shape (chain, draw) — do NOT add
        # a leading dimension; samples.shape is already (nchains, nsamples).
        # In arviz 0.23.x, az.ess()[var] returns a 1-element xarray DataArray,
        # so extract via np.asarray(...).ravel()[0] rather than float(.values).
        az = pytest.importorskip("arviz")
        samples = np.array(self._iid_normal())
        bj = float(diagnostics.ess_bulk(jnp.asarray(samples)))
        idata = az.convert_to_dataset({"x": samples})
        az_val = float(np.asarray(az.ess(idata, method="bulk")["x"]).ravel()[0])
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
        idata = az.convert_to_dataset({"x": samples})
        az_val = float(np.asarray(az.ess(idata, method="bulk")["x"]).ravel()[0])
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
        idata = az.convert_to_dataset({"x": samples})
        az_val = float(np.asarray(az.ess(idata, method="tail")["x"]).ravel()[0])
        rel = abs(bj - az_val) / max(abs(az_val), 1.0)
        assert rel < 0.10, (
            f"ess_tail normal: blackjax={round(bj, 2)}"
            f" arviz={round(az_val, 2)} rel={round(rel, 3)}"
        )

    def test_arviz_calibration_heavy_tail(self):
        az = pytest.importorskip("arviz")
        samples = np.array(jax.random.t(self.rng, df=3.0, shape=(_NCHAINS, _NSAMPLES)))
        bj = float(diagnostics.ess_tail(jnp.asarray(samples)))
        idata = az.convert_to_dataset({"x": samples})
        az_val = float(np.asarray(az.ess(idata, method="tail")["x"]).ravel()[0])
        rel = abs(bj - az_val) / max(abs(az_val), 1.0)
        assert rel < 0.10, (
            f"ess_tail t(3): blackjax={round(bj, 2)}"
            f" arviz={round(az_val, 2)} rel={round(rel, 3)}"
        )

    def test_prob_param_default_matches_arviz(self):
        # Default prob=0.90 gives (0.05, 0.95) quantiles — same as az.ess(method="tail").
        # Verify bit-match (within floating-point rounding) on normal and t(3) data.
        az = pytest.importorskip("arviz")
        for dist_name, samples in [
            ("normal", np.array(self._iid_normal())),
            (
                "t3",
                np.array(jax.random.t(self.rng, df=3.0, shape=(_NCHAINS, _NSAMPLES))),
            ),
        ]:
            bj = float(diagnostics.ess_tail(jnp.asarray(samples)))
            idata = az.convert_to_dataset({"x": samples})
            az_val = float(np.asarray(az.ess(idata, method="tail")["x"]).ravel()[0])
            rel = abs(bj - az_val) / max(abs(az_val), 1.0)
            assert rel < 0.01, (
                f"ess_tail default prob=0.90 ({dist_name}): "
                f"blackjax={round(bj, 4)} arviz={round(az_val, 4)} rel={round(rel, 6)}"
            )

    def test_prob_param_0_90_matches_5_95(self):
        # prob=0.90 → quantiles at (0.05, 0.95): explicit vs default must match.
        samples = self._iid_normal()
        bj_default = float(diagnostics.ess_tail(samples))
        bj_explicit = float(diagnostics.ess_tail(samples, prob=0.90))
        np.testing.assert_allclose(bj_default, bj_explicit, rtol=1e-6)

    def test_prob_param_changes_result(self):
        # Different prob values should produce different (but valid) ESS estimates.
        samples = self._iid_normal()
        bj_90 = float(diagnostics.ess_tail(samples, prob=0.90))
        bj_80 = float(diagnostics.ess_tail(samples, prob=0.80))
        # prob=0.80 → 10th/90th percentiles (less extreme tail); ESS can differ.
        assert bj_80 > 0 and bj_90 > 0, "ess_tail must be positive for any prob"
        assert (
            bj_80 != bj_90
        ), f"Different prob values must give different ESS, got same value {bj_90}"

    def test_funnel_tail_ess(self):
        # Neal's funnel: x[0] ~ N(0,9), x[1:] ~ N(0, exp(x[0]/2)).
        # From iid funnel draws, ess_tail should be positive.
        rng = self.rng
        k1, k2 = jax.random.split(rng)
        v = jax.random.normal(k1, shape=(_NCHAINS, _NSAMPLES)) * 3.0
        x = jax.random.normal(k2, shape=(_NCHAINS, _NSAMPLES)) * jnp.exp(v / 2.0)
        result = float(diagnostics.ess_tail(x))
        assert result > 0, f"ess_tail for funnel draws must be positive, got {result}"


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
        # arviz's PSIS k̂ (az.psislw / az.loo) operates on importance
        # log-weights, not raw samples, so there is no direct arviz equivalent
        # for pareto_khat(raw_samples).  This test gates on arviz being present
        # (dev-time only) and verifies the BlackJAX result is sensible:
        # Normal(0,1) is light-tailed so k̂ should be finite and well below 0.5.
        pytest.importorskip("arviz")
        x = np.array(jax.random.normal(self.rng, shape=(1000,)))
        bj_k = float(diagnostics.pareto_khat(jnp.asarray(x)))
        got_k = round(bj_k, 4)
        assert np.isfinite(
            bj_k
        ), f"pareto_khat must be finite for normal samples, got {got_k}"
        assert (
            bj_k < 0.3
        ), f"pareto_khat for normal should be <0.3 (light tail), got {got_k}"

    def test_arviz_calibration_cauchy(self):
        # For Cauchy samples (extreme tails), both should give k > 0.3.
        pytest.importorskip("arviz")
        x = np.array(jax.random.cauchy(self.rng, shape=(1000,)))
        bj_k = float(diagnostics.pareto_khat(jnp.asarray(x)))
        got_k = round(bj_k, 4)
        assert (
            bj_k > 0.3
        ), f"pareto_khat for Cauchy heavy tail expected >0.3 got {got_k}"


def _build_is_divergent(chain_specs, n_draws):
    """Build a synthetic ``(n_chains, n_draws)`` boolean array from
    ``(total, first_quarter_count, last_quarter_count)`` triples.

    Reproduces exact real per-chain totals *and* first-/last-quarter
    counts (the rest of a chain's divergences are placed in the middle
    half, where placement is irrelevant to what these tests check) so the
    quarter-profile fields can be tested against real validation-corpus
    numbers without depending on any external file path.
    """
    quarter = n_draws // 4
    rows = []
    for total, first_q, last_q in chain_specs:
        mid = total - first_q - last_q
        assert 0 <= first_q <= quarter and 0 <= last_q <= quarter and mid >= 0
        row = np.zeros(n_draws, dtype=bool)
        row[:first_q] = True
        if last_q:
            row[n_draws - last_q :] = True
        row[quarter : quarter + mid] = True
        rows.append(row)
    return jnp.asarray(np.stack(rows))


class DivergenceConcentrationTest(chex.TestCase):
    """Tests for the sampling-phase divergence-concentration warning.

    Fixtures reproduce real per-chain divergence counts and (for the
    flagged-chain quarter-profile tests) real first-/last-quarter counts,
    pulled from a retrospective 120-cell validation corpus of
    BlackJAX/tuningfork benchmark runs — the same corpus the default
    ``rate_threshold=0.02`` and the ``num_chains // 4`` minority cap were
    calibrated against.
    """

    _N_DRAWS = 2000

    # A minority-of-two case: one dominant chain (index 6) landing on a
    # bad post-warmup start state (18.75% overall, 72.2% in just the
    # first quarter, recovering to 0.2% in the last), plus one chain
    # (index 3) barely over the 2% threshold. Both cross rate_threshold;
    # num_flagged=2 == max(1, 8 // 4), the boundary of the minority cap.
    # (total, first_quarter_count, last_quarter_count) per chain, quarter=500.
    _STORM_SPECS = [
        (7, 0, 1),
        (6, 1, 2),
        (9, 2, 4),
        (44, 1, 0),
        (6, 3, 1),
        (9, 3, 4),
        (375, 361, 1),
        (15, 9, 1),
    ]
    _STORM_FLAGGED = (3, 6)

    # A single-chain minority case at a much smaller D: chain 7 at 2.9%,
    # right at the edge of the default 2% threshold, degrading rather
    # than recovering (0% first quarter, 11.4% last quarter).
    _SINGLE_FLAGGED_SPECS = [
        (1, 0, 0),
        (1, 0, 0),
        (0, 0, 0),
        (3, 0, 0),
        (0, 0, 0),
        (4, 0, 0),
        (0, 0, 0),
        (58, 0, 57),
    ]
    _SINGLE_FLAGGED_CHAIN = 7

    # Every chain in the ensemble globally elevated (12.85%-26.70% each):
    # a model-geometry signature, not a minority bad-start chain — all 8
    # of 8 chains flagged, over the max(1, 8 // 4) = 2 minority cap, so
    # this must NOT warn even though the fields are all populated.
    _ENSEMBLE_COUNTS = [316, 261, 257, 534, 272, 384, 302, 271]

    # Ordinary scattered divergences well below any threshold: healthy.
    _HEALTHY_COUNTS = [2, 6, 3, 4, 7, 3, 4, 7]

    @chex.all_variants(with_pmap=False)
    def test_storm_warns_minority_of_two(self):
        is_divergent = _build_is_divergent(self._STORM_SPECS, self._N_DRAWS)
        report = self.variant(diagnostics.divergence_concentration)(is_divergent)
        assert bool(report.warn) is True
        assert int(report.num_flagged) == 2
        np.testing.assert_array_equal(
            np.asarray(report.flagged),
            [False, False, False, True, False, False, True, False],
        )
        assert int(report.total_divergences) == 471
        np.testing.assert_allclose(float(report.rates[6]), 375 / 2000, rtol=1e-6)

    def test_storm_quarter_profile_dominant_chain(self):
        # Real evidence: chain 6 diverges heavily early and recovers.
        is_divergent = _build_is_divergent(self._STORM_SPECS, self._N_DRAWS)
        report = diagnostics.divergence_concentration(is_divergent)
        np.testing.assert_allclose(float(report.early_rate[6]), 361 / 500, rtol=1e-6)
        np.testing.assert_allclose(float(report.late_rate[6]), 1 / 500, rtol=1e-6)
        assert float(report.early_rate[6]) > float(report.late_rate[6])

    def test_storm_message_lists_both_flagged_chains_no_causal_claims(self):
        is_divergent = _build_is_divergent(self._STORM_SPECS, self._N_DRAWS)
        report = diagnostics.divergence_concentration(is_divergent)
        msg = diagnostics.format_divergence_warning(report)
        assert "chain 6:" in msg, msg
        assert "chain 3:" in msg, msg
        assert "first quarter" in msg and "in the last" in msg
        assert "median" in msg
        # Factual observation only -- no causal/remedy language in the
        # message text itself (that stays in the docstring as documentation).
        for banned in ("geometry", "post-warmup start", "reparameterization"):
            assert banned not in msg, msg

    def test_single_flagged_chain_warns(self):
        is_divergent = _build_is_divergent(self._SINGLE_FLAGGED_SPECS, self._N_DRAWS)
        report = diagnostics.divergence_concentration(is_divergent)
        assert bool(report.warn) is True
        assert int(report.num_flagged) == 1
        assert bool(report.flagged[self._SINGLE_FLAGGED_CHAIN])
        np.testing.assert_allclose(
            float(report.rates[self._SINGLE_FLAGGED_CHAIN]), 58 / 2000, rtol=1e-6
        )
        np.testing.assert_allclose(
            float(report.late_rate[self._SINGLE_FLAGGED_CHAIN]), 57 / 500, rtol=1e-6
        )
        msg = diagnostics.format_divergence_warning(report)
        assert msg.startswith(f"chain {self._SINGLE_FLAGGED_CHAIN}" ":"), msg

    def test_ensemble_wide_elevation_does_not_warn(self):
        # 8 of 8 chains flagged > max(1, 8 // 4) = 2: fields are populated
        # but this is out of scope for the minority-outlier trigger.
        report = diagnostics.divergence_concentration_from_counts(
            jnp.array(self._ENSEMBLE_COUNTS), self._N_DRAWS
        )
        assert bool(report.warn) is False
        assert int(report.num_flagged) == 8
        np.testing.assert_array_equal(np.asarray(report.flagged), [True] * 8)
        assert int(report.total_divergences) == sum(self._ENSEMBLE_COUNTS)
        assert diagnostics.format_divergence_warning(report) == ""

    def test_minority_cap_boundary(self):
        # num_chains=8 -> cap=2. Exactly 3 flagged chains must NOT warn,
        # even though each individually clears rate_threshold.
        counts = jnp.array([50, 50, 50, 0, 0, 0, 0, 0])  # 3 chains at 2.5%
        report = diagnostics.divergence_concentration_from_counts(counts, self._N_DRAWS)
        assert int(report.num_flagged) == 3
        assert bool(report.warn) is False

    def test_minority_cap_at_small_num_chains(self):
        # num_chains=4 -> cap=max(1, 1)=1. One flagged chain warns; two does not.
        one_flagged = jnp.array([50, 0, 0, 0])
        two_flagged = jnp.array([50, 50, 0, 0])
        report_one = diagnostics.divergence_concentration_from_counts(
            one_flagged, self._N_DRAWS
        )
        report_two = diagnostics.divergence_concentration_from_counts(
            two_flagged, self._N_DRAWS
        )
        assert bool(report_one.warn) is True
        assert bool(report_two.warn) is False

    def test_healthy_scatter_no_warning(self):
        report = diagnostics.divergence_concentration_from_counts(
            jnp.array(self._HEALTHY_COUNTS), self._N_DRAWS
        )
        assert bool(report.warn) is False
        assert int(report.num_flagged) == 0
        assert diagnostics.format_divergence_warning(report) == ""

    def test_zero_divergences_no_warning(self):
        report = diagnostics.divergence_concentration_from_counts(
            jnp.zeros(8, dtype=int), self._N_DRAWS
        )
        assert bool(report.warn) is False
        assert int(report.num_flagged) == 0
        assert diagnostics.format_divergence_warning(report) == ""

    def test_from_counts_quarters_are_nan_and_degrade_gracefully(self):
        # The counts-only entry point has no per-draw resolution: quarter
        # fields are NaN, and the message omits the quarter parenthetical
        # rather than printing "nan%".
        counts = jnp.array([1, 1, 0, 3, 0, 4, 0, 58])  # matches the single-flagged case
        report = diagnostics.divergence_concentration_from_counts(counts, self._N_DRAWS)
        assert bool(report.warn) is True
        assert np.isnan(float(report.early_rate[7]))
        assert np.isnan(float(report.late_rate[7]))
        msg = diagnostics.format_divergence_warning(report)
        assert "nan" not in msg.lower(), msg
        assert "first quarter" not in msg, msg
        assert msg.startswith("chain 7:"), msg

    def test_from_is_divergent_matches_from_counts_trigger(self):
        # Trigger-level fields (warn, flagged, rates, D) must agree between
        # the two entry points for the same underlying counts; only the
        # quarter fields differ (real vs NaN).
        is_divergent = _build_is_divergent(self._STORM_SPECS, self._N_DRAWS)
        report_flags = diagnostics.divergence_concentration(is_divergent)
        counts = [total for total, _, _ in self._STORM_SPECS]
        report_counts = diagnostics.divergence_concentration_from_counts(
            jnp.array(counts), self._N_DRAWS
        )
        assert bool(report_flags.warn) == bool(report_counts.warn)
        assert int(report_flags.num_flagged) == int(report_counts.num_flagged)
        np.testing.assert_allclose(
            np.asarray(report_flags.rates), np.asarray(report_counts.rates), rtol=1e-6
        )
        assert int(report_flags.total_divergences) == int(
            report_counts.total_divergences
        )

    def test_rate_threshold_override(self):
        # Raising the threshold above the single-flagged cell's rate silences it.
        counts = jnp.array([1, 1, 0, 3, 0, 4, 0, 58])
        report_default = diagnostics.divergence_concentration_from_counts(
            counts, self._N_DRAWS
        )
        report_high_threshold = diagnostics.divergence_concentration_from_counts(
            counts, self._N_DRAWS, rate_threshold=0.05
        )
        assert bool(report_default.warn) is True
        assert bool(report_high_threshold.warn) is False

    def test_median_other_rate_single_chain_guard(self):
        # num_chains=1: there are no "other" chains -- must not crash, and
        # median_other_rate must be NaN rather than raising.
        report = diagnostics.divergence_concentration_from_counts(
            jnp.array([5]), self._N_DRAWS
        )
        assert np.isnan(float(report.median_other_rate[0]))

    def test_multinomial_p_value_is_context_not_trigger(self):
        # The healthy-scatter cell can still have a small multinomial
        # p-value (real chains are not exchangeable) yet must not warn --
        # the per-chain rate threshold, not the p-value, decides `warn`.
        report = diagnostics.divergence_concentration_from_counts(
            jnp.array(self._HEALTHY_COUNTS), self._N_DRAWS
        )
        assert bool(report.warn) is False  # regardless of what p_value says
        assert np.isfinite(float(report.multinomial_p_value))

    def test_p_value_nan_when_no_divergences(self):
        report = diagnostics.divergence_concentration_from_counts(
            jnp.zeros(6, dtype=int), self._N_DRAWS
        )
        assert np.isnan(float(report.multinomial_p_value))


if __name__ == "__main__":
    absltest.main()
