"""Test MCMC diagnostics."""
import functools
import itertools

import chex
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from absl.testing import absltest, parameterized
from scipy.stats import norm, rankdata

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

    @chex.all_variants(with_pmap=False)
    @parameterized.parameters(1, 2)
    def test_ess_returns_zero_for_numerically_degenerate_chains(self, num_chains):
        num_samples = 2000
        samples_shape = (num_chains, num_samples)
        random_samples = jax.random.normal(
            jax.random.key(self.test_seed), shape=samples_shape
        )
        constant_samples = jnp.zeros(samples_shape)
        constant_with_different_chain_means = jnp.broadcast_to(
            jnp.arange(num_chains)[:, None], samples_shape
        )
        near_constant_samples = 1e-30 * random_samples
        samples = jnp.stack(
            [
                constant_samples,
                constant_with_different_chain_means,
                near_constant_samples,
                random_samples,
            ],
            axis=-1,
        )

        effective_sample_size = self.variant(diagnostics.effective_sample_size)
        ess = effective_sample_size(samples)

        np.testing.assert_array_equal(ess[:3], jnp.zeros(3))
        assert ess[3] > 0

    def test_ess_can_exceed_draw_count_for_antithetic_chain(self):
        num_samples = 2000
        samples = jnp.tile(jnp.array([-1.0, 1.0]), num_samples // 2)[None, :]

        ess = diagnostics.effective_sample_size(samples)

        assert ess > num_samples


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


# ---------------------------------------------------------------------------
# Regressions for tied-draw handling in the rank-normalized diagnostics
# ---------------------------------------------------------------------------


def _pairwise_average_ranks(x):
    """1-indexed average ranks along axis 0, by direct pairwise counting.

    Uses ``rank(v) = (1 + #{x < v} + #{x <= v}) / 2`` — the definition of the
    mean of the ordinal ranks a tie group spans.  This shares no code path
    with the sort/cumulative-reduction implementation under test, so it is an
    independent oracle rather than a restatement of it.
    """
    a = np.asarray(x)
    flat = a.reshape(a.shape[0], -1)
    out = np.empty(flat.shape, dtype=np.float64)
    for j in range(flat.shape[1]):
        col = flat[:, j]
        n_lt = (col[None, :] < col[:, None]).sum(axis=1)
        n_le = (col[None, :] <= col[:, None]).sum(axis=1)
        out[:, j] = (1.0 + n_lt + n_le) / 2.0
    return out.reshape(a.shape)


def _reference_rank_normalize(x):
    """Blom rank-normalization of ``(nchains, nsamples, ...)`` draws, in NumPy."""
    n = x.shape[0] * x.shape[1]
    flat = np.asarray(x).reshape(n, *x.shape[2:])
    ranks = _pairwise_average_ranks(flat)
    return norm.ppf((ranks - 3.0 / 8) / (n + 1.0 / 4)).reshape(x.shape)


def _reference_split(x):
    """Split each chain of a ``(nchains, nsamples, ...)`` array in half."""
    half = x.shape[1] // 2
    x = x[:, : 2 * half]
    return np.concatenate([x[:, :half], x[:, half:]], axis=0)


def _reference_split_rhat(x):
    """Plain split-R-hat on already-split, already-normalized draws."""
    num_samples = x.shape[1]
    between = num_samples * x.mean(axis=1).var(axis=0, ddof=1)
    within = x.var(axis=1, ddof=1).mean(axis=0)
    return np.sqrt((between / within + num_samples - 1) / num_samples)


def _reference_rhat(x):
    """Independent NumPy implementation of rank-normalized split-R-hat."""
    x_split = _reference_split(np.asarray(x))
    r_bulk = _reference_split_rhat(_reference_rank_normalize(x_split))
    pooled = x_split.reshape(x_split.shape[0] * x_split.shape[1], *x_split.shape[2:])
    folded = np.abs(x_split - np.median(pooled, axis=0))
    r_tail = _reference_split_rhat(_reference_rank_normalize(folded))
    return np.maximum(r_bulk, r_tail)


class RankNormalizeTiesTest(chex.TestCase):
    """Tied draws must receive equal, permutation-invariant rank scores.

    Ordinal (double-``argsort``) ranking hands equal values different ranks
    according to where they sit in the pooled array, which manufactures
    chain/time structure out of ties and corrupts :func:`rhat` and
    :func:`ess_bulk` for repeated states, indicator observables and
    rejection-heavy chains.
    """

    def setUp(self):
        super().setUp()
        self.rng = jax.random.key(20260906)

    def _tied_draws(self, nchains=4, nsamples=48, num_levels=5, shape=()):
        """Draws from a small discrete grid, so most values are tied."""
        levels = jax.random.randint(
            self.rng, shape=(nchains, nsamples, *shape), minval=0, maxval=num_levels
        )
        return levels.astype(jnp.float32)

    # -- the ranking primitive ------------------------------------------

    def test_average_ranks_match_pairwise_reference(self):
        pooled = np.asarray(self._tied_draws()).reshape(4 * 48)
        ranks = np.asarray(diagnostics._average_ranks(jnp.asarray(pooled)))
        np.testing.assert_array_equal(ranks, _pairwise_average_ranks(pooled))

    def test_average_ranks_match_scipy_rankdata(self):
        pooled = np.asarray(self._tied_draws(num_levels=3)).reshape(-1)
        ranks = np.asarray(diagnostics._average_ranks(jnp.asarray(pooled)))
        np.testing.assert_array_equal(ranks, rankdata(pooled, method="average"))

    def test_average_ranks_are_independent_per_event(self):
        # Trailing dimensions must be ranked independently of one another.
        draws = self._tied_draws(shape=(3,))
        pooled = np.asarray(draws).reshape(4 * 48, 3)
        ranks = np.asarray(diagnostics._average_ranks(jnp.asarray(pooled)))
        np.testing.assert_array_equal(ranks, _pairwise_average_ranks(pooled))
        for event in range(3):
            column = np.asarray(
                diagnostics._average_ranks(jnp.asarray(pooled[:, event]))
            )
            np.testing.assert_array_equal(ranks[:, event], column)

    def test_distinct_values_keep_ordinal_ranks(self):
        # With no ties the average rank reduces to the ordinal rank, so the
        # pre-existing behaviour on continuous draws is untouched.
        # A shuffled arange, so the values are distinct by construction —
        # float32 normal draws do collide occasionally at this size.
        draws = jax.random.permutation(self.rng, jnp.arange(4 * 48)).astype(jnp.float32)
        ranks = np.asarray(diagnostics._average_ranks(draws))
        ordinal = np.asarray(jnp.argsort(jnp.argsort(draws)) + 1, dtype=np.float64)
        np.testing.assert_array_equal(ranks, ordinal)

    # -- permutation symmetry -------------------------------------------

    def test_tied_draws_receive_identical_scores(self):
        draws = self._tied_draws()
        z = np.asarray(diagnostics._rank_normalize(draws))
        values = np.asarray(draws).reshape(-1)
        scores = z.reshape(-1)
        for level in np.unique(values):
            group = scores[values == level]
            np.testing.assert_array_equal(
                group,
                np.full_like(group, group[0]),
                err_msg=f"tied value {level} received unequal scores",
            )

    def test_rank_normalize_is_permutation_invariant(self):
        draws = self._tied_draws()
        z = np.asarray(diagnostics._rank_normalize(draws))

        rng = np.random.default_rng(0)
        pooled = np.asarray(draws).reshape(-1)
        permutation = rng.permutation(pooled.size)
        shuffled = pooled[permutation].reshape(draws.shape)
        z_shuffled = np.asarray(diagnostics._rank_normalize(jnp.asarray(shuffled)))

        inverse = np.empty_like(permutation)
        inverse[permutation] = np.arange(permutation.size)
        restored = z_shuffled.reshape(-1)[inverse].reshape(draws.shape)
        np.testing.assert_array_equal(
            z,
            restored,
            err_msg="rank normalization is not invariant to pooled permutation",
        )

    # -- degenerate inputs ----------------------------------------------

    def test_constant_input_normalizes_to_zero(self):
        # A constant input must not acquire artificial spread.
        draws = jnp.full((4, 48), 2.5)
        z = np.asarray(diagnostics._rank_normalize(draws))
        np.testing.assert_allclose(z, np.zeros_like(z), atol=1e-6)

    def test_constant_input_rhat_is_undefined(self):
        # R-hat is 0/0 for a constant input: undefined, not a large number.
        result = np.asarray(diagnostics.rhat(jnp.full((4, 64), 2.5)))
        assert np.isnan(result), f"expected undefined R-hat, got {result}"

    def test_constant_input_ess_bulk_reports_degeneracy(self):
        # BlackJAX flags zero-variance variables with ESS 0; rank
        # normalization must not turn the constant into a varying sequence
        # first.  (This deliberately differs from ArviZ, which reports N.)
        result = np.asarray(diagnostics.ess_bulk(jnp.full((4, 64), 2.5)))
        np.testing.assert_array_equal(result, 0.0)

    def test_mixed_constant_and_varying_events(self):
        # Per-event degeneracy must be decided per event, not globally.
        varying = jax.random.normal(self.rng, shape=(4, 64))
        draws = jnp.stack([jnp.full((4, 64), 1.0), varying], axis=-1)
        ess = np.asarray(diagnostics.ess_bulk(draws))
        assert ess[0] == 0.0, f"constant event must report ESS 0, got {ess[0]}"
        assert ess[1] > 0.0, f"varying event must report positive ESS, got {ess[1]}"

    # -- NaN is a missing observation, not an ordered tie ----------------

    def _contaminated(self):
        """Deterministic draws whose component 0 holds one NaN, 1 is clean."""
        base = np.arange(4 * 64, dtype=np.float64).reshape(4, 64) % 7
        dirty = base.copy()
        dirty[1, 5] = np.nan
        return jnp.asarray(np.stack([dirty, base], axis=-1))

    def test_nan_component_cannot_be_ranked(self):
        pooled = jnp.array([1.0, jnp.nan, 0.0, 2.0, 0.0])
        ranks = np.asarray(diagnostics._average_ranks(pooled))
        assert np.all(
            np.isnan(ranks)
        ), f"a component holding a missing observation has no ranking: {ranks}"

    def test_nan_does_not_poison_a_finite_sibling_component(self):
        draws = self._contaminated()
        ranks = np.asarray(diagnostics._average_ranks(draws.reshape(4 * 64, 2)))
        assert np.all(np.isnan(ranks[:, 0])), "contaminated component must be NaN"
        assert np.all(np.isfinite(ranks[:, 1])), "clean sibling must be untouched"
        # The clean sibling ranks exactly as it would on its own.
        alone = np.asarray(diagnostics._average_ranks(draws[..., 1].reshape(4 * 64)))
        np.testing.assert_array_equal(ranks[:, 1], alone)

    @parameterized.parameters("rhat", "ess_bulk", "ess_tail")
    def test_nan_propagates_per_component(self, name):
        # A NaN-contaminated series must not report an apparently valid
        # finite diagnostic, and must not disturb an independent component.
        draws = self._contaminated()
        result = np.asarray(getattr(diagnostics, name)(draws))
        assert np.isnan(
            result[0]
        ), f"{name} reported {result[0]} for a NaN-contaminated component"
        assert np.isfinite(
            result[1]
        ), f"{name} poisoned a clean sibling component: {result[1]}"
        alone = np.asarray(getattr(diagnostics, name)(draws[..., 1]))
        np.testing.assert_allclose(result[1], alone, rtol=1e-6)

    def test_nan_propagation_survives_jit(self):
        draws = self._contaminated()
        for name in ("rhat", "ess_bulk", "ess_tail"):
            fn = getattr(diagnostics, name)
            eager = np.asarray(fn(draws))
            jitted = np.asarray(jax.jit(fn)(draws))
            np.testing.assert_array_equal(
                np.isnan(eager),
                np.isnan(jitted),
                err_msg=f"{name}: NaN contract differs under jit",
            )
            np.testing.assert_allclose(eager[1], jitted[1], rtol=1e-6)

    def test_nan_propagation_is_permutation_invariant(self):
        # Where the NaN sits in the pool must not matter.
        base = (np.arange(4 * 64, dtype=np.float64) % 7).reshape(4, 64)
        results = []
        for position in ((0, 0), (1, 5), (3, 63)):
            dirty = base.copy()
            dirty[position] = np.nan
            results.append(float(diagnostics.ess_bulk(jnp.asarray(dirty))))
        assert all(
            np.isnan(r) for r in results
        ), f"NaN handling depends on position in the pool: {results}"

    def test_raw_ess_does_not_invent_a_sample_size_for_nan(self):
        # Geyer's truncation gates on partial sums being > 0; every such
        # comparison is False for NaN, so without a guard the truncation
        # collapses and tau_hat falls back to its 1/log10(MN) floor,
        # manufacturing a large finite ESS (measured: 616.51 for this input).
        all_nan = jnp.full((4, 64), jnp.nan)
        assert np.isnan(float(diagnostics.effective_sample_size(all_nan)))
        assert np.isnan(float(diagnostics.ess_bulk(all_nan)))

    # Axis conventions for a rank-3 (chain, sample, event) array, including
    # genuinely negative arguments — normalising them inside the test would
    # mean a negative-axis regression could never fail it.
    _AXES_3D = ((0, 1), (1, 0), (-3, -2), (-2, -3), (0, -2), (-3, 1))

    @parameterized.parameters(*_AXES_3D)
    def test_raw_ess_propagates_nan_on_every_axis_convention(
        self, chain_axis, sample_axis
    ):
        draws = np.asarray(self._contaminated(), dtype=np.float64)  # (4, 64, 2)
        chain, sample = chain_axis % 3, sample_axis % 3
        event = ({0, 1, 2} - {chain, sample}).pop()
        perm = [0, 0, 0]
        perm[chain], perm[sample], perm[event] = 0, 1, 2
        moved = np.transpose(draws, perm)
        # Pass the raw, possibly negative, axis arguments through.
        result = np.asarray(
            diagnostics.effective_sample_size(
                jnp.asarray(moved), chain_axis, sample_axis
            )
        )
        assert np.isnan(result[0]), f"contaminated component reported {result[0]}"
        assert np.isfinite(result[1]), f"clean sibling poisoned: {result[1]}"

    def test_raw_ess_nan_does_not_poison_a_finite_sibling(self):
        draws = self._contaminated()
        result = np.asarray(diagnostics.effective_sample_size(draws))
        alone = np.asarray(diagnostics.effective_sample_size(draws[..., 1]))
        assert np.isnan(result[0])
        np.testing.assert_array_equal(result[1], alone)

    def test_raw_ess_nan_propagation_survives_jit(self):
        draws = self._contaminated()
        eager = np.asarray(diagnostics.effective_sample_size(draws))
        jitted = np.asarray(jax.jit(diagnostics.effective_sample_size)(draws))
        np.testing.assert_array_equal(np.isnan(eager), np.isnan(jitted))
        np.testing.assert_allclose(eager[1], jitted[1], rtol=1e-6)

    @parameterized.parameters(
        (4, 64),
        (4, 64, 1),
        (4, 64, 2),
        (4, 64, 1, 2),
        (4, 64, 3, 1),
        (4, 64, 1, 1),
        (1, 64, 3),
        (4, 64, 2, 3),
    )
    def test_nan_guards_preserve_output_shape(self, *shape):
        # The NaN guards must not change the shape contract.  A mask that
        # keeps size-1 event axes broadcasts against the estimators' squeezed
        # output: (4, 64, 3, 1) once produced ess_bulk of shape (3, 3).
        draws = jax.random.normal(self.rng, shape=shape)
        expected = diagnostics.effective_sample_size(draws).shape
        assert diagnostics.ess_bulk(draws).shape == expected
        assert diagnostics.ess_tail(draws).shape == expected
        assert diagnostics.rhat(draws).shape == expected

    def test_infinities_are_ordered_and_still_ranked(self):
        # +/-inf are ordered values and keep their ranks in _average_ranks.
        pooled = jnp.array([-jnp.inf, 0.0, 0.0, jnp.inf, 1.0])
        ranks = np.asarray(diagnostics._average_ranks(pooled))
        np.testing.assert_array_equal(ranks, np.array([1.0, 2.5, 2.5, 5.0, 4.0]))

    def test_sparse_infinities_do_not_break_the_diagnostics(self):
        # A handful of infinities leaves the pooled median finite, so the fold
        # is well defined and all three helpers return finite values.
        draws = np.asarray(
            jax.random.normal(self.rng, shape=(4, 64)), dtype=np.float32
        ).copy()
        draws.reshape(-1)[:8] = np.inf
        x = jnp.asarray(draws)
        assert np.isfinite(float(diagnostics.rhat(x)))
        assert np.isfinite(float(diagnostics.ess_bulk(x)))

    def test_majority_infinity_makes_rhat_nan_without_any_nan_input(self):
        # Pins a KNOWN limitation rather than a desired behaviour: rhat folds
        # about the pooled median, so when >= half of a component's pooled
        # draws are +inf the median is inf and the fold computes inf - inf.
        # rhat is then NaN for input containing no NaN at all, while ess_bulk
        # and ess_tail stay finite.  NaN from rhat therefore does not uniquely
        # mean "missing observation".  Infinity handling is an open question.
        #
        # NOTE: this test pins CURRENT behaviour, not a guarantee.  If the
        # inf-fold is ever fixed, this test must be updated or deleted rather
        # than treated as a regression — it is green by construction today.
        draws = np.asarray(
            jax.random.normal(self.rng, shape=(4, 64)), dtype=np.float32
        ).copy()
        draws[:, :40] = np.inf
        x = jnp.asarray(draws)
        assert not np.isnan(draws).any(), "fixture must contain no NaN"
        assert np.isnan(float(diagnostics.rhat(x)))
        assert np.isfinite(float(diagnostics.ess_bulk(x)))
        assert np.isfinite(float(diagnostics.ess_tail(x)))

    def test_nan_in_a_trimmed_odd_draw_is_not_seen(self):
        # Pins the exact scope of the NaN contract.  With an odd number of
        # draws _split_chains trims the last one, so a NaN sitting only there
        # is never used and the diagnostics stay finite.  A NaN among the
        # draws actually used propagates as normal.
        base = np.asarray(
            jax.random.normal(self.rng, shape=(4, 65)), dtype=np.float64
        ).copy()
        trimmed = base.copy()
        trimmed[1, 64] = np.nan
        assert np.isfinite(float(diagnostics.ess_bulk(jnp.asarray(trimmed))))
        used = base.copy()
        used[1, 3] = np.nan
        assert np.isnan(float(diagnostics.ess_bulk(jnp.asarray(used))))

    def test_raw_effective_sample_size_semantics_are_unchanged(self):
        # The raw estimator is deliberately untouched by the NaN extension:
        # its constant-chain and antithetic contracts must still hold.
        np.testing.assert_array_equal(
            np.asarray(diagnostics.effective_sample_size(jnp.zeros((4, 64)))), 0.0
        )
        antithetic = jnp.tile(jnp.array([-1.0, 1.0]), 256)[None, :]
        assert float(diagnostics.effective_sample_size(antithetic)) > 512

    # -- end-to-end diagnostics -----------------------------------------

    def test_binary_draws_match_independent_rhat(self):
        # Bernoulli(0.1) indicator draws: heavily tied, and the case where
        # ordinal ranking previously reported R-hat far above 1 for iid data.
        draws = (jax.random.uniform(self.rng, shape=(8, 96, 2)) < 0.1).astype(
            jnp.float32
        )
        result = np.asarray(diagnostics.rhat(draws))
        expected = _reference_rhat(np.asarray(draws, dtype=np.float64))
        np.testing.assert_allclose(result, expected, rtol=1e-5)
        assert np.all(
            result < 1.01
        ), f"iid Bernoulli draws must not look non-converged, got {result}"

    def test_binary_draws_bulk_ess_is_not_collapsed(self):
        # Ordinal ranking collapsed bulk ESS for these draws to ~1% of N.
        draws = (jax.random.uniform(self.rng, shape=(8, 96, 2)) < 0.1).astype(
            jnp.float32
        )
        total = 8 * 96
        result = np.asarray(diagnostics.ess_bulk(draws))
        assert np.all(
            result > 0.5 * total
        ), f"bulk ESS for iid Bernoulli draws collapsed: {result} (N={total})"

    def test_binary_draws_match_arviz(self):
        az = pytest.importorskip("arviz")
        draws = np.asarray(
            (jax.random.uniform(self.rng, shape=(8, 96, 2)) < 0.1).astype(jnp.float32),
            dtype=np.float64,
        )
        idata = az.convert_to_dataset({"x": draws})
        az_rhat = np.asarray(az.rhat(idata, method="rank")["x"]).ravel()
        az_ess = np.asarray(az.ess(idata, method="bulk")["x"]).ravel()
        bj_rhat = np.asarray(diagnostics.rhat(jnp.asarray(draws)))
        bj_ess = np.asarray(diagnostics.ess_bulk(jnp.asarray(draws)))
        np.testing.assert_allclose(bj_rhat, az_rhat, rtol=1e-4)
        np.testing.assert_allclose(bj_ess, az_ess, rtol=1e-3)

    def _rejection_trace(self, proposals, accepted):
        """Carry the last accepted proposal forward — a Metropolis-like trace."""
        draws = np.empty_like(proposals)
        draws[:, 0] = proposals[:, 0]
        for t in range(1, proposals.shape[1]):
            draws[:, t] = np.where(accepted[:, t], proposals[:, t], draws[:, t - 1])
        return draws

    def test_repeated_discrete_states_do_not_inflate_rhat(self):
        # An 80%-rejection trace over a 5-level proposal: the repeats form
        # large, scattered tie groups, which is where ordinal ranking does
        # real damage.  Measured at 9e128d206 vs the fix: R-hat 1.1404 ->
        # 1.0291 and bulk ESS 120.57 -> 237.55.
        rng = np.random.default_rng(20260906)
        accepted = rng.uniform(size=(4, 512)) < 0.2
        draws = self._rejection_trace(
            rng.integers(0, 5, size=(4, 512)).astype(float), accepted
        )
        assert (draws[:, 1:] == draws[:, :-1]).mean() > 0.5, "too few repeats"

        result = np.asarray(diagnostics.rhat(jnp.asarray(draws, dtype=jnp.float32)))
        expected = _reference_rhat(draws)
        np.testing.assert_allclose(result, expected, rtol=1e-4)
        assert result < 1.05, f"tied repeats still inflate R-hat: {result}"

    def test_continuous_rejection_trace_matches_the_oracle(self):
        # The continuous-proposal counterpart.  This one is an oracle-parity
        # check, NOT a regression guard: rejection repeats in a continuous
        # chain form short contiguous runs, so ordinal ranking perturbs them
        # by only ~2e-5 and this assertion passes at 9e128d206 too.
        rng = np.random.default_rng(20260906)
        draws = self._rejection_trace(
            rng.normal(size=(4, 512)), rng.uniform(size=(4, 512)) < 0.2
        )
        result = np.asarray(diagnostics.rhat(jnp.asarray(draws, dtype=jnp.float32)))
        np.testing.assert_allclose(result, _reference_rhat(draws), rtol=1e-4)

    def test_folded_tail_component_uses_average_ranks(self):
        # rhat folds about the median before rank-normalizing; the folded
        # draws of a symmetric tied grid are themselves heavily tied.
        draws = self._tied_draws(nsamples=64, num_levels=4)
        result = np.asarray(diagnostics.rhat(draws))
        expected = _reference_rhat(np.asarray(draws, dtype=np.float64))
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_antithetic_binary_bulk_ess_may_exceed_draw_count(self):
        # No global ESS <= N cap: an antithetic sequence validly exceeds N.
        nsamples = 512
        draws = jnp.tile(
            jnp.tile(jnp.array([0.0, 1.0]), nsamples // 2)[None, :], (4, 1)
        )
        result = float(diagnostics.ess_bulk(draws))
        assert (
            result > 4 * nsamples
        ), f"antithetic bulk ESS was capped: {result} <= {4 * nsamples}"

    # -- axes, dtypes, transformations ----------------------------------

    @parameterized.parameters(*test_cases)
    def test_axis_invariance_with_ties(self, chain_axis, sample_axis):
        draws = self._tied_draws(nchains=4, nsamples=64)
        expected = diagnostics.rhat(draws)
        expected_ess = diagnostics.ess_bulk(draws)

        if (chain_axis, sample_axis) == (0, 1):
            moved = draws
        else:
            ndim = 2
            moved = jnp.transpose(
                draws, np.argsort([chain_axis % ndim, sample_axis % ndim])
            )
        np.testing.assert_allclose(
            diagnostics.rhat(moved, chain_axis, sample_axis), expected, rtol=1e-6
        )
        np.testing.assert_allclose(
            diagnostics.ess_bulk(moved, chain_axis, sample_axis),
            expected_ess,
            rtol=1e-6,
        )

    def test_dtype_is_preserved(self):
        # Rank normalization must not silently widen or narrow the draws.
        default_float = jnp.zeros(0).dtype
        draws = self._tied_draws().astype(default_float)
        assert diagnostics._rank_normalize(draws).dtype == default_float
        assert diagnostics.ess_bulk(draws).dtype == default_float
        assert diagnostics.rhat(draws).dtype == default_float

    def test_integer_draws_are_accepted(self):
        # Discrete draws are exactly the tied case; they must rank without
        # first being cast to something lossy.
        integers = jax.random.randint(self.rng, (4, 64), minval=0, maxval=5)
        ranks = np.asarray(diagnostics._average_ranks(integers.reshape(-1)))
        np.testing.assert_array_equal(
            ranks, rankdata(np.asarray(integers).reshape(-1), method="average")
        )

    @chex.all_variants(with_pmap=False)
    def test_jit_compatible(self):
        draws = self._tied_draws()
        rank_normalize = self.variant(diagnostics._rank_normalize)
        # ndtri is fused differently under XLA, so this is a tolerance
        # comparison rather than bit equality; the ranks themselves are exact.
        np.testing.assert_allclose(
            np.asarray(rank_normalize(draws)),
            np.asarray(diagnostics._rank_normalize(draws)),
            rtol=1e-5,
            atol=1e-6,
        )

    def test_vmap_over_events(self):
        draws = self._tied_draws(shape=(3,))
        stacked = jnp.moveaxis(draws, -1, 0)
        mapped = jax.vmap(diagnostics.ess_bulk)(stacked)
        np.testing.assert_allclose(mapped, diagnostics.ess_bulk(draws), rtol=1e-6)


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
    """Build a ``(n_chains, n_draws)`` bool array from ``(total,
    first_quarter_count, last_quarter_count)`` triples, reproducing real
    validation-corpus per-chain and quarter counts without an external file.
    """
    quarter = n_draws // 4
    rows = []
    for total, first_q, last_q in chain_specs:
        mid = total - first_q - last_q
        row = np.zeros(n_draws, dtype=bool)
        row[:first_q] = True
        if last_q:
            row[n_draws - last_q :] = True
        row[quarter : quarter + mid] = True
        rows.append(row)
    return jnp.asarray(np.stack(rows))


class DivergenceConcentrationTest(chex.TestCase):
    """Tests for the sampling-phase divergence-concentration warning."""

    _N_DRAWS = 2000

    # Real validation-corpus counts: chain 6 dominant (18.75%, early-
    # concentrated then recovers), chain 3 marginal (2.2%) -- num_flagged=2
    # at the max(1, 8 // 4) minority-cap boundary.
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

    # Every chain elevated (12.85%-26.70%): 8 of 8 flagged, over the
    # minority cap -- must NOT warn.
    _ENSEMBLE_COUNTS = [316, 261, 257, 534, 272, 384, 302, 271]

    _HEALTHY_COUNTS = [2, 6, 3, 4, 7, 3, 4, 7]
    _SINGLE_CHAIN_COUNTS = jnp.array([1, 1, 0, 3, 0, 4, 0, 58])

    def test_storm_minority_of_two(self):
        is_divergent = _build_is_divergent(self._STORM_SPECS, self._N_DRAWS)
        report = diagnostics.divergence_concentration(is_divergent)
        assert bool(report.warn) is True
        assert int(report.num_flagged) == 2
        np.testing.assert_array_equal(
            np.asarray(report.flagged),
            [False, False, False, True, False, False, True, False],
        )
        np.testing.assert_allclose(float(report.early_rate[6]), 361 / 500, rtol=1e-6)
        np.testing.assert_allclose(float(report.late_rate[6]), 1 / 500, rtol=1e-6)

        msg = diagnostics.format_divergence_warning(report)
        assert "chain 3:" in msg and "chain 6:" in msg, msg
        assert "first quarter" in msg and "median" in msg
        for banned in ("geometry", "post-warmup start", "reparameterization"):
            assert banned not in msg, msg

    def test_ensemble_wide_elevation_does_not_warn(self):
        report = diagnostics.divergence_concentration_from_counts(
            jnp.array(self._ENSEMBLE_COUNTS), self._N_DRAWS
        )
        assert bool(report.warn) is False
        assert int(report.num_flagged) == 8
        assert diagnostics.format_divergence_warning(report) == ""

    def test_healthy_scatter_no_warning(self):
        report = diagnostics.divergence_concentration_from_counts(
            jnp.array(self._HEALTHY_COUNTS), self._N_DRAWS
        )
        assert bool(report.warn) is False
        assert diagnostics.format_divergence_warning(report) == ""

    def test_zero_divergences_no_warning(self):
        report = diagnostics.divergence_concentration_from_counts(
            jnp.zeros(8, dtype=int), self._N_DRAWS
        )
        assert bool(report.warn) is False
        assert np.isnan(float(report.multinomial_p_value))

    def test_from_counts_quarters_are_nan_and_degrade_gracefully(self):
        report = diagnostics.divergence_concentration_from_counts(
            self._SINGLE_CHAIN_COUNTS, self._N_DRAWS
        )
        assert bool(report.warn) is True
        assert np.isnan(float(report.early_rate[7]))
        msg = diagnostics.format_divergence_warning(report)
        assert "nan" not in msg.lower() and "first quarter" not in msg, msg

    def test_nan_input_flags_loud_not_silent(self):
        # NaN must flag, not silently pass -- in both entry points.
        is_divergent = jnp.array([[0.0, 1.0, float("nan"), 0.0]] * 4)
        report = diagnostics.divergence_concentration(is_divergent)
        assert not np.any(np.isnan(np.asarray(report.rates)))
        assert bool(np.all(np.asarray(report.flagged)))

        counts_report = diagnostics.divergence_concentration_from_counts(
            jnp.array([1.0, 2.0, float("nan"), 0.0]), self._N_DRAWS
        )
        assert not np.isnan(float(counts_report.rates[2]))
        assert bool(counts_report.flagged[2])

    @chex.all_variants(with_pmap=False)
    def test_jit_compilability(self):
        core = self.variant(
            functools.partial(
                diagnostics.divergence_concentration_from_counts,
                n_draws=self._N_DRAWS,
            )
        )
        report = core(jnp.array([7, 6, 9, 44, 6, 9, 375, 15]))
        assert bool(report.warn) is True

    def test_rate_threshold_override(self):
        report_default = diagnostics.divergence_concentration_from_counts(
            self._SINGLE_CHAIN_COUNTS, self._N_DRAWS
        )
        report_high = diagnostics.divergence_concentration_from_counts(
            self._SINGLE_CHAIN_COUNTS, self._N_DRAWS, rate_threshold=0.05
        )
        assert bool(report_default.warn) is True
        assert bool(report_high.warn) is False

    def test_single_chain_input_is_handled_gracefully(self):
        # num_chains=1 has no "other" chains to compare against -- must
        # not crash, just report cleanly.
        report = diagnostics.divergence_concentration_from_counts(
            jnp.array([5]), self._N_DRAWS
        )
        assert bool(report.warn) is False


if __name__ == "__main__":
    absltest.main()
