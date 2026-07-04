import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

import blackjax
from blackjax.adaptation import window_adaptation
from blackjax.adaptation.base import get_filter_adapt_info_fn, return_all_adapt_info
from blackjax.adaptation.chees_adaptation import (
    _diagonal_mass_matrix_or_fallback,
    _mass_matrix_engagement_threshold,
)
from blackjax.adaptation.chees_adaptation import base as chees_base
from blackjax.adaptation.mass_matrix import WelfordAlgorithmState, welford_algorithm
from blackjax.util import run_inference_algorithm


@pytest.mark.parametrize(
    "num_steps, expected_schedule",
    [
        (19, [(0, False)] * 19),  # no mass matrix adaptation
        (
            100,
            [(0, False)] * 15 + [(1, False)] * 74 + [(1, True)] + [(0, False)] * 10,
        ),  # windows are resized
        (
            200,
            [(0, False)] * 75
            + [(1, False)] * 24
            + [(1, True)]
            + [(1, False)] * 49
            + [(1, True)]
            + [(0, False)] * 50,
        ),
    ],
)
def test_adaptation_schedule(num_steps, expected_schedule):
    adaptation_schedule = window_adaptation.build_schedule(num_steps)
    assert num_steps == len(adaptation_schedule)
    assert np.array_equal(adaptation_schedule, expected_schedule)


@pytest.mark.parametrize(
    "adaptation_filters",
    [
        {
            "filter_fn": return_all_adapt_info,
            "return_sets": None,
        },
        {
            "filter_fn": get_filter_adapt_info_fn(),
            "return_sets": (set(), set(), set()),
        },
        {
            "filter_fn": get_filter_adapt_info_fn(
                {"logdensity"},
                {"proposal"},
                {"random_generator_arg", "step", "da_state"},
            ),
            "return_sets": (
                {"logdensity"},
                {"proposal"},
                {"random_generator_arg", "step", "da_state"},
            ),
        },
    ],
)
def test_chees_adaptation(adaptation_filters):
    target_mean = jnp.array([0.0, 0.0])
    target_std = jnp.array([1.0, 10.0])
    logprob_fn = lambda x: jax.scipy.stats.norm.logpdf(
        x, loc=target_mean, scale=target_std
    ).sum(axis=-1)

    num_burnin_steps = 1000
    num_results = 500
    num_chains = 16
    step_size = 0.1
    target_acceptance_rate = 0.75

    init_key, warmup_key, inference_key = jax.random.split(jax.random.key(346), 3)

    warmup = blackjax.chees_adaptation(
        logprob_fn,
        num_chains=num_chains,
        target_acceptance_rate=target_acceptance_rate,
        adaptation_info_fn=adaptation_filters["filter_fn"],
    )

    initial_positions = jax.random.normal(init_key, (num_chains, 2))
    (last_states, parameters), warmup_info = warmup.run(
        warmup_key,
        initial_positions,
        step_size=step_size,
        optim=optax.adam(learning_rate=0.5, b1=0, b2=0.95),
        num_steps=num_burnin_steps,
    )
    algorithm = blackjax.dhmc(logprob_fn, **parameters)
    chain_keys = jax.random.split(inference_key, num_chains)
    final_states, (states, infos) = jax.vmap(
        lambda key, state: run_inference_algorithm(
            rng_key=key,
            initial_state=state,
            inference_algorithm=algorithm,
            num_steps=num_results,
        )
    )(chain_keys, last_states)

    harmonic_mean = 1.0 / jnp.mean(1.0 / infos.acceptance_rate, axis=0)
    assert harmonic_mean.shape == (num_results,)
    harmonic_mean = jnp.mean(harmonic_mean)

    def check_attrs(attribute, keyset):
        for name, param in getattr(warmup_info, attribute)._asdict().items():
            print(name, param)
            if name in keyset:
                assert param is not None
            else:
                assert param is None

    keysets = adaptation_filters["return_sets"]
    if keysets is None:
        keysets = (
            warmup_info.state._fields,
            warmup_info.info._fields,
            warmup_info.adaptation_state._fields,
        )
    for i, attribute in enumerate(["state", "info", "adaptation_state"]):
        check_attrs(attribute, keysets[i])

    # The harmonic mean of the acceptance rate should be close to the target acceptance rate
    np.testing.assert_allclose(harmonic_mean, target_acceptance_rate, atol=1e-1)

    # These are empirical values that should be roughly correct for this target distribution
    np.testing.assert_allclose(parameters["step_size"], 1.5, atol=0.3)
    np.testing.assert_allclose(infos.num_integration_steps.mean(), 9, atol=3)

    # Check that sample means and stds are close to target values
    draws = states.position.reshape(-1, states.position.shape[-1])
    empirical_mean = jnp.mean(draws, axis=0)
    empirical_std = jnp.std(draws, axis=0)
    np.testing.assert_allclose(empirical_mean, target_mean, atol=0.5)
    np.testing.assert_allclose(empirical_std, target_std, rtol=0.1)


def _chees_gaussian_logdensity(target_std):
    def logdensity_fn(x):
        return jax.scipy.stats.norm.logpdf(x, scale=target_std).sum(axis=-1)

    return logdensity_fn


def test_chees_mass_matrix_estimation_none_matches_omitted_bit_for_bit():
    """mass_matrix_estimation=None must be numerically identical to omitting
    it -- mirrors meads_adaptation's low_rank_rank=None backward-compat test."""
    target_std = jnp.array([1.0, 10.0])
    logdensity_fn = _chees_gaussian_logdensity(target_std)
    num_chains = 16
    positions = jax.random.normal(jax.random.key(7), (num_chains, 2))
    warmup_key = jax.random.key(11)

    def run(pass_explicit_none):
        kwargs = {"mass_matrix_estimation": None} if pass_explicit_none else {}
        warmup = blackjax.chees_adaptation(
            logdensity_fn, num_chains=num_chains, **kwargs
        )
        return warmup.run(
            warmup_key,
            positions,
            step_size=0.1,
            optim=optax.adam(learning_rate=0.5, b1=0, b2=0.95),
            num_steps=50,
        )

    (states_default, params_default), _ = run(False)
    (states_none, params_none), _ = run(True)

    np.testing.assert_array_equal(states_default.position, states_none.position)
    np.testing.assert_array_equal(
        params_default["inverse_mass_matrix"], params_none["inverse_mass_matrix"]
    )
    assert params_default["step_size"] == params_none["step_size"]


def test_chees_mass_matrix_estimation_invalid_value_raises():
    with pytest.raises(ValueError, match="mass_matrix_estimation"):
        blackjax.chees_adaptation(
            _chees_gaussian_logdensity(jnp.ones(2)),
            num_chains=4,
            mass_matrix_estimation="dense",
        )


def test_chees_mass_matrix_window_fraction_invalid_raises():
    with pytest.raises(ValueError, match="mass_matrix_window_fraction"):
        blackjax.chees_adaptation(
            _chees_gaussian_logdensity(jnp.ones(2)),
            num_chains=4,
            mass_matrix_estimation="diagonal",
            mass_matrix_window_fraction=1.5,
        )
    with pytest.raises(ValueError, match="mass_matrix_window_fraction"):
        blackjax.chees_adaptation(
            _chees_gaussian_logdensity(jnp.ones(2)),
            num_chains=4,
            mass_matrix_estimation="diagonal",
            mass_matrix_window_fraction=-0.1,
        )


def test_chees_whitened_criterion_reduces_to_raw_when_identity():
    """Unit-level, deterministic: with inverse_mass_matrix=ones, the whitened
    (_whiten_criterion=True) and raw (_whiten_criterion=False) code paths
    must produce bit-for-bit identical ChEESAdaptationState updates -- see
    the derivation comment in chees_adaptation.compute_parameters."""
    num_chains, dim = 8, 4
    keys = jax.random.split(jax.random.key(0), 4)
    proposed_positions = jax.random.normal(keys[0], (num_chains, dim))
    proposed_momentums = jax.random.normal(keys[1], (num_chains, dim))
    initial_positions = jax.random.normal(keys[2], (num_chains, dim))
    acceptance_probabilities = jax.random.uniform(keys[3], (num_chains,))
    is_divergent = jnp.zeros((num_chains,), dtype=bool)
    step_size = 0.1

    def run_update(whiten):
        init, update = chees_base(
            jitter_generator=lambda i: 0.5,
            next_random_arg_fn=lambda i: i + 1,
            optim=optax.adam(learning_rate=0.5),
            target_acceptance_rate=0.651,
            decay_rate=0.5,
            max_leapfrog_steps=1000,
            _whiten_criterion=whiten,
        )
        state0 = init(0, step_size)
        return update(
            state0,
            proposed_positions,
            proposed_momentums,
            initial_positions,
            acceptance_probabilities,
            is_divergent,
            jnp.ones(dim),
        )

    state_whitened = run_update(True)
    state_raw = run_update(False)

    equal_leaves = jax.tree.leaves(
        jax.tree.map(lambda a, b: jnp.array_equal(a, b), state_whitened, state_raw)
    )
    assert all(bool(leaf) for leaf in equal_leaves)


def test_chees_mass_matrix_engagement_gate():
    """Before the pooled Welford sample count reaches the engagement
    threshold, the diagonal estimate must fall back to ones; once it does,
    the estimate must differ from ones and stay finite."""
    num_dim = 5
    threshold = _mass_matrix_engagement_threshold(num_dim)
    assert threshold >= 64

    below_accum = WelfordAlgorithmState(
        mean=jnp.zeros(num_dim), m2=jnp.zeros(num_dim), sample_size=threshold - 1
    )
    imm_below = _diagonal_mass_matrix_or_fallback(below_accum, threshold, num_dim)
    np.testing.assert_array_equal(imm_below, jnp.ones(num_dim))

    scales = jnp.array([0.1, 1.0, 10.0, 2.0, 5.0])
    samples = jax.random.normal(jax.random.key(0), (threshold + 20, num_dim)) * scales
    _, wc_update, _ = welford_algorithm(is_diagonal_matrix=True)
    acc = WelfordAlgorithmState(
        mean=jnp.zeros(num_dim), m2=jnp.zeros(num_dim), sample_size=0
    )
    for i in range(threshold + 20):
        acc = wc_update(acc, samples[i])

    imm_above = _diagonal_mass_matrix_or_fallback(acc, threshold, num_dim)
    assert jnp.all(jnp.isfinite(imm_above))
    assert not jnp.allclose(imm_above, jnp.ones(num_dim))


def test_chees_mass_matrix_estimation_correctness():
    """On a diagonal Gaussian with well-separated known variances, the
    engaged diagonal inverse_mass_matrix should approximate the true
    per-dimension variance (loose ratio tolerance, pinned seed -- no
    ESS-threshold assertions)."""
    target_std = jnp.array([0.1, 1.0, 10.0])
    true_variance = target_std**2
    logdensity_fn = _chees_gaussian_logdensity(target_std)
    num_chains, dim = 32, 3

    init_key, warmup_key = jax.random.split(jax.random.key(2026))
    positions = jax.random.normal(init_key, (num_chains, dim)) * target_std

    warmup = blackjax.chees_adaptation(
        logdensity_fn,
        num_chains=num_chains,
        mass_matrix_estimation="diagonal",
        mass_matrix_window_fraction=0.5,
    )
    (last_states, parameters), _ = warmup.run(
        warmup_key,
        positions,
        step_size=0.1,
        optim=optax.adam(learning_rate=0.5, b1=0, b2=0.95),
        num_steps=300,
    )

    ratio = parameters["inverse_mass_matrix"] / true_variance
    np.testing.assert_allclose(ratio, 1.0, rtol=0.6)
    assert jnp.all(jnp.isfinite(last_states.position))


def test_chees_mass_matrix_estimation_e2e_smoke():
    """Scale-separated synthetic target: mass_matrix_estimation="diagonal"
    should run to completion with no NaN, a finite adapted step size, and a
    returned inverse_mass_matrix within a loose ratio band of the truth."""
    dim, num_chains = 12, 32
    scale_key, init_key, warmup_key = jax.random.split(jax.random.key(3), 3)
    log_scales = jax.random.uniform(scale_key, (dim,), minval=-1.0, maxval=1.0)
    target_std = 10.0**log_scales  # spans roughly 1e-1..1e1
    logdensity_fn = _chees_gaussian_logdensity(target_std)

    positions = jax.random.normal(init_key, (num_chains, dim)) * target_std

    warmup = blackjax.chees_adaptation(
        logdensity_fn,
        num_chains=num_chains,
        mass_matrix_estimation="diagonal",
        mass_matrix_window_fraction=0.5,
    )
    (last_states, parameters), _ = warmup.run(
        warmup_key,
        positions,
        step_size=0.1,
        optim=optax.adam(learning_rate=0.5, b1=0, b2=0.95),
        num_steps=150,
    )

    assert jnp.all(jnp.isfinite(last_states.position))
    assert jnp.isfinite(parameters["step_size"])
    imm = parameters["inverse_mass_matrix"]
    assert jnp.all(jnp.isfinite(imm))
    ratio = imm / target_std**2
    assert jnp.all(ratio > 0.1) and jnp.all(ratio < 10.0)


def test_chees_whiten_criterion_ablation_seam_changes_behavior():
    """_whiten_criterion=False (naive: kernel uses the estimated diagonal,
    criterion stays raw) must yield different adapted parameters than the
    default whitened path once the estimated diagonal deviates from ones --
    otherwise the ablation seam would silently be a no-op. This is the seam
    a validation study toggles to compare {identity, naive, whitened}."""
    target_std = jnp.array([0.1, 1.0, 10.0])
    logdensity_fn = _chees_gaussian_logdensity(target_std)
    num_chains = 32
    init_key, warmup_key = jax.random.split(jax.random.key(5))
    positions = jax.random.normal(init_key, (num_chains, 3)) * target_std

    def run(whiten):
        warmup = blackjax.chees_adaptation(
            logdensity_fn,
            num_chains=num_chains,
            mass_matrix_estimation="diagonal",
            _whiten_criterion=whiten,
        )
        return warmup.run(
            warmup_key,
            positions,
            step_size=0.1,
            optim=optax.adam(learning_rate=0.5, b1=0, b2=0.95),
            num_steps=200,
        )

    (_, params_whitened), _ = run(True)
    (_, params_naive), _ = run(False)

    assert not jnp.allclose(
        params_whitened["step_size"], params_naive["step_size"], rtol=1e-3
    )


def test_halton_sequence_raise_value():
    """Test that halton sequence raises value error when max_bits is too large."""
    from blackjax.mcmc.dynamic_hmc import halton_sequence

    with pytest.raises(ValueError, match="max_bits"):
        halton_sequence(jnp.array([0], dtype=jnp.int32), max_bits=32)
