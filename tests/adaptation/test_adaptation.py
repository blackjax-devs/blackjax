import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

import blackjax
from blackjax.adaptation import window_adaptation
from blackjax.adaptation.base import get_filter_adapt_info_fn, return_all_adapt_info
from blackjax.adaptation.chees_adaptation import (
    _LENGTH_FLOOR_POWER_ITERATIONS,
    CHEES_LENGTH_FLOOR_FACTOR,
    _apply_length_floor,
    _cov_accumulator_init,
    _cov_accumulator_update,
    _diagonal_mass_matrix_or_fallback,
    _eig_state_init,
    _mass_matrix_engagement_threshold,
    _power_iteration_lambda_max,
    _recompute_eig_state,
)
from blackjax.adaptation.chees_adaptation import base as chees_base
from blackjax.adaptation.mass_matrix import WelfordAlgorithmState, welford_algorithm
from blackjax.diagnostics import effective_sample_size
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
    it -- mirrors meads_adaptation's low_rank_rank=None backward-compat test.
    Also covers the slow-direction length floor: the floor machinery must be
    fully inert on the None path regardless of _length_floor (extended by
    this feature -- integration_steps_params is exactly the field the floor
    touches)."""
    target_std = jnp.array([1.0, 10.0])
    logdensity_fn = _chees_gaussian_logdensity(target_std)
    num_chains = 16
    positions = jax.random.normal(jax.random.key(7), (num_chains, 2))
    warmup_key = jax.random.key(11)

    def run(pass_explicit_none, length_floor=True):
        kwargs = {"mass_matrix_estimation": None} if pass_explicit_none else {}
        warmup = blackjax.chees_adaptation(
            logdensity_fn,
            num_chains=num_chains,
            _length_floor=length_floor,
            **kwargs,
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
    (states_none_no_floor, params_none_no_floor), _ = run(True, length_floor=False)

    np.testing.assert_array_equal(states_default.position, states_none.position)
    np.testing.assert_array_equal(
        params_default["inverse_mass_matrix"], params_none["inverse_mass_matrix"]
    )
    assert params_default["step_size"] == params_none["step_size"]
    np.testing.assert_array_equal(
        params_default["integration_steps_params"][0],
        params_none["integration_steps_params"][0],
    )

    # _length_floor must have zero effect when mass_matrix_estimation=None
    # (the floor is fully inert, not just numerically a no-op).
    np.testing.assert_array_equal(states_none.position, states_none_no_floor.position)
    np.testing.assert_array_equal(
        params_none["integration_steps_params"][0],
        params_none_no_floor["integration_steps_params"][0],
    )
    assert params_none["step_size"] == params_none_no_floor["step_size"]


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


def _equicorrelated_block_logdensity(dim, block_k, rho, scale_span, scale_key):
    """dims 0..block_k-1: equicorrelated at rho (unit marginal variance,
    analytic top eigenvalue 1 + (block_k - 1) * rho -- a genuine residual
    correlation direction no diagonal metric can remove); remaining dims:
    independent, scale-separated (log-uniform std spanning
    10**-scale_span..10**scale_span, as in test_chees_mass_matrix_estimation
    _e2e_smoke) -- the severe scale separation is what makes a diagonal
    metric collapse ChEES's OWN adapted length well below what the block
    needs (see test_chees_length_floor_e2e_smoke_recovers_slow_direction)."""
    C_block = jnp.full((block_k, block_k), rho) + (1.0 - rho) * jnp.eye(block_k)
    precision_block = jnp.linalg.inv(C_block)
    n_rest = dim - block_k
    log_scales = jax.random.uniform(
        scale_key, (n_rest,), minval=-scale_span, maxval=scale_span
    )
    stds = 10.0**log_scales

    def logdensity_fn(x):
        block = x[:block_k]
        rest = x[block_k:]
        lp_block = -0.5 * block @ precision_block @ block
        lp_rest = -0.5 * jnp.sum((rest / stds) ** 2)
        return lp_block + lp_rest

    top_eigenvalue = 1.0 + (block_k - 1) * rho
    return logdensity_fn, top_eigenvalue, stds


def test_length_floor_accumulator_and_power_iteration_recover_planted_eigenvalue():
    """lambda_max estimator unit test (accumulator -> whitening -> power
    iteration), no sampling from chees_adaptation.run() involved: a
    synthetic ensemble with a KNOWN planted correlation direction -- a
    rank-1 correlation between dims 0-1 (rho=0.9) embedded in d=10, unit
    marginal variances elsewhere -- has an analytic whitened-covariance top
    eigenvalue of 1 + rho (eigenvector (1, 1, 0, ..., 0) / sqrt(2)). Since
    every dimension has unit marginal variance, D=ones(d) makes the
    "whitened" covariance exactly the correlation matrix itself."""
    d, rho = 10, 0.9
    analytic_lambda_max = 1.0 + rho

    C = jnp.eye(d).at[0, 1].set(rho).at[1, 0].set(rho)
    key = jax.random.key(0)
    samples = jax.random.multivariate_normal(key, jnp.zeros(d), C, shape=(20_000,))

    acc = _cov_accumulator_init(d)
    batch_size = 200
    for i in range(samples.shape[0] // batch_size):
        acc = _cov_accumulator_update(
            acc, samples[i * batch_size : (i + 1) * batch_size]
        )
    # The accumulator must reproduce the naive batch covariance (same check
    # as meads_adaptation's test_accumulator_matches_naive_batch_covariance).
    naive_cov = jnp.cov(samples, rowvar=False)
    accumulated_cov = acc.m2 / (acc.count - 1.0)
    np.testing.assert_allclose(accumulated_cov, naive_cov, atol=1e-6, rtol=1e-6)

    eig_state = _eig_state_init(d)
    # Several recomputes, as the real run() would perform every
    # _LENGTH_FLOOR_RECOMPUTE_INTERVAL steps, warm-starting each time.
    for _ in range(5):
        eig_state = _recompute_eig_state(acc, jnp.ones(d), eig_state)

    np.testing.assert_allclose(
        float(eig_state.lambda_max), analytic_lambda_max, rtol=0.1
    )
    # The eigenvector should have converged into the planted subspace
    # span{e0, e1}: negligible energy on the other (uncorrelated) dims.
    energy_outside_block = float(jnp.sum(eig_state.eigenvector[2:] ** 2))
    assert energy_outside_block < 0.05


def test_length_floor_power_iteration_converges_from_warm_start():
    """Deterministic (no sampling noise): power iteration on the EXACT
    planted correlation matrix recovers its analytic top eigenvalue within
    a handful of iterations from a generic starting vector, and stays
    converged when warm-started again from the recovered eigenvector (the
    per-step "refresh every K steps" pattern `_recompute_eig_state` uses)."""
    d, rho = 10, 0.9
    analytic_lambda_max = 1.0 + rho
    C = jnp.eye(d).at[0, 1].set(rho).at[1, 0].set(rho)

    v0 = jnp.ones(d) / jnp.sqrt(d)
    lambda_max, v = _power_iteration_lambda_max(
        C, v0, num_iterations=_LENGTH_FLOOR_POWER_ITERATIONS
    )
    np.testing.assert_allclose(float(lambda_max), analytic_lambda_max, rtol=0.05)

    # Warm-starting from the converged eigenvector for just 1 more iteration
    # should stay converged (not regress) -- this is what makes refreshing
    # only every K steps (rather than every step) cheap without losing
    # accuracy once the estimate has locked on.
    lambda_max_2, _ = _power_iteration_lambda_max(C, v, num_iterations=1)
    np.testing.assert_allclose(float(lambda_max_2), analytic_lambda_max, rtol=1e-3)


def test_apply_length_floor_arithmetic():
    """Direct unit test of the floor arithmetic (no sampling, mocked
    lambda_max): the consumed length is
    max(adapted, CHEES_LENGTH_FLOOR_FACTOR * sqrt(lambda_max)) when engaged
    and enabled; the private _length_floor=False seam (enable=False) leaves
    the adapted length completely unchanged."""
    adapted_length = 3.0
    lambda_max = 100.0
    expected_floor = float(CHEES_LENGTH_FLOOR_FACTOR * jnp.sqrt(lambda_max))
    assert expected_floor > adapted_length  # the floor must actually bind here

    consumed = _apply_length_floor(
        adapted_length, lambda_max, engaged=True, enable=True
    )
    np.testing.assert_allclose(float(consumed), expected_floor)

    # The floor must not shrink an already-long adapted length.
    consumed_large = _apply_length_floor(50.0, lambda_max, engaged=True, enable=True)
    assert float(consumed_large) == 50.0

    # Ablation seam: enable=False (the private _length_floor=False path)
    # returns the adapted length unchanged, regardless of lambda_max/engaged.
    consumed_disabled = _apply_length_floor(
        adapted_length, lambda_max, engaged=True, enable=False
    )
    assert float(consumed_disabled) == adapted_length


def test_apply_length_floor_inert_pre_gate():
    """Before the engagement gate (engaged=False), the floor is a no-op
    even with enable=True and a huge lambda_max -- pre-gate, lambda_max is
    not yet a meaningful estimate (mirrors _diagonal_mass_matrix_or_fallback's
    engagement-gate semantics for the diagonal estimate itself)."""
    adapted_length = 3.0
    consumed = _apply_length_floor(
        adapted_length, lambda_max=1.0e6, engaged=False, enable=True
    )
    assert float(consumed) == adapted_length


def test_chees_length_floor_e2e_smoke_recovers_slow_direction():
    """E2E smoke (small, targeted, deterministic seed): a synthetic target
    with a genuine residual slow-correlation block (10 equicorrelated dims,
    rho=0.95, analytic top eigenvalue ~9.55) plus severe scale separation
    among the remaining dims (std spanning 1e-2..1e2) -- the scale
    separation is what makes mass_matrix_estimation="diagonal" collapse
    ChEES's own adapted trajectory length well below what the block needs
    (mirrors the mechanism found on radon/irt_2pl in the validation probe).
    metric+floor must (a) floor the adapted length at roughly the analytic
    (pi/2)*sqrt(top_eigenvalue) quarter-turn value and (b) show a much
    higher min-ESS than metric-without-floor at a matched, small sampling
    budget. Loose bands throughout -- no tight ESS thresholds."""
    dim, block_k, rho, scale_span = 100, 10, 0.95, 2.0
    num_chains, num_warmup, num_samples = 32, 250, 200
    scale_key, init_key, warmup_key, sample_key = jax.random.split(jax.random.key(4), 4)
    logdensity_fn, top_eigenvalue, stds = _equicorrelated_block_logdensity(
        dim, block_k, rho, scale_span, scale_key
    )
    # Initialize the scale-separated dims at their target scale (not unit
    # variance) -- otherwise the initial ensemble doesn't yet reflect the
    # severe scale separation the mechanism depends on, and the collapse
    # this test is designed to exercise doesn't happen (verified empirically:
    # omitting this scaling made the un-floored arm's adapted length land at
    # ~4.7, not the collapsed ~0.4-0.5 seen with it).
    positions = jax.random.normal(init_key, (num_chains, dim))
    positions = positions.at[:, block_k:].multiply(stds)

    def run(length_floor):
        warmup = blackjax.chees_adaptation(
            logdensity_fn,
            num_chains=num_chains,
            mass_matrix_estimation="diagonal",
            _length_floor=length_floor,
        )
        (last_states, parameters), _ = warmup.run(
            warmup_key,
            positions,
            step_size=0.1,
            optim=optax.adam(learning_rate=0.5, b1=0, b2=0.95),
            num_steps=num_warmup,
        )
        algorithm = blackjax.dhmc(logdensity_fn, **parameters)
        chain_keys = jax.random.split(sample_key, num_chains)
        _, (states, _) = jax.vmap(
            lambda key, state: run_inference_algorithm(
                rng_key=key,
                initial_state=state,
                inference_algorithm=algorithm,
                num_steps=num_samples,
            )
        )(chain_keys, last_states)
        trajectory_length = float(
            parameters["integration_steps_params"][0] * parameters["step_size"]
        )
        return trajectory_length, states

    length_with_floor, states_with_floor = run(True)
    length_without_floor, states_without_floor = run(False)

    assert jnp.all(jnp.isfinite(states_with_floor.position))
    assert jnp.all(jnp.isfinite(states_without_floor.position))

    # (a) the floored length must land close to the analytic quarter-turn
    # floor value, and well above the un-floored (collapsed) length.
    expected_floor = float(np.pi / 2 * np.sqrt(top_eigenvalue))
    assert length_with_floor >= 0.7 * expected_floor
    assert length_with_floor > length_without_floor

    # (b) min-ESS meaningfully higher with the floor -- loose band (>=3x),
    # the actual effect measured in exploration was 30-60x.
    ess_with_floor = effective_sample_size(
        states_with_floor.position, chain_axis=0, sample_axis=1
    )
    ess_without_floor = effective_sample_size(
        states_without_floor.position, chain_axis=0, sample_axis=1
    )
    min_ess_with_floor = float(jnp.min(ess_with_floor))
    min_ess_without_floor = float(jnp.min(ess_without_floor))
    assert min_ess_with_floor >= 3.0 * min_ess_without_floor


def test_halton_sequence_raise_value():
    """Test that halton sequence raises value error when max_bits is too large."""
    from blackjax.mcmc.dynamic_hmc import halton_sequence

    with pytest.raises(ValueError, match="max_bits"):
        halton_sequence(jnp.array([0], dtype=jnp.int32), max_bits=32)
