import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

import blackjax
from blackjax.adaptation import window_adaptation
from blackjax.adaptation.base import get_filter_adapt_info_fn, return_all_adapt_info
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
    algorithm = blackjax.dynamic_hmc(logprob_fn, **parameters)
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


def test_halton_sequence_raise_value():
    """Test that halton sequence raises value error when max_bits is too large."""
    from blackjax.mcmc.dynamic_hmc import halton_sequence

    with pytest.raises(ValueError, match="max_bits"):
        halton_sequence(jnp.array([0], dtype=jnp.int32), max_bits=32)
