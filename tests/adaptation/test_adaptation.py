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
    logprob_fn = lambda x: jax.scipy.stats.norm.logpdf(
        x, loc=0.0, scale=jnp.array([1.0, 10.0])
    ).sum()

    num_burnin_steps = 1000
    num_results = 500
    num_chains = 16
    step_size = 0.1

    init_key, warmup_key, inference_key = jax.random.split(jax.random.key(346), 3)

    warmup = blackjax.chees_adaptation(
        logprob_fn,
        num_chains=num_chains,
        target_acceptance_rate=0.75,
        adaptation_info_fn=adaptation_filters["filter_fn"],
    )

    initial_positions = jax.random.normal(init_key, (num_chains, 2))
    (last_states, parameters), warmup_info = warmup.run(
        warmup_key,
        initial_positions,
        step_size=step_size,
        optim=optax.adamw(learning_rate=0.5),
        num_steps=num_burnin_steps,
    )
    algorithm = blackjax.dynamic_hmc(logprob_fn, **parameters)

    chain_keys = jax.random.split(inference_key, num_chains)
    _, _, infos = jax.vmap(
        lambda key, state: run_inference_algorithm(
            rng_key=key,
            initial_state=state,
            inference_algorithm=algorithm,
            num_steps=num_results,
        )
    )(chain_keys, last_states)

    harmonic_mean = 1.0 / jnp.mean(1.0 / infos.acceptance_rate)

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

    np.testing.assert_allclose(harmonic_mean, 0.75, atol=1e-1)
    np.testing.assert_allclose(parameters["step_size"], 1.5, rtol=2e-1)
    np.testing.assert_array_less(infos.num_integration_steps.mean(), 15.0)
