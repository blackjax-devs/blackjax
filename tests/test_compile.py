# Copyright Contributors to the Numpyro project.
# SPDX-License-Identifier: Apache-2.0
"""Make sure that the potential is only compiled once.

When there are many observation the JIT-compilation of the conditioned model's
potential can take a substantial amount of time. We thus need to make sure that
BlackJAX only needs to compile the potential once.

JIT-compilation will trace through the potential function, and remove
the update of the 'GLOBAL' variable. The variable is thus only updated
once, that is when the function is being traced before compilation.

"""
import blackjax.hmc as hmc
import blackjax.nuts as nuts

import jax
import jax.numpy as jnp
import jax.scipy.stats as stats
import pytest

GLOBAL = {'count': 0}

def potential_fn(x):
    GLOBAL["count"] += 1
    logpdf = stats.norm.logpdf(x, 0, 3)
    return logpdf

def test_hmc_one_chain():
    GLOBAL["count"] = 0
    initial_position = {'x': 1.}

    potential = lambda x: potential_fn(**x)
    params = hmc.HMCParameters(
        num_integration_steps=90, step_size=1e-3, inv_mass_matrix=jnp.array([1.])
    )
    kernel = jax.jit(hmc.kernel(potential, params))  # count += 1

    state = hmc.new_state(initial_position, potential)  # count += 1
    rng_key = jax.random.PRNGKey(0)
    for _ in range(100):
        _, rng_key = jax.random.split(rng_key)
        state, _ = kernel(rng_key, state)

    assert GLOBAL["count"] == 2


def test_nuts_one_chain():
    GLOBAL["count"] = 0
    initial_position = {'x': 1.}

    potential = lambda x: potential_fn(**x)
    params = nuts.NUTSParameters(
        step_size=1e-3, inv_mass_matrix=jnp.array([1.])
    )
    kernel = jax.jit(nuts.kernel(potential, params))  # count += 1

    state = nuts.new_state(initial_position, potential)  # count += 1
    rng_key = jax.random.PRNGKey(0)
    for _ in range(100):
        _, rng_key = jax.random.split(rng_key)
        state, _ = kernel(rng_key, state)

    assert GLOBAL["count"] == 2
