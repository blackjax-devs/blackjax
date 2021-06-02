# Copyright Contributors to the Numpyro project.
# SPDX-License-Identifier: Apache-2.0
"""Make sure that the potential is only compiled once.
"""
import jax
import jax.numpy as jnp
import jax.scipy as jscipy

import blackjax.hmc as hmc
import blackjax.nuts as nuts

GLOBAL = {"count": 0}


def potential(x):
    GLOBAL["count"] += 1
    return jscipy.stats.norm.logpdf(x)


def test_hmc():
    """The reason why this works is because JAX only reads the potential once when compiled?"""
    rng_key = jax.random.PRNGKey(0)
    state = hmc.new_state(1.0, potential)

    GLOBAL["count"] = 0
    kernel = jax.jit(
        hmc.kernel(
            potential,
            step_size=1e-2,
            inverse_mass_matrix=jnp.array([1.0]),
            num_integration_steps=10,
        )
    )

    for _ in range(10):
        _, rng_key = jax.random.split(rng_key)
        state, _ = kernel(rng_key, state)

    assert GLOBAL["count"] == 1


def test_nuts():
    rng_key = jax.random.PRNGKey(0)
    state = hmc.new_state(1.0, potential)

    GLOBAL["count"] = 0
    kernel = jax.jit(
        nuts.kernel(potential, step_size=1e-2, inverse_mass_matrix=jnp.array([1.0]))
    )

    for _ in range(10):
        _, rng_key = jax.random.split(rng_key)
        state, _ = kernel(rng_key, state)

    # Potential function was traced twice as we call potential function
    # at Step 0 when building a new trajectory in tree doubling.
    assert GLOBAL["count"] == 2
