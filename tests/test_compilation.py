# Copyright Contributors to the Numpyro project.
# SPDX-License-Identifier: Apache-2.0
"""Make sure that the potential is only compiled once.
"""
import chex
import jax
import jax.numpy as jnp
import jax.scipy as jscipy

import blackjax.hmc as hmc
import blackjax.nuts as nuts


def test_hmc():
    @chex.assert_max_traces(n=1)
    def potential(x):
        return jscipy.stats.norm.logpdf(x)
    
    rng_key = jax.random.PRNGKey(0)
    state = hmc.new_state(1.0, potential)

    chex.clear_trace_counter()

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


def test_nuts():
    # Potential function was traced twice as we call potential function
    # at Step 0 when building a new trajectory in tree doubling.
    @chex.assert_max_traces(n=2)
    def potential(x):
        return jscipy.stats.norm.logpdf(x)

    rng_key = jax.random.PRNGKey(0)
    state = hmc.new_state(1.0, potential)

    chex.clear_trace_counter()

    kernel = jax.jit(
        nuts.kernel(potential, step_size=1e-2, inverse_mass_matrix=jnp.array([1.0]))
    )

    for _ in range(10):
        _, rng_key = jax.random.split(rng_key)
        state, _ = kernel(rng_key, state)

