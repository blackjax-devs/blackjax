# Copyright Contributors to the Numpyro project.
# SPDX-License-Identifier: Apache-2.0
"""Make sure that the potential is only compiled once.
"""
import jax
import jax.scipy as jscipy

import blackjax.hmc as hmc

GLOBAL = {'count': 0}


def potential(x):
    GLOBAL['count'] += 1
    return jscipy.stats.normal(x)


def test_hmc():
    """The reason why this works is because JAX only reads the potential once when compiled?
    """
    GLOBAL["count"] = 0
    kernel = jax.jit(hmc.kernel())
    for _ in range(10):
        _, rng_key = jax.random.split(rng_key)
        state, _ = kernel(rng_key, state)

    assert GLOBAL["count"] == 1


def test_nuts():
    GLOBAL["count"] = 0
    kernel = jax.jit(nuts.kernel())
    for _ in range(10):
        _, rng_key = jax.random.split(rng_key)
        state, _ = kernel(rng_key, state)

    assert GLOBAL["count"] == 1
