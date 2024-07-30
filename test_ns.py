import matplotlib.pyplot as plt
import jax
from datetime import date

rng_key = jax.random.PRNGKey(0)


import numpy as np
import jax.numpy as jnp
from jax.scipy.stats import multivariate_normal

import blackjax


def loglikelihood(x):
    return -5 * jnp.square(jnp.sum(((x + 2) / 4) ** 2, axis=-1) - 1)


algo = blackjax.rejection_ns(loglikelihood)

n_samples = 100
rng_key, init_key, sample_key = jax.random.split(rng_key, 3)
state = jax.random.uniform(init_key, (n_samples, 1))
state = algo.init(state, loglikelihood)


def one_step(carry, xs):
    state, k = carry
    k, subk = jax.random.split(k, 2)
    state, dead_point = algo.step(subk, state)
    return (state, k), (xs[0], dead_point)


with jax.disable_jit():
    n_iter = 100
    iterations = jnp.arange(n_iter)
    res = jax.lax.scan(
        one_step, (state, rng_key), (iterations, jnp.empty(n_iter))
    )

final = res[1][1].update_info["particles"]
live = res[0][0][0]

plt.hist(final.flatten(), bins=30, density=True)
plt.hist(live.flatten(), bins=30, density=True)
plt.show()
