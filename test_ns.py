from datetime import date

import jax
import matplotlib.pyplot as plt

import blackjax.progress_bar

rng_key = jax.random.PRNGKey(0)


import jax.numpy as jnp
import numpy as np
from jax.scipy.stats import multivariate_normal

import blackjax

d = 2


def loglikelihood(x):
    return -5 * jnp.square(jnp.sum(((x + 2) / 4) ** 2, axis=-1) - 1)


algo = blackjax.rejection_ns(loglikelihood)

n_samples = 100
rng_key, init_key, sample_key = jax.random.split(rng_key, 3)
state = jax.random.uniform(init_key, (n_samples, d))
state = algo.init(state, loglikelihood)

# from jax_tqdm import scan_tqdm
from blackjax.progress_bar import progress_bar_scan

n_steps = 1000


# @scan_tqdm(100)
@progress_bar_scan(n_steps)
def one_step(carry, xs):
    state, k = carry
    k, subk = jax.random.split(k, 2)
    state, dead_point = algo.step(subk, state)
    return (state, k), dead_point


iterations = jnp.arange(n_steps)
(live, _), dead = jax.lax.scan((one_step), (state, rng_key), iterations)

# with jax.disable_jit():
#     n_iter = 10
#     iterations = jnp.arange(n_iter)
#     res = jax.lax.scan((one_step), (state, rng_key), iterations)


dead_points = dead.particles.squeeze()
live_points = live.particles.squeeze()

plt.scatter(*dead_points.T, label="Dead points", color="C1")
plt.scatter(*live_points.T, label="Live points", color="C0")

# plt.loglog()
plt.legend()
plt.show()
