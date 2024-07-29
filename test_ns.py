import matplotlib.pyplot as plt
import jax
from datetime import date
rng_key = jax.random.PRNGKey(0)


import numpy as np
import jax.numpy as jnp
from jax.scipy.stats import multivariate_normal

import blackjax

def loglikelihood(x):
    return -5 * jnp.square(jnp.sum(((x+2)/4)**2, axis=-1) - 1)

algo = blackjax.rejection_ns(loglikelihood)

n_samples = 100
rng_key, init_key, sample_key = jax.random.split(rng_key, 3)
state = jax.random.uniform(init_key, (n_samples, 1))
state = algo.init(state, loglikelihood)

def cond(carry):
    i, state, _k = carry
    return i < 100

def one_step(carry):
    i, state, k = carry
    k, subk = jax.random.split(k, 2)
    state, dead_point = algo.step(subk, state)
    print(dead_point)
    return i + 1, state, k

with jax.disable_jit():
    n_iter, final_state, _ = jax.lax.while_loop(
            cond, one_step, (0, state, rng_key)
            )
