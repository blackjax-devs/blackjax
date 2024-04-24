import jax
import jax.numpy as jnp

from mcmc.ensemble_mclmc import *

key= jax.random.PRNGKey(42)
key, key_prior = jax.random.split(key)

d = 100
chains = 256

logp = lambda x: - jnp.sum(jnp.square(x))

initial_position = jax.random.normal(key_prior, shape = (chains, d)) * 5.

algorithm(logp, 1000, initial_position, chains, key)



import optax

optim = optax.adam(learning_rate)
