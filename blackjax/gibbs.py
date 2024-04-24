import jax
import jax.numpy as jnp
from mcmc.mclmc_gibbs import mclmcGibbs
from util import run_inference_algorithm as run
import matplotlib.pyplot as plt


# set the problem
logp = lambda x, y: -0.5 * jnp.dot(x, x) - 0.5 * jnp.dot(y, y) / 25.
dimx, dimy = 2, 100


# initialize
keys = jax.random.split(jax.random.PRNGKey(42), 4)
x0 = jax.random.normal(keys[0], shape = (dimx, )) * 2
y0 = jax.random.normal(keys[1], shape = (dimy, )) * 4


# hyperparameters
L1 = jnp.sqrt(dimx)
L2 = jnp.sqrt(dimy) * 5.
eps1 = L1 * 0.2
eps2 = L2 * 0.2
hyp = (L1, eps1, L2, eps2)

sampler = mclmcGibbs(logp, hyp, nx= 1, ny= 2)

final_state, samples, info = run(keys[2], (x0, y0), sampler, 10000)

burnin = 1000
x, y = samples.x.position[burnin:], samples.y.position[burnin:]


print(jnp.average(x[:, 0]), jnp.std(x[:, 0]))
print(jnp.average(y[:, 0]), jnp.std(y[:, 0]))