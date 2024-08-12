from datetime import date

import jax
import matplotlib.pyplot as plt

import blackjax.progress_bar

rng_key = jax.random.PRNGKey(0)


import jax.numpy as jnp
import numpy as np
from jax.scipy.stats import multivariate_normal

import blackjax
import distrax
import anesthetic as ns

d = 4


def loglikelihood(x):
    return multivariate_normal.logpdf(
        x, mean=jnp.ones(d), cov=jnp.eye(d) * 0.05
    )


n_samples = 500
rng_key, init_key, sample_key = jax.random.split(rng_key, 3)

prior = distrax.MultivariateNormalDiag(
    loc=jnp.zeros(d), scale_diag=jnp.ones(d)
)
# prior = distrax.Uniform(low=-2.0 * jnp.ones(d), high=2.0 * jnp.ones(d))

state = prior._sample_n(rng_key, n_samples)

algo = blackjax.rejection_ns(prior.sample, loglikelihood)
state = algo.init(state, loglikelihood) 


# from jax_tqdm import scan_tqdm
from blackjax.progress_bar import progress_bar_scan

n_steps = 5000


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

samples = ns.NestedSamples(
    data=np.concatenate([live_points, dead_points], axis=0),
    logL=np.concatenate([live.logL, dead.logL.squeeze()]),
    logL_birth=np.concatenate([live.logL_birth, dead.logL_birth.squeeze()]),
)
print(samples.logZ())
from lsbi.model import ReducedLinearModel

model = ReducedLinearModel(mu_L = np.ones(d), Sigma_L = np.eye(d)*0.05, logLmax=loglikelihood(np.ones(d)))

print(model.logZ()) 
a = samples.set_beta(0.0).plot_2d(np.arange(d))
# samples.plot_2d(a)
samples.plot_2d(a)

plt.show()
