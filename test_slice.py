import os
from datetime import date

import anesthetic as ns
import distrax
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jax.scipy.stats import multivariate_normal

import blackjax
import blackjax.progress_bar
from blackjax import univariate_slice
from blackjax.progress_bar import progress_bar_scan

##################################################################################
# Setup the problem
##################################################################################

rng_key = jax.random.PRNGKey(2)
d = 2

np.random.seed(1)
C = np.random.randn(d, d) * 0.1
like_cov = C @ C.T
like_mean = np.random.randn(d) * 2


def loglikelihood(x):
    return multivariate_normal.logpdf(x, mean=like_mean, cov=like_cov)


n_samples = 500
rng_key, init_key, sample_key = jax.random.split(rng_key, 3)

prior = distrax.MultivariateNormalDiag(loc=jnp.zeros(d), scale_diag=jnp.ones(d))


##################################################################################
# Configure the NS kernel
##################################################################################


def log_density(x):
    return prior.log_prob(x) + loglikelihood(x)


algo = univariate_slice(log_density, n_doublings=10)
initial_state = prior.sample(seed=rng_key)
state = algo.init(initial_state)

n_steps = 500


@progress_bar_scan(n_steps)
def one_step(carry, xs):
    state, k = carry
    k, subk = jax.random.split(k, 2)
    state, info = algo.step(subk, state)
    return (state, k), info


##################################################################################
# run the ns kernel
##################################################################################

iterations = jnp.arange(n_steps)
(live, _), dead = jax.lax.scan((one_step), (state, rng_key), iterations)


# comment out the above scan and uncomment this for debugging
# with jax.disable_jit():
#     for i in range(10):
#         rng_key, sample_key = jax.random.split(rng_key)
#         state, info = algo.step(sample_key, state)

##################################################################################
# Collect the samples into anesthetic objects
##################################################################################

samples = ns.MCMCSamples(jnp.concatenate(dead[0], axis=0))
from lsbi.model import ReducedLinearModel

model = ReducedLinearModel(
    mu_L=like_mean,
    Sigma_L=like_cov,
    logLmax=loglikelihood(like_mean),
)

print(f"True logZ = {model.logZ():.2f}")
# a = samples.set_beta(0.0).plot_2d(np.arange(d), figsize=(10, 10))
# samples.plot_2d(a)
a = ns.MCMCSamples(model.posterior().rvs(200)).plot_2d(np.arange(d), figsize=(10, 10))
samples.plot_2d(a)
samples.to_csv("post.csv")
a.iloc[0, 0].legend(["Truth", "NS"], loc="lower left", bbox_to_anchor=(0, 1), ncol=3)
plt.savefig("post.pdf")
plt.savefig("post.png", dpi=300)
plt.show()
