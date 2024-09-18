import multiprocessing
import os
from datetime import date

num_cores = multiprocessing.cpu_count()
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count={}".format(num_cores)

import anesthetic as ns
import distrax
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jax.scipy.stats import multivariate_normal

import blackjax
import blackjax.progress_bar
from blackjax import irmh
from blackjax.progress_bar import progress_bar_scan
from blackjax.smc.tuning.from_particles import (
    mass_matrix_from_particles,
    particles_covariance_matrix,
    particles_means,
    particles_stds,
)

##################################################################################
# Setup the problem
##################################################################################

rng_key = jax.random.PRNGKey(2)
d = 5

np.random.seed(1)
C = np.random.randn(d, d) * 0.05
like_cov = C @ C.T
like_mean = np.random.randn(d) * 2


def loglikelihood(x):
    return multivariate_normal.logpdf(x, mean=like_mean, cov=like_cov)


n_samples = 500
n_delete = num_cores
rng_key, init_key, sample_key = jax.random.split(rng_key, 3)

prior = distrax.MultivariateNormalDiag(loc=jnp.zeros(d), scale_diag=jnp.ones(d))


##################################################################################
# Configure the NS kernel
##################################################################################

kernel = irmh.build_kernel()


def mcmc_step_fn(key, state, logdensity, means, cov):
    proposal_distribution = lambda key: jax.random.multivariate_normal(key, means, cov)

    def proposal_logdensity_fn(proposal, state):
        return jax.scipy.stats.multivariate_normal.logpdf(
            state.position, mean=means, cov=cov
        ).squeeze()

    return kernel(key, state, logdensity, proposal_distribution, proposal_logdensity_fn)


def mcmc_parameter_update_fn(state, info):
    cov = jnp.atleast_2d(particles_covariance_matrix(state.particles))
    mean = particles_means(state.particles)
    return {"means": mean, "cov": cov}


initial_state = prior._sample_n(rng_key, n_samples)
means = particles_means(initial_state)
cov = particles_covariance_matrix(initial_state)
init_params = {"means": means, "cov": cov}


"""
Setup the Nested Sampling algorithm
Provide compulsary functions:
logprior_fn: log prior density function
loglikelihood_fn: log likelihood density function #TODO combine the two with logl as a mask

mcmc_step_fn: inner MCMC algorithm step function to evolve the particles
mcmc_init_fn: corresponding initialization function for the inner kernel
mcmc_parameter_update_fn: function to tune the parameters of the mcmc step
mcmc_initial_parameters: initial parameters for the inner kernel -- effectively call the parameter update fn on the initial pop

Specific settings for the NS algorithm:
n_delete: number of points to delete at each iteration
        jax will pmap over this, so it is detected automatically in this script as the number of available cpu cores
num mcmc steps: number of successful steps to take in the inner kernel - n_repeats in polychord language
"""
algo = blackjax.inner_kernel_ns(
    logprior_fn=lambda x: prior.log_prob(x).sum().squeeze(),
    loglikelihood_fn=loglikelihood,
    mcmc_step_fn=mcmc_step_fn,
    mcmc_init_fn=blackjax.rmh.init,
    mcmc_parameter_update_fn=mcmc_parameter_update_fn,
    n_delete=n_delete,
    mcmc_initial_parameters=init_params,
    num_mcmc_steps=5,
)

# Initialize the ns state
state = algo.init(initial_state, loglikelihood)


# request 1000 steps of the NS kernel, currently this is fixed, and compresses for n_delete * n_steps rounds
# simplest design pattern is to put this in an outer while loop, and break when some convergence criteria is met
# currently there is no safety check in this compression so it can hang with too many steps, or not a good enough inner kernel
n_steps = 1000


@progress_bar_scan(n_steps)
def one_step(carry, xs):
    state, k = carry
    k, subk = jax.random.split(k, 2)
    state, dead_point = algo.step(subk, state)
    return (state, k), dead_point


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

dead_points = dead.particles.squeeze()
live_points = live.sampler_state.particles.squeeze()
# live_logL = live.sampler_state.logL


samples = ns.NestedSamples(
    data=np.concatenate([live_points, dead_points.reshape(-1, d)], axis=0),
    logL=np.concatenate([live.sampler_state.logL, dead.logL.squeeze().reshape(-1)]),
    logL_birth=np.concatenate(
        [live.sampler_state.logL_birth, dead.logL_birth.squeeze().reshape(-1)]
    ),
)
samples.to_csv("samples.csv")
lzs = samples.logZ(100)
# print(samples.logZ())
print(f"logZ = {lzs.mean():.2f} ± {lzs.std():.2f}")
from lsbi.model import ReducedLinearModel

model = ReducedLinearModel(
    mu_L=like_mean,
    Sigma_L=like_cov,
    logLmax=loglikelihood(like_mean),
)

print(f"True logZ = {model.logZ():.2f}")
a = samples.set_beta(0.0).plot_2d(np.arange(d), figsize=(10, 10))
# samples.plot_2d(a)
ns.MCMCSamples(model.posterior().rvs(200)).plot_2d(a)
samples.plot_2d(a)
samples.to_csv("post.csv")
a.iloc[0, 0].legend(
    ["Prior", "Truth", "NS"], loc="lower left", bbox_to_anchor=(0, 1), ncol=3
)
plt.suptitle(
    f"NS logZ = {lzs.mean():.2f} ± {lzs.std():.2f}, true logZ = {model.logZ():.2f}"
)
plt.savefig("post.pdf")
plt.savefig("post.png", dpi=300)
plt.show()
