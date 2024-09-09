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

from blackjax.smc.tuning.from_particles import (
    particles_covariance_matrix,
    particles_stds,
    particles_means,
    mass_matrix_from_particles,
)


def loglikelihood(x):
    return multivariate_normal.logpdf(
        x, mean=jnp.ones(d), cov=jnp.eye(d) * 0.05
    )


n_samples = 500
n_delete = 250
rng_key, init_key, sample_key = jax.random.split(rng_key, 3)

prior = distrax.MultivariateNormalDiag(
    loc=jnp.zeros(d), scale_diag=jnp.ones(d)
)
# prior = distrax.Uniform(low=-2.0 * jnp.ones(d), high=2.0 * jnp.ones(d))
from blackjax import irmh

kernel = irmh.build_kernel()

mean = jnp.zeros(d)
cov = jnp.diag(jnp.ones(d)) * 2


def irmh_proposal_distribution(rng_key):
    return jax.random.multivariate_normal(rng_key, mean, cov)


def proposal_logdensity_fn(proposal, state):
    return jnp.log(
        jax.scipy.stats.multivariate_normal.pdf(
            state.position, mean=mean, cov=cov
        )
    )


def step(key, state, logdensity):
    return irmh(
        logdensity, irmh_proposal_distribution, proposal_logdensity_fn
    ).step(key, state)


def step_fn(key, state, logdensity, means, cov):
    proposal_distribution = lambda key: jax.random.multivariate_normal(
        key, means, cov
    )

    def proposal_logdensity_fn(proposal, state):
        return jax.scipy.stats.multivariate_normal.logpdf(
            state.position, mean=means, cov=cov
        ).squeeze()

    return kernel(
        key, state, logdensity, proposal_distribution, proposal_logdensity_fn
    )


def irmh_update_fn(state, info):
    cov = jnp.atleast_2d(particles_covariance_matrix(state.particles))
    mean = particles_means(state.particles)
    # cov = particles_covariance_matrix(state.particles)
    return blackjax.smc.extend_params(
        {
            "means": mean,
            "cov": cov,
        },
    )


state = prior._sample_n(rng_key, n_samples)
means = particles_means(state)
cov = particles_covariance_matrix(state)

init_params = {"means": means, "cov": cov}

# algo = blackjax.rejection_ns(prior.sample, loglikelihood, n_delete=n_delete)
algo = blackjax.inner_kernel_ns(
    logprior_fn=lambda x: prior.log_prob(x).sum().squeeze(),
    loglikelihood_fn=loglikelihood,
    mcmc_step_fn=step_fn,
    mcmc_init_fn=blackjax.rmh.init,
    mcmc_parameter_update_fn=irmh_update_fn,
    n_delete=n_delete,
    mcmc_initial_parameters=blackjax.smc.extend_params(init_params),
)
state = algo.init(state, loglikelihood)


# from jax_tqdm import scan_tqdm
from blackjax.progress_bar import progress_bar_scan


n_steps = 3000 // n_delete

# with jax.disable_jit():
#     n_iter = 10
#     iterations = jnp.arange(n_iter)
#     res = jax.lax.scan((one_step), (state, rng_key), iterations)


# @scan_tqdm(100)
@progress_bar_scan(n_steps)
def one_step(carry, xs):
    state, k = carry
    k, subk = jax.random.split(k, 2)
    state, dead_point = algo.step(subk, state)
    return (state, k), dead_point


iterations = jnp.arange(n_steps)
(live, _), dead = jax.lax.scan((one_step), (state, rng_key), iterations)

with jax.disable_jit():
    n_iter = 10
    iterations = jnp.arange(n_iter)
    res = jax.lax.scan((one_step), (state, rng_key), iterations)


dead_points = dead.particles.squeeze()
live_points = live.particles.squeeze()

samples = ns.NestedSamples(
    data=np.concatenate([live_points, dead_points.reshape(-1, d)], axis=0),
    logL=np.concatenate([live.logL, dead.logL.squeeze().reshape(-1)]),
    logL_birth=np.concatenate(
        [live.logL_birth, dead.logL_birth.squeeze().reshape(-1)]
    ),
)
print(samples.logZ())
from lsbi.model import ReducedLinearModel

model = ReducedLinearModel(
    mu_L=np.ones(d),
    Sigma_L=np.eye(d) * 0.05,
    logLmax=loglikelihood(np.ones(d)),
)

print(model.logZ())
a = samples.set_beta(0.0).plot_2d(np.arange(d))
# samples.plot_2d(a)
samples.plot_2d(a)
samples.to_csv("post.csv")
plt.savefig("post.pdf")
plt.show()
