import jax
import jax.numpy as jnp
import tqdm
from jax.scipy.linalg import inv, solve

import blackjax
from blackjax.ns.utils import finalise, log_weights

# jax.config.update("jax_enable_x64", True)

rng_key = jax.random.PRNGKey(0)
d = 5

C = jax.random.normal(rng_key, (d, d)) * 0.1
like_cov = C @ C.T
like_mean = jax.random.normal(rng_key, (d,))
prior_mean = jnp.zeros(d)
prior_cov = jnp.eye(d) * 1
logprior_fn = lambda x: jax.scipy.stats.multivariate_normal.logpdf(
    x, prior_mean, prior_cov
)


def loglikelihood_fn(x):
    return jax.scipy.stats.multivariate_normal.logpdf(x, mean=like_mean, cov=like_cov)


def compute_logZ(mu_L, Sigma_L, logLmax=0, mu_pi=None, Sigma_pi=None):
    Sigma_P = inv(inv(Sigma_pi) + inv(Sigma_L))
    mu_P = jnp.dot(Sigma_P, (solve(Sigma_pi, mu_pi) + solve(Sigma_L, mu_L)))
    logdet_Sigma_P = jnp.linalg.slogdet(Sigma_P)[1]
    logdet_Sigma_pi = jnp.linalg.slogdet(Sigma_pi)[1]

    return (
        logLmax
        + logdet_Sigma_P / 2
        - logdet_Sigma_pi / 2
        - jnp.dot((mu_P - mu_pi), solve(Sigma_pi, mu_P - mu_pi)) / 2
        - jnp.dot((mu_P - mu_L), solve(Sigma_L, mu_P - mu_L)) / 2
    )


log_analytic_evidence = compute_logZ(
    like_mean,
    like_cov,
    mu_pi=prior_mean,
    Sigma_pi=prior_cov,
    logLmax=loglikelihood_fn(like_mean),
)

############################################
# Nested Sampling algorithm definition
############################################

# We use the loaded `nested slice sampling` here, bypassing the choice of inner kernel and
# inner kernel tuning, in favour of a simpler UI that loads the vectorized slice sampler

# n_live is the number of live samples to draw initially and maintain through the run
n_live = 500
# num_delete is the number of samples to delete each outer kernel iteration, as the inner kernel is parallelised we do this
# to update all of these points in parallel, useful for GPU acceleration hopefully.
num_delete = 20
# num_inner_steps is the number of MCMC steps to perform with the inner kernel in order to decorrelate the resampled points
# we set this conservatively high here at 5 times the dimension of the parameter space
num_inner_steps = d * 5

algo = blackjax.nss(
    logprior_fn=logprior_fn,
    loglikelihood_fn=loglikelihood_fn,
    num_delete=num_delete,
    num_inner_steps=num_inner_steps,
)

rng_key, init_key, sample_key = jax.random.split(rng_key, 3)

initial_particles = jax.random.multivariate_normal(
    init_key, prior_mean, prior_cov, (n_live,)
)

# We can run the algorithm for a fixed number of steps but we run into a quirk of nested sampling here. The state after N iterations
# does not necessarily contain any useful posterior points, it will have accumulated an estimate of the marginal likelihood, and this
# is what is usefully tracked.


# n_steps = 1000
# (live, _), dead = jax.lax.scan((one_step), (state, rng_key), length=n_steps)


# Also typically we would wrap the outer in a while loop, as the compression of nested sampling can push well past the posterior typical
# set if left fixed. This leaves a slightly strange construction, but works well. We want to accumulate the algorithm info (as this is)
# how we will reconstruct posterior points, but the lax while loop wrapper won't accumulate well. So we will jit compile the outer step
# and run it in a python loop

live = algo.init(initial_particles)
step_fn = jax.jit(algo.step)
dead = []
# with jax.disable_jit():
for _ in tqdm.trange(1000):
    # We track the estimate of the evidence in the live points as logZ_live, and the accumulated sum across all steps in logZ
    # this gives a handy termination that allows us to stop early
    if live.logZ_live - live.logZ < -3:  # type: ignore[attr-defined]
        break
    rng_key, subkey = jax.random.split(rng_key, 2)
    live, dead_info = step_fn(subkey, live)
    dead.append(dead_info)

# It is now not too bad to remap the list of NSInfos into a single instance
# note in theory we should include the live points, but assuming we have done things correctly and hit the termination criteria,
# they will contain negligible weight
# dead = jax.tree.map(lambda *args: jnp.concatenate(args), *dead)

# From here we can use the utils to compute the log weights and the evidence of the accumulated dead points
# sampling log weights lets us get a sensible error on the evidence estimate
nested_samples = finalise(live, dead)
logw = log_weights(rng_key, nested_samples)
logZs = jax.scipy.special.logsumexp(logw, axis=0)

print(f"Analytic evidence: {log_analytic_evidence:.2f}")
print(f"Runtime evidence: {live.logZ:.2f}")  # type: ignore[attr-defined]
print(f"Estimated evidence: {logZs.mean():.2f} +- {logZs.std():.2f}")
