import distrax
import jax
import jax.numpy as jnp
import tqdm
from jax.scipy.linalg import inv, solve

import blackjax
from blackjax.ns.utils import log_weights

# jax.config.update("jax_enable_x64", True)

rng_key = jax.random.PRNGKey(0)
d = 20

C = jax.random.normal(rng_key, (d, d)) * 0.1
like_cov = C @ C.T
like_mean = jax.random.normal(rng_key, (d,))
prior_mean = jnp.zeros(d)
prior_cov = jnp.eye(d) * 1
prior = distrax.MultivariateNormalDiag(loc=jnp.zeros(d), scale_diag=jnp.diag(prior_cov))


def loglikelihood(x):
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
    logLmax=loglikelihood(like_mean),
)

n_live = 500
n_delete = 20
num_mcmc_steps = d * 5
algo = blackjax.ns.adaptive.nss(
    logprior_fn=prior.log_prob,
    loglikelihood_fn=loglikelihood,
    n_delete=n_delete,
    num_mcmc_steps=num_mcmc_steps,
)

rng_key, init_key, sample_key = jax.random.split(rng_key, 3)

initial_particles = prior.sample(seed=init_key, sample_shape=(n_live,))
state = algo.init(initial_particles, loglikelihood)


@jax.jit
def one_step(carry, xs):
    state, k = carry
    k, subk = jax.random.split(k, 2)
    state, dead_point = algo.step(subk, state)
    return (state, k), dead_point


# n_steps = 1000
# (live, _), dead = jax.lax.scan((one_step), (state, rng_key), length=n_steps)

dead = []

for _ in tqdm.trange(1000):
    if state.sampler_state.logZ_live - state.sampler_state.logZ < -3:  # type: ignore[attr-defined]
        break
    (state, rng_key), dead_info = one_step((state, rng_key), None)
    dead.append(dead_info)


dead = jax.tree.map(lambda *args: jnp.concatenate(args), *dead)

logw = log_weights(rng_key, dead)  # type: ignore[arg-type]
logZs = jax.scipy.special.logsumexp(logw, axis=0)

print(f"Analytic evidence: {log_analytic_evidence:.2f}")
print(f"Estimated evidence: {logZs.mean():.2f} +- {logZs.std():.2f}")
