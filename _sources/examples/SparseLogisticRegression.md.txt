---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.1
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
mystnb:
  execution_timeout: 200
---

# Sparse logistic regression

This example models a logistic regression with hierarchies on the scale of the independent variable's parameters that function as a proxy for variable selection. We give the independent variable's regressors $\beta$ a prior mean of 0 and global $\tau$ and local $\lambda$ scale parameters with strong prior information around 0, thus allowing these parameters to a posteriori degenerate at 0, hence excluding its independent variable from the model. These type of hierarchies on the prior scale of a parameter create funnel geometries that are hard to efficiently explore without local or global structure of the target.

The model is run on its non-centered parametrization \citep{papaspiliopoulos2007general} with data from the numerical version of the [German credit dataset](https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data)). The target posterior is defined by its likelihood

$$
L(\mathbf{y}|\beta, \lambda, \tau) = \prod_i \text{Beta}(y_i;\sigma((\tau \lambda \odot \beta)^T X_i))
$$

with $\sigma$ the sigmoid function, and prior

$$
\pi_0(\beta, \lambda, \tau) = \text{Gamma}(\tau;1/2, 1/2)\prod_i \mathcal{N}(\beta_i;0, 1)\text{Gamma}(\lambda_i;1/2, 1/2)
$$


```{code-cell} ipython3
import jax
import jax.numpy as jnp
import jax.random as jrnd
import numpy as np
import pandas as pd
from jax.scipy.special import expit
from jax.scipy.stats import bernoulli, gamma, norm
from numpyro.diagnostics import print_summary

import blackjax


class HorseshoeLogisticReg:
    def __init__(self, X, y) -> None:
        self.X = X
        self.y = y

    def initialize_model(self, rng_key, n_chain):
        kb, kl, kt = jax.random.split(rng_key, 3)
        self.init_params = {
            "beta": jax.random.normal(kb, (n_chain, self.X.shape[1])),
            "lamda": jax.random.normal(kl, (n_chain, self.X.shape[1])),
            "tau": jax.random.normal(kt, (n_chain,)),
        }

    def logprob(self, beta, lamda, tau):  # non-centered
        # priors
        lprob = (
            jnp.sum(
                norm.logpdf(beta, loc=0.0, scale=1.0)
                + gamma.logpdf(jnp.exp(lamda), a=0.5, loc=0.0, scale=2.0)
                + lamda
            )
            + gamma.logpdf(jnp.exp(tau), a=0.5, loc=0.0, scale=2.0)
            + tau
        )
        # likelihood
        logit = jnp.sum(self.X * (jnp.exp(tau) * beta * jnp.exp(lamda)), axis=1)
        p = jnp.clip(expit(logit), a_min=1e-6, a_max=1 - 1e-6)
        lprob += jnp.sum(bernoulli.logpmf(self.y, p))
        return lprob

    def logprob_fn(self, x):
        return self.logprob(**x)


def inference_loop(rng, init_state, kernel, n_iter):
    keys = jrnd.split(rng, n_iter)

    def step(state, key):
        state, info = kernel(key, state)
        return state, (state, info)

    _, (states, info) = jax.lax.scan(step, init_state, keys)
    return states, info


print("Loading German credit data...")
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data-numeric"
data = pd.read_table(url, header=None, delim_whitespace=True)
y = -1 * (data.iloc[:, -1].values - 2)
X = (
    data.iloc[:, :-1]
    .apply(lambda x: -1 + (x - x.min()) * 2 / (x.max() - x.min()), axis=0)
    .values
)
X = np.concatenate([np.ones((1000, 1)), X], axis=1)
N_OBS, N_REG = X.shape

N_PARAM = N_REG * 2 + 1
print("Setting up German credit logistic horseshoe model...")
dist = HorseshoeLogisticReg(X, y)

batch_fn = jax.vmap
[n_chain, n_warm, n_iter] = [128, 20000, 10000]
ksam, kinit = jrnd.split(jrnd.PRNGKey(0), 2)
dist.initialize_model(kinit, n_chain)

print("Running MEADS...")
tic1 = pd.Timestamp.now()
k_warm, k_sample = jrnd.split(ksam)
warmup = blackjax.meads(dist.logprob_fn, n_chain, n_warm, batch_fn=batch_fn)
chain_state, kernel, _ = warmup.run(k_warm, dist.init_params)
init_state = chain_state.states


def one_chain(k_sam, init_state):
    state, info = inference_loop(k_sam, init_state, kernel, n_iter)
    return state.position, info


k_sample = jrnd.split(k_sample, n_chain)
samples, infos = batch_fn(one_chain)(k_sample, init_state)
tic2 = pd.Timestamp.now()
print("Runtime for MEADS", tic2 - tic1)
print_summary(samples)
```
