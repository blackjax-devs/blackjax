---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.15.2
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
mystnb:
  execution_timeout: 400
---

# Use with Funsor

[Funsor](https://github.com/pyro-ppl/funsor) is a library of functional
tensors for probabilistic programming. It generalises the tensor interface by
treating arrays as *functions over named variables* rather than positionally
indexed grids. The central consequence for MCMC: Funsor can **exactly
marginalise discrete latent variables** via variable elimination, turning
a sum over K discrete states into a fused JAX operation that is fully
compatible with `jax.jit` and `jax.grad`.

This makes Funsor the natural complement to BlackJax for models that mix
discrete and continuous latent variables — Gaussian Mixture Models, Hidden
Markov Models, mixed-membership models — where gradient-based samplers such
as NUTS would otherwise be inapplicable because discrete variables block
automatic differentiation.

In this notebook we fit a **Gaussian Mixture Model** with K=3 components.
The discrete component assignments `z_i` are marginalised analytically by
Funsor; BlackJax NUTS then samples the continuous parameters `μ` and `π`.

```{admonition} Before you start
You will need [Funsor](https://github.com/pyro-ppl/funsor) to run this
example:

    pip install "funsor>=0.4.7"
```

## Setup and data

```{code-cell} ipython3
:tags: [remove-output]

import math
import jax
import jax.numpy as jnp
import numpy as np
import blackjax

import funsor
import funsor.ops as ops

funsor.set_backend("jax")          # must be called before Tensor/Variable are used

from funsor.domains import Bint
from funsor.tensor import Tensor
from funsor.terms import Variable

from datetime import date
rng_key = jax.random.key(int(date.today().strftime("%Y%m%d")))
```

We generate synthetic data from a three-component Gaussian mixture so we can
verify that the sampler recovers the true parameters.

```{code-cell} ipython3
:tags: [hide-cell]

K, N = 3, 300
true_mu = np.array([-4.0, 0.0, 4.0])
true_pi = np.array([0.3,  0.4, 0.3])

rng    = np.random.default_rng(42)
z_true = rng.choice(K, size=N, p=true_pi)
data   = jnp.array(rng.normal(true_mu[z_true], 1.0), dtype=jnp.float32)
```

```{code-cell} ipython3
print(f"N={N} observations, K={K} components")
print(f"True means : {true_mu}")
print(f"True weights: {true_pi}")
```

## The model and log-density

The generative model is:

```
π   ~ Dirichlet(1, 1, 1)        # mixing weights  (continuous)
μ_k ~ Normal(0, 5)              # component means (continuous)
z_i ~ Categorical(π)            # assignment      (discrete — marginalised)
x_i | z_i ~ Normal(μ[z_i], 1)  # likelihood
```

Because `z_i` is discrete, direct gradient-based inference is impossible.
Funsor solves this by treating the sum over assignments as a symbolic
computation graph that JAX can differentiate through. Three Funsor primitives
do all the work:

- `Tensor(arr, {"name": Bint[size]})` — wraps a JAX array so its axis is
  addressable by name rather than by integer position.
- `Variable("z", Bint[K])` — a free (unevaluated) discrete variable ranging
  over `{0, …, K-1}`.
- `f(k=z)` — substitution: renames the `"k"` axis to `"z"`, transferring the
  name so that expressions over different named axes broadcast correctly.

The mixing weights `π` live on the K-simplex. Rather than sampling all K
components of `log_pi` (which leaves a constant-shift null direction that
confuses the mass matrix), we fix the first logit to zero and sample only
K−1 free parameters. The map `ℝ^{K-1} → Δ^{K-1}` is bijective, so no
Jacobian correction is needed.

```{code-cell} ipython3
LOG_2PI = math.log(2.0 * math.pi)   # Python float — treated as a Funsor Number

def gmm_logdensity(position):
    mu     = position["mu"]          # (K,)   component means
    log_pi = position["log_pi"]      # (K-1,) unconstrained free logits
    # Prepend a fixed zero so softmax maps K-1 reals bijectively onto Δ^{K-1}.
    pi = jax.nn.softmax(jnp.concatenate([jnp.zeros(1), log_pi]))   # (K,)

    # ── μ prior: μ_k ~ Normal(0, 5) ─────────────────────────────────────────
    log_mu_prior = jnp.sum(-0.5 * (mu / 5.0) ** 2 - math.log(5.0) - 0.5 * LOG_2PI)

    # ── Wrap JAX arrays as named Funsor tensors ──────────────────────────────
    # Tensor(arr, inputs): arr is indexed by the dimensions named in inputs.
    mu_f = Tensor(mu,   {"k": Bint[K]})   # mu_f[k]  for k ∈ {0, …, K-1}
    pi_f = Tensor(pi,   {"k": Bint[K]})   # pi_f[k]
    x_f  = Tensor(data, {"n": Bint[N]})   # x_f[n]   for n ∈ {0, …, N-1}

    # ── Discrete variable: component assignment ──────────────────────────────
    z = Variable("z", Bint[K])

    # ── log p(z | π): Funsor with named dim "z" ─────────────────────────────
    # pi_f(k=z) substitutes k→z, renaming the axis.
    # .log() applies element-wise: log π[z].
    log_cat_prior = pi_f(k=z).log()       # inputs: {"z": Bint[K]}

    # ── log p(x_n | z): Funsor over {"z", "n"} ──────────────────────────────
    # mu_f(k=z) has inputs {"z": Bint[K]}.
    # x_f       has inputs {"n": Bint[N]}.
    # Their difference creates an outer product automatically:
    #   inputs {"z": Bint[K], "n": Bint[N]}, underlying shape (K, N).
    mu_z   = mu_f(k=z)
    sq_err = (x_f - mu_z) ** 2           # inputs: {"z": Bint[K], "n": Bint[N]}
    log_lik = -0.5 * sq_err - 0.5 * LOG_2PI   # Normal(μ[z], σ=1) log-prob at x[n]

    # ── Joint log p(x_n, z): broadcast log_cat_prior over "n" ───────────────
    log_joint = log_cat_prior + log_lik   # inputs: {"z": Bint[K], "n": Bint[N]}

    # ── Exact marginalisation of z ───────────────────────────────────────────
    # reduce(logaddexp, "z") computes log Σ_z exp(log_joint) for every n.
    # This is exact variable elimination: Funsor evaluates all K states and
    # combines them with logsumexp entirely inside JAX — no sampling, no
    # approximation, and the result is differentiable w.r.t. mu and pi.
    log_marginal_n = log_joint.reduce(ops.logaddexp, "z")  # inputs: {"n": Bint[N]}

    # ── Sum over observations + μ prior ─────────────────────────────────────
    log_p = log_marginal_n.reduce(ops.add, "n")            # scalar Funsor, inputs: {}

    return log_p.data + log_mu_prior
```

Let us verify that the log-density is finite at a sensible initialisation and
that `jax.grad` can differentiate through the Funsor marginalisation:

```{code-cell} ipython3
position0 = {
    "mu":     jnp.array([-3., 0., 3.]),
    "log_pi": jnp.zeros(K - 1),   # K-1 free logits; first logit fixed at 0
}
print("log p(data | init):", gmm_logdensity(position0))
print("grad w.r.t. mu    :", jax.grad(gmm_logdensity)(position0)["mu"])
```

## Window adaptation

BlackJax's window adaptation tunes the NUTS step size and mass matrix during
a warmup phase. Here we run 1000 warmup steps, after which the kernel is ready
for sampling.

```{code-cell} ipython3
%%time

rng_key, warmup_key = jax.random.split(rng_key)

adapt = blackjax.window_adaptation(blackjax.nuts, gmm_logdensity)
(last_state, parameters), _ = adapt.run(warmup_key, position0, num_steps=1000)
kernel = blackjax.nuts(gmm_logdensity, **parameters).step
```

## Inference loop

```{code-cell} ipython3
:tags: [hide-cell]

def inference_loop(rng_key, kernel, initial_state, num_samples):
    @jax.jit
    def one_step(state, rng_key):
        state, info = kernel(rng_key, state)
        return state, (state, info)

    keys = jax.random.split(rng_key, num_samples)
    _, (states, infos) = jax.lax.scan(one_step, initial_state, keys)
    return states, infos
```

```{code-cell} ipython3
%%time

rng_key, sample_key = jax.random.split(rng_key)
states, infos = inference_loop(sample_key, kernel, last_state, num_samples=1000)
```

## Results

```{note}
GMMs are invariant to permutation of component labels. NUTS converges to one
of the K! symmetric modes; the component indices carry no absolute meaning
across runs with different random seeds.
```

```{code-cell} ipython3
import matplotlib.pyplot as plt

mu_samples = states.position["mu"]
# Reconstruct full K-simplex from K-1 free logits.
pi_samples = jax.vmap(
    lambda lp: jax.nn.softmax(jnp.concatenate([jnp.zeros(1), lp]))
)(states.position["log_pi"])

fig, axes = plt.subplots(2, K, figsize=(9, 4), sharey="row")
for k in range(K):
    axes[0, k].hist(np.array(mu_samples[:, k]), bins=40, density=True)
    axes[0, k].axvline(true_mu[k], color="red", linestyle="--", label="true")
    axes[0, k].set_title(f"μ[{k}]")
    axes[1, k].hist(np.array(pi_samples[:, k]), bins=40, density=True)
    axes[1, k].axvline(true_pi[k], color="red", linestyle="--", label="true")
    axes[1, k].set_title(f"π[{k}]")
axes[0, 0].legend()
plt.tight_layout();
```

```{code-cell} ipython3
:tags: [hide-input]

print("Posterior E[μ]:", mu_samples.mean(0).round(2), "  true:", true_mu)
print("Posterior E[π]:", pi_samples.mean(0).round(2), "  true:", true_pi)

accept = float(infos.acceptance_rate.mean())
print(f"Mean acceptance rate: {accept:.2f}")
```
