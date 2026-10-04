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
---

# MCMC Diagnostics

Once you have generated samples from an MCMC chain, it is crucial to assess whether the chain has converged and how many independent samples you have effectively obtained. BlackJAX provides native utilities for common diagnostics, and it integrates seamlessly with [ArviZ](https://arviz-devs.github.io/arviz/) for more advanced analysis.

```{code-cell} ipython3
import jax
import jax.numpy as jnp
import jax.scipy.stats as stats
import matplotlib.pyplot as plt
import numpy as np
import arviz as az

import blackjax
```

## Generating multiple chains

Diagnostics like $\hat{R}$ (R-hat) require multiple chains to compare within-chain and between-chain variance. Let's set up a simple 1D Gaussian model and sample 4 chains.

```{code-cell} ipython3
def logdensity_fn(x):
    return jnp.sum(stats.norm.logpdf(x, 0, 1))

# Sampling parameters
num_chains = 4
num_samples = 1000
step_size = 0.5
inverse_mass_matrix = jnp.ones(1)

nuts = blackjax.nuts(logdensity_fn, step_size, inverse_mass_matrix)

# Initialize multiple chains
rng_key = jax.random.key(0)
initial_positions = jax.random.normal(rng_key, (num_chains, 1)) * 5 # Disperse starting points
initial_states = jax.vmap(nuts.init)(initial_positions)

# Inference loop using jax.lax.scan
def inference_loop(rng_key, initial_state):
    @jax.jit
    def one_step(state, rng_key):
        state, info = nuts.step(rng_key, state)
        return state, (state, info)

    keys = jax.random.split(rng_key, num_samples)
    _, (states, infos) = jax.lax.scan(one_step, initial_state, keys)
    return states, infos

# Run chains in parallel using vmap
rng_key, sample_key = jax.random.split(rng_key)
sample_keys = jax.random.split(sample_key, num_chains)
states, infos = jax.vmap(inference_loop)(sample_keys, initial_states)

# states.position has shape (num_chains, num_samples, 1)
print(f"Samples shape: {states.position.shape}")
```

## Native BlackJAX Diagnostics

BlackJAX provides two primary diagnostic functions: `potential_scale_reduction` (R-hat) and `effective_sample_size` (ESS). These functions expect an input array with dimensions corresponding to chains and samples.

### Potential Scale Reduction ($\hat{R}$)

R-hat measures the convergence of multiple chains. A value close to 1.0 (typically $< 1.05$ or even $< 1.01$) indicates that the chains have converged to the same distribution.

```{code-cell} ipython3
rhat = blackjax.rhat(states.position)
print(f"R-hat: {rhat}")
```

### Effective Sample Size (ESS)

ESS estimates the number of independent samples contained in the chain, accounting for autocorrelation.

```{code-cell} ipython3
ess = blackjax.ess(states.position)
print(f"ESS: {ess}")
```

## Integration with ArviZ

While BlackJAX provides core utilities, ArviZ is the industry standard for Bayesian visualization and diagnostic reporting. You can easily convert BlackJAX output to an ArviZ `InferenceData` object.

### Converting to InferenceData

```{code-cell} ipython3
# ArviZ expects (chain, draw, *shape)
# BlackJAX vmap output is already (chain, draw, *shape)
dataset = az.from_dict(
    posterior={"x": states.position},
    sample_stats={
        "diverging": infos.is_divergent,
        "acceptance_rate": infos.acceptance_rate,
    }
)

az.summary(dataset)
```

### Visualizing Convergence

You can use ArviZ to plot trace plots, which help visually inspect chain mixing and convergence.

```{code-cell} ipython3
az.plot_trace(dataset)
plt.show()
```

### Autocorrelation

Autocorrelation plots help understand how quickly the information in the chain is being "refreshed."

```{code-cell} ipython3
az.plot_autocorr(dataset)
plt.show()
```

## Pareto Smoothed Importance Sampling (PSIS)

BlackJAX also includes `psis_weights`, which is useful for algorithms like Pathfinder or Variational Inference to assess the quality of the approximation and perform importance resampling.

```{code-cell} ipython3
# Dummy log-ratios for demonstration
log_ratios = jax.random.normal(rng_key, (1000,))
log_weights, pareto_k = blackjax.diagnostics.psis_weights(log_ratios)

print(f"Pareto k statistic: {pareto_k}")
# A value of k < 0.7 is generally considered a good approximation.
```


## Kernelized Stein discrepancy

KSD uses the target score (gradient of its log density), so its normalizing
constant is not needed :cite:p:`liu2016kernelized`. It complements R-hat rather
than replacing it. Use approximately independent draws (for example, thinned
chain draws). Autocorrelation can inflate it, and even IMQ KSD can fail to reveal
an unvisited mode. A small value alone does not certify convergence.

`blackjax.diagnostics.kernelized_stein_discrepancy` returns **squared KSD**.
`statistic="v"` includes diagonal pairs and has a sample-dependent bias of order
`1/n`, even for exact draws. `statistic="u"` excludes the diagonal; it can be
negative. Neither option returns a calibrated p-value or automatic stopping rule.
Compare values only with attention to the kernel, sample count and dependence.

A Gaussian RBF kernel is not suitable for detecting non-convergence in dimension
three or higher. Here we use the inverse multiquadric (IMQ) kernel with `c=1` and
`beta=-0.5`. IMQ convergence guarantees require suitable target conditions;
heavy-tailed targets with bounded scores remain a known gap
:cite:p:`gorham2017kernels`. A positive semidefinite kernel by itself is not enough
to identify distributions; integral strict positive definiteness and the target's
Stein boundary conditions are additional requirements.

Rows are processed sequentially, with vectorized columns. Pairwise work is
quadratic in sample count, and mixed Hessians have `d²` entries; temporary
pairwise storage scales as `O(n*d²)`. This example pools a thinned subset from
each chain:

```{code-cell} ipython3
import functools
from blackjax.diagnostics import imq_kernel

ksd_samples = states.position[:, ::20, :].reshape(-1, 1)
ksd_squared = jax.jit(functools.partial(
    blackjax.diagnostics.kernelized_stein_discrepancy,
    grad_logdensity_fn=jax.grad(logdensity_fn),
    kernel=imq_kernel,
    statistic="u",
))(ksd_samples)
print(f"Squared KSD U-statistic: {ksd_squared}")
```

This printed value is illustrative, not evidence of convergence. Even exact iid
samples give a range of empirical values; no null calibration is performed here.
