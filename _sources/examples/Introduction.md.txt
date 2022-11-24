---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.0
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# A Quick Introduction to Blackjax

BlackJAX is an MCMC sampling library based on [JAX](https://github.com/google/jax). BlackJAX provides well-tested and ready to use sampling algorithms. It is also explicitly designed to be modular: it is easy for advanced users to mix-and-match different metrics, integrators, trajectory integrations, etc.

In this notebook we provide a simple example based on basic Hamiltonian Monte Carlo and the NUTS algorithm to showcase the architecture and interfaces in the library

```{code-cell} ipython3
import jax
import jax.numpy as jnp
import jax.scipy.stats as stats
import matplotlib.pyplot as plt
import numpy as np

import blackjax
```

## The Problem

We'll generate observations from a normal distribution of known `loc` and `scale` to see if we can recover the parameters in sampling. **MCMC algorithms usually assume samples are being drawn from an unconstrained Euclidean space.** Hence why we'll log transform the scale parameter, so that sampling is done on the real line. Samples can be transformed back to their original space in post-processing. Let's take a decent-size dataset with 1,000 points:

```{code-cell} ipython3
loc, scale = 10, 20
observed = np.random.normal(loc, scale, size=1_000)
```

```{code-cell} ipython3
def logprob_fn(loc, log_scale, observed=observed):
    """Univariate Normal"""
    scale = jnp.exp(log_scale)
    logpdf = stats.norm.logpdf(observed, loc, scale)
    return jnp.sum(logpdf)


logprob = lambda x: logprob_fn(**x)
```

## HMC

### Sampler Parameters

```{code-cell} ipython3
inv_mass_matrix = np.array([0.5, 0.01])
num_integration_steps = 60
step_size = 1e-3

hmc = blackjax.hmc(logprob, step_size, inv_mass_matrix, num_integration_steps)
```

### Set the Initial State

The initial state of the HMC algorithm requires not only an initial position, but also the potential energy and gradient of the potential energy at this position. BlackJAX provides a `new_state` function to initialize the state from an initial position.

```{code-cell} ipython3
initial_position = {"loc": 1.0, "log_scale": 1.0}
initial_state = hmc.init(initial_position)
initial_state
```

### Build the Kernel and Inference Loop


The HMC kernel is easy to obtain:

```{code-cell} ipython3
hmc_kernel = jax.jit(hmc.step)
```

BlackJAX does not provide a default inference loop, but it easy to implement with JAX's `lax.scan`:

```{code-cell} ipython3
def inference_loop(rng_key, kernel, initial_state, num_samples):
    @jax.jit
    def one_step(state, rng_key):
        state, _ = kernel(rng_key, state)
        return state, state

    keys = jax.random.split(rng_key, num_samples)
    _, states = jax.lax.scan(one_step, initial_state, keys)

    return states
```

### Inference

```{code-cell} ipython3
%%time
rng_key = jax.random.PRNGKey(0)
states = inference_loop(rng_key, hmc_kernel, initial_state, 10_000)

loc_samples = states.position["loc"].block_until_ready()
scale_samples = jnp.exp(states.position["log_scale"])
```

```{code-cell} ipython3
:tags: [hide-input]

fig, (ax, ax1) = plt.subplots(ncols=2, figsize=(15, 6))
ax.plot(loc_samples)
ax.set_xlabel("Samples")
ax.set_ylabel("loc")

ax1.plot(scale_samples)
ax1.set_xlabel("Samples")
ax1.set_ylabel("scale")
```

## NUTS

NUTS is a *dynamic* algorithm: the number of integration steps is determined at runtime. We still need to specify a step size and a mass matrix:

```{code-cell} ipython3
inv_mass_matrix = np.array([0.5, 0.01])
step_size = 1e-3

nuts = blackjax.nuts(logprob, step_size, inv_mass_matrix)
```

```{code-cell} ipython3
initial_position = {"loc": 1.0, "log_scale": 1.0}
initial_state = nuts.init(initial_position)
initial_state
```

```{code-cell} ipython3
%%time
rng_key = jax.random.PRNGKey(0)
states = inference_loop(rng_key, nuts.step, initial_state, 4_000)

loc_samples = states.position["loc"].block_until_ready()
scale_samples = jnp.exp(states.position["log_scale"])
```

```{code-cell} ipython3
:tags: [hide-input]

fig, (ax, ax1) = plt.subplots(ncols=2, figsize=(15, 6))
ax.plot(loc_samples)
ax.set_xlabel("Samples")
ax.set_ylabel("loc")

ax1.plot(scale_samples)
ax1.set_xlabel("Samples")
ax1.set_ylabel("scale")
```

### Use Stan's Window Adaptation

Specifying the step size and inverse mass matrix is cumbersome. We can use Stan's window adaptation to get reasonable values for them so we have, in practice, no parameter to specify.

The adaptation algorithm takes a function that returns a transition kernel given a step size and an inverse mass matrix:

```{code-cell} ipython3
%%time

warmup = blackjax.window_adaptation(blackjax.nuts, logprob)
state, kernel, _ = warmup.run(rng_key, initial_position, num_steps=1000)
```

We can use the obtained parameters to define a new kernel. Note that we do not have to use the same kernel that was used for the adaptation:

```{code-cell} ipython3
%%time

states = inference_loop(rng_key, kernel, state, 1_000)

loc_samples = states.position["loc"].block_until_ready()
scale_samples = jnp.exp(states.position["log_scale"])
```

```{code-cell} ipython3
:tags: [hide-input]

fig, (ax, ax1) = plt.subplots(ncols=2, figsize=(15, 6))
ax.plot(loc_samples)
ax.set_xlabel("Samples")
ax.set_ylabel("loc")

ax1.plot(scale_samples)
ax1.set_xlabel("Samples")
ax1.set_ylabel("scale")
```
