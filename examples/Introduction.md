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

```{code-cell} ipython3
%load_ext watermark
%watermark -d -m -v -p jax,jaxlib,blackjax
```

```{code-cell} ipython3
jax.devices()
```

## The Problem

We'll generate observations from a normal distribution of known `loc` and `scale` to see if we can recover the parameters in sampling. Let's take a decent-size dataset with 1,000 points:

```{code-cell} ipython3
loc, scale = 10, 20
observed = np.random.normal(loc, scale, size=1_000)
```

```{code-cell} ipython3
def logprob_fn(loc, scale, observed=observed):
    """Univariate Normal"""
    logpdf = stats.norm.logpdf(observed, loc, scale)
    return jnp.sum(logpdf)


logprob = lambda x: logprob_fn(**x)
```

## HMC

### Sampler Parameters

```{code-cell} ipython3
inv_mass_matrix = np.array([0.5, 0.5])
num_integration_steps = 60
step_size = 1e-3

hmc = blackjax.hmc(logprob, step_size, inv_mass_matrix, num_integration_steps)
```

### Set the Initial State

The initial state of the HMC algorithm requires not only an initial position, but also the potential energy and gradient of the potential energy at this position. BlackJAX provides a `new_state` function to initialize the state from an initial position.

```{code-cell} ipython3
initial_position = {"loc": 1.0, "scale": 2.0}
initial_state = hmc.init(initial_position)
initial_state
```

### Build the Kernel and Inference Loop


The HMC kernel is easy to obtain:

```{code-cell} ipython3
%%time
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
scale_samples = states.position["scale"]
```

```{code-cell} ipython3
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
inv_mass_matrix = np.array([0.5, 0.5])
step_size = 1e-3

nuts = blackjax.nuts(logprob, step_size, inv_mass_matrix)
```

```{code-cell} ipython3
initial_position = {"loc": 1.0, "scale": 2.0}
initial_state = nuts.init(initial_position)
initial_state
```

```{code-cell} ipython3
%%time
rng_key = jax.random.PRNGKey(0)
states = inference_loop(rng_key, nuts.step, initial_state, 4_000)

loc_samples = states.position["loc"].block_until_ready()
scale_samples = states.position["scale"]
```

```{code-cell} ipython3
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

warmup = blackjax.window_adaptation(
    blackjax.nuts,
    logprob,
    1000,
)
state, kernel, _ = warmup.run(
    rng_key,
    initial_position,
)
```

We can use the obtained parameters to define a new kernel. Note that we do not have to use the same kernel that was used for the adaptation:

```{code-cell} ipython3
%%time

states = inference_loop(rng_key, nuts.step, initial_state, 1_000)

loc_samples = states.position["loc"].block_until_ready()
scale_samples = states.position["scale"]
```

```{code-cell} ipython3
fig, (ax, ax1) = plt.subplots(ncols=2, figsize=(15, 6))
ax.plot(loc_samples)
ax.set_xlabel("Samples")
ax.set_ylabel("loc")

ax1.plot(scale_samples)
ax1.set_xlabel("Samples")
ax1.set_ylabel("scale")
```

## Sample Multiple Chains

We can easily sample multiple chains using JAX's `vmap` construct. See the [documentation](https://jax.readthedocs.io/en/latest/jax.html?highlight=vmap#jax.vmap) to understand how the mapping works.

```{code-cell} ipython3
num_chains = 4
initial_positions = {"loc": np.ones(num_chains), "scale": 2.0 * np.ones(num_chains)}
initial_states = jax.vmap(nuts.init, in_axes=(0))(initial_positions)
```

```{code-cell} ipython3
def inference_loop_multiple_chains(
    rng_key, kernel, initial_state, num_samples, num_chains
):
    def one_step(states, rng_key):
        keys = jax.random.split(rng_key, num_chains)
        states, _ = jax.vmap(kernel)(keys, states)
        return states, states

    keys = jax.random.split(rng_key, num_samples)
    _, states = jax.lax.scan(one_step, initial_state, keys)

    return states
```

```{code-cell} ipython3
%%time
states = inference_loop_multiple_chains(
    rng_key, nuts.step, initial_states, 2_000, num_chains
)
states.position["loc"].block_until_ready()
```

This scales very well to hundreds of chains on CPU, tens of thousand on GPU:

```{code-cell} ipython3
%%time
num_chains = 40
initial_positions = {"loc": np.ones(num_chains), "scale": 2.0 * np.ones(num_chains)}
initial_states = jax.vmap(nuts.init, in_axes=(0,))(initial_positions)
states = inference_loop_multiple_chains(
    rng_key, nuts.step, initial_states, 1_000, num_chains
)
states.position["loc"].block_until_ready()
```

In this example the result is a dictionnary and each entry has shape `(num_samples, num_chains)`. Here's how to access the samples of the second chains for `loc`:
