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
---

# Sampling Multiple Chains

In this example, we will briefly demonstrate how you can run multiple MCMC chains using `jax` built-in constructs: `vmap` and `pmap`.
We will use the NUTS example from the introduction notebook, and compare the performance of the two approaches.

## Vectorization vs parallelization

`jax` provides two distinct transformations:
- [vmap](https://jax.readthedocs.io/en/latest/jax.html?highlight=vmap#jax.vmap), used to automatically vectorize `jax` code
- and [pmap](https://jax.readthedocs.io/en/latest/_autosummary/jax.pmap.html#jax.pmap),
which enables parallelization across multiple devices, such as multiple GPUs (or, in our case, CPU cores).

Please see the the respective tutorials on [Automatic Vectorization](https://jax.readthedocs.io/en/latest/jax-101/03-vectorization.html)
and [Parallel Evaluation](https://jax.readthedocs.io/en/latest/jax-101/06-parallelism.html) for a detailed walkthrough of both features.

## Using `pmap` on CPU

By default, `jax` will treat your CPU as a single device, regardless of the number of cores available.

Unfortunately, this means that using `pmap` is not possible out of the box -- we'll first need
to instruct `jax` to split the CPU into multiple devices.
Please see [this issue](https://github.com/google/jax/issues/1408) for more discussion on this topic.

Currently, this can only be done via `XLA_FLAGS` environmental variable.

**Note that this variable has to be set before any `jax` code is executed**

```{code-cell} ipython3
import os
import multiprocessing

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count={}".format(
    multiprocessing.cpu_count()
)
```

We can now import `jax` and confirm that it have successfuly recognized our CPU as multiple devices.

```{code-cell} ipython3
import jax
import jax.numpy as jnp

len(jax.devices())
```

```{code-cell} ipython3
jax.devices()[:2]
```

### Choosing the number of devices

`pmap` has one more limitation - it is not able to parallelize the execution when the number of items is larger than the number of devices.

```{code-cell} ipython3
def fn(x):
    return x + 1

try:
    data = jnp.arange(1024)
    parallel_fn = jax.pmap(fn)

    parallel_fn(data)

except Exception as e:
    print(e)
```

This means that you will only be able to run as many MCMC chains as you have CPU cores.
See this [question](https://github.com/google/jax/discussions/4198) for a more detailed discussion on the topic,
and a workaround involving nesting `pmap` and `vmap` calls.

Another option is to set the device count to a number larger than the core count, e.g. `200`, but
it's [unclear what side effects it might have](https://github.com/google/jax/issues/1408#issuecomment-536158048).

### Using numpyro helpers

[Numpyro](https://num.pyro.ai/en/stable/index.html) also relies on `pmap` to sample multiple chains,
and provides small helper functions to simplify the `jax` configuration:
- [set_platform](https://num.pyro.ai/en/stable/utilities.html#set-platform)
- [set_host_device_count](https://num.pyro.ai/en/stable/utilities.html#set-host-device-count)

They might be helpful if you have `numpyro` installed in your system.

## Perfomance comparison - NUTS

The code below follows the NUTS example from the previous notebook

```{code-cell} ipython3
import jax.scipy.stats as stats

import matplotlib.pyplot as plt
import numpy as np

import blackjax
```

```{code-cell} ipython3
loc, scale = 10, 20
observed = np.random.normal(loc, scale, size=1_000)


def logprob_fn(loc, scale, observed=observed):
    """Univariate Normal"""
    logpdf = stats.norm.logpdf(observed, loc, scale)
    return jnp.sum(logpdf)


def logprob(x):
    return logprob_fn(**x)


def inference_loop(rng_key, kernel, initial_state, num_samples):

    @jax.jit
    def one_step(state, rng_key):
        state, _ = kernel(rng_key, state)
        return state, state

    keys = jax.random.split(rng_key, num_samples)
    _, states = jax.lax.scan(one_step, initial_state, keys)

    return states
```

```{code-cell} ipython3
inv_mass_matrix = np.array([0.5, 0.5])
step_size = 1e-3

nuts = blackjax.nuts(logprob, step_size, inv_mass_matrix)
```

```{code-cell} ipython3
rng_key = jax.random.PRNGKey(0)
num_chains = multiprocessing.cpu_count()
```

### Using `vmap`

Here we apply `vmap` inside the `one_step` function, vectorizing the transition function,
such that it can handle multiple states (and rng keys) at the same time.

```{code-cell} ipython3
def inference_loop_multiple_chains(
    rng_key, kernel, initial_state, num_samples, num_chains
):

    @jax.jit
    def one_step(states, rng_key):
        keys = jax.random.split(rng_key, num_chains)
        states, _ = jax.vmap(kernel)(keys, states)
        return states, states

    keys = jax.random.split(rng_key, num_samples)
    _, states = jax.lax.scan(one_step, initial_state, keys)

    return states
```

We now prepare the initial states (using `vmap` again to call `init` on a batch of initial positions)

```{code-cell} ipython3
initial_positions = {"loc": np.ones(num_chains), "scale": 2.0 * np.ones(num_chains)}
initial_states = jax.vmap(nuts.init, in_axes=(0))(initial_positions)
```

And finally run the sampler

```{code-cell} ipython3
%%time

states = inference_loop_multiple_chains(
    rng_key, nuts.step, initial_states, 2_000, num_chains
)
_ = states.position["loc"].block_until_ready()
```

You can now access the samples from individual chains by simply indexing the returned arrays:

```{code-cell} ipython3
states.position["loc"].shape
```

E.g. to get the `loc` samples for the second chain:

```{code-cell} ipython3
samples = states.position["loc"][:, 1]
samples
```

```{code-cell} ipython3
samples.shape
```

```{code-cell} ipython3
fig, ax = plt.subplots(figsize=(15, 6))

ax.plot(states.position["loc"][:, 0], color="blue", alpha=0.25)
ax.plot(states.position["loc"][:, 1], color="red", alpha=0.25)
ax.set_xlabel("Samples")
ax.set_ylabel("loc")
ax.legend(["Chain 1", "Chain 2"])
```

### Using `pmap`

In case of `pmap`, we can simply choose to apply the transformation directly to the original `inference_loop` function.

```{code-cell} ipython3
inference_loop_multiple_chains = jax.pmap(inference_loop, in_axes=(0, None, 0, None), static_broadcasted_argnums=(1, 3))
```

We now need to generate one random key per chain:

```{code-cell} ipython3
keys = jax.random.split(rng_key, num_chains)
```

And we're ready to run the sampler:

```{code-cell} ipython3
%%time

pmap_states = inference_loop_multiple_chains(
    keys, nuts.step, initial_states, 2_000
)
_ = pmap_states.position["loc"].block_until_ready()
```

Note that the samples are transposed compared to the `vmap` case

```{code-cell} ipython3
pmap_states.position["loc"].shape
```

```{code-cell} ipython3
fig, ax = plt.subplots(figsize=(15, 6))

ax.plot(pmap_states.position["loc"][0, :], color="blue", alpha=0.25)
ax.plot(pmap_states.position["loc"][1, :], color="red", alpha=0.25)
ax.set_xlabel("Samples")
ax.set_ylabel("loc")
ax.legend(["Chain 1", "Chain 2"])
```

### Conclusions

In this particular case we can see quite dramatic differences in performance
between the two approaches (several minutes for `vmap`, and several seconds for `pmap`).

This is actually an expected result for NUTS, especially un-tuned one as in our example.

What happens here is that with `vmap` we need always need to wait for the slowest chain when calling `one_step` function.
With several thousand steps, the differences can easily add-up, leading to low utilization of the CPU
(most cores are idle, waiting for the chain with longest leapfrog).

`pmap`, on the other hand, runs the chains independently, and hence does not suffer from this effect.

---

```{code-cell} ipython3
%load_ext watermark
%watermark -d -m -v -p jax,jaxlib,blackjax
```
