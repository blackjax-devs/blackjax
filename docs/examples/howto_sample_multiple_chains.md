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

```{code-cell} ipython3
:tags: [remove-cell]

import os
import multiprocessing

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count={}".format(
    multiprocessing.cpu_count()
)
```

# Sample with multiple chains in parallel

Sampling with a few chains has become ubiquitous in modern probabilistic programming because it allows to compute better convergence diagnostics such as $\hat{R}$. More recently a new trend has emerged where researchers try to sample with thousands of chains for only a few steps. Whatever your use case is, Blackjax has you covered: thanks to JAX's primitives you will be able to run multiple chains on CPU, GPU or TPU.

## Vectorization vs parallelization

`JAX` provides two distinct primitives to "run things in parallel", and it is important to understand the difference to make the best use of Blackjax:

- [jax.vmap](https://jax.readthedocs.io/en/latest/jax.html?highlight=vmap#jax.vmap) is used to SIMD vectorize `JAX` code. It is important to remember that vectorization happens at the *instruction level*, each CPU or GPU  instruction will the process the information from your different chains, *one intructions at a time*. This can have some unexpected consequences;
- JAX's [sharding API](https://docs.jax.dev/en/latest/sharding.html) is a higher-level abstraction where computation is distributed across multiple devices (GPUs, TPUs, or CPU cores) by annotating arrays with a `NamedSharding`. This replaces the now-deprecated `jax.pmap`.

For detailed walkthroughs of both primitives we invite you to read JAX's tutorials on [Automatic Vectorization](https://jax.readthedocs.io/en/latest/jax-101/03-vectorization.html)
and [Distributed arrays and automatic parallelism](https://docs.jax.dev/en/latest/notebooks/Distributed_arrays_and_automatic_parallelization.html).

## NUTS in parallel

In the following we will sample from a linear regression with a NUTS sampler. This will illustrate the inherent limits with using `jax.vmap` when sampling with adaptative algorithms such as NUTS.

The model is

```{code-cell} ipython3
import numpy as np

import jax
import jax.numpy as jnp
import jax.scipy.stats as stats


loc, scale = 10, 20
observed = np.random.normal(loc, scale, size=1_000)


def logdensity_fn(loc, log_scale, observed=observed):
    """Univariate Normal"""
    scale = jnp.exp(log_scale)
    logjac = log_scale
    logpdf = stats.norm.logpdf(observed, loc, scale)
    return logjac + jnp.sum(logpdf)


def logdensity(x):
    return logdensity_fn(**x)


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
:tags: [remove-output]

from datetime import date
rng_key = jax.random.key(int(date.today().strftime("%Y%m%d")))
```

To make our demonstration more dramatic we will used a NUTS sampler with poorly chosen parameters:

```{code-cell} ipython3
import blackjax


inv_mass_matrix = np.array([0.5, 0.01])
step_size = 1e-3

nuts = blackjax.nuts(logdensity, step_size, inv_mass_matrix)
```

And finally, to compare `jax.vmap` against device-parallel execution we sample as many chains as the machine has CPU cores:

```{code-cell} ipython3
import multiprocessing

num_chains = multiprocessing.cpu_count()
```

### Using `jax.vmap`

Newcomers to JAX immediately recognize the benefits of using `jax.vmap`, and for a good reason: easily transforming any function into a universal function that will execute instructions in parallel is awesome!

Here we apply `jax.vmap` inside the `one_step` function and vectorize the transition kernel:

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

We now prepare the initial states using `jax.vmap` again, to vectorize the `init` function:

```{code-cell} ipython3
initial_positions = {"loc": np.ones(num_chains), "log_scale": np.ones(num_chains)}
initial_states = jax.vmap(nuts.init, in_axes=(0))(initial_positions)
```

And finally run the sampler

```{code-cell} ipython3
%%time
rng_key, sample_key = jax.random.split(rng_key)
states = inference_loop_multiple_chains(
    sample_key, nuts.step, initial_states, 2_000, num_chains
)
_ = states.position["loc"].block_until_ready()
```

We'll let you judge of the correctness of the samples obtained (see the introduction notebook), but one thing should be obvious to you if you've samples with single chains with Blackjax before: **it is slow!**

Remember when we said SIMD vectorization happens at the instruction level? At each step, the NUTS sampler can perform from 1 to 1024 integration steps, and the CPU (GPU) has to wait for all the chains to complete before moving on to the next chain. As a result, each step is as long as the slowest chain.

```{note}
You may be thinking that instead of applying `jax.vmap` to `one_step` we could apply it to the `inference_loop_multiple_chains`, and the chains will run independently. Unfortunately, this is not how SIMD vectorization work although, granted, JAX's user interface could led you to think otherwise.
```

### Using JAX's sharding API

Now you may be thinking: we are limited by one chain if we synchronize at the step level, but things being random, chains that run truly in parallel should take in total roughly similar numbers of integration steps. So running chains on separate devices should help here. This is true, let's prove it!

#### A note on using the sharding API on CPU

JAX will treat your CPU as a single device by default, regardless of the number of cores available.

Unfortunately, this means that multi-device parallelism is not possible out of the box -- we'll first need
to instruct JAX to split the CPU into multiple devices. See [this issue](https://github.com/google/jax/issues/1408) for more discussion on this topic.

Currently, this can only be done via `XLA_FLAGS` environmental variable.

```{warning}
This variable has to be set before JAX or any library that imports it is imported
```

```python
import os
import multiprocessing

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count={}".format(
    multiprocessing.cpu_count()
)
```

We advise you to confirm that JAX has successfully recognized our CPU as multiple devices with the following command before moving forward:

```{code-cell} ipython3
len(jax.devices())
```

### Back to our example

JAX's sharding API (introduced as the replacement for the deprecated `jax.pmap`) lets us distribute computation across devices using `jax.shard_map`. This runs the function independently on each device, one chain per device.

The recipe is:
1. Create a `Mesh` that names the device axis `'chain'`.
2. Shard the per-chain inputs (RNG keys and initial states) with `jax.device_put`.
3. Use `jax.shard_map` to dispatch one chain per device, then compile with `jax.jit`.

Inside `shard_map`, each device receives a `(1, ...)` slice of the input, so we squeeze the leading axis on the way in and restore it on the way out.

```{code-cell} ipython3
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

mesh = jax.make_mesh((num_chains,), ('chain',))
sharding = NamedSharding(mesh, P('chain'))
```

```{code-cell} ipython3
rng_key, sample_key = jax.random.split(rng_key)
sample_keys = jax.device_put(jax.random.split(sample_key, num_chains), sharding)
initial_states_sharded = jax.device_put(initial_states, sharding)
```

```{note}
You can inspect the sharding with `jax.debug.visualize_array_sharding(initial_states_sharded.position["loc"])` to confirm each chain is on its own device.
```

```{code-cell} ipython3
def run_one_chain(key, state):
    result = inference_loop(key[0], nuts.step, jax.tree.map(lambda x: x[0], state), 2_000)
    return jax.tree.map(lambda x: x[None], result)
```

```{code-cell} ipython3
%%time
sharded_states = jax.jit(jax.shard_map(
    run_one_chain,
    mesh=mesh,
    in_specs=(P('chain'), P('chain')),
    out_specs=P('chain'),
    check_vma=False,
))(sample_keys, initial_states_sharded)
_ = sharded_states.position["loc"].block_until_ready()
```

Wow, this was much faster, our intuition was correct! Note that the sample shape is `(num_chains, num_samples)`, the same convention as the `jax.vmap` example.

```{code-cell} ipython3
states.position["loc"].shape, sharded_states.position["loc"].shape
```

### Conclusions

In this example the difference between `jax.vmap` and the sharding API is dramatic: it takes several minutes with `jax.vmap` and a few seconds with device-parallel execution to sample the same number of chains. This is expected for NUTS, and other adaptive algorithms: each chain runs a different number of internal steps for each sample generated and we need to wait for the slowest chain.

We saw one possible solution for those who just want a few chains to run diagnostics: distribute chains across devices using JAX's sharding API. For the thousands of chains we mentioned earlier you will need something different: either distribute on several machines (expensive), or design new algorithms altogether. HMC, for instance, runs the same number of integration steps on every chain and thus doesn't exhibit the same synchronization problem. That's the idea behind algorithms like ChEEs and MEADS! This is a very active area of research, and now you understand why.
