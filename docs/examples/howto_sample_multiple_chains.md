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
- [jax.pmap](https://jax.readthedocs.io/en/latest/_autosummary/jax.pmap.html#jax.pmap) is a higher level abstraction, where processes are split across multiple devices: GPUs, TPUs, or CPU cores.

For detailed walkthrough both primitives we invite your to read JAX's tutorials on [Automatic Vectorization](https://jax.readthedocs.io/en/latest/jax-101/03-vectorization.html)
and [Parallel Evaluation](https://jax.readthedocs.io/en/latest/jax-101/06-parallelism.html).

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

And finally, to put `jax.vmap` and `jax.pmap` on an equal foot we sample as many chains as the machine has CPU cores:

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

### Using `jax.pmap`

Now you may be thinking: we are limited by one chain if we synchronize at the step level, but things being random, chains that run truly in parallel should take in total roughly similar numbers of integration steps. So `jax.pmap` should help here. This is true, let's prove it!

#### A note on using `jax.pmap` on CPU

JAX will treat your CPU as a single device by default, regardless of the number of cores available.

Unfortunately, this means that using `pmap` is not possible out of the box -- we'll first need
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

We advise you to confirm that JAX has successfuly recognized our CPU as multiple devices with the following command before moving forward:

```{code-cell} ipython3
len(jax.devices())
```

##### Choosing the number of devices

`jax.pmap` has one more limitation: it is not able to parallelize the execution when you ask it to perform more computations than there are available deviced. The following code snippet asks `jax.pmap` perform 1024 operations in parallel:

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

This means that you will only be able to run as many MCMC chains as you have CPU cores. See this [question](https://github.com/google/jax/discussions/4198) for a more detailed discussion,
and a workaround involving nesting `jax.pmap` and `jax.vmap` calls.

Another option (we advise against) is to set the device count to a number larger than the core count, e.g. `200`, but it's [unclear what side effects it might have](https://github.com/google/jax/issues/1408#issuecomment-536158048).

### Back to our example

In case of `jax.pmap`, we apply the transformation directly to the original `inference_loop` function.

```{code-cell} ipython3
inference_loop_multiple_chains = jax.pmap(inference_loop, in_axes=(0, None, 0, None), static_broadcasted_argnums=(1, 3))
```

```{note}
We could have done that in the `jax.vmap` example (and it wouldn't have helped), but we prefered to highlight in the code the fact that vectorization happens at the instruction level.
```

We are now ready to sample:

```{code-cell} ipython3
%%time
rng_key, sample_key = jax.random.split(rng_key)
sample_keys = jax.random.split(sample_key, num_chains)

pmap_states = inference_loop_multiple_chains(
    sample_keys, nuts.step, initial_states, 2_000
)
_ = pmap_states.position["loc"].block_until_ready()
```

Wow, this was much faster, our intuition was correct! Note that the samples are transposed compared to the ones obtained with `jax.vmap`.

Also, note how the shape of the posterior samples are different:

```{code-cell} ipython3
states.position["loc"].shape, pmap_states.position["loc"].shape
```

### Conclusions

In this example the different between `jax.vmap` and `jax.pmap` is dramatic: it takes several minutes to `jax.vmap` and a few seconds for `jax.pmap` to sample the same number of chains. This is expected for NUTS, and other adaptive algorithms: each chain runs a different number of internal steps for each sample generated and we need to wait for the slowest chain.

We saw one possible solutions for those who just want a few chains to run diagnostics: parallelize using `jax.pmap`. For the thousands of chains we mentionned earlier you will need something different: either distribute on several machine (expensive), or design new algorithms altogether. HMC, for instance, runs the same number of integration steps on every chain and thus doesn't exhibit the same synchronization problem. That's the idea behind algorithms like ChEEs and MEADS! This is a very active area of research, and now you understand why.
