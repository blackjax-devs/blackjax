# Welcome to Blackjax!

```{warning}
The documentation corresponds to the current state of the `main` branch. There may be differences with the latest released version.
```

Blackjax is a library of samplers for [JAX](https://github.com/google/jax) that works on CPU as well as GPU. It is designed with two categories of users in mind:

- People who just need state-of-the-art samplers that are fast, robust and well tested;
- Researchers who can use the library's building blocks to design new algorithms.

It integrates really well with PPLs as long as they can provide a (potentially unnormalized) log-probability density function compatible with JAX.


# Hello World

```{code-block} Python
import jax
import jax.numpy as jnp
import jax.scipy.stats as stats
import numpy as np

import blackjax

observed = np.random.normal(10, 20, size=1_000)
def logdensity_fn(x):
    logpdf = stats.norm.logpdf(observed, x["loc"], x["scale"])
    return jnp.sum(logpdf)

# Build the kernel
step_size = 1e-3
inverse_mass_matrix = jnp.array([1., 1.])
nuts = blackjax.nuts(logdensity_fn, step_size, inverse_mass_matrix)

# Initialize the state
initial_position = {"loc": 1., "scale": 2.}
state = nuts.init(initial_position)

# Iterate
rng_key = jax.random.key(0)
step = jax.jit(nuts.step)
for i in range(1_000):
    nuts_key = jax.random.fold_in(rng_key, i)
    state, _ = step(nuts_key, state)
```

:::{note}
If you want to use Blackjax with a model implemented with a PPL, go to the related tutorials in the left menu.
:::


# Installation

::::{tab-set}

:::{tab-item} Latest
```{code-block} bash
pip install blackjax
```


:::{tab-item} Conda
```{code-block} bash
conda install blackjax -c conda-forge
```
:::

::::

:::{admonition} GPU instructions
:class: tip

BlackJAX is written in pure Python but depends on XLA via JAX. By default, the
version of JAX that will be installed along with BlackJAX will make your code
run on CPU only. **If you want to use BlackJAX on GPU/TPU** we recommend you follow
[these instructions](https://github.com/google/jax#installation) to install JAX
with the relevant hardware acceleration support.
:::

```{toctree}
---
maxdepth: 1
hidden:
---
Quickstart <examples/quickstart.md>
```

```{toctree}
---
maxdepth: 1
caption: PPL INTEGRATION
hidden:
---
Aesara<examples/howto_use_aesara.md>
Numpyro<examples/howto_use_numpyro.md>
Oryx<examples/howto_use_oryx.md>
PyMC<examples/howto_use_pymc.md>
Tensorflow-Probability<examples/howto_use_tfp.md>
```

```{toctree}
---
maxdepth: 2
caption: HOW TO
hidden:
---
Sample with multiple chains?<examples/howto_sample_multiple_chains.md>
Use custom gradients?<examples/howto_custom_gradients.md>
Use non-JAX log-prob functions?<examples/howto_other_frameworks.md>
Build a Metropolis-Within-Gibbs sampler?<examples/howto_metropolis_within_gibbs.md>
Sample from the word BlackJAX using BlackJAX?<examples/howto_reproduce_the_blackjax_image.md>
```

```{toctree}
---
maxdepth: 1
caption: LEARN BY EXAMPLE
hidden:
---
The Sampling Book <https://blackjax-devs.github.io/sampling-book>
```

```{toctree}
---
hidden:
caption: API
maxdepth: 2
---
API Reference<autoapi/blackjax/index>
Bibliography<bib.rst>
```
