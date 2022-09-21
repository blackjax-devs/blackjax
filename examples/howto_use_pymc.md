---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.0
kernelspec:
  display_name: blackjax
  language: python
  name: python3
---

# Use BlackJAX with PyMC
Author: Kaustubh Chaudhari

BlackJAX can take any log-probability function as long as it is compatible with JAX's JIT. In this notebook we show how we can use PyMC as a modeling language and BlackJAX as an inference library.

This example relies on PyMC v4, see [installation instructions](https://github.com/pymc-devs/pymc#installation) on the PyMC repository.

```{code-cell} ipython3
import jax
import numpy as np
import pymc as pm
import pymc.sampling_jax

import blackjax

print(f"Running on PyMC v{pm.__version__}")
```

## Data

Please refer to the [original TFP example](https://www.tensorflow.org/probability/examples/Eight_Schools) for a description of the problem and the model that is used.

```{code-cell} ipython3
# Data of the Eight Schools Model
J = 8
y = np.array([28.0, 8.0, -3.0, 7.0, -1.0, 1.0, 18.0, 12.0])
sigma = np.array([15.0, 10.0, 16.0, 11.0, 9.0, 11.0, 10.0, 18.0])
```

## Model

```{code-cell} ipython3
with pm.Model() as model:

    mu = pm.Normal("mu", mu=0.0, sigma=10.0)
    tau = pm.HalfCauchy("tau", 5.0)

    theta = pm.Normal("theta", mu=0, sigma=1, shape=J)
    theta_1 = mu + tau * theta
    obs = pm.Normal("obs", mu=theta, sigma=sigma, shape=J, observed=y)
```

## Sampling using PyMC NUTS Sampler

```{code-cell} ipython3
%%time

with model:
    posterior = pm.sample(50_000, chains=1)
```

## Sampling Using PyMC JAX Numpyro NUTS Sampler

```{code-cell} ipython3
%%time

with model:
    hierarchical_trace_jax = pm.sampling_jax.sample_numpyro_nuts(
        50_000, target_accept=0.9, chains=1, progress_bar=False
    )
```

## Sampling using BlackJax

### Configuring the Model for BlackJax

We first need to transpile the PyMC model into a logprobability density function that is compatible with JAX:

```{code-cell} ipython3
from pymc.sampling_jax import get_jaxified_logp

rvs = [rv.name for rv in model.value_vars]
init_position_dict = model.initial_point()
init_position = [init_position_dict[rv] for rv in rvs]

logprob_fn = get_jaxified_logp(model)
```

### Sampling

```{code-cell} ipython3
%%time

seed = jax.random.PRNGKey(1234)

adapt = blackjax.window_adaptation(blackjax.nuts, logprob_fn, 1000)
last_state, kernel, _ = adapt.run(seed, init_position)


def inference_loop(rng_key, kernel, initial_state, num_samples):
    def one_step(state, rng_key):
        state, info = kernel(rng_key, state)
        return state, (state, info)

    keys = jax.random.split(rng_key, num_samples)
    _, (states, infos) = jax.lax.scan(one_step, initial_state, keys)

    return states, infos

# Sample from the posterior distribution
states, infos = inference_loop(seed, kernel, last_state, 50_000)
```
