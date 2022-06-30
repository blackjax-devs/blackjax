---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.13.8
  kernelspec:
    display_name: blackjax
    language: python
    name: blackjax
---

<!-- #region id="397995ab" -->
# Use BlackJAX with PyMC
Author: Kaustubh Chaudhari
<!-- #endregion -->

<!-- #region id="bb51846d" -->
BlackJAX can take any log-probability function as long as it is compatible with JAX's JIT. In this notebook we show how we can use PyMC as a modeling language and BlackJAX as an inference library.

This example relies on PyMC v4, see [installation instructions](https://github.com/pymc-devs/pymc#installation) on the PyMC repository.
<!-- #endregion -->

```python id="3a905211"
import jax
import numpy as np
import pymc as pm
import pymc.sampling_jax

import blackjax

print(f"Running on PyMC v{pm.__version__}")
```

<!-- #region id="8VrYFaoIX--y" -->
## Data

Please refer to the [original TFP example](https://www.tensorflow.org/probability/examples/Eight_Schools) for a description of the problem and the model that is used.
<!-- #endregion -->

```python id="imotOe9sUNYF"
# Data of the Eight Schools Model
J = 8
y = np.array([28.0, 8.0, -3.0, 7.0, -1.0, 1.0, 18.0, 12.0])
sigma = np.array([15.0, 10.0, 16.0, 11.0, 9.0, 11.0, 10.0, 18.0])
```

<!-- #region id="aabSQ11iYGZw" -->
## Model

<!-- #endregion -->

```python id="PiBv9iOvRK0f"
with pm.Model() as model:

    mu = pm.Normal("mu", mu=0.0, sigma=10.0)
    tau = pm.HalfCauchy("tau", 5.0)

    theta = pm.Normal("theta", mu=0, sigma=1, shape=J)
    theta_1 = mu + tau * theta
    obs = pm.Normal("obs", mu=theta, sigma=sigma, shape=J, observed=y)
```

<!-- #region id="VKMdLEu1Y5jb" -->
## Sampling using PyMC NUTS Sampler
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 244} id="0ZyMxwLFY_ZI" outputId="793af037-31e4-4e55-9c76-231c9d78532d"
%%time

with model:
    posterior = pm.sample(50_000, chains=1)
```

<!-- #region id="3I6zXC-JZCfs" -->
## Sampling using PyMC JAX Numpyro NUTS sampler
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="daQ5OO6aZS9t" outputId="d865c9dc-45ae-4baa-c643-f145492ea4ab"
%%time

with model:
    hierarchical_trace_jax = pm.sampling_jax.sample_numpyro_nuts(
        50_000, target_accept=0.9, chains=1, progress_bar=False
    )
```

<!-- #region id="h8cMqFwiZjxS" -->
## Sampling using BlackJax

### Configuring the model for BlackJax

We first need to transpile the PyMC model into a logprobability density function that is compatible with JAX:
<!-- #endregion -->

```python
from pymc.sampling_jax import get_jaxified_logp

rvs = [rv.name for rv in model.value_vars]
init_position_dict = model.compute_initial_point()
init_position = [init_position_dict[rv] for rv in rvs]

logprob_fn = get_jaxified_logp(model)
```

### Sampling

```python id="cTlcZCYmidZ6"
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

```python

```
