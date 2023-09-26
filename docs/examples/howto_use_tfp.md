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

# Use with TFP models

BlackJAX can take any log-probability function as long as it is compatible with JAX's primitives. In this notebook we show how we can use tensorflow-probability as a modeling language and BlackJAX as an inference library.

``` {admonition} Before you start
You will need [tensorflow-probability](https://www.tensorflow.org/probability) to run this example. Please follow the installation instructions on TFP's repository.
```

We reproduce the Eight Schools example from the [TFP documentation](https://www.tensorflow.org/probability/examples/Eight_Schools).

+++

Please refer to the [original TFP example](https://www.tensorflow.org/probability/examples/Eight_Schools) for a description of the problem and the model that is used.

```{code-cell} ipython3
:tags: [hide-cell]

import numpy as np


num_schools = 8  # number of schools
treatment_effects = np.array(
    [28, 8, -3, 7, -1, 1, 18, 12], dtype=np.float32
)  # treatment effects
treatment_stddevs = np.array(
    [15, 10, 16, 11, 9, 11, 10, 18], dtype=np.float32
)  # treatment SE
```

```{code-cell} ipython3
:tags: [remove-output]

import jax
import jax.numpy as jnp

from datetime import date
rng_key = jax.random.key(int(date.today().strftime("%Y%m%d")))
```

We implement the non-centered version of the hierarchical model:

```{code-cell} ipython3
from tensorflow_probability.substrates import jax as tfp
tfd = tfp.distributions
jdc = tfd.JointDistributionCoroutineAutoBatched

@jdc
def model():
    mu = yield tfd.Normal(0.0, 10.0, name="avg_effect")
    log_tau = yield tfd.Normal(5.0, 1.0, name="avg_stddev")
    theta_prime = yield tfd.Sample(tfd.Normal(0, 1),
                                   num_schools,
                                   name="school_effects_standard")
    yhat = mu + jnp.exp(log_tau) * theta_prime
    yield tfd.Normal(yhat, treatment_stddevs, name="treatment_effects")
```

We need to translate the model into a log-probability density function that will be used by Blackjax to perform inference.

```{code-cell} ipython3
# Condition on the observed
pinned_model = model.experimental_pin(treatment_effects=treatment_effects)

logdensity_fn = pinned_model.unnormalized_log_prob
```

Let us first run the window adaptation to find a good value for the step size and for the inverse mass matrix. As in the original example we will run the HMC integrator 3 times at each step.

```{code-cell} ipython3
import blackjax


initial_position = {
    "avg_effect": jnp.zeros([]),
    "avg_stddev": jnp.zeros([]),
    "school_effects_standard": jnp.ones([num_schools]),
}


rng_key, warmup_key = jax.random.split(rng_key)
adapt = blackjax.window_adaptation(
    blackjax.hmc, logdensity_fn, num_integration_steps=3
)

(last_state, parameters), _ = adapt.run(warmup_key, initial_position, 1000)
kernel = blackjax.hmc(logdensity_fn, **parameters).step
```

We can now perform inference with the tuned kernel:

```{code-cell} ipython3
:tags: [hide-cell]

def inference_loop(rng_key, kernel, initial_state, num_samples):
    def one_step(state, rng_key):
        state, info = kernel(rng_key, state)
        return state, (state, info)

    keys = jax.random.split(rng_key, num_samples)
    _, (states, infos) = jax.lax.scan(one_step, initial_state, keys)

    return states, infos
```

```{code-cell} ipython3
rng_key, sample_key = jax.random.split(rng_key)
states, infos = inference_loop(sample_key, kernel, last_state, 50_000)
```

Extra information about the inference is contained in the `infos` namedtuple. Let us compute the average acceptance rate:

```{code-cell} ipython3
:tags: [hide-input]

acceptance_rate = np.mean(infos.acceptance_rate)
print(f"Average acceptance rate: {acceptance_rate:.2f}")
```

The samples are contained as a dictionnary in `states.position`. Let us compute the posterior of the school treatment effect:

```{code-cell} ipython3
samples = states.position
school_effects_samples = (
    samples["avg_effect"][:, np.newaxis]
    + np.exp(samples["avg_stddev"])[:, np.newaxis] * samples["school_effects_standard"]
)
```

And now let us plot the correponding chains and distributions:

```{code-cell} ipython3
:tags: [hide-input, remove-stderr]

import matplotlib.pyplot as plt
import arviz as az

idata = az.from_dict(posterior={k: v[None, ...] for k, v in states.position.items()})
az.plot_trace(idata, var_names=["school_effects_standard"], compact=False)
plt.tight_layout();
```
