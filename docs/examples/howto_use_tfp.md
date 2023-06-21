---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.5
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

```{code-cell} python
:tags: [hide-input]

import jax
import blackjax
```

Please refer to the [original TFP example](https://www.tensorflow.org/probability/examples/Eight_Schools) for a description of the problem and the model that is used.

```{code-cell} python
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

We implement the non-centered version of the hierarchical model:

```{code-cell} python
from tensorflow_probability.substrates import jax as tfp
tfd = tfp.distributions
jdc = tfd.JointDistributionCoroutineAutoBatched

import jax.numpy as jnp

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

```{code-cell} python
# Condition on the observed
pinned_model = model.experimental_pin(treatment_effects=treatment_effects)

logdensity_fn = pinned_model.unnormalized_log_prob
```

Let us first run the window adaptation to find a good value for the step size and for the inverse mass matrix. As in the original example we will run the HMC integrator 3 times at each step.

```{code-cell} python
import blackjax
import jax


initial_position = {
    "avg_effect": jnp.zeros([]),
    "avg_stddev": jnp.zeros([]),
    "school_effects_standard": jnp.ones([num_schools]),
}


rng_key = jax.random.PRNGKey(0)
adapt = blackjax.window_adaptation(
    blackjax.hmc, logdensity_fn, num_integration_steps=3
)

(last_state, parameters), _ = adapt.run(rng_key, initial_position, 1000)
kernel = blackjax.hmc(logdensity_fn, **parameters).step
```

We can now perform inference with the tuned kernel:

```{code-cell} python
:tags: [hide-cell]

def inference_loop(rng_key, kernel, initial_state, num_samples):
    def one_step(state, rng_key):
        state, info = kernel(rng_key, state)
        return state, (state, info)

    keys = jax.random.split(rng_key, num_samples)
    _, (states, infos) = jax.lax.scan(one_step, initial_state, keys)

    return states, infos
```

```{code-cell} python
states, infos = inference_loop(rng_key, kernel, last_state, 500_000)
states.position["avg_effect"].block_until_ready()
```

Extra information about the inference is contained in the `infos` namedtuple. Let us compute the average acceptance rate:

```{code-cell} python
:tags: [hide-input]

acceptance_rate = np.mean(infos.acceptance_rate)
print(f"Average acceptance rate: {acceptance_rate:.2f}")
```

The samples are contained as a dictionnary in `states.position`. Let us compute the posterior of the school treatment effect:

```{code-cell} python
samples = states.position
school_effects_samples = (
    samples["avg_effect"][:, np.newaxis]
    + np.exp(samples["avg_stddev"])[:, np.newaxis] * samples["school_effects_standard"]
)
```

And now let us plot the correponding chains and distributions:

```{code-cell} python
:tags: [hide-input, remove-stderr]

import seaborn as sns
from matplotlib import pyplot as plt

fig, axes = plt.subplots(8, 2, sharex="col", sharey="col")
fig.set_size_inches(12, 10)
for i in range(num_schools):
    axes[i][0].plot(school_effects_samples[:, i])
    axes[i][0].title.set_text(f"School {i} treatment effect chain")
    sns.kdeplot(school_effects_samples[:, i], ax=axes[i][1], fill=True)
    axes[i][1].title.set_text(f"School {i} treatment effect distribution")
axes[num_schools - 1][0].set_xlabel("Iteration")
axes[num_schools - 1][1].set_xlabel("School effect")
fig.tight_layout()
plt.show()
```
