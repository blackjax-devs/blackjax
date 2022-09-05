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

# Use BlackJAX with Numpyro

BlackJAX can take any log-probability function as long as it is compatible with JAX's JIT. In this notebook we show how we can use Numpyro as a modeling language and BlackJAX as an inference library.

We reproduce the Eight Schools example from the [Numpyro documentation](https://github.com/pyro-ppl/numpyro) (all credit for the model goes to the Numpyro team). For this notebook to run you will need to install Numpyro:

```bash
pip install numpyro
```

```{code-cell} ipython3
import jax
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.infer.reparam import TransformReparam
from numpyro.infer.util import initialize_model

import blackjax
```

```{code-cell} ipython3
num_warmup = 1000

# We can use this notebook for simple benchmarking by setting
# below to True and run from Terminal.
# $ipython examples/use_with_numpyro.ipynb
RUN_BENCHMARK = False

if RUN_BENCHMARK:
    num_sample = 5_000_000
    print(f"Benchmark with {num_warmup} warmup steps and {num_sample} sampling steps.")
else:
    num_sample = 10_000
```

## Data

```{code-cell} ipython3
# Data of the Eight Schools Model
J = 8
y = np.array([28.0, 8.0, -3.0, 7.0, -1.0, 1.0, 18.0, 12.0])
sigma = np.array([15.0, 10.0, 16.0, 11.0, 9.0, 11.0, 10.0, 18.0])
```

## Model

We use the non-centered version of the model described towards the end of the README on Numpyro's repository:

```{code-cell} ipython3
# Eight Schools example - Non-centered Reparametrization
def eight_schools_noncentered(J, sigma, y=None):
    mu = numpyro.sample("mu", dist.Normal(0, 5))
    tau = numpyro.sample("tau", dist.HalfCauchy(5))
    with numpyro.plate("J", J):
        with numpyro.handlers.reparam(config={"theta": TransformReparam()}):
            theta = numpyro.sample(
                "theta",
                dist.TransformedDistribution(
                    dist.Normal(0.0, 1.0), dist.transforms.AffineTransform(mu, tau)
                ),
            )
        numpyro.sample("obs", dist.Normal(theta, sigma), obs=y)
```

We need to translate the model into a log-probability function that will be used by BlackJAX to perform inference. For that we use the `initialize_model` function in Numpyro's internals. We will also use the initial position it returns:

```{code-cell} ipython3
rng_key = jax.random.PRNGKey(0)

init_params, potential_fn_gen, *_ = initialize_model(
    rng_key,
    eight_schools_noncentered,
    model_args=(J, sigma, y),
    dynamic_args=True,
)
```

Now we create the potential using the `potential_fn_gen` provided by Numpyro and initialize the NUTS state with BlackJAX:

```{code-cell} ipython3
if RUN_BENCHMARK:
    print("\nBlackjax:")
    print("-> Running warmup.")
```

We now run the window adaptation in BlackJAX:

```{code-cell} ipython3
%%time

initial_position = init_params.z
logprob = lambda position: -potential_fn_gen(J, sigma, y)(position)

adapt = blackjax.window_adaptation(
    blackjax.nuts, logprob, num_warmup, target_acceptance_rate=0.8
)
last_state, kernel, _ = adapt.run(rng_key, initial_position)
```

Let us now perform inference using the previously computed step size and inverse mass matrix. We also time the sampling to give you an idea of how fast BlackJAX can be on simple models:

```{code-cell} ipython3
if RUN_BENCHMARK:
    print("-> Running sampling.")
```

```{code-cell} ipython3
%%time

def inference_loop(rng_key, kernel, initial_state, num_samples):
    @jax.jit
    def one_step(state, rng_key):
        state, info = kernel(rng_key, state)
        return state, (state, info)

    keys = jax.random.split(rng_key, num_samples)
    _, (states, infos) = jax.lax.scan(one_step, initial_state, keys)

    return states, (
        infos.acceptance_probability,
        infos.is_divergent,
        infos.num_integration_steps,
    )


# Sample from the posterior distribution
states, infos = inference_loop(rng_key, kernel, last_state, num_sample)
_ = states.position["mu"].block_until_ready()
```

Let us compute the average acceptance probability and check the number of divergences (to make sure that the model sampled correctly, and that the sampling time is not a result of a majority of divergent transitions):

```{code-cell} ipython3
acceptance_rate = np.mean(infos[0])
num_divergent = np.mean(infos[1])

print(f"\nAcceptance rate: {acceptance_rate:.2f}")
print(f"{100*num_divergent:.2f}% divergent transitions")
```

Let us now plot the distribution of the parameters. Note that since we use a transformed variable, Numpyro does not output the school treatment effect directly:

```{code-cell} ipython3
if not RUN_BENCHMARK:
    import seaborn as sns
    from matplotlib import pyplot as plt

    samples = states.position

    fig, axes = plt.subplots(ncols=2)
    fig.set_size_inches(12, 5)
    sns.kdeplot(samples["mu"], ax=axes[0])
    sns.kdeplot(samples["tau"], ax=axes[1])
    axes[0].set_xlabel("mu")
    axes[1].set_xlabel("tau")
    fig.tight_layout()
```

```{code-cell} ipython3
if not RUN_BENCHMARK:
    fig, axes = plt.subplots(8, 2, sharex="col", sharey="col")
    fig.set_size_inches(12, 10)
    for i in range(J):
        axes[i][0].plot(samples["theta_base"][:, i])
        axes[i][0].title.set_text(f"School {i} relative treatment effect chain")
        sns.kdeplot(samples["theta_base"][:, i], ax=axes[i][1], shade=True)
        axes[i][1].title.set_text(f"School {i} relative treatment effect distribution")
    axes[J - 1][0].set_xlabel("Iteration")
    axes[J - 1][1].set_xlabel("School effect")
    fig.tight_layout()
    plt.show()
```

```{code-cell} ipython3
if not RUN_BENCHMARK:
    for i in range(J):
        print(
            f"Relative treatment effect for school {i}: {np.mean(samples['theta_base'][:, i]):.2f}"
        )
```

## Compare Sampling Time with Numpyro

We compare the time it took BlackJAX to do the warmup for 1,000 iterations and then taking 100,000 samples with Numpyro's:

```{code-cell} ipython3
from numpyro.infer import MCMC, NUTS
```

```{code-cell} ipython3
if RUN_BENCHMARK:
    print("\nNumpyro:")
    print("-> Running warmup+sampling.")
```

```{code-cell} ipython3
%%time

nuts_kernel = NUTS(eight_schools_noncentered, target_accept_prob=0.8)
mcmc = MCMC(
    nuts_kernel, num_warmup=num_warmup, num_samples=num_sample, progress_bar=False
)

rng_key = jax.random.PRNGKey(0)
mcmc.run(rng_key, J, sigma, y=y, extra_fields=("num_steps", "accept_prob"))
samples = mcmc.get_samples()
_ = samples["mu"].block_until_ready()
```

```{code-cell} ipython3
print(f"\nAcceptance rate: {mcmc.get_extra_fields()['accept_prob'].mean():.2f}")
print(f"{100*mcmc.get_extra_fields()['diverging'].mean():.2f}% divergent transitions")
```

```{code-cell} ipython3
print(f"\nBlackjax average {infos[2].mean():.2f} leapfrog per iteration.")
print(
    f"Numpyro average {mcmc.get_extra_fields()['num_steps'].mean():.2f} leapfrog per iteration."
)
```
