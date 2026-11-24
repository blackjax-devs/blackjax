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

# How to sample with the Importance Markov Chain?

The Importance Markov Chain (IMC) {cite:p}`andral2024importance,bertholom2026theoretical` runs a sampler on an instrumental distribution $\tilde\pi$ and turns its output into a chain that targets $\pi$. Each instrumental draw $\tilde x$ is repeated a random number of times $N$ with

$$
\mathbb{E}[N \mid \tilde x] = \kappa \frac{\pi(\tilde x)}{\tilde\pi(\tilde x)}.
$$

Draws where the target has more mass than the instrumental distribution are repeated more often, and draws where it has less are often deleted. The constant $\kappa$ sets the expected number of output samples per instrumental draw.

This notebook shows the two ways to use IMC in BlackJAX: on top of an MCMC sampler, and with independent proposals.

```{code-cell} ipython3
:tags: [remove-output]

import jax
import jax.numpy as jnp
import jax.scipy.stats as stats
import matplotlib.pyplot as plt

import blackjax
from blackjax.mcmc.importance_markov_chain import log_kappa_for_num_samples
from blackjax.util import run_inference_algorithm

from datetime import date
rng_key = jax.random.key(int(date.today().strftime("%Y%m%d")))
```

## A target with two separated modes

The target is a mixture of two narrow Gaussians, with 30% of the mass on the left mode and 70% on the right one.

```{code-cell} ipython3
weights = jnp.array([0.3, 0.7])
means = jnp.array([-4.0, 4.0])


def logdensity_fn(x):
    return jax.scipy.special.logsumexp(
        jnp.log(weights) + stats.norm.logpdf(x, means, 0.5)
    )
```

A random walk on this target rarely crosses from one mode to the other, so its estimate of the mode weights depends on where it starts.

```{code-cell} ipython3
random_walk = blackjax.additive_step_random_walk.normal_random_walk(
    logdensity_fn, sigma=jnp.array([0.5])
)
rng_key, sample_key = jax.random.split(rng_key)
_, (states, _) = run_inference_algorithm(
    sample_key, random_walk, 20_000, initial_position=jnp.array(-4.0)
)
print(f"Random walk, mass on the right mode: {jnp.mean(states.position > 0):.3f}")
```

## IMC on top of a tempered random walk

We use the tempered distribution $\tilde\pi \propto \pi^{\beta}$ with $\beta = 0.1$ as the instrumental distribution. It is much flatter, so a random walk moves between the two modes easily.

```{code-cell} ipython3
beta = 0.1


def instrumental_logdensity_fn(x):
    return beta * logdensity_fn(x)


instrumental_algorithm = blackjax.additive_step_random_walk.normal_random_walk(
    instrumental_logdensity_fn, sigma=jnp.array([3.0])
)
```

We choose $\kappa$ from a short pilot run so that each instrumental step produces one output sample on average. `log_kappa_for_num_samples` does this from the log-weights $\log\pi - \log\tilde\pi$ of the pilot draws.

```{code-cell} ipython3
num_pilot_steps = 5_000
rng_key, pilot_key = jax.random.split(rng_key)
_, (pilot_states, _) = run_inference_algorithm(
    pilot_key,
    instrumental_algorithm,
    num_pilot_steps,
    initial_position=jnp.array(-4.0),
)
pilot_log_weights = jax.vmap(
    lambda x: logdensity_fn(x) - instrumental_logdensity_fn(x)
)(pilot_states.position)
log_kappa = log_kappa_for_num_samples(pilot_log_weights, num_pilot_steps)
```

`blackjax.imc` combines the target, the instrumental distribution and the instrumental algorithm. Each step returns one sample of the IMC chain: either the next copy of the current sample, or a new instrumental draw.

```{code-cell} ipython3
imc = blackjax.imc(
    logdensity_fn, instrumental_logdensity_fn, instrumental_algorithm, log_kappa
)
rng_key, sample_key = jax.random.split(rng_key)
_, (states, infos) = run_inference_algorithm(
    sample_key, imc, 50_000, initial_position=jnp.array(-4.0)
)
samples = states.position

print(f"IMC, mass on the right mode: {jnp.mean(samples > 0):.3f}")
print(f"Instrumental steps per sample: {jnp.mean(infos.num_instrumental_steps):.2f}")
```

```{code-cell} ipython3
grid = jnp.linspace(-6, 6, 500)
plt.hist(samples, bins=100, density=True, alpha=0.5, label="IMC samples")
plt.plot(grid, jnp.exp(jax.vmap(logdensity_fn)(grid)), label="target")
plt.legend();
```

`infos.is_exhausted` reports transitions where `max_num_steps` instrumental steps (1,000 by default) passed without any draw being kept. Each one costs `max_num_steps` instrumental steps, so if they are frequent, increase $\kappa$.

```{code-cell} ipython3
print(f"Exhausted transitions: {jnp.sum(infos.is_exhausted)}")
```

## IMC with independent proposals

When the instrumental draws are independent, for example from a fitted approximation of the target, there is no chain to run: all proposals can be drawn, weighed and replicated at once (see Section 4.3.1 in {cite:p}`bertholom2026theoretical`). `blackjax.imc.independent_sampling` returns the proposals and their numbers of replicas.

As before, we choose $\kappa$ from a pilot batch of proposals so that each proposal gives one sample on average.

```{code-cell} ipython3
def proposal_distribution(rng_key):
    return 4.0 * jax.random.normal(rng_key)


def proposal_logdensity_fn(x):
    return stats.norm.logpdf(x, 0.0, 4.0)


num_pilot_proposals = 5_000
rng_key, pilot_key = jax.random.split(rng_key)
pilot_positions = jax.vmap(proposal_distribution)(
    jax.random.split(pilot_key, num_pilot_proposals)
)
pilot_log_weights = jax.vmap(logdensity_fn)(pilot_positions) - jax.vmap(
    proposal_logdensity_fn
)(pilot_positions)
log_kappa = log_kappa_for_num_samples(pilot_log_weights, num_pilot_proposals)

rng_key, sample_key = jax.random.split(rng_key)
positions, num_replicas = blackjax.imc.independent_sampling(
    sample_key,
    logdensity_fn,
    proposal_distribution,
    proposal_logdensity_fn,
    num_proposals=50_000,
    log_kappa=log_kappa,
)
```

Averages use the numbers of replicas as weights. To get the chain itself, repeat each proposal.

```{code-cell} ipython3
right_mode_mass = jnp.sum(num_replicas * (positions > 0)) / jnp.sum(num_replicas)
chain = jnp.repeat(positions, num_replicas)

print(f"Number of samples: {chain.shape[0]}")
print(f"Mass on the right mode: {right_mode_mass:.3f}")
```

When the ratio $\pi/\tilde\pi$ is bounded and $\kappa<1/\sup(\pi/\tilde\pi)$, IMC is algorithmically equivalent to rejection sampling. As $\kappa$ grows, averages over the chain approach self-normalised importance sampling.
