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

# How to jump between modes with ensemble slice sampling?

This example implements a direction based on the zeus global move
{cite:p}`karamanis2021ensemble` and mixes it with the differential direction
in BlackJAX's ensemble slice sampler. We compare the two samplers on a
target with two separated modes.

``` {admonition} Before you start
You will need [gmmx](https://github.com/adonath/gmmx) to run this example.
```

```{code-cell} ipython3
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from gmmx import EMFitter, GaussianMixtureModelJax

import blackjax
from blackjax.mcmc.ensemble_slice import differential_direction
```

## The target

The target is a mixture of two Gaussians in 20 dimensions, each with
identity covariance. Their centres are separated by 30 along the first
axis, with weights 0.3 and 0.7.

```{code-cell} ipython3
dim = 20
weights = jnp.array([0.3, 0.7])
centres = jnp.zeros((2, dim)).at[:, 0].set(jnp.array([-15.0, 15.0]))


def logdensity_fn(x):
    return jax.scipy.special.logsumexp(
        jnp.log(weights) - 0.5 * jnp.sum((x - centres) ** 2, axis=-1)
    )
```

## A global direction

A direction may depend only on the complementary walkers and the random
key, never on the walker being moved. This restriction keeps the move
valid. We fit a Gaussian mixture to the complementary walkers using
gmmx's EM fitter, then select two complementary walkers and look up their
component labels. If the labels differ, we draw from the two components
with covariances multiplied by `rescale_cov` and use their difference for
the jump. If the labels agree, we draw a zero-mean Gaussian direction
with that component's covariance. The fit depends only on the
complementary walkers, so under `vmap` it is computed once per half.

The sampler's `width` is the initial bracket width along the direction
and equals $2\mu$ in zeus's notation. The zeus global jump uses
$2(\mathrm{draw}_i - \mathrm{draw}_j)$, independently of $\mu$, so with
the default `width=2` the difference is used unscaled.

Maximum-likelihood EM has no prior to remove redundant components or
prevent a component from collapsing onto a walker. We initialise the
means by farthest-point traversal, which spreads them across separated
groups of walkers, and use `reg_covar` as a covariance floor relative to
the fitting scale. We standardise all coordinates with a single scale,
since scaling each coordinate separately would shrink the axis that
separates the modes relative to the others.

```{code-cell} ipython3
def fit_mixture(positions, n_components, reg_covar):
    location = positions.mean(axis=0)
    scale = jnp.sqrt(positions.var(axis=0).mean())
    x = (positions - location) / scale

    def add_farthest(distance, _):
        index = jnp.argmax(distance)
        distance = jnp.minimum(distance, jnp.sum((x - x[index]) ** 2, axis=-1))
        return distance, x[index]

    _, means = jax.lax.scan(
        add_farthest, jnp.full(x.shape[0], jnp.inf), length=n_components
    )
    covariances = jnp.broadcast_to(jnp.eye(x.shape[1]), (n_components,) + 2 * x.shape[1:])
    gmm = GaussianMixtureModelJax.from_squeezed(
        means, covariances, jnp.full(n_components, 1.0 / n_components)
    )
    gmm = EMFitter(reg_covar=reg_covar).fit(x, gmm).gmm
    labels = gmm.predict(x)[:, 0]
    means = gmm.means[0, :, :, 0] * scale + location
    covariances = gmm.covariances.values[0] * scale**2
    return means, covariances, labels


def global_direction(n_components=5, rescale_cov=1e-3, reg_covar=1e-3):
    def direction(rng_key, complementary_positions):
        means, covariances, labels = fit_mixture(
            complementary_positions, n_components, reg_covar
        )
        factors = jnp.linalg.cholesky(covariances)
        key_pair, key_i, key_j = jax.random.split(rng_key, 3)
        i, j = jax.random.choice(key_pair, labels, (2,), replace=False)
        z_i = jax.random.normal(key_i, means.shape[1:])
        z_j = jax.random.normal(key_j, means.shape[1:])
        jump = means[i] - means[j]
        jump += jnp.sqrt(rescale_cov) * (factors[i] @ z_i - factors[j] @ z_j)
        return jnp.where(i != j, jump, factors[i] @ z_i)

    return direction
```

## Mixing with the differential direction

As for mixing moves in `blackjax.ensemble`, we choose one direction
function for the whole ensemble at each iteration. Here the differential
direction is selected with probability 0.8 and the global direction with
probability 0.2.

```{code-cell} ipython3
def mixture_step(directions, probabilities):
    steps = [
        blackjax.ensemble_slice(logdensity_fn, direction=d).step for d in directions
    ]

    def step(rng_key, state):
        key_choice, key_step = jax.random.split(rng_key)
        index = jax.random.choice(key_choice, len(steps), p=probabilities)
        return jax.lax.switch(index, steps, key_step, state)

    return step
```

## Sampling

We initialise 64 walkers, half in each mode. The light mode therefore
starts with a fraction of 0.5 instead of its target weight of 0.3. For
each sampler, we run 16 independent ensembles for 4000 steps, tracking
the fraction of walkers in the light mode and counting walkers that
change mode at each step.

```{code-cell} ipython3
num_walkers, num_steps, num_ensembles = 64, 4000, 16
init = blackjax.ensemble_slice(logdensity_fn).init


def light_fraction(step, rng_key):
    key_start, key_run = jax.random.split(rng_key)
    start = jax.random.normal(key_start, (num_walkers, dim))
    start += centres[jnp.arange(num_walkers) % 2]

    def one_step(state, rng_key):
        new_state, _ = step(rng_key, state)
        in_light = new_state.position[:, 0] < 0
        crossings = jnp.sum(in_light != (state.position[:, 0] < 0))
        return new_state, (in_light.mean(), crossings)

    keys = jax.random.split(key_run, num_steps)
    _, (fraction, crossings) = jax.lax.scan(one_step, init(start), keys)
    return fraction, crossings.sum()


samplers = {
    "differential": blackjax.ensemble_slice(logdensity_fn).step,
    "differential + global": mixture_step(
        [differential_direction, global_direction()], jnp.array([0.8, 0.2])
    ),
}


def run_ensembles(step, keys):
    return jax.lax.map(lambda k: light_fraction(step, k), keys)


keys = jax.random.split(jax.random.key(0), num_ensembles)
results = {
    name: jax.jit(run_ensembles, static_argnums=0)(step, keys)
    for name, step in samplers.items()
}
```

```{code-cell} ipython3
for name, (fraction, crossings) in results.items():
    tail = fraction[:, num_steps // 2 :].mean(axis=1)
    print(
        f"{name}: light-mode fraction over the second half "
        f"{tail.mean():.3f} ± {tail.std() / num_ensembles**0.5:.3f}, "
        f"{crossings.mean():.0f} mode changes per ensemble"
    )
```

```{code-cell} ipython3
:tags: [hide-input]

fig, ax = plt.subplots(figsize=(8, 4))
for name, (fraction, _) in results.items():
    ax.plot(fraction.mean(axis=0), label=name)
ax.axhline(0.3, color="black", linestyle="--", label="target weight")
ax.set_xlabel("step")
ax.set_ylabel("fraction of walkers in the light mode")
ax.legend();
```

With the differential direction alone, the light-mode fraction drifts
down slowly from 0.5 and remains above 0.3 at the end. With the mixture,
it reaches about 0.3 within roughly the first third of the run and then
fluctuates around it. The mixture also produces more mode changes.

The mixture fit runs at every global step, once per half of the ensemble.
The choice of `n_components` also affects the directions. Zeus uses
scikit-learn's `BayesianGaussianMixture`, whose Dirichlet prior suppresses
unneeded components; gmmx uses maximum-likelihood EM without this prior.

Zeus's documentation recommends using the global move after burn-in and
mixing it with other moves. There is also a difference in how directions
are selected: zeus draws one pair of component labels per half and
applies the same kind of direction to every walker in that half. Here,
each walker draws its own pair.
