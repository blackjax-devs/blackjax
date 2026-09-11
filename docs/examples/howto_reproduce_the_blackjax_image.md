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

# Reproducing the front page image of the repository

The front page image of the repository is a sampled version of the following image:

![front_page_image](./data/blackjax.png)

Here we show how we can sample from the uniform distribution corresponding to black pixels in the image.


## Make the image into a numpy array

```{code-cell} ipython3
import matplotlib
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["axes.spines.right"] = False
plt.rcParams["axes.spines.top"] = False
matplotlib.rcParams['animation.embed_limit'] = 25
```

```{code-cell} ipython3
# Load the image
im = mpimg.imread('./data/blackjax.png')

# Convert to mask
im = np.amax(im[:, :, :2], 2) < 0.5

# Convert back to float
im = im.astype(float)
```

## The sampling procedure

To sample from **BlackJAX**, we form a bridge between the uniform distribution over the full image, corresponding to a 2D domain of size (80, 250) and the uniform distribution over the black pixels.

Formally, this corresponds to a prior distribution $p_0 \sim U([[0, 79]] \times [[0, 249]]$ and a target distribution $p_1(x) \propto \mathbb{1}_{x \in \text{image}}$.

```{code-cell} ipython3
import jax
import jax.numpy as jnp
from datetime import date

rng_key = jax.random.key(int(date.today().strftime("%Y%m%d")))
```

```{code-cell} ipython3
# Sample from the uniform distribution over the domain
key_init, rng_key = jax.random.split(rng_key)
n_samples = 5_000
INF = 1e2  # A large number
jax_im = jnp.array(im.astype(float))

key_init_xs, key_init_ys = jax.random.split(key_init)
xs_init = jax.random.uniform(key_init_xs, (n_samples,), minval=0, maxval=80)
ys_init = jax.random.randint(key_init_ys, (n_samples,), minval=0, maxval=250)

zs_init = np.stack([xs_init, ys_init], axis=1)


# Set the prior and likelihood
def prior_logpdf(z): return 0.0


# The pdf is uniform, so the logpdf is constant on the domain and negative infinite outside
def log_likelihood(z):
    x, y = z
    # The pixel is black if x, y falls within the image
    floor_x = jnp.floor(x).astype(jnp.int32)
    floor_y = jnp.floor(y).astype(jnp.int32)
    out_of_bounds = (floor_x < 0) | (floor_x >= 80) | (floor_y < 0) | (floor_y >= 250)
    value = jax.lax.cond(
        out_of_bounds,
        lambda: -INF,
        lambda: -INF * (jax_im[floor_x, floor_y] == 0),
    )
    return value
```

## Run the SMC sampler

We use a RWMH sampler within an SMC routine to sample from the target distribution.
For more information see the [documentation](https://blackjax-devs.github.io/sampling-book/algorithms/TemperedSMC.html) specific to SMC.

```{code-cell} ipython3
import functools
import blackjax
import blackjax.smc.resampling as resampling

# Temperature schedule
n_temperatures = 150
lambda_schedule = np.logspace(-3, 0, n_temperatures)

# The proposal distribution is a random walk with a fixed scale
scale = 0.5
normal = blackjax.mcmc.random_walk.normal(scale * jnp.ones((2,)))

# Bake the proposal into the kernel so no function leaks into mcmc_parameters
rw_kernel = functools.partial(
    blackjax.additive_step_random_walk.build_kernel(),
    random_step=normal,
)
rw_init = blackjax.additive_step_random_walk.init

tempered = blackjax.tempered_smc(
    prior_logpdf,
    log_likelihood,
    rw_kernel,
    rw_init,
    {},   # mcmc_parameters must be JAX-traceable; proposal is baked into rw_kernel
    resampling.systematic,
    num_mcmc_steps=5,
)

initial_smc_state = tempered.init(zs_init)
```

```{code-cell} ipython3
def smc_inference_loop(loop_key, smc_kernel, init_state, schedule):
    """Run the tempered SMC algorithm."""

    def body_fn(carry, tempering_param):
        i, state = carry
        subkey = jax.random.fold_in(loop_key, i)
        new_state, info = smc_kernel(subkey, state, tempering_param)
        return (i + 1, new_state), (new_state, info)

    _, (all_samples, _) = jax.lax.scan(body_fn, (0, init_state), schedule)
    return all_samples


rng_key, sample_key = jax.random.split(rng_key)
blackjax_samples = smc_inference_loop(sample_key, tempered.step, initial_smc_state, lambda_schedule)
```

## Plot the samples

```{code-cell} ipython3
weights = np.array(blackjax_samples.weights)
samples = np.array(blackjax_samples.particles)
```

```{code-cell} ipython3
import matplotlib.animation as animation
from IPython.display import HTML

plt.style.use('dark_background')
animation.embed_limit = 25
fig, ax = plt.subplots()
fig.tight_layout()

ax.set_axis_off()
ax.set_xlim(0, 250)
ax.set_ylim(0, 80)

scat = ax.scatter(ys_init, 80 - xs_init, s=1000 * 1 / n_samples)


def animate(i):
    scat.set_offsets(np.c_[samples[i, :, 1], 80 - samples[i, :, 0]])
    scat.set_sizes(1000 * weights[i])
    return scat,


ani = animation.FuncAnimation(fig, animate, repeat=True,
                              frames=n_temperatures, blit=True, interval=100)
HTML(ani.to_jshtml())
```
