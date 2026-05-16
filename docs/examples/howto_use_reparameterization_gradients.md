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

# Use reparameterization gradients through samplers

Some BlackJAX samplers expose *pathwise* or *reparameterization* gradients: if you write the target log-density as

```python
logdensity_fn(position, *logdensity_args)
```

and pass those runtime log-density inputs explicitly to `init` and `step`, then you can differentiate sampled positions with respect to those positional inputs with ordinary JAX transforms.

At a high level the workflow is:

1. Instantiate a sampler with a fixed `logdensity_fn` taking arguments in addition to the usual position argument.
2. Initialize the sampler state with both the position and the explicit log-density inputs.
3. Compose `step` with `jax.lax.scan` for many steps, and optionally `jax.vmap` when you want many chains.
4. Differentiate the resulting sampled positions or any scalar objective built from them with `jax.grad`, `jax.jacrev`, or higher-order compositions.

## Example: a mixture of two 2D Gaussians

In this example we sample from a mixture of two correlated 2D Gaussians with fixed covariances, and compute directional derivatives of the samples w.r.t. a variation
of the centers of the mixture components.

```{code-cell} ipython3
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np

import jax
import jax.numpy as jnp
import jax.scipy as jsp

import blackjax
```

```{code-cell} ipython3
:tags: [remove-output]

rng_key = jax.random.key(20240329)
```

We pick two moderately overlapping mixture components with different correlation structures:

```{code-cell} ipython3
weights = jnp.array([0.5, 0.5])

cov_left = jnp.array([[0.45, 0.18], [0.18, 0.35]])
cov_right = jnp.array([[0.40, -0.10], [-0.10, 0.30]])

initial_centers = jnp.array([-1.0, 0.0, 1.0, 0.0])
center_direction = jnp.array([0.0, -0.25, -0.25, 0.0])
```

```{code-cell} ipython3
inv_cov_left = jnp.linalg.inv(cov_left)
inv_cov_right = jnp.linalg.inv(cov_right)
logdet_left = jnp.linalg.slogdet(cov_left)[1]
logdet_right = jnp.linalg.slogdet(cov_right)[1]

def gaussian_logdensity(x, mean, inv_cov, logdet):
    delta = x - mean
    dim = x.shape[0]
    quadratic = delta @ inv_cov @ delta
    return -0.5 * (quadratic + logdet + dim * jnp.log(2.0 * jnp.pi))

def logdensity_fn(x, centers):
    left_center = centers[:2]
    right_center = centers[2:]

    left = jnp.log(weights[0]) + gaussian_logdensity(
        x, left_center, inv_cov_left, logdet_left
    )
    right = jnp.log(weights[1]) + gaussian_logdensity(
        x, right_center, inv_cov_right, logdet_right
    )
    return jsp.special.logsumexp(jnp.array([left, right]))
```

The log-density inputs are passed explicitly to both `init` and `step`.

```{code-cell} ipython3
initial_position = jnp.array([0.0, 0.8])
num_steps = 220

sampler = blackjax.reparameterized_slice(logdensity_fn)
```

```{code-cell} ipython3
rng_key, key_steps = jax.random.split(rng_key)
step_keys = jax.random.split(key_steps, num_steps)
```

```{code-cell} ipython3
def rollout_positions(centers):
    state = sampler.init(initial_position, centers)

    def one_step(state, key):
        state, _ = sampler.step(key, state, centers)
        return state, state.position

    _, step_positions = jax.lax.scan(one_step, state, step_keys)
    return step_positions


rollout_positions = jax.jit(rollout_positions)
rollout = rollout_positions(initial_centers)
rollout.block_until_ready()
```

We run a single chain and keep the entire rollout. To get a directional derivative for each point in that rollout, we differentiate the whole sequence of positions with respect to the center parameters and then project that Jacobian onto the chosen direction.

```{code-cell} ipython3
rollout_jacobian = jax.jacobian(rollout_positions)(initial_centers)
rollout_tangents = jnp.einsum("tij,j->ti", rollout_jacobian, center_direction)
```

```{code-cell} ipython3
def covariance_ellipse(covariance, center):
    eigenvalues, eigenvectors = np.linalg.eigh(np.asarray(covariance))
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
    width, height = 2.0 * np.sqrt(eigenvalues)
    return Ellipse(
        xy=np.asarray(center),
        width=width,
        height=height,
        angle=angle,
        facecolor="none",
        edgecolor="black",
        lw=2.0,
        alpha=0.9,
    )

rollout_points = np.asarray(rollout)
rollout_arrows = np.asarray(rollout_tangents)
centers = np.asarray(initial_centers).reshape(2, 2)
center_arrows = np.asarray(center_direction).reshape(2, 2)

fig, ax = plt.subplots(figsize=(7, 6))
ax.scatter(
    rollout_points[:, 0],
    rollout_points[:, 1],
    s=9,
    color="black",
    alpha=0.75,
    label="Rollout positions",
)

ax.quiver(
    rollout_points[:, 0],
    rollout_points[:, 1],
    rollout_arrows[:, 0],
    rollout_arrows[:, 1],
    angles="xy",
    scale_units="xy",
    scale=1.0,
    width=0.002,
    color="#dc2626",
    alpha=0.8,
    label="Directional derivative",
)
ax.scatter(centers[:, 0], centers[:, 1], s=90, color="black", marker="o", label="Centers")
ax.add_patch(covariance_ellipse(cov_left, centers[0]))
ax.add_patch(covariance_ellipse(cov_right, centers[1]))

ax.quiver(
    centers[:, 0],
    centers[:, 1],
    center_arrows[:, 0],
    center_arrows[:, 1],
    angles="xy",
    scale_units="xy",
    scale=1.0,
    width=0.004,
    color="#dc2626",
    alpha=0.95,
)

ax.set_aspect("equal")
ax.set_xlim(-3.4, 2.8)
ax.set_ylim(-2.6, 2.6)
ax.set_xlabel("$x_1$")
ax.set_ylabel("$x_2$")
ax.set_title("Samples and pathwise directional derivatives")
ax.legend(loc="upper right")
plt.show()
```

```{figure} ../_static/reparameterization_gradients_flow_field.png
:alt: Samples from a two-component Gaussian mixture with directional-derivative arrows and center-motion arrows
:align: center

Positions from a single-chain rollout, with unscaled directional-derivative arrows at each position and unscaled center-motion arrows for the component means.
```
