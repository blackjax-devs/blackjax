---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.0
kernelspec:
  display_name: Python 3.9.7 ('blackjax')
  language: python
  name: python3
---

# Pathfinder

In this notebook we introduce the pathfinder algorithm and we show how to use it as a variational inference method or as an initialization tool for MCMC kernels.

```{code-cell} ipython3
import jax
import jax.numpy as jnp
import jax.random as random
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from sklearn.datasets import make_biclusters

import blackjax
```

```{code-cell} ipython3
:tags: [hide-cell]

%config InlineBackend.figure_format = "retina"
plt.rcParams["axes.spines.right"] = False
plt.rcParams["axes.spines.top"] = False
plt.rcParams["figure.figsize"] = (10, 6)
```

## The Data

We create two clusters of points using [scikit-learn's `make_bicluster` function](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_biclusters.html?highlight=bicluster%20data#sklearn.datasets.make_biclusters).

```{code-cell} ipython3
num_points = 50
X, rows, cols = make_biclusters(
    (num_points, 2), 2, noise=0.6, random_state=314, minval=-3, maxval=3
)
y = rows[0] * 1.0  # y[i] = whether point i belongs to cluster 1
```

```{code-cell} ipython3
:tags: [hide-input]

colors = ["tab:red" if el else "tab:blue" for el in rows[0]]
plt.scatter(*X.T, edgecolors=colors, c="none")
plt.xlabel(r"$X_0$")
plt.ylabel(r"$X_1$")
plt.show()
```

## The Model

We use a simple logistic regression model to infer to which cluster each of the points belongs. We note $ y $ a binary variable that indicates whether a point belongs to the first cluster:

$$
y \sim \operatorname{Bernoulli}(p)
$$

The probability $p$ to belong to the first cluster commes from a logistic regression:

$$
p = \operatorname{logistic}(\Phi\,\boldsymbol{w})
$$

where $w$ is a vector of weights whose priors are a normal prior centered on 0:

$$
\boldsymbol{w} \sim \operatorname{Normal}(0, \sigma)
$$

And $\Phi$ is the matrix that contains the data, so each row $\Phi_{i,:}$ is the vector $\left[X_0^i, X_1^i\right]$

```{code-cell} ipython3
Phi = X
N, M = Phi.shape


def sigmoid(z):
    return jnp.exp(z) / (1 + jnp.exp(z))


def log_sigmoid(z):
    return z - jnp.log(1 + jnp.exp(z))


def logprob_fn(w, alpha=1.0):
    """The log-probability density function of the posterior distribution of the model."""
    log_an = log_sigmoid(Phi @ w)
    an = Phi @ w
    log_likelihood_term = y * log_an + (1 - y) * jnp.log(1 - sigmoid(an))
    prior_term = alpha * w @ w / 2

    return -prior_term + log_likelihood_term.sum()
```

## Pathfinder: Parallel Quasi-Newton Variational Inference

Starting from a random initialization, Pathfinder locates normal approximations to the target
density along a quasi-Newton optimization path, with local covariance estimated using the inverse Hessian
estimates produced by the optimizer. Pathfinder returns draws from the approximation with the lowest
estimated Kullback-Leibler (KL) divergence to the true posterior.
The optimizer is the limited memory BFGS algorithm.

To help understand the approximations that pathfinder evaluates during its run, here we plot for each step of the L-BFGS optimizer the approximation of the posterior distribution of the model derived by pathfinder and its ELBO:

```{code-cell} ipython3
rng_key = random.PRNGKey(314)
w0 = random.multivariate_normal(rng_key, 2.0 + jnp.zeros(M), jnp.eye(M))
path = blackjax.vi.pathfinder.init(rng_key, logprob_fn, w0, return_path=True, ftol=1e-4)
```

```{code-cell} ipython3
:tags: [hide-input]

def ellipse_confidence(mu, cov, ax, c, n_std=2.0):
    import numpy as np

    lambda_, v = np.linalg.eig(cov)
    lambda_ = np.sqrt(lambda_)
    ellipse = Ellipse(
        xy=(*mu,),
        width=lambda_[0] * n_std * 2,
        height=lambda_[1] * n_std * 2,
        angle=np.degrees(np.arctan2(*v[:, 0][::-1])),
        facecolor=c,
        edgecolor="b",
        alpha=0.1,
    )
    return ax.add_artist(ellipse)


step = 0.1
x_, y_ = jnp.mgrid[-1:3:step, -1:3:step]
pos_ = jnp.dstack((x_, y_))
logp_ = jnp.nan_to_num(
    jax.vmap(logprob_fn)(pos_.reshape(-1, M)).reshape(pos_.shape[0], pos_.shape[1]),
    nan=-1e10,
)
levels_ = jnp.percentile(logp_.flatten(), jnp.linspace(60, 100, 10))


steps = (jnp.isfinite(path.elbo)).sum()
rows = int(jnp.ceil(steps / 3))
fig, axs = plt.subplots(rows, 3, figsize=(15, 5 * rows), sharex=True, sharey=True)

for i, ax in zip(range(1, steps + 1), axs.flatten()):

    ax.contour(x_, y_, logp_, levels=levels_)
    state = jax.tree_map(lambda x: x[i], path)
    sample_state, _ = blackjax.vi.pathfinder.sample_from_state(rng_key, state, 10_000)
    position_path = path.position[: i + 1]
    ax.plot(
        position_path[:, 0],
        position_path[:, 1],
        marker="*",
        linestyle="--",
        markersize=10,
    )
    mu_i, cov_i = sample_state.mean(0), jnp.cov(sample_state, rowvar=False)
    ellipse_confidence(mu_i, cov_i, ax, "r")
    ax.set_title(f"Iteration: {i+1}\nEstimated ELBO: {state.elbo:.2f}")
fig.show()
```

## Pathfinder as a Variational Inference Method

Pathfinder can be used as a variational inference method, using its kernel API:

```{code-cell} ipython3
pathfinder = blackjax.kernels.pathfinder(rng_key, logprob_fn, ftol=1e-4)
state = pathfinder.init(w0)
```

Since `blackjax` does not provide an inference loop we need to implement one ourselves:

```{code-cell} ipython3
def inference_loop(rng_key, kernel, initial_state, num_samples):
    @jax.jit
    def one_step(state, rng_key):
        state, info = kernel(rng_key, state)
        return state, (state, info)

    keys = jax.random.split(rng_key, num_samples)
    return jax.lax.scan(one_step, initial_state, keys)
```

We can now run the inference:

```{code-cell} ipython3
_, rng_key = random.split(rng_key)
_, (_, samples) = inference_loop(rng_key, pathfinder.step, state, 5_000)
```

And display the trace:

```{code-cell} ipython3
:tags: [hide-input]

fig, ax = plt.subplots(1, 2, figsize=(8, 2), sharey=True)
for i, axi in enumerate(ax):
    axi.plot(samples[:, i])
    axi.set_title(f"$w_{i}$")
plt.show()
```

Please note that pathfinder is implemented as follows:
* it runs L-BFGS optimization and finds the best approximation in the `init` phase
* `step` phase it's just sampling from a multinormal distribution, whose parameters have been already estimated


hence it makes sense to `jit` the `init` function and then use the `blackjax.vi.pathfinder.sample_from_state` helper function instead of implementing the inference loop:

```{code-cell} ipython3
state = jax.jit(pathfinder.init)(w0)
samples, _ = blackjax.vi.pathfinder.sample_from_state(rng_key, state, 5_000)
```

Quick comparison against `rmh` kernel:

```{code-cell} ipython3
rmh = blackjax.kernels.rmh(logprob_fn, sigma=jnp.ones(M) * 0.7)
state_rmh = rmh.init(w0)
_, (samples_rmh, _) = inference_loop(rng_key, rmh.step, state_rmh, 5_000)
```

```{code-cell} ipython3
:tags: [hide-input]

fig, ax = plt.subplots(2, 2, figsize=(10, 4), sharey=True)
for i in range(2):
    ax[i, 0].plot(samples_rmh.position[:, i])
    ax[i, 0].axvline(x=300, c="tab:red")
    ax[i, 0].set_ylabel(f"$w_{i}$")
    ax[i, 1].plot(samples[:, i])

ax[0, 0].set_title("RMH")
ax[0, 1].set_title("Pathfinder")
fig.show()
```

### Pathfinder as an Initialization Tool for MCMC Kernels

Pathfinder uses internally the inverse hessian estimation of the L-BFGS optimizer to evaluate the approximations to the target distribution along the quasi-Newton optimization path.

We can calculate explicitly this inverse hessian matrix for a step of the optimization path using the `blackjax.vi.pathfinder.lbfgs_inverse_hessian_formula_1` function:

```{code-cell} ipython3
from blackjax.optimizers.lbfgs import lbfgs_inverse_hessian_formula_1

inverse_mass_matrix = lbfgs_inverse_hessian_formula_1(
    state.alpha, state.beta, state.gamma
)
inverse_mass_matrix
```

This estimation of the inverse mass matrix, coupled with Nesterov's dual averaging adaptation for estimating the step size, yields an alternative adaptation scheme for initializing MCMC kernels.

This scheme is implemented in `blackjax.kernel.pathfinder_adaptation` function:

```{code-cell} ipython3
adapt = blackjax.kernels.pathfinder_adaptation(blackjax.nuts, logprob_fn)
state, kernel, info = adapt.run(rng_key, w0)
```

## Some Caveats


* L-BFGS algorithm struggles with float32s and log-likelihood functions; it's suggested to use double precision numbers. In order to do that in `jax` a configuration variable needs to be set up at initialization time (see [here](https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html#double-64bit-precision))

* otherwise you can stick with float32 mode and try to tweak `ftol`, `gtol`, or the initialization point

* It may make sense to start pathfinder with a "bad" initialization point, in order to make the L-BFGS algorithm run longer and have more datapoints to estimate the inverse hessian matrix.
