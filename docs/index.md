# Welcome to Blackjax!

```{warning}
The documentation corresponds to the current state of the `main` branch. There may be differences with the latest released version.
```

Blackjax is a library of samplers for [JAX](https://github.com/google/jax) that works on CPU as well as GPU. It is designed with two categories of users in mind:

- People who just need state-of-the-art samplers that are fast, robust and well tested;
- Researchers who can use the library's building blocks to design new algorithms.

It integrates really well with PPLs as long as they can provide a (potentially unnormalized) log-probability density function compatible with JAX.


# Hello World

```{code-block} Python
import jax
import jax.numpy as jnp
import jax.scipy.stats as stats
import numpy as np

import blackjax

observed = np.random.normal(10, 20, size=1_000)
def logdensity_fn(x):
    logpdf = stats.norm.logpdf(observed, x["loc"], x["scale"])
    return jnp.sum(logpdf)

# Build the kernel
step_size = 1e-3
inverse_mass_matrix = jnp.array([1., 1.])
nuts = blackjax.nuts(logdensity_fn, step_size, inverse_mass_matrix)

# Initialize the state
initial_position = {"loc": 1., "scale": 2.}
state = nuts.init(initial_position)

# Iterate
rng_key = jax.random.key(0)
step = jax.jit(nuts.step)
for i in range(1_000):
    nuts_key = jax.random.fold_in(rng_key, i)
    state, _ = step(nuts_key, state)
```

:::{note}
If you want to use Blackjax with a model implemented with a PPL, go to the related tutorials in the left menu.
:::


# Installation

::::{tab-set}

:::{tab-item} Latest
```{code-block} bash
pip install blackjax
```
:::


:::{tab-item} Conda
```{code-block} bash
conda install blackjax -c conda-forge
```
:::

::::

:::{admonition} GPU instructions
:class: tip

BlackJAX is written in pure Python but depends on XLA via JAX. By default, the
version of JAX that will be installed along with BlackJAX will make your code
run on CPU only. **If you want to use BlackJAX on GPU/TPU** we recommend you follow
[these instructions](https://github.com/google/jax#installation) to install JAX
with the relevant hardware acceleration support.
:::

# Algorithm Reference

Every public algorithm in `blackjax` is listed below. **Guide** links point to
a worked example; **API** links go to the generated reference. Algorithms
marked *Sampling Book* are covered in depth at
[blackjax-devs.github.io/sampling-book](https://blackjax-devs.github.io/sampling-book).

## MCMC

| `blackjax.X` | Description | Guide | API |
|---|---|---|---|
| `hmc` | Hamiltonian Monte Carlo (static trajectory) | [Quickstart](examples/quickstart.md) | [API](autoapi/blackjax/mcmc/hmc/index) |
| `nuts` | No-U-Turn Sampler (dynamic trajectory) | [Quickstart](examples/quickstart.md) | [API](autoapi/blackjax/mcmc/nuts/index) |
| `dhmc` / `dynamic_hmc` | Dynamic HMC (alias of `nuts` trajectory logic, fixed integration) | — | [API](autoapi/blackjax/mcmc/dynamic_hmc/index) |
| `mhmc` / `multinomial_hmc` | HMC with multinomial trajectory proposal | — | [API](autoapi/blackjax/mcmc/hmc/index) |
| `dmhmc` | Dynamic HMC with multinomial proposal | — | [API](autoapi/blackjax/mcmc/dynamic_hmc/index) |
| `rmhmc` | Riemannian Manifold HMC | — | [API](autoapi/blackjax/mcmc/rmhmc/index) |
| `mala` | Metropolis-Adjusted Langevin Algorithm | — | [API](autoapi/blackjax/mcmc/mala/index) |
| `ghmc` | Generalised HMC (persistent momentum) | — | [API](autoapi/blackjax/mcmc/ghmc/index) |
| `barker` | Barker proposal (gradient-based MH) | — | [API](autoapi/blackjax/mcmc/barker/index) |
| `rmh` | Random-walk Metropolis-Hastings | — | [API](autoapi/blackjax/mcmc/random_walk/index) |
| `irmh` | Independent Random-walk MH | — | [API](autoapi/blackjax/mcmc/random_walk/index) |
| `additive_step_random_walk` / `normal_random_walk` | Additive-step random walk (Gaussian or custom) | — | [API](autoapi/blackjax/mcmc/random_walk/index) |
| `mgrad_gaussian` | Marginal latent Gaussian sampler | — | [API](autoapi/blackjax/mcmc/marginal_latent_gaussian/index) |
| `elliptical_slice` | Elliptical slice sampling | — | [API](autoapi/blackjax/mcmc/elliptical_slice/index) |
| `orbital_hmc` | Periodic orbital / periodic HMC | — | [API](autoapi/blackjax/mcmc/periodic_orbital/index) |

## MCMC — MCLMC family

| `blackjax.X` | Description | Guide | API |
|---|---|---|---|
| `mclmc` | Microcanonical Langevin Monte Carlo | [Sampling Book](https://blackjax-devs.github.io/sampling-book) | [API](autoapi/blackjax/mcmc/mclmc/index) |
| `adjusted_mclmc` | Adjusted MCLMC (MH correction) | [Sampling Book](https://blackjax-devs.github.io/sampling-book) | [API](autoapi/blackjax/mcmc/adjusted_mclmc/index) |
| `adjusted_mclmc_dynamic` | Adjusted MCLMC with dynamic step-size | [Sampling Book](https://blackjax-devs.github.io/sampling-book) | [API](autoapi/blackjax/mcmc/adjusted_mclmc/index) |

## MCMC — Laplace-preconditioned family

| `blackjax.X` | Description | Guide | API |
|---|---|---|---|
| `laplace_hmc` | HMC with Laplace approximation preconditioning | [How-to](examples/howto_laplace_hmc.md) | [API](autoapi/blackjax/mcmc/laplace/index) |
| `laplace_dhmc` | Dynamic HMC with Laplace preconditioning | [How-to](examples/howto_laplace_hmc.md) | [API](autoapi/blackjax/mcmc/laplace_dynamic_hmc/index) |
| `laplace_mhmc` | Multinomial HMC with Laplace preconditioning | [How-to](examples/howto_laplace_hmc.md) | [API](autoapi/blackjax/mcmc/laplace/index) |
| `laplace_dmhmc` | Dynamic multinomial HMC with Laplace preconditioning | [How-to](examples/howto_laplace_hmc.md) | [API](autoapi/blackjax/mcmc/laplace_dynamic_hmc/index) |

## Stochastic Gradient MCMC

| `blackjax.X` | Description | Guide | API |
|---|---|---|---|
| `sgld` | Stochastic Gradient Langevin Dynamics | — | [API](autoapi/blackjax/sgmcmc/sgld/index) |
| `sghmc` | Stochastic Gradient HMC | — | [API](autoapi/blackjax/sgmcmc/sghmc/index) |
| `sgnht` | Stochastic Gradient Nosé–Hoover Thermostat | — | [API](autoapi/blackjax/sgmcmc/sgnht/index) |
| `csgld` | Cyclical SGLD | — | [API](autoapi/blackjax/sgmcmc/csgld/index) |
| `svgd` | Stein Variational Gradient Descent | — | [API](autoapi/blackjax/vi/svgd/index) |

## Sequential Monte Carlo

| `blackjax.X` | Description | Guide | API |
|---|---|---|---|
| `tempered_smc` | Tempered (annealed) SMC | — | [API](autoapi/blackjax/smc/tempered/index) |
| `adaptive_tempered_smc` | Adaptive tempering schedule SMC | — | [API](autoapi/blackjax/smc/adaptive_tempered/index) |
| `partial_posteriors_smc` | SMC over a sequence of partial posteriors | — | [API](autoapi/blackjax/smc/partial_posteriors_path/index) |
| `persistent_sampling_smc` | Persistent-particle SMC | — | [API](autoapi/blackjax/smc/persistent_sampling/index) |
| `adaptive_persistent_sampling_smc` | Adaptive persistent-particle SMC | — | [API](autoapi/blackjax/smc/adaptive_persistent_sampling/index) |
| `inner_kernel_tuning` | SMC with per-step inner-kernel tuning | — | [API](autoapi/blackjax/smc/inner_kernel_tuning/index) |
| `pretuning` | SMC pretuning step | — | [API](autoapi/blackjax/smc/pretuning/index) |

## Variational Inference

| `blackjax.X` | Description | Guide | API |
|---|---|---|---|
| `meanfield_vi` | Mean-field (diagonal) ADVI | — | [API](autoapi/blackjax/vi/meanfield_vi/index) |
| `fullrank_vi` | Full-rank (dense covariance) ADVI | — | [API](autoapi/blackjax/vi/fullrank_vi/index) |
| `pathfinder` | Pathfinder variational inference | — | [API](autoapi/blackjax/vi/pathfinder/index) |
| `multipathfinder` | Multi-path Pathfinder | — | [API](autoapi/blackjax/vi/multipathfinder/index) |
| `schrodinger_follmer` | Schrödinger–Föllmer sampler | — | [API](autoapi/blackjax/vi/schrodinger_follmer/index) |

## Adaptation / Warmup

| `blackjax.X` | Description | Guide | API |
|---|---|---|---|
| `window_adaptation` | Dual-averaging step-size + mass-matrix warmup (HMC/NUTS) | [Quickstart](examples/quickstart.md) | [API](autoapi/blackjax/adaptation/window_adaptation/index) |
| `mclmc_find_L_and_step_size` | MCLMC trajectory-length + step-size tuning | [Sampling Book](https://blackjax-devs.github.io/sampling-book) | [API](autoapi/blackjax/adaptation/mclmc_adaptation/index) |
| `adjusted_mclmc_find_L_and_step_size` | Adjusted MCLMC tuning | [Sampling Book](https://blackjax-devs.github.io/sampling-book) | [API](autoapi/blackjax/adaptation/adjusted_mclmc_adaptation/index) |
| `chees_adaptation` | CHEES (chain-ensemble adaptation) | — | [API](autoapi/blackjax/adaptation/chees_adaptation/index) |
| `meads_adaptation` | MEADS (mass-matrix via ensemble) | — | [API](autoapi/blackjax/adaptation/meads_adaptation/index) |
| `pathfinder_adaptation` | Pathfinder-based warmup | — | [API](autoapi/blackjax/adaptation/pathfinder_adaptation/index) |
| `window_adaptation_low_rank` | Window adaptation with low-rank mass matrix | — | [API](autoapi/blackjax/adaptation/low_rank_adaptation/index) |

## Diagnostics & Utilities

| Name | Description | Guide | API |
|---|---|---|---|
| `blackjax.ess` | Effective Sample Size | [Diagnostics Guide](examples/diagnostics.md) | [API](autoapi/blackjax/diagnostics/index) |
| `blackjax.rhat` | Potential Scale Reduction (R̂) | [Diagnostics Guide](examples/diagnostics.md) | [API](autoapi/blackjax/diagnostics/index) |
| `run_inference_algorithm` | `lax.scan`-based inference loop utility | [Speed-up Guide](examples/speed_up_guide.md) | [API](autoapi/blackjax/util/index) |
| `store_only_expectation_values` | Memory-efficient streaming expectations | — | [API](autoapi/blackjax/util/index) |

```{toctree}
---
maxdepth: 1
hidden:
---
Quickstart <examples/quickstart.md>
MCMC Diagnostics <examples/diagnostics.md>
Speed-up Guide <examples/speed_up_guide.md>
```

```{toctree}
---
maxdepth: 1
caption: PPL INTEGRATION
hidden:
---
PyTensor<examples/howto_use_pytensor.md>
Numpyro<examples/howto_use_numpyro.md>
Funsor<examples/howto_use_funsor.md>
Oryx<examples/howto_use_oryx.md>
PyMC<examples/howto_use_pymc.md>
Tensorflow-Probability<examples/howto_use_tfp.md>
```

```{toctree}
---
maxdepth: 2
caption: HOW TO
hidden:
---
Sample with multiple chains?<examples/howto_sample_multiple_chains.md>
Use custom gradients?<examples/howto_custom_gradients.md>
Use non-JAX log-prob functions?<examples/howto_other_frameworks.md>
Build a Metropolis-Within-Gibbs sampler?<examples/howto_metropolis_within_gibbs.md>
Use Laplace-preconditioned HMC?<examples/howto_laplace_hmc.md>
Sample from the word BlackJAX using BlackJAX?<examples/howto_reproduce_the_blackjax_image.md>
```

```{toctree}
---
maxdepth: 1
caption: LEARN BY EXAMPLE
hidden:
---
The Sampling Book <https://blackjax-devs.github.io/sampling-book>
```

```{toctree}
---
hidden:
caption: API
maxdepth: 2
---
API Reference<autoapi/blackjax/index>
Bibliography<bib.rst>
```

```{toctree}
---
maxdepth: 1
caption: DEVELOPER DOCUMENTATION
hidden:
---
Design Principles <developer/design_principles.md>
Contributing a New Algorithm <developer/new_algorithm_guide.md>
```
