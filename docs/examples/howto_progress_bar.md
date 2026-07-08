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

```{code-cell} ipython3
:tags: [remove-cell]

import os
import multiprocessing

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count={}".format(
    multiprocessing.cpu_count()
)
```

# Show progress during sampling

BlackJAX provides a context manager, `progress_bar()`, that adds a live progress bar to **any** sampling call without modifying the algorithm itself. The bar works transparently under JAX transformations like `jax.vmap`, fixing the shape-mismatch crashes of the old `progress_bar=` parameter (see issue [#927](https://github.com/blackjax-devs/blackjax/issues/927)).

## How it works

The context manager instruments the outermost `jax.lax.scan` encountered at trace time and injects a step counter via a JAX callback. No algorithm parameters need to change—simply wrap your sampling call with `with blackjax.progress_bar():`.

## The model: Bayesian linear regression

For all examples in this guide, we'll use the same Bayesian linear regression problem:

$$y = X w + \varepsilon, \quad \varepsilon \sim N(0, \sigma^2)$$

with known noise scale $\sigma$ and Gaussian priors on the weight vector $w$.

```{code-cell} ipython3
import jax
import jax.numpy as jnp
from jax.scipy.stats import norm

import blackjax
from blackjax.util import run_inference_algorithm

N, DIM, SIGMA = 300, 10, 0.5
kx, kw, kn = jax.random.split(jax.random.key(0), 3)
X = jax.random.normal(kx, (N, DIM))
true_w = 2.0 * jax.random.normal(kw, (DIM,))
y = X @ true_w + SIGMA * jax.random.normal(kn, (N,))


def logprior(w):
    return norm.logpdf(w, 0.0, 5.0).sum()


def loglikelihood(w):
    return norm.logpdf(y - X @ w, 0.0, SIGMA).sum()


logdensity = lambda w: logprior(w) + loglikelihood(w)
```

## Basic usage: labeled bar around a warmup

Wrap any sampling call with `with blackjax.progress_bar()` and provide a label. The bar automatically detects the outermost scan and displays live updates.

```{code-cell} ipython3
warmup = blackjax.window_adaptation(blackjax.nuts, logdensity)
k_warm = jax.random.key(1)

with blackjax.progress_bar(label="NUTS warmup"):
    (state, params), _ = warmup.run(k_warm, jnp.zeros(DIM), num_steps=1000)
```

## Controlling update frequency with `print_rate`

By default, the bar updates every 5% of steps (≈ `num_steps // 20`). For slower iterations (e.g., when each step is expensive), use `print_rate=1` to update every step. This is especially useful for slow distributed clusters (see issue [#960](https://github.com/blackjax-devs/blackjax/issues/960)).

```{code-cell} ipython3
nuts = blackjax.nuts(logdensity, **params)

with blackjax.progress_bar(label="NUTS sampling", print_rate=1):
    _, samples = run_inference_algorithm(
        rng_key=jax.random.key(2),
        initial_state=state,
        inference_algorithm=nuts,
        num_steps=1500,
        transform=lambda st, _: st.position,
    )

posterior_mean = samples.mean(axis=0)
print("max |posterior_mean - true_w| =", jnp.abs(posterior_mean - true_w).max())
```

## Multi-phase workflows: adaptation and sampling in one context

Some algorithms (like `mclmc_find_L_and_step_size`) perform multiple internal scans during adaptation. A single `progress_bar()` context wraps all phases. Watch the bar reset per phase with its own total and elapsed time—this is expected behavior for multi-phase runs.

```{code-cell} ipython3
k_init, k_tune, k_run = jax.random.split(jax.random.key(3), 3)
mclmc_state = blackjax.mcmc.mclmc.init(
    position=jnp.zeros(DIM), logdensity_fn=logdensity, rng_key=k_init
)
kernel = blackjax.mcmc.mclmc.build_kernel(
    integrator=blackjax.mcmc.mclmc.isokinetic_mclachlan
)

with blackjax.progress_bar(label="MCLMC adapt+sample"):
    tuned_state, mclmc_params, _ = blackjax.mclmc_find_L_and_step_size(
        mclmc_kernel=kernel,
        logdensity_fn=logdensity,
        num_steps=2000,
        state=mclmc_state,
        rng_key=k_tune,
        diagonal_preconditioning=True,
    )
    alg = blackjax.mclmc(
        logdensity,
        L=mclmc_params.L,
        step_size=mclmc_params.step_size,
        inverse_mass_matrix=mclmc_params.inverse_mass_matrix,
    )
    _, mclmc_samples = run_inference_algorithm(
        rng_key=k_run,
        initial_state=tuned_state,
        inference_algorithm=alg,
        num_steps=1500,
        transform=lambda st, _: st.position,
    )

print("MCLMC max |mean - true_w| =", jnp.abs(mclmc_samples.mean(0) - true_w).max())
```

## Multi-chain sampling with `jax.vmap`

This is the centerpiece: a full multi-chain workflow using `jax.vmap`. The old progress bar crashed under vmap with shape mismatches (issue [#927](https://github.com/blackjax-devs/blackjax/issues/927)). The new callback-based implementation is vmap-safe because the injected step index never depends on the batched axis—it's a compile-time constant. **The bar fires exactly once per real step regardless of chain count.**

First, run a vmapped warmup across 4 chains in parallel:

```{code-cell} ipython3
n_chains = 4
k_wu = jax.random.split(jax.random.key(11), n_chains)
init_positions = jax.random.normal(jax.random.key(13), (n_chains, DIM))

mc_warmup = blackjax.window_adaptation(blackjax.nuts, logdensity)


def warmup_one(key, pos):
    (st, chain_params), _ = mc_warmup.run(key, pos, num_steps=1000)
    return st, chain_params


with blackjax.progress_bar(label=f"vmap warmup ({n_chains} chains)"):
    warm_states, chain_params = jax.vmap(warmup_one)(k_wu, init_positions)

print("Per-chain adapted params shape:", jax.tree.map(jnp.shape, chain_params))
```

Now sample from each chain with its own adapted parameters:

```{code-cell} ipython3
k_sm = jax.random.split(jax.random.key(12), n_chains)


def sample_one(key, st, p):
    alg = blackjax.nuts(logdensity, **p)  # per-chain step_size + mass matrix
    _, hist = run_inference_algorithm(
        rng_key=key,
        initial_state=st,
        inference_algorithm=alg,
        num_steps=1500,
        transform=lambda s, _: s.position,
    )
    return hist


with blackjax.progress_bar(label=f"vmap sampling ({n_chains} chains)"):
    mc_chains = jax.vmap(sample_one)(k_sm, warm_states, chain_params)

print("Output shape (chains, steps, dim):", mc_chains.shape)
```

## Tempered SMC: bar ticks per temperature

Sequential Monte Carlo iteratively tempering a likelihood is a natural fit for progress bars: the bar ticks once per **temperature**, not per particle move. This is because the outermost scan is over the tempering schedule.

```{code-cell} ipython3
import numpy as np

import blackjax.smc.resampling as resampling
from blackjax.smc import extend_params

num_particles, num_tempering = 300, 30
lambda_schedule = np.logspace(-5, 0, num_tempering)

hmc_parameters = extend_params(
    {
        "step_size": 1e-2,
        "inverse_mass_matrix": jnp.eye(DIM),
        "num_integration_steps": 10,
    }
)

tempering = blackjax.tempered_smc(
    logprior,
    loglikelihood,
    blackjax.hmc.build_kernel(),
    blackjax.hmc.init,
    hmc_parameters,
    resampling.systematic,
    10,  # num_mcmc_steps per tempering move
)

k_particles, k_smc = jax.random.split(jax.random.key(5))
initial_particles = 5.0 * jax.random.normal(k_particles, (num_particles, DIM))
smc_state = tempering.init(initial_particles)


def smc_step(carry, lmbda):
    i, st = carry
    st, info = tempering.step(jax.random.fold_in(k_smc, i), st, lmbda)
    return (i + 1, st), info


with blackjax.progress_bar(label=f"tempered SMC ({num_tempering} temps)"):
    (_, smc_final), _ = jax.lax.scan(smc_step, (0, smc_state), lambda_schedule)

smc_mean = smc_final.particles.mean(axis=0)
print("SMC  max |particle mean - true_w| =", jnp.abs(smc_mean - true_w).max())
```

**Note:** Adaptive tempering (with unknown length) runs inside a `while_loop`, so a determinate progress bar is not possible. Use `output_file=` and the external reader (see below) to get a heartbeat instead.

## File-based progress: external monitoring

For long-running jobs, write progress to a file that can be monitored from a separate terminal:

```{code-cell} ipython3
with blackjax.progress_bar(
    label="file-backed", output_file="/tmp/bjx_progress.txt", print_rate=1
):
    _, _ = run_inference_algorithm(
        rng_key=jax.random.key(6),
        initial_state=state,
        inference_algorithm=nuts,
        num_steps=500,  # reduced for docs build time
        transform=lambda st, _: st.position,
    )
```

While sampling runs in one process, monitor it from another:

```
cd /path/to/blackjax && uv run python -m blackjax.progress_reader /tmp/bjx_progress.txt
```

This displays live updates of `<current_step> <total_steps>` and is useful for HPCs where Jupyter is unavailable.

## Common gotchas

### JIT cache staleness

A function traced (compiled) inside one `progress_bar()` context caches the bar injection in JAX's JIT cache. Calling that same compiled function inside a *later* context will hit the cache and reuse the old callback—resulting in **no bar and a warning at exit**.

Safe pattern: keep one `progress_bar()` context open across all calls to the same compiled function (as shown in the multi-phase MCLMC example). If you must re-enter contexts, clear the cache:

```{code-cell} ipython3
import warnings

sample_jitted = jax.jit(
    lambda key: run_inference_algorithm(
        rng_key=key,
        initial_state=state,
        inference_algorithm=nuts,
        num_steps=100,
        transform=lambda st, _: st.position,
    )[1]
)

# Context 1: traces the function
with blackjax.progress_bar(label="chain 1 (traces)"):
    _ = sample_jitted(jax.random.key(7))

# Context 2: cache hit → no bar → UserWarning at exit
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    with blackjax.progress_bar(label="chain 2 (cache hit)"):
        _ = sample_jitted(jax.random.key(8))
    if w:
        print("Warning:", w[0].message)
```

Workaround: call `jax.clear_caches()` between contexts to force a retrace:

```python
jax.clear_caches()  # Clear JAX's JIT cache
with blackjax.progress_bar(label="chain 3"):
    _ = sample_jitted(jax.random.key(9))  # Retraced, bar now works
```

Let's verify the cache-clear workaround:

```{code-cell} ipython3
# Clear caches and re-trace in a new context
jax.clear_caches()

with blackjax.progress_bar(label="chain 3 (fresh trace)"):
    _ = sample_jitted(jax.random.key(9))  # Bar works
```

### Manual `__enter__` without `__exit__` leaks the patch

Never use manual `__enter__` / `__exit__` except for error handling. Leaving a context unclosed patches `jax.lax.scan` for the entire kernel session:

```{code-cell} ipython3
# Show the leak and recovery
cm = blackjax.progress_bar(label="leaked")
cm.__enter__()
print("Scan is patched:", "progress_bar" in jax.lax.scan.__module__)

# Recovery: always call __exit__
cm.__exit__(None, None, None)
print("Scan is restored:", "progress_bar" not in jax.lax.scan.__module__)
```

### Other caveats

- **Checkpoint + gradient:** Under `jax.checkpoint` combined with differentiation, the callback fires twice per logical step (primal + recompute), so step counts appear roughly doubled. Computed values and gradients are unaffected.
- **Multi-device sharding:** Under `jax.shard_map`, the callback fires once per device per step, multiplying the host dispatch overhead. The bar may reach 100% while slower shards are still running.
- **`functools.partial` bypass:** A `functools.partial(jax.lax.scan, ...)` captured before the context is entered silently bypasses the bar with no error.
- **Shared `output_file`:** If two contexts share the same file path, their writes interleave and corrupt the file. Use a unique path per context.

## Jupyter rendering

By default, the progress bar displays as a **rich widget** in Jupyter (requires `ipywidgets` >= 7.0, installed separately). Without it, you get a text-based bar on stderr. Both work identically; the widget is simply prettier in notebooks.

To use the rich widget:

```bash
pip install ipywidgets
```

No code changes needed—the bar detects the environment automatically.

## Summary

- Wrap any sampling call with `with blackjax.progress_bar(label="...")` to add a live bar.
- Works under `jax.vmap` without shape mismatches (fixes issue [#927](https://github.com/blackjax-devs/blackjax/issues/927)).
- Use `print_rate=` to control update frequency for expensive iterations.
- Multi-phase runs (e.g., MCLMC tuning + sampling) show bar resets per phase—expected behavior.
- Keep one context open across repeated calls to the same compiled function to avoid JIT cache staleness.
- Use `output_file=` + `python -m blackjax.progress_reader` for monitoring long-running jobs from another terminal.
- Always use the `with` form; manual `__enter__` without `__exit__` leaks the patch.
