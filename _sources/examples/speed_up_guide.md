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

# Speed-up Guide

*Last updated: 2026-04-19*

This guide collects actionable tips for getting the best performance out of BlackJAX.
The recommendations build on the [JAX performance documentation](https://docs.jax.dev/en/latest/gpu_performance_tips.html)
and on benchmarks run against other JAX-based PPLs such as NumPyro.

---

## 1. Always JIT the step function at the outermost scope

BlackJAX kernels are plain Python callables.  Calling them without `jax.jit`
triggers a fresh XLA compilation on every Python invocation.

```python
# ✗ slow — recompiles on every call
for i in range(1000):
    state, info = nuts.step(key, state)

# ✓ fast — compile once, run many
step = jax.jit(nuts.step)
for i in range(1000):
    state, info = step(key, state)
```

`run_inference_algorithm` and `window_adaptation` already JIT internally, but
if you write your own loop wrap the whole function in a single `jax.jit` at the top:

```python
@jax.jit
def run(rng_key, initial_state):
    def step(state, key):
        state, info = kernel(key, state)
        return state, info
    return jax.lax.scan(step, initial_state, jax.random.split(rng_key, n_iter))
```

See the [JAX JIT documentation](https://docs.jax.dev/en/latest/jit-compilation.html)
for a full explanation of what triggers recompilation.

---

## 2. Use `jax.lax.scan` instead of Python loops

A Python `for` loop over MCMC steps **unrolls** the entire computation graph at
trace time — for 1 000 steps that is a 1 000× larger HLO program, slow to
compile and large in memory.  `jax.lax.scan` lowers to a single XLA `WhileOp`
and compiles the body *once*, keeping compilation cost constant regardless of
step count.

:::{note}
`jax.lax.scan` applies **implicit JIT compilation** to its body function even
when called outside an explicit `jax.jit` (you can verify this with
`jax.disable_jit()`).  However, you should still wrap the *outer* function in
`@jax.jit` so that JAX caches the full trace — without it, Python-level setup
code (key splitting, state initialisation) is re-executed and the whole
function re-traced on every call.
:::

```python
# ✗ Python loop inside jit — unrolls at trace time, O(n) compile cost
@jax.jit
def run_python_loop(rng_key, initial_state, n_steps):
    state = initial_state
    for key in jax.random.split(rng_key, n_steps):  # unrolled by JAX tracer
        state, _ = kernel(key, state)
    return state

# ✓ lax.scan inside jit — O(1) compile cost, compact XLA WhileOp
@jax.jit
def run_scan(rng_key, initial_state, n_steps):
    def step(state, key):
        state, info = kernel(key, state)
        return state, info
    return jax.lax.scan(step, initial_state, jax.random.split(rng_key, n_steps))
```

NumPyro, Oryx, and other JAX PPLs all rely on `lax.scan` internally for the
same reason.  See [JAX's official scan docs](https://docs.jax.dev/en/latest/_autosummary/jax.lax.scan.html)
and [JAX discussion #16106](https://github.com/jax-ml/jax/discussions/16106)
for more background.

---

## 3. Flatten multi-leaf pytree positions before passing to BlackJAX

:::{admonition} Key insight from benchmarking
:class: important

When the sampler position is a **dict with multiple named arrays** (e.g. a
model with parameters `alpha`, `beta`, `sigma`, …), JAX/XLA allocates a
**separate buffer for every leaf** inside the `lax.scan` carry.  This
multiplies the number of HLO nodes in the compiled program and increases both
compile time and per-step execution time.

Replacing the dict with a **single flat 1-D array** collapses all leaves into
one buffer and eliminates the overhead.  In benchmarks on a Finnish horseshoe
model with 6 named parameter groups (404 parameters total), this yielded a
**~1.3× speedup** on sampling time.
:::

### The fix: wrap your log-density with `jax.flatten_util`

```python
init_flat, unflatten = jax.flatten_util.ravel_pytree(init_params)

def logdensity_flat(flat):
    return logdensity_fn(unflatten(flat))
```

After sampling, recover named parameters with:

```python
samples_dict = jax.vmap(unflatten)(samples.position)
# samples_dict["alpha"].shape == (n_iter,)
```

### Benchmark

The benchmark is in [`tests/test_benchmarks.py`](../../tests/test_benchmarks.py)
(`test_horseshoe_nuts_flat_vs_dict`).  Run it with:

```bash
JAX_PLATFORM_NAME=cpu pytest tests/test_benchmarks.py::test_horseshoe_nuts_flat_vs_dict -v -s
```

Typical output on CPU (Finnish horseshoe N=100 M=200, 1 chain, 1 000 NUTS
steps, shared warmup so only pytree-carry overhead differs):

```
  Warmup:  step_size=0.00643  IMM diag mean=0.0032

  Model: Finnish horseshoe  N=100 M=200  1 chain  1000 samples  (shared warmup)

  Metric                          flat (1 leaf)  dict (6 leaves)
  --------------------------------------------------------------
  sample time (s)                          4.94             6.35
  min ESS                                  93.3             46.3
  min ESS/s                                18.9              7.3
  --------------------------------------------------------------
  sample speedup (dict/flat)               1.29x

  Parameter       size   ESS flat  ESS dict   Rhat flat  Rhat dict
  ----------------------------------------------------------------
  alpha              1      546.9     499.4       0.999      0.999
  sigma              1     1063.9    1114.3       1.003      1.002
  tau_tilde          1      849.5     885.6       0.999      0.999
  c2_tilde           1      562.0     485.7       0.999      0.999
  lambda_          200      232.5     230.1       1.036      1.016
  beta_tilde       200       93.3      46.3       1.013      1.014
```

Both parameterisations produce equivalent posteriors (R̂ ≈ 1).  The flat array
is ~1.3× faster because the single-leaf carry reduces XLA buffer allocation
overhead inside `lax.scan`.  `lambda_` and `beta_tilde` show the highest R̂,
reflecting the well-known slow mixing of local shrinkage scales in horseshoe
models — this is a property of the model geometry, not the parameterisation.

### When does this matter?

| Scenario | Impact |
|---|---|
| Position is a **single array** (e.g. `jnp.zeros(D)`) | None — already flat |
| Position is a **dict with 2–3 scalar leaves** | Small (~10–20%) |
| Position is a **dict with many leaves** or large arrays | Significant (~1.3×) |

---

## 4. Use `vmap` over chains, not a Python loop

To run multiple chains in parallel, use `jax.vmap` rather than a Python loop.
`vmap` batches the computation into a single XLA kernel call.

```python
n_chains = 4
keys = jax.random.split(jax.random.key(0), n_chains)
states, infos = jax.vmap(one_chain)(keys, initial_positions)
```

See [howto_sample_multiple_chains.md](howto_sample_multiple_chains.md) for a
complete worked example.

---

## 5. Avoid recompilation: keep shapes and dtypes stable

JAX recompiles whenever an input shape, dtype, or static argument changes.
Common pitfalls in BlackJAX workflows:

- **Changing `num_steps`** between warmup and sampling — use `jax.lax.scan`
  with a fixed step count, or pass `num_steps` as a traced value.
- **Mixing float32 and float64** — choose one dtype and stick to it.  On CPU
  `float64` is fine; on GPU `float32` is faster and avoids the x64 overhead.
  See [JAX configuration options](https://docs.jax.dev/en/latest/config_options.html).
- **Rebuilding the kernel inside a loop** — construct `blackjax.nuts(...)` once
  outside the loop and reuse the `.step` function.

---

## 6. GPU-specific tips

- **Prefer `float32`** — GPU throughput is typically 2–32× higher for
  `float32` vs `float64`.  Enable `float64` only if numerical precision is
  critical.
- **Batch across chains** — a single chain is often too small to saturate a
  GPU.  Running 8–64 chains with `vmap` amortises kernel launch overhead.
  See [JAX GPU performance tips](https://docs.jax.dev/en/latest/gpu_performance_tips.html).
- **Use `jax.block_until_ready`** when benchmarking — JAX dispatches
  asynchronously, so wall-clock time measured without `block_until_ready` is
  misleading.  See the [JAX benchmarking guide](https://docs.jax.dev/en/latest/benchmarking.html).

---

## 7. Profile before optimising

Use JAX's built-in profiler or Perfetto to identify the actual bottleneck
before applying the tips above.

```python
with jax.profiler.trace("/tmp/jax-trace"):
    result = run(rng_key, initial_state)
    jax.block_until_ready(result)
# open /tmp/jax-trace in https://ui.perfetto.dev
```

Most performance problems in BlackJAX workflows fall into one of three
categories:

1. **Recompilation** — check for shape/dtype changes; use `jax.make_jaxpr` to
   inspect traces.
2. **Pytree overhead in scan** — apply the flattening trick from §3 if your
   position is a multi-leaf dict.
3. **Insufficient parallelism on GPU** — increase chain count or batch size.

---

## See also

- [JAX: just-in-time compilation](https://docs.jax.dev/en/latest/jit-compilation.html)
- [JAX: GPU performance tips](https://docs.jax.dev/en/latest/gpu_performance_tips.html)
- [JAX: benchmarking JAX code](https://docs.jax.dev/en/latest/benchmarking.html)
- [JAX: pytrees](https://docs.jax.dev/en/latest/pytrees.html)
- [JAX: `lax.scan` API](https://docs.jax.dev/en/latest/_autosummary/jax.lax.scan.html)
- [BlackJAX: sampling with multiple chains](howto_sample_multiple_chains.md)
- [BlackJAX issue #597 — NUTS GPU performance](https://github.com/blackjax-devs/blackjax/issues/597)
