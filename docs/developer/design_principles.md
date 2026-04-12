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

# BlackJAX Design Principles

This document describes the architectural principles, coding conventions, and style guidelines
that define the "BlackJAX style". The reference implementations are
`trajectory.py`, `nuts.py`, `hmc.py`, `proposal.py`, and `base.py` — these files set the
quality bar that all new contributions should meet.

---

## 1. Core Architecture

### 1.1 Kernels Are Pure Functions

Every BlackJAX kernel is a stateless pure function. All state is carried explicitly:

```python
kernel(rng_key, state, logdensity_fn, **params) -> (new_state, info)
```

There is no hidden state, no mutable closures, and no side effects. This makes kernels
trivially JIT-compilable and composable.

### 1.2 The Three-Layer API

Every algorithm in `blackjax/mcmc/`, `blackjax/vi/`, and `blackjax/sgmcmc/` exposes exactly
three levels, defined in `blackjax/mcmc/<name>.py`:

| Layer | Signature | Purpose |
|-------|-----------|---------|
| `init` | `(position, logdensity_fn, *, rng_key=None) -> State` | Creates the initial algorithm state |
| `build_kernel` | `(**params) -> kernel_fn` | Returns a specialized kernel via closure; full control over all parameters |
| `as_top_level_api` | `(logdensity_fn, **params) -> SamplingAlgorithm` | Convenience wrapper; binds `logdensity_fn` and parameters once |

`SamplingAlgorithm` (defined in `blackjax/base.py`) is a NamedTuple `(init, step)` where
`step(rng_key, state) -> (new_state, info)`.

**`init` signature rule:** All `init` functions must follow `(position, logdensity_fn, *, rng_key=None, **kwargs)`.
If an algorithm needs a random key at initialization (e.g., to sample initial momentum), pass
`rng_key` as a keyword argument. Extra required arguments like `period` belong in
`build_kernel` or `as_top_level_api`, not in `init`.

### 1.3 Composable Closures

`build_kernel()` uses closures to specialize behavior at construction time, not at call time.
The returned `kernel_fn` captures configuration (integrator choice, metric, etc.) via closure
rather than re-deriving it on every step. This is the idiomatic JAX pattern for avoiding
recompilation.

### 1.4 NamedTuple for State

All algorithm state types (`HMCState`, `NUTSInfo`, `Trajectory`, `Proposal`) are
`NamedTuple`s. This provides:

- Automatic JAX pytree registration (no `register_pytree_node` needed)
- Immutability (prevents accidental in-place mutation)
- Named field access for readability
- Structural typing compatibility with `Protocol`

When constructing a modified state, use explicit constructors rather than `_replace()`:

```python
# Preferred: explicit and grep-able
return HMCState(position=new_position, logdensity=new_logdensity, logdensity_grad=new_grad)

# Avoid: hides which fields change
return state._replace(position=new_position)
```

### 1.5 `build_sampling_algorithm` Helper

The boilerplate `as_top_level_api` pattern — wrapping `init` and a bound kernel into a
`SamplingAlgorithm` — is handled by `build_sampling_algorithm` in `blackjax/base.py`.
Each module uses this helper rather than repeating the same wrapper structure. For algorithms
whose `init` requires a `rng_key`, pass `pass_rng_key_to_init=True`.

### 1.6 Explicit State Threading

All state is passed as function arguments and returned as function results. Never capture
mutable state in closures. This is what makes BlackJAX kernels safe to use with
`jax.vmap`, `jax.lax.scan`, and `jax.lax.while_loop`.

---

## 2. JAX Idioms

### 2.1 Control Flow

Use JAX's functional control flow primitives for all branches and loops in traced code:

| Use case | Primitive |
|----------|-----------|
| Conditional branch | `jax.lax.cond(pred, true_fn, false_fn)` |
| Sequential accumulation | `jax.lax.scan` |
| Fixed-count loop | `jax.lax.fori_loop` |
| While loop | `jax.lax.while_loop` |
| Batching | `jax.vmap` |

Never use Python `for`/`while`/`if` in code that will be traced (i.e., inside a kernel or
any function called from a kernel). Python conditionals on traced values will either error
or silently trace only one branch.

**Modern `jax.lax.cond` form** — use the no-operand form:

```python
# Modern (preferred)
jax.lax.cond(condition, lambda: true_branch, lambda: false_branch)

# Legacy (avoid) — the operand=None pattern is deprecated
jax.lax.cond(condition, lambda _: true_branch, lambda _: false_branch, operand=None)
```

### 2.2 Random Keys

- **Internally**: always use `jax.random.key()`, never `jax.random.PRNGKey()` (deprecated)
- **At the user boundary**: accept both old-style and new-style PRNG keys

### 2.3 Array Operations

```python
# Use named keyword args for clip
jnp.clip(x, min=lower, max=upper)   # correct
jnp.clip(x, a_min=lower, a_max=upper)  # old arg names — avoid

# Use jnp constants, not string dtype names
jnp.zeros(shape, dtype=jnp.int32)   # correct
jnp.zeros(shape, dtype="int32")      # avoid
```

### 2.4 PyTree Operations

Prefer `jax.tree.map` (not the deprecated `jax.tree_map`) for element-wise operations on
pytrees. Only flatten a pytree to a 1D array (via `ravel_pytree`) when true linear-algebra
operations are required — for example, a Cholesky decomposition or a dot product against a
mass matrix. For element-wise operations (scaling, addition, masking), stay in pytree space:

```python
# Preferred: stay in pytree space when operations are element-wise
scaled = jax.tree.map(lambda x: step_size * x, momentum)

# ravel_pytree: only at the linear-algebra boundary
flat, unravel = ravel_pytree(position)
new_flat = mass_matrix @ flat
new_position = unravel(new_flat)
```

---

## 3. Naming Conventions

Consistent naming is critical for an API used across many algorithms. Follow these rules
without exception:

| Item | Convention | Example |
|------|-----------|---------|
| Log-density function | `logdensity_fn` | (not `log_prob`, `log_density`, `logpdf`) |
| Log-density gradient | `logdensity_grad` | (not `logdensitygrad`, `grad`) |
| Step function in `SamplingAlgorithm` | `step_fn` | (not `update_fn`, `kernel_fn`) |
| Noise function parameter | `noise_fn` | |
| Momentum PRNG key | `key_momentum` | (not `key_mometum`) |
| Descriptive names | Write them out | `initial_energy`, not `e0`; `trajectory_integrator`, not `traj_int` |

**No abbreviations** in public API names. Single-letter variable names (`l`, `g`) are only
acceptable as local temporaries in short, obvious expressions.

---

## 4. Type Annotations

All public function signatures must carry type annotations. Use modern Python 3.10+ syntax:

| Old style | Modern style |
|-----------|--------------|
| `Tuple[X, Y]` | `tuple[X, Y]` |
| `Dict[str, X]` | `dict[str, X]` |
| `List[X]` | `list[X]` |
| `Optional[X]` | `X \| None` |
| `Union[X, Y]` | `X \| Y` |

Do not import `Tuple`, `Dict`, `List`, `Optional`, or `Union` from `typing`. The only
`typing` imports that remain useful are `Protocol`, `TypeAlias`, `Callable`, and `Any`.

Use `Protocol` for structural typing of function signatures. `base.py` and `metrics.py`
show the established patterns — extend them rather than writing plain `Callable`.

---

## 5. Documentation Style

Docstrings follow the **numpydoc** format with `Parameters` and `Returns` sections.
Every public function, class, and module needs a docstring.

```python
def build_kernel(step_size: float, inverse_mass_matrix: Array) -> Callable:
    """Build an HMC transition kernel.

    Parameters
    ----------
    step_size
        Size of the leapfrog integration step.
    inverse_mass_matrix
        Inverse of the mass matrix. Either a 1D array (diagonal) or 2D array (dense).

    Returns
    -------
    A kernel function ``kernel(rng_key, state, logdensity_fn) -> (HMCState, HMCInfo)``.
    """
```

Magic numbers must be explained. If a constant has a mathematical derivation (e.g., a
coefficient that comes from `Var[E] = O(ε⁶)`), document that derivation inline rather than
leaving a bare numeric literal.

---

## 6. Module Organization

### 6.1 `__all__` Exports

Every module must define `__all__` at the top level, listing the public API.

### 6.2 Section Comments

For long files, use section comments to delimit logical groups:

```python
# --- Trajectory integration ---

# --- Proposal generation ---
```

### 6.3 Module Boundaries

Resist the temptation to build monolithic files. The HMC family demonstrates the right
decomposition:

| File | Responsibility |
|------|---------------|
| `integrators.py` | Leapfrog and higher-order integrators |
| `metrics.py` | Kinetic energy and mass matrix logic |
| `proposal.py` | Acceptance/rejection |
| `trajectory.py` | Trajectory integration strategies |
| `termination.py` | NUTS stopping criteria |
| `hmc.py` / `nuts.py` | Assembly: wires the pieces together |

A new algorithm should identify which existing building blocks it can reuse before adding
new code. Only introduce new abstractions when there is genuine reuse across at least two
algorithms.

Utilities belong in dedicated modules by function:
- Diagnostics (PSIS, ESS, R-hat) → `diagnostics.py`
- ECA / ensemble utilities → `eca.py`
- Core type aliases and protocols → `base.py`, `types.py`

---

## 7. Testing Conventions

Tests live in `tests/`, mirroring the `blackjax/` module structure.

- All test classes inherit `BlackJAXTest` (from `tests/fixtures.py`) for date-based PRNG
  keys and JAX-aware assertions.
- Use `self.next_key()` for each independent random operation.
- Use `std_normal_logdensity` as the canonical test target.
- Decorate kernel tests with `@chex.assert_max_traces(n=2)` to verify kernels do not trigger
  excess recompilation.
- Every MCMC algorithm must have a **protocol conformance test** that verifies:
  - `init(position, logdensity_fn)` returns the declared `State` type
  - `step(rng_key, state) -> (State, Info)` matches the `SamplingAlgorithm` contract
  - Parameter names match the declared API

---

## 8. What the Gold Standard Looks Like

The files that best embody all of the above principles are:

- **`blackjax/mcmc/trajectory.py`** — composable trajectory strategies, clear abstractions
- **`blackjax/mcmc/nuts.py`** — three-layer API, reuses every HMC building block
- **`blackjax/mcmc/hmc.py`** — clean assembly of composable pieces
- **`blackjax/mcmc/proposal.py`** — acceptance logic, proper use of NamedTuples
- **`blackjax/base.py`** — protocol definitions, `SamplingAlgorithm`, `build_sampling_algorithm`

When in doubt, read these files first.
