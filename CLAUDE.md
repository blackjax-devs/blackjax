# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

BlackJAX — Bayesian inference library in pure JAX/Python.

## Commands

```bash
pip install -e ".[dev]"                          # install for development

# Tests must be run via the mamba "blackjax" environment
mamba run -n blackjax python -m pytest tests/                            # full suite
mamba run -n blackjax python -m pytest tests/ -x -W error::DeprecationWarning
mamba run -n blackjax python -m pytest tests/mcmc/test_nuts.py           # single file
mamba run -n blackjax python -m pytest tests/mcmc/test_nuts.py::TestNUTS::test_sample

make build-docs   # build Sphinx documentation
mamba run -n blackjax pre-commit run --show-diff-on-failure --color=always --all-files  # linting/formatting
```

## PR Workflow

Always branch from `origin/main`:
```bash
git fetch origin
git checkout -b my-branch origin/main
```

Before opening or updating a PR, always rebase onto the latest `origin/main`
and force-push:
```bash
git fetch origin && git rebase origin/main
git push --force-with-lease
```

## Key Rules

- All code must be jittable
- Accept both old-style (`PRNGKey`) and new-style (`key`) PRNG keys from users
- Use `jax.random.key()` internally, never `jax.random.PRNGKey()`
- Use `jax.tree.map` not `jax.tree_map`
- Use `jnp.clip(x, min=..., max=...)` not the old arg names

## Architecture

BlackJAX is a library of composable MCMC/VI/SMC kernels. The core design: **kernels are stateless pure functions**—all state is carried explicitly as `(rng_key, state) -> (new_state, info)`.

### Three-Layer Kernel Pattern

Every algorithm exposes three levels in `blackjax/mcmc/<name>.py`:

1. **`init(position, logdensity_fn) -> State`** — creates initial state
2. **`build_kernel() -> kernel_fn`** — low-level kernel with full control; signature is `kernel_fn(rng_key, state, logdensity_fn, **params) -> (State, Info)`
3. **`as_top_level_api(logdensity_fn, **params) -> SamplingAlgorithm`** — high-level wrapper

`SamplingAlgorithm` (defined in `blackjax/base.py`) is a NamedTuple `(init, step)` where `step(rng_key, state) -> (new_state, info)`.

The top-level `blackjax/__init__.py` wraps each algorithm in `GenerateSamplingAPI`, exposing `.init`, `.build_kernel`, and calling the instance directly gives `as_top_level_api`.

### Module Structure

- **`blackjax/mcmc/`** — gradient-based samplers: HMC, NUTS, MALA, GHMC, Barker, MCLMC, random walk, elliptical slice, orbital
- **`blackjax/adaptation/`** — warmup/adaptation: window adaptation (Stan-style), mass matrix, step size, LAPS, MCLMC adaptation
- **`blackjax/smc/`** — Sequential Monte Carlo
- **`blackjax/sgmcmc/`** — stochastic gradient MCMC (SGLD, SGHMC, SGNHT, CSGLD)
- **`blackjax/vi/`** — variational inference: meanfield, pathfinder, Schrodinger-Follmer, SVGD
- **`blackjax/optimizers/`** — dual averaging, LBFGS
- **`blackjax/base.py`** — core abstractions (`SamplingAlgorithm`, `VIAlgorithm`, `AdaptationAlgorithm`)
- **`blackjax/types.py`** — JAX type aliases (`Array`, `ArrayTree`, `PRNGKey`)

### HMC/NUTS Composition

The HMC family is built from composable pieces that are assembled in `nuts.py`/`hmc.py`:
- **`integrators.py`** — leapfrog, McLachlan, velocity Verlet, 4th-order integrators
- **`metrics.py`** — kinetic energy / mass matrix (diagonal or dense)
- **`proposal.py`** — acceptance/rejection logic
- **`trajectory.py`** — trajectory integration strategies
- **`termination.py`** — NUTS stopping criteria (dynamic slice, iterative)

### Adding a New Kernel

Follow `blackjax/mcmc/mala.py` (simple) or `blackjax/mcmc/nuts.py` (complex):

1. Define `State` and `Info` NamedTuples
2. Implement `init`, `build_kernel`, `as_top_level_api`
3. Register in `blackjax/__init__.py` using `GenerateSamplingAPI`

### Testing Conventions

- Tests inherit `BlackJAXTest` (from `tests/fixtures.py`) for date-based PRNG keys and JAX-aware assertions
- Use `self.next_key()` for each independent random operation; use `std_normal_logdensity` for standard Gaussian targets
- `@chex.assert_max_traces(n=2)` verifies kernels don't trigger excess recompilation
- Tests in `tests/` mirror the `blackjax/` module structure
