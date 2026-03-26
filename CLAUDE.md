# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

BlackJAX — Bayesian inference library in pure JAX/Python.

## Commands

```bash
pip install -e ".[dev]"                          # install for development
pytest tests/                                    # run full test suite
pytest tests/ -x -W error::DeprecationWarning    # catch deprecations
pytest tests/mcmc/test_nuts.py                   # run a single test file
pytest tests/mcmc/test_nuts.py::TestNUTS::test_sample  # run a single test

make build-docs   # build Sphinx documentation
pre-commit run --all-files  # run linting/formatting
```

## Current Task

Library cleanup for consistency. See the cleanup plan below. The jaxopt → optax migration (documented below) is complete.

## Library Cleanup Plan

### Objective

Establish consistent style across BlackJAX: small focused functions, uniform naming conventions, and complete docstrings. The codebase has grown through contributions from separate teams (HMC/NUTS, MCLMC, SMC, VI) and needs alignment.

### Areas of Work

**1. Code style audit across subsystems** ✅ DONE (PR #824)

Audited `blackjax/mcmc/`, `blackjax/smc/`, `blackjax/vi/`. Findings acted upon:

- ✅ Naming conventions: aligned `logdensity` throughout; fixed `logprob` → `logdensity` in `smc/ess.py`; fixed `RWState` docstring field name mismatch
- ✅ Docstring completeness: added Args/Returns sections to ~30 public functions across all three subsystems
- ✅ Stale parameter docs fixed (e.g. `adjusted_mclmc.py` referenced non-existent params)
- ✅ JAX API confirmed consistent (`jax.tree.map`, `jax.random.key()`, `jnp.clip(min=, max=)`)

Remaining known issues (tracked for future PRs, not blocking Step 2):
- Large functions in `mcmc/trajectory.py` (150+ lines each: `dynamic_progressive_integration`, `dynamic_recursive_integration`, `dynamic_multiplicative_expansion`) and `smc/persistent_sampling.py`
- Type annotation coverage is 0% in ~13 files (`adjusted_mclmc.py`, `diffusions.py`, `elliptical_slice.py`, `meanfield_vi.py`, etc.)

**2. Test suite depth audit**

Tests are inconsistently deep across subsystems:
- Some test only end-to-end behavior (too shallow — misses unit-level bugs)
- Some test internal implementation details exhaustively (too deep — brittle)
- Target: unit tests for key building blocks + end-to-end smoke tests for each kernel

Specific gaps to close:
- MCLMC tests vs HMC tests: compare depth and coverage
- SMC tests: check if internal resampling/weighting steps are tested
- VI tests: pathfinder vs meanfield coverage comparison

**3. PRNG key strategy in tests**

Current tests hardcode specific PRNG seeds, making them brittle (pass/fail depends on lucky/unlucky keys). Strategy:

- Use date-based seeds: `seed = int(datetime.date.today().strftime("%Y%m%d"))` — deterministic per day, rotates automatically
- Alternatively, parameterize tests over multiple seeds and assert statistical properties rather than exact values
- Prefer asserting distribution properties (mean, variance within tolerance) over exact sample equality

**4. Clean up issues and pull requests**

After the code and test cleanup is complete:
- Review open GitHub issues: close those resolved by the cleanup, update or label those still relevant
- Review open PRs: merge or close stale ones, ensure remaining PRs are rebased and conflict-free

### Key Rules for Cleanup

- Do not change public API signatures — this is style/consistency only
- All refactored code must remain JIT-compatible
- Run `pre-commit run --all-files` and full test suite after each subsystem change
- Keep changes subsystem-scoped per PR (no mega-PRs)

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

- Tests use `chex.TestCase` for JAX-aware assertions
- `@chex.assert_max_traces(n=2)` verifies kernels don't trigger excess recompilation
- Tests in `tests/` mirror the `blackjax/` module structure
