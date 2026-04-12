# Contributing a New Algorithm to BlackJAX

This guide walks through everything needed to add a new algorithm — from file layout through
registration, testing, and PR review. Read `examples/design_principles.md` first for the
underlying rules; this document is the practical how-to.

Skeleton files for copy-pasting are provided alongside this guide:

- [`sampling_algorithm.py`](sampling_algorithm.py) — MCMC sampler skeleton
- [`approximate_inf_algorithm.py`](approximate_inf_algorithm.py) — VI algorithm skeleton

---

## 1. Orientation: Where Does My Algorithm Live?

| Algorithm family | Directory |
|-----------------|-----------|
| MCMC (HMC, MALA, random walk, …) | `blackjax/mcmc/` |
| Variational inference | `blackjax/vi/` |
| Stochastic-gradient MCMC | `blackjax/sgmcmc/` |
| Sequential Monte Carlo | `blackjax/smc/` |
| Adaptation / warmup | `blackjax/adaptation/` |

Each algorithm lives in its own module — one `.py` file per algorithm. Do not add a new
algorithm to an existing file.

---

## 2. Adding an MCMC Sampler

Every MCMC module must export exactly four public names (listed in `__all__`):

```python
__all__ = ["MyState", "MyInfo", "init", "build_kernel", "as_top_level_api"]
```

### 2.1 State and Info NamedTuples

```python
from typing import NamedTuple
from blackjax.types import Array, ArrayTree

class MyState(NamedTuple):
    """State of My Sampler.

    position
        Current position of the chain.
    logdensity
        Log-density at the current position.
    """
    position: ArrayTree
    logdensity: float
    # add any extra fields the kernel needs to carry forward


class MyInfo(NamedTuple):
    """Transition information returned by My Sampler.

    acceptance_rate
        Metropolis–Hastings acceptance probability.
    is_accepted
        Whether the proposal was accepted.
    """
    acceptance_rate: float
    is_accepted: bool
```

Rules:
- Both must be `NamedTuple` — never a plain dataclass or dict.
- `State` carries only what is needed for the *next* step.  Do **not** put tuning
  counters, convergence diagnostics, or adaptation parameters in `State`; those
  belong in a separate `AdaptationState` returned by an adaptation routine.
- `Info` carries anything useful for diagnostics that does *not* need to persist.

### 2.2 `init`

```python
from typing import Callable
from blackjax.types import ArrayLikeTree, PRNGKey

def init(position: ArrayLikeTree, logdensity_fn: Callable,
         *, rng_key: PRNGKey | None = None) -> MyState:
    logdensity = logdensity_fn(position)
    return MyState(position, logdensity)
```

Rules:
- Signature is always `(position, logdensity_fn, *, rng_key=None)`.
- If the algorithm needs a random key at init (e.g. to sample initial momentum),
  accept it as `rng_key` — a keyword-only argument.  Never add extra positional
  arguments to `init`; those belong in `build_kernel` or `as_top_level_api`.

### 2.3 `build_kernel`

```python
def build_kernel(
    # Algorithm-level configuration (integrator, threshold, …) goes here.
    # Captured by closure; does NOT appear in the inner kernel's signature.
) -> Callable:
    """Build My Sampler kernel.

    Returns
    -------
    A kernel ``(rng_key, state, logdensity_fn, *params) -> (MyState, MyInfo)``.
    """

    def kernel(
        rng_key: PRNGKey,
        state: MyState,
        logdensity_fn: Callable,
        step_size: float,  # per-step parameters follow logdensity_fn
    ) -> tuple[MyState, MyInfo]:
        """Generate a new sample."""
        # Split rng_key for each independent random operation.
        # Implement your proposal, energy evaluation, and accept/reject here.
        # See blackjax/mcmc/mala.py (simple) or nuts.py (complex) for reference.
        ...

    return kernel
```

Rules:
- The *outer* `build_kernel` captures algorithm-level configuration via closure.
  Keep the inner `kernel` signature as short as possible.
- **No Python `for`/`while`/`if` on traced values inside `kernel`.**  Use
  `jax.lax.cond`, `jax.lax.scan`, `jax.lax.fori_loop`, or `jax.vmap`.
- Use `jax.tree.map`, not the deprecated `jax.tree_map`.
- Use `jax.random.key()` internally; never `jax.random.PRNGKey()`.

### 2.4 `as_top_level_api`

Use `build_sampling_algorithm` from `blackjax.base` — do not repeat the
`init_fn` / `step_fn` boilerplate by hand:

```python
from blackjax.base import SamplingAlgorithm, build_sampling_algorithm

def as_top_level_api(
    logdensity_fn: Callable,
    step_size: float,
) -> SamplingAlgorithm:
    """My Sampler — user-facing convenience wrapper.

    Examples
    --------

    .. code::

        sampler = blackjax.my_sampler(logdensity_fn, step_size=0.1)
        state = sampler.init(initial_position)
        new_state, info = sampler.step(rng_key, state)

    Parameters
    ----------
    logdensity_fn
        The log-density function of the target distribution.
    step_size
        Proposal step size.

    Returns
    -------
    A ``SamplingAlgorithm``.
    """
    kernel = build_kernel()
    return build_sampling_algorithm(kernel, init, logdensity_fn,
                                    kernel_args=(step_size,))
```

If `init` needs a `rng_key` (e.g. for MCLMC-style initialization), pass
`pass_rng_key_to_init=True` to `build_sampling_algorithm`.

---

## 3. Reusing Building Blocks

Before writing new code, decompose your algorithm into its basic components and check
whether BlackJAX already implements them. The `blackjax/mcmc/proposal.py` module contains
the lowest-level accept/reject primitives used by every MCMC algorithm:

**Symmetric proposal (Metropolis)** — when `P(x'|x) = P(x|x')`:

```python
import blackjax.mcmc.proposal as proposal

# Compute the log acceptance ratio.
new_proposal, is_diverging = proposal.safe_energy_diff(initial_energy, proposal_energy)
# Draw from the proposal distribution.
sampled_state, info = proposal.static_binomial_sampling(rng_key, proposal, new_proposal)
```

See `blackjax/mcmc/hmc.py` for a complete example.

**Asymmetric proposal (Metropolis–Hastings)** — when the transition kernel is not symmetric:

```python
compute_acceptance_ratio = proposal.compute_asymmetric_acceptance_ratio(transition_energy)
sampled_state, info = proposal.static_binomial_sampling(rng_key, log_p_accept, state, new_state)
```

See `blackjax/mcmc/mala.py` for a complete example.

**Non-reversible slice sampling** — swap `static_binomial_sampling` for
`nonreversible_slice_sampling` on either of the above to get Neal's non-reversible
update. The slice variable must then be carried in the kernel state rather than
regenerated from a PRNG key each step. `blackjax/mcmc/ghmc.py` demonstrates this:
it is HMC with a persistent momentum and a non-reversible slice sampling step.

The key principle: **find and reuse** existing building blocks before introducing new
abstractions. Only add a new module-level function when it will be shared by at least
two algorithms.

---

## 4. Adding a Variational Inference Algorithm

VI modules export:

```python
__all__ = ["MyVIState", "MyVIInfo", "init", "step", "sample", "as_top_level_api"]
```

The pattern mirrors the MCMC case but uses `VIAlgorithm` (a NamedTuple of `init`,
`step`, `sample`):

```python
from blackjax.base import VIAlgorithm

def as_top_level_api(
    logdensity_fn: Callable,
    optimizer,       # optax GradientTransformation
    num_samples: int = 100,
) -> VIAlgorithm:
    def init_fn(position):
        return init(position, logdensity_fn)

    def step_fn(rng_key, state):
        return step(rng_key, state, logdensity_fn, optimizer, num_samples)

    def sample_fn(rng_key, state, num_samples):
        return sample(rng_key, state, num_samples)

    return VIAlgorithm(init_fn, step_fn, sample_fn)
```

---

## 5. Registration in `blackjax/__init__.py`

### MCMC / SGMCMC / SMC

```python
# At the top of __init__.py, import your module:
from .mcmc import my_sampler as _my_sampler

# Below the GenerateSamplingAPI block:
my_sampler = generate_top_level_api_from(_my_sampler)
```

`generate_top_level_api_from` wraps the module into a `GenerateSamplingAPI`
dataclass that exposes `.init`, `.build_kernel`, and is callable as
`blackjax.my_sampler(logdensity_fn, ...)`.

### VI

```python
from .vi import my_vi as _my_vi

# Use GenerateVariationalAPI:
my_vi = GenerateVariationalAPI(
    _my_vi.as_top_level_api,
    _my_vi.init,
    _my_vi.step,
    _my_vi.sample,
)
```

---

## 6. Testing

### 5.1 File location

Tests mirror the module structure:

| Module | Test file |
|--------|-----------|
| `blackjax/mcmc/my_sampler.py` | `tests/mcmc/test_my_sampler.py` |
| `blackjax/vi/my_vi.py` | `tests/vi/test_my_vi.py` |

### 5.2 Base class and fixtures

```python
from absl.testing import absltest
import chex
import jax
import jax.numpy as jnp
import blackjax
from tests.fixtures import BlackJAXTest, std_normal_logdensity
```

- Inherit `BlackJAXTest` (not bare `chex.TestCase`) — it provides `self.next_key()`
  seeded from today's date so tests are deterministic and don't clash.
- Use `std_normal_logdensity` as the canonical 1-D and N-D test target.

### 5.3 Required test cases

Every new algorithm needs **at minimum**:

```python
class TestMySampler(BlackJAXTest):

    @chex.assert_max_traces(n=2)
    def test_jit_and_no_recompile(self):
        """The kernel must not retrace on the second call."""
        sampler = blackjax.my_sampler(std_normal_logdensity, step_size=0.1)
        state = sampler.init(jnp.zeros(2))
        step = jax.jit(sampler.step)
        state, _ = step(self.next_key(), state)
        state, _ = step(self.next_key(), state)

    def test_convergence_1d_gaussian(self):
        """Samples must have mean ≈ 0 and std ≈ 1 for a 1-D standard normal."""
        sampler = blackjax.my_sampler(std_normal_logdensity, step_size=0.1)
        state = sampler.init(jnp.array([1.0]))

        def one_step(state, key):
            state, _ = sampler.step(key, state)
            return state, state.position

        keys = jax.random.split(self.next_key(), 5_000)
        _, samples = jax.lax.scan(one_step, state, keys)
        samples = samples[1_000:]  # discard burn-in

        self.assertAllClose(jnp.mean(samples), 0.0, atol=0.1)
        self.assertAllClose(jnp.std(samples), 1.0, atol=0.1)

    def test_pytree_position(self):
        """The kernel must work with dict / nested PyTree positions."""
        def logdensity(x):
            return -0.5 * (x["a"] ** 2 + jnp.sum(x["b"] ** 2))

        sampler = blackjax.my_sampler(logdensity, step_size=0.1)
        state = sampler.init({"a": 0.0, "b": jnp.zeros(3)})
        new_state, info = sampler.step(self.next_key(), state)
        chex.assert_trees_all_equal_shapes(state, new_state)
```

Additionally, **add a row to `tests/mcmc/test_sampling.py`** in the
`regression_test_cases` list so your algorithm participates in the shared
accuracy regression suite:

```python
{
    "algorithm": blackjax.my_sampler,
    "initial_position": {"log_scale": 0.0, "coefs": 4.0},
    "parameters": {"step_size": 0.1},
    "num_warmup_steps": 1_000,
    "num_sampling_steps": 3_000,
},
```

### 5.4 Run tests

```bash
mamba run -n blackjax python -m pytest tests/mcmc/test_my_sampler.py -x
mamba run -n blackjax python -m pytest tests/mcmc/test_sampling.py -x
```

---

## 7. PR Checklist

Before opening a PR, verify each item:

**API correctness**
- [ ] `init` signature is `(position, logdensity_fn, *, rng_key=None)` — no extra positional args
- [ ] `build_kernel()` returns a kernel with signature `(rng_key, state, logdensity_fn, *params)`
- [ ] `as_top_level_api` uses `build_sampling_algorithm` (not hand-rolled boilerplate)
- [ ] Tuning / adaptation is **not** in `State` — it lives in a separate `AdaptationState`
- [ ] `logdensity_fn` returns a scalar — no side-channel return values

**JAX correctness**
- [ ] No Python `for`/`while`/`if` on traced values inside the kernel
- [ ] Uses `jax.tree.map` (not deprecated `jax.tree_map`)
- [ ] Uses `jax.random.key()` internally (not `jax.random.PRNGKey()`)
- [ ] Uses `jnp.clip(x, min=..., max=...)` with named args
- [ ] No `jnp.ndarray` type hints — use `jax.Array`

**Types and style**
- [ ] `__all__` defined at module top
- [ ] Modern union syntax: `X | None` not `Optional[X]`, `tuple[X, Y]` not `Tuple[X, Y]`
- [ ] Numpydoc docstrings on all public functions and classes
- [ ] Magic constants explained inline

**Tests**
- [ ] `test_jit_and_no_recompile` with `@chex.assert_max_traces(n=2)`
- [ ] `test_convergence_1d_gaussian` or equivalent accuracy test
- [ ] `test_pytree_position`
- [ ] Row added to `tests/mcmc/test_sampling.py` regression suite

**Registration**
- [ ] Module imported and registered in `blackjax/__init__.py`

---

## 8. Common Pitfalls (informed by real PRs)

### Adaptation state leaking into sampler state

```python
# WRONG — tuning parameters in the sampler state
class MyState(NamedTuple):
    position: ArrayTree
    logdensity: float
    step_size: float          # ← belongs in AdaptationState
    tuning_active: bool       # ← belongs in AdaptationState
    patience_count: int       # ← belongs in AdaptationState
```

Adaptation belongs in `blackjax/adaptation/`, following the pattern in
`window_adaptation.py`. The sampler state should be the minimum information
needed to generate the next sample.

### Non-standard `logdensity_fn` interface

BlackJAX's contract is that `logdensity_fn(position) -> scalar`. Do not add a
"blobs" pattern (where `logdensity_fn` can return extra metadata alongside the
scalar). If your reference implementation (e.g. `emcee`) returns side-channel
data from the log-density, implement a thin wrapper at the user boundary instead
of modifying the BlackJAX interface.

### Python loops in the kernel

```python
# WRONG — Python loop unrolled at trace time; breaks with dynamic nsplits
for i in range(nsplits):
    group = update_group(keys[i], groups[i], ...)

# RIGHT — use jax.lax.scan or jax.vmap
groups, infos = jax.lax.scan(
    lambda carry, xs: update_group(*xs),
    init_carry,
    (keys, stacked_groups),
)
```

### Inconsistent state types across variants

If you implement two related samplers (e.g. stretch move and slice sampling),
they should share a common `EnsembleState` base rather than defining
`StretchState` and `SliceEnsembleState` independently.  Inconsistent state
types break generic adaptation wrappers and make testing harder.
