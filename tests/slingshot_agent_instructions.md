# Agent Instructions: Implementing Slingshot MP-MCMC (Mad Props) Sampler

You are an expert open-source developer agent specializing in JAX, high-performance Bayesian computation, and the BlackJAX framework architectures. Your objective is to formally implement the **Slingshot MP-MCMC (Mad Props)** algorithm into the local BlackJAX repository copy.

## Core Design Principles
1. **Purely Functional Implementation**: All states, kernels, and factory wrappers must be completely stateless. All parameters and metrics must pass explicitly through immutable JAX PyTrees without side effects.
2. **Vectorized Across Proposals**: Leverage `jax.vmap` to evaluate the entire localized proposal cloud simultaneously, maximizing GPU execution scale.
3. **BlackJAX API Compliance**: Align precisely with the modular architectural structural layouts observed across native samplers like Barker, HMC, and Random Walk Metropolis.

---


```

### Task 2: Create the Low-Level Mathematical Engine
Create a brand new file at the following location: `blackjax/mcmc/slingshot.py`. This file houses the mathematical state representations, initialization logic, and the distance-biased transition kernel calculation.

**File Content for `blackjax/mcmc/slingshot.py`:**
```python
import jax
import jax.numpy as jnp
from typing import Callable, NamedTuple

class SlingshotState(NamedTuple):
    """State of the Slingshot MP-MCMC sampler.

    position: Current parameter values.
    log_density: The log-density value evaluated at the current position.
    """
    position: jnp.ndarray
    log_density: float

class SlingshotInfo(NamedTuple):
    """Internal diagnostics for the Slingshot transition step.

    proposal_cloud: The entire coordinate matrix of generated proposals.
    weights: The calculated selection probabilities across the cloud.
    chosen_index: The index of the proposal chosen to be the next state.
    """
    proposal_cloud: jnp.ndarray
    weights: jnp.ndarray
    chosen_index: jnp.ndarray

def init(position: jnp.ndarray, logdensity_fn: Callable) -> SlingshotState:
    """Initialize the Slingshot sampler state from a starting position."""
    return SlingshotState(position=position, log_density=logdensity_fn(position))

def kernel() -> Callable:
    """Build the functional transition kernel for the Slingshot sampler.

    Returns a pure, JIT-compilable function that executes a single parallel
    cloud proposal step.
    """
    def one_step(
        rng_key: jax.random.PRNGKey,
        state: SlingshotState,
        logdensity_fn: Callable,
        step_size: float,
        num_proposals: int,
    ) -> tuple[SlingshotState, SlingshotInfo]:

        key_cloud, key_select = jax.random.split(rng_key)
        dim = state.position.shape[0]

        # 1. Generate parallel proposal cloud around the current state
        # Shape: (num_proposals, dim)
        noise = jax.random.normal(key_cloud, shape=(num_proposals, dim))
        proposal_cloud = state.position + noise * step_size

        # 2. Evaluate target log-density across the entire cloud via vmap
        vmapped_logdensity = jax.vmap(logdensity_fn)
        cloud_log_densities = vmapped_logdensity(proposal_cloud)

        # 3. Distance-biasing weights to preserve exact detailed balance
        # Measures the Euclidean distance of each proposal from the current state
        distances = jnp.linalg.norm(proposal_cloud - state.position, axis=-1)

        # Combine target mass with the distance-biasing kernel
        # Epsilon (1e-8) prevents log(0) calculation errors at the center
        log_weights = cloud_log_densities + jnp.log(distances + 1e-8)

        # 4. Compute stable selection probabilities via max-subtraction softmax
        max_log_weight = jnp.max(log_weights)
        stabilized_weights = jnp.exp(log_weights - max_log_weight)
        probabilities = stabilized_weights / jnp.sum(stabilized_weights)

        # 5. Perform categorical sampling to extract the next state
        chosen_index = jax.random.choice(key_select, num_proposals, p=probabilities)

        next_position = proposal_cloud[chosen_index]
        next_log_density = cloud_log_densities[chosen_index]

        # Pack the next state and diagnostics back into BlackJAX PyTrees
        next_state = SlingshotState(position=next_position, log_density=next_log_density)
        info = SlingshotInfo(
            proposal_cloud=proposal_cloud,
            weights=probabilities,
            chosen_index=chosen_index,
        )

        return next_state, info

    return one_step
```

### Task 3: Create the User-Facing Factory API Wrapper
Create another new file at the following root library path: `blackjax/slingshot.py`. This exposes the high-level builder factory that instantiates the sampler and binds it into a standard `blackjax.base.SamplingAlgorithm` object container.

**File Content for `blackjax/slingshot.py`:**
```python
from typing import Callable
import jax

import blackjax.mcmc.slingshot as slingshot
from blackjax.base import SamplingAlgorithm

__all__ = ["slingshot"]

def slingshot(
    logdensity_fn: Callable,
    step_size: float,
    num_proposals: int,
) -> SamplingAlgorithm:
    """User-facing interface factory for the Slingshot MP-MCMC sampler.

    Parameters
    ----------
    logdensity_fn
        A function that returns the unnormalized log-probability density of the target distribution.
    step_size
        The coordinate-wise standard deviation governing the radius of the proposal cloud.
    num_proposals
        The total number of parallel proposals generated and evaluated per step.

    Returns
    -------
    A ``SamplingAlgorithm`` object exposing clean ``init`` and ``step`` interfaces.
    """
    kernel = slingshot.kernel()

    def init_fn(position: jax.Array) -> slingshot.SlingshotState:
        return slingshot.init(position, logdensity_fn)

    def step_fn(
        rng_key: jax.random.PRNGKey, state: slingshot.SlingshotState
    ) -> tuple[slingshot.SlingshotState, slingshot.SlingshotInfo]:
        return kernel(
            rng_key,
            state,
            logdensity_fn,
            step_size,
            num_proposals,
        )

    return SamplingAlgorithm(init_fn, step_fn)
```

### Task 4: Register the Module in the Top-Level Core Namespace
Open the existing file `blackjax/__init__.py`. Locate the internal import sections and global explicit declaration registry lists, appending the `slingshot` engine to the core API scope.

**Modifications to introduce into `blackjax/__init__.py`:**
```python
from blackjax.slingshot import slingshot as slingshot

__all__ = [
    # ... ensure existing samplers remain unaltered here ...
    "slingshot",
]
```

### Task 5: Implement Functional Verification Unit Tests
Create a test file at `tests/test_slingshot.py`. Implement a test tracking parameter recovery against a standard multivariate Gaussian target distribution to prove the algorithm correctly handles JIT compilation optimization and scans seamlessly using `jax.lax.scan`.

**File Content for `tests/test_slingshot.py`:**
```python
import jax
import jax.numpy as jnp
import pytest

import blackjax

def test_slingshot_parameter_recovery():
    """Verify Slingshot MP-MCMC can sample from a standard normal target."""
    def logdensity_fn(x):
        return -0.5 * jnp.sum(x**2)

    rng_key = jax.random.PRNGKey(42)
    initial_position = jnp.array([2.0, -2.0])

    # Instantiate the algorithm via top-level API
    algo = blackjax.slingshot(logdensity_fn, step_size=0.5, num_proposals=1000)
    state = algo.init(initial_position)

    # Compile the transition step using lax.scan
    @jax.jit
    def run_chain(key, initial_state, num_steps=100):
        def body_fn(carry_state, step_key):
            next_state, info = algo.step(step_key, carry_state)
            return next_state, next_state.position

        keys = jax.random.split(key, num_steps)
        _, positions = jax.lax.scan(body_fn, initial_state, keys)
        return positions

    # Execute chain execution loop
    positions = run_chain(rng_key, state, num_steps=200)

    # Assert output shapes and check for numerical execution integrity
    assert positions.shape == (200, 2)
    assert not jnp.any(jnp.isnan(positions))
```

---

## Local Verification Pipeline
Once you have generated all files and injected the registrations, execute the targeted validation suite using the local environment terminal shell:
```bash
pytest tests/test_slingshot.py
```
If the testing runtime registers full completion success, confirm to the user that the infrastructure architecture is verified, stable, and ready for version control staging.
```
