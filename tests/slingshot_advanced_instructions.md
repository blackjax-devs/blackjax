# Agent Instructions: Upgrading Slingshot MP-MCMC with Advanced JAX Extensions

You are an expert open-source contributor upgrading the **Slingshot MP-MCMC (Mad Props)** sampler in BlackJAX. Your task is to implement three high-performance features:
1. **Exact Metropolis-Hastings Correction**: Ensures mathematical exactness and detailed balance for finite proposal counts (P) by evaluating a vectorized reverse-proposal cloud.
2. **Adaptive Step-Size Engine**: Employs a Nesterov dual-averaging scheme during warmup to tune the proposal cloud radius automatically to a target acceptance rate.
3. **Multi-Chain Parallelization Engine**: Leverages nested `jax.vmap` loops to execute dozens of independent sampling chains simultaneously across GPU cores.

---

## Technical Execution Checklist

### Task 1: Update the Low-Level Mathematical Engine
Modify `blackjax/mcmc/slingshot.py`. We will expand `SlingshotInfo` to track acceptance diagnostics, and rewrite the transition kernel to evaluate both a forward proposal cloud and a reverse validation cloud to enforce exact detailed balance for any finite P.

**File Content for `blackjax/mcmc/slingshot.py`:**
```python
import jax
import jax.numpy as jnp
from typing import Callable, NamedTuple

class SlingshotState(NamedTuple):
    """State of the Slingshot MP-MCMC sampler."""
    position: jnp.ndarray
    log_density: float

class SlingshotInfo(NamedTuple):
    """Internal diagnostics for the exact Slingshot transition step."""
    proposal_cloud: jnp.ndarray
    weights: jnp.ndarray
    chosen_index: jnp.ndarray
    acceptance_rate: float
    is_accepted: bool

class SlingshotAdaptState(NamedTuple):
    """State parameters for Nesterov dual-averaging step-size tuning."""
    log_step_size: float
    log_step_size_bar: float
    h_bar: float
    t: int

def init(position: jnp.ndarray, logdensity_fn: Callable) -> SlingshotState:
    """Initialize the Slingshot sampler state from a starting position."""
    return SlingshotState(position=position, log_density=logdensity_fn(position))

def init_adaptation(initial_step_size: float) -> SlingshotAdaptState:
    """Initialize dual-averaging state parameters."""
    log_ss = jnp.log(initial_step_size)
    return SlingshotAdaptState(
        log_step_size=log_ss,
        log_step_size_bar=log_ss,
        h_bar=0.0,
        t=0
    )

def dual_averaging_step(
    adapt_state: SlingshotAdaptState,
    acceptance_rate: float,
    target_rate: float = 0.65,
    gamma: float = 0.05,
    t0: int = 10,
    kappa: float = 0.75
) -> SlingshotAdaptState:
    """Update step size logs dynamically using Nesterov dual averaging."""
    t = adapt_state.t + 1
    alpha = target_rate - acceptance_rate
    h_bar = (1.0 - 1.0 / (t + t0)) * adapt_state.h_bar + (1.0 / (t + t0)) * alpha
    
    log_step_size = - (jnp.sqrt(t) / gamma) * h_bar
    eta = t ** (-kappa)
    log_step_size_bar = (1.0 - eta) * adapt_state.log_step_size_bar + eta * log_step_size
    
    return SlingshotAdaptState(
        log_step_size=log_step_size,
        log_step_size_bar=log_step_size_bar,
        h_bar=h_bar,
        t=t
    )

def kernel() -> Callable:
    """Build the functional transition kernel for the Slingshot sampler.

    Evaluates both forward and reverse clouds via jax.vmap to run an exact
    Metropolis-Hastings validation step matching finite-proposal conditions.
    """
    def one_step(
        rng_key: jax.random.PRNGKey,
        state: SlingshotState,
        logdensity_fn: Callable,
        step_size: float,
        num_proposals: int,
    ) -> tuple[SlingshotState, SlingshotInfo]:
        
        key_cloud, key_select, key_accept, key_reverse = jax.random.split(rng_key, 4)
        dim = state.position.shape[0]
        vmapped_logdensity = jax.vmap(logdensity_fn)
        
        # 1. Forward Pass: Generate and evaluate proposal cloud
        noise = jax.random.normal(key_cloud, shape=(num_proposals, dim))
        proposal_cloud = state.position + noise * step_size
        cloud_log_densities = vmapped_logdensity(proposal_cloud)
        
        distances = jnp.linalg.norm(proposal_cloud - state.position, axis=-1)
        log_weights = cloud_log_densities + jnp.log(distances + 1e-8)
        
        max_log_weight = jnp.max(log_weights)
        stabilized_weights = jnp.exp(log_weights - max_log_weight)
        sum_forward_weights = jnp.sum(stabilized_weights)
        probabilities = stabilized_weights / sum_forward_weights
        
        # 2. Extract Candidate Proposal
        chosen_index = jax.random.choice(key_select, num_proposals, p=probabilities)
        candidate_position = proposal_cloud[chosen_index]
        candidate_log_density = cloud_log_densities[chosen_index]
        
        # 3. Reverse Pass: Generate validation cloud around candidate position
        reverse_noise = jax.random.normal(key_reverse, shape=(num_proposals - 1, dim))
        reverse_cloud_minus_one = candidate_position + reverse_noise * step_size
        reverse_cloud = jnp.vstack([reverse_cloud_minus_one, state.position]) # Append original state
        
        reverse_log_densities = vmapped_logdensity(reverse_cloud)
        reverse_distances = jnp.linalg.norm(reverse_cloud - candidate_position, axis=-1)
        reverse_log_weights = reverse_log_densities + jnp.log(reverse_distances + 1e-8)
        
        max_rev_weight = jnp.max(reverse_log_weights)
        stabilized_rev_weights = jnp.exp(reverse_log_weights - max_rev_weight)
        sum_reverse_weights = jnp.sum(stabilized_rev_weights)
        
        # 4. Exact Finite-P Metropolis Acceptance Correction
        log_accept_ratio = (max_log_weight + jnp.log(sum_forward_weights)) - (max_rev_weight + jnp.log(sum_reverse_weights))
        acceptance_rate = jnp.minimum(1.0, jnp.exp(log_accept_ratio))
        
        is_accepted = jax.random.uniform(key_accept) < acceptance_rate
        
        # 5. Resolve State Updates
        next_position = jnp.where(is_accepted, candidate_position, state.position)
        next_log_density = jnp.where(is_accepted, candidate_log_density, state.log_density)
        
        next_state = SlingshotState(position=next_position, log_density=next_log_density)
        info = SlingshotInfo(
            proposal_cloud=proposal_cloud,
            weights=probabilities,
            chosen_index=chosen_index,
            acceptance_rate=acceptance_rate,
            is_accepted=is_accepted
        )
        
        return next_state, info
        
    return one_step
```

### Task 2: Update the User-Facing Wrapper API
Modify `blackjax/slingshot.py`. We will extend the high-level factory to expose both the standard initialization boundaries and helper hooks for adaptation states.

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
    """User-facing interface factory for the exact Slingshot MP-MCMC sampler."""
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

### Task 3: Implement Multi-Chain, Adaptive Verification Unit Tests
Overwrite `tests/test_slingshot.py`. This test suite verifies the exact correction mechanics, runs dual-averaging step-size adaptation, and uses nested `jax.vmap` structures to run 16 independent chains in parallel across the GPU hardware.

**File Content for `tests/test_slingshot.py`:**
```python
import jax
import jax.numpy as jnp
import pytest

import blackjax
import blackjax.mcmc.slingshot as slingshot_core

def test_slingshot_multi_chain_and_adaptation():
    """Verify exact execution, adaptation updates, and multi-chain vmap loops."""
    
    # 1. Define standard Gaussian Target
    def logdensity_fn(x):
        return -0.5 * jnp.sum(x**2)

    num_chains = 16
    num_proposals = 500
    warmup_steps = 50
    sampling_steps = 100

    master_key = jax.random.PRNGKey(123)
    key_init, key_warmup, key_sample = jax.random.split(master_key, 3)

    # 2. Setup Multi-Chain Initial Vector Positions
    initial_positions = jax.random.normal(key_init, shape=(num_chains, 2)) * 3.0
    
    # Vectorize single-chain initializers via jax.vmap
    init_vmap = jax.vmap(lambda p: slingshot_core.init(p, logdensity_fn))
    initial_states = init_vmap(initial_positions)
    
    # Initialize Multi-Chain Adaptation states
    init_adapt_vmap = jax.vmap(slingshot_core.init_adaptation)
    adapt_states = init_adapt_vmap(jnp.ones(num_chains) * 1.0)

    # 3. Phase A: Warmup Optimization Loop (Adapting Step Sizes Across Chains)
    def warmup_loop(carry, step_key):
        states, a_states = carry
        keys = jax.random.split(step_key, num_chains)
        
        # Extract individual step sizes from logs
        current_step_sizes = jnp.exp(a_states.log_step_size)
        
        # Vectorized step execution across all 16 chains simultaneously
        def vmapped_step(k, s, ss):
            algo = blackjax.slingshot(logdensity_fn, step_size=ss, num_proposals=num_proposals)
            return algo.step(k, s)
            
        next_states, infos = jax.vmap(vmapped_step)(keys, states, current_step_sizes)
        
        # Vectorized dual-averaging optimization step
        next_adapt_states = jax.vmap(slingshot_core.dual_averaging_step)(infos.acceptance_rate)
        
        return (next_states, next_adapt_states), None

    warmup_keys = jax.random.split(key_warmup, warmup_steps)
    (w_states, final_adapt_states), _ = jax.lax.scan(warmup_loop, (initial_states, adapt_states), warmup_keys)

    # Extract final optimized step sizes per chain
    optimized_step_sizes = jnp.exp(final_adapt_states.log_step_size_bar)

    # 4. Phase B: Production Sampling Loop (Fixed Optimized Step Sizes)
    def sampling_loop(carry_states, step_key):
        keys = jax.random.split(step_key, num_chains)
        
        def vmapped_step(k, s, ss):
            algo = blackjax.slingshot(logdensity_fn, step_size=ss, num_proposals=num_proposals)
            return algo.step(k, s)
            
        next_states, infos = jax.vmap(vmapped_step)(keys, carry_states, optimized_step_sizes)
        return next_states, next_states.position

    sample_keys = jax.random.split(key_sample, sampling_steps)
    _, trace_positions = jax.lax.scan(sampling_loop, w_states, sample_keys)

    # 5. Asset Assertions tracking dimension integrity and sanity values
    # Expected Trace Shape: (sampling_steps, num_chains, dimensions) -> (100, 16, 2)
    assert trace_positions.shape == (sampling_steps, num_chains, 2)
    assert not jnp.any(jnp.isnan(trace_positions))
    assert optimized_step_sizes.shape == (num_chains,)
```

---

## Local Verification Pipeline
Once all target updates are integrated, trigger the verification test suite in the workspace shell terminal:
```bash
pytest tests/test_slingshot.py
```
Confirm to the user the moment execution completes successfully.
```