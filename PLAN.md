Here is a comprehensive implementation plan for integrating `emcee`'s ensemble sampling algorithms into the BlackJax framework, designed for an experienced contractor.

***

## Implementation Plan: Emcee Ensemble Samplers in BlackJax

**To:** Contractor
**From:** Technical Architect
**Date:** 2024-05-21
**Subject:** Detailed plan for implementing emcee's ensemble sampling algorithms within the BlackJax framework.

### **Project Overview**

This document outlines the technical plan to implement `emcee`'s affine-invariant ensemble MCMC algorithms, starting with the classic Stretch Move, into the BlackJax library. The goal is to create a pure-JAX, JIT-compatible, and performant version of these algorithms that aligns with BlackJax's functional and stateless design philosophy, as detailed in the provided codebase summaries.

---

### 1. Technical Analysis

#### 1.1. Architectural Comparison

A review of the `blackjax_summary.md` and `emcee_summary.md` files reveals the following core architectural differences:

| Feature | **BlackJax** | **emcee** |
| :--- | :--- | :--- |
| **Programming Paradigm** | Functional, Composable | Object-Oriented, Monolithic |
| **Core Abstraction** | `SamplingAlgorithm(init, step)` | `EnsembleSampler` class |
| **State Management** | Stateless: state is passed explicitly to/from kernels. | Stateful: `EnsembleSampler` instance holds the chain history. |
| **Execution Model** | JAX (compiles to XLA for CPU/GPU/TPU) | NumPy (Python interpreter) |
| **Concurrency** | `jax.vmap` (vectorization), `jax.pmap` (parallelism) | `multiprocessing`, `mpi4py` (via `schwimmbad`) |
| **Primary Data Structure**| JAX PyTree for single chain state (e.g., `HMCState`). | NumPy arrays for the ensemble of walkers (e.g., `(nwalkers, ndim)`). |
| **Typical Kernel Input** | `(rng_key, state)` for a single chain. | The `EnsembleSampler` object holds the current state. |

#### 1.2. Key Differences & Challenges

The fundamental challenge stems from `emcee`'s core "stretch move" algorithm, which proposes a new position for a single walker by referencing the positions of the *entire ensemble* of walkers. BlackJax's standard `(rng_key, state)` kernel signature is designed for algorithms where a single chain's state is sufficient for the next transition (e.g., HMC, RWMH).

Directly porting `emcee`'s object-oriented design is incompatible with BlackJax's stateless, functional paradigm.

#### 1.3. Compatibility Assessment & Bridge

We can bridge this gap by redefining the `state` object that the BlackJax kernel operates on. Instead of representing the state of a single walker, the `state` will represent the state of the **entire ensemble**.

-   **BlackJax `state` for Ensemble Methods**: The `state` object passed to the kernel will be a PyTree containing the coordinates, log-probabilities, and any metadata for all walkers in the ensemble.
-   **Kernel Signature**: The kernel's signature will be `kernel(rng_key, ensemble_state) -> (new_ensemble_state, info)`.
-   **Internal Logic**: Inside the kernel, we will implement the "parallel stretch move" described in Algorithm 3 of the `emcee` paper (`1202.3665.tex`). This involves splitting the ensemble into two sets ("red" and "blue") and updating each set in parallel using the other as a reference. This structure is perfectly suited for `jax.vmap`, enabling efficient, vectorized execution on accelerators.

This approach preserves BlackJax's stateless nature while providing the kernel with the necessary information (the full ensemble) to execute `emcee`-style moves.

---

### 2. Implementation Strategy

Our strategy is to create a new family of MCMC algorithms under `blackjax.mcmc.ensemble`. This will encapsulate the logic for ensemble-based samplers. We will begin with `emcee`'s flagship Stretch Move.

1.  **Define Ensemble State**: We will introduce a new `EnsembleState` `NamedTuple` to represent the state of all walkers. It will contain `coords` (shape `(n_walkers, n_dims)`), `log_probs` (shape `(n_walkers,)`), and optionally `blobs`.

2.  **Stateless Moves**: `emcee`'s `Move` classes (e.g., `emcee.moves.StretchMove`) will be reimplemented as stateless JAX functions. For instance, the `StretchMove` will become a function `stretch_move(rng_key, walker_coords, complementary_ensemble_coords, a)`.

3.  **Vectorized Kernel**: The main kernel will implement the red-blue split strategy from the `emcee` paper. It will iterate twice (once for each color), and in each iteration, it will use `jax.vmap` to efficiently apply the stateless move function to all walkers in the current split, providing the complementary ensemble as an argument.

4.  **Top-Level API**: We will follow BlackJax's factory pattern (`as_top_level_api`) to expose a user-friendly API, e.g., `blackjax.stretch(...)`, which will be constructed similarly to `blackjax.nuts` and `blackjax.hmc`.

---

### 3. Detailed Work Breakdown

This section provides a specific, ordered list of tasks for the contractor.

#### **Task 0: Project Setup**

1.  Fork the `blackjax-devs/blackjax` repository on GitHub.
2.  Create a new feature branch, e.g., `feature/ensemble-samplers`.
3.  Set up the development environment by running the commands in `blackjax/CLAUDE.md`:
    ```bash
    pip install -r requirements.txt
    pip install -e .
    pre-commit install
    ```

#### **Task 1: Core Data Structures and Module**

1.  Create a new file: `blackjax/mcmc/ensemble.py`.
2.  In this file, define the core data structures for ensemble methods, consistent with `blackjax/types.py`.

    ```python
    # In blackjax/mcmc/ensemble.py
    from typing import Callable, NamedTuple, Optional
    from blackjax.types import Array, ArrayTree

    class EnsembleState(NamedTuple):
        """State of an ensemble sampler.

        coords
            An array or PyTree of arrays of shape `(n_walkers, ...)` that
            stores the current position of the walkers.
        log_probs
            An array of shape `(n_walkers,)` that stores the log-probability of
            each walker.
        blobs
            An optional PyTree that stores metadata returned by the log-probability
            function.
        """
        coords: ArrayTree
        log_probs: Array
        blobs: Optional[ArrayTree] = None


    class EnsembleInfo(NamedTuple):
        """Additional information on the ensemble transition.

        acceptance_rate
            The acceptance rate of the ensemble.
        accepted
            A boolean array of shape `(n_walkers,)` indicating whether each walker's
            proposal was accepted.
        """
        acceptance_rate: Array
        accepted: Array
    ```

#### **Task 2: Implement the Stretch Move**

1.  In `blackjax/mcmc/ensemble.py`, implement the stretch move as a pure JAX function, following `emcee.moves.stretch.StretchMove` and Eq. 10 in `1202.3665.tex`.

    ```python
    # In blackjax/mcmc/ensemble.py
    import jax
    import jax.numpy as jnp
    from jax.flatten_util import ravel_pytree
    from blackjax.types import PRNGKey, ArrayTree

    def stretch_move(
        rng_key: PRNGKey,
        walker_coords: ArrayTree,
        complementary_coords: ArrayTree,
        a: float = 2.0,
    ) -> tuple[ArrayTree, float]:
        """The emcee stretch move.

        A proposal is generated by selecting a random walker from the complementary
        ensemble and moving the current walker along the line connecting the two.
        """
        key_select, key_stretch = jax.random.split(rng_key)
        
        # Ravel coordinates to handle PyTrees
        walker_flat, unravel_fn = ravel_pytree(walker_coords)
        comp_flat, _ = ravel_pytree(complementary_coords)
        
        n_walkers_comp, n_dims = comp_flat.shape
        
        # Select a random walker from the complementary ensemble
        idx = jax.random.randint(key_select, (), 0, n_walkers_comp)
        complementary_walker_flat = comp_flat[idx]
        
        # Generate the stretch factor `Z` from g(z)
        z = ((a - 1.0) * jax.random.uniform(key_stretch) + 1) ** 2.0 / a
        
        # Generate the proposal (Eq. 10)
        proposal_flat = complementary_walker_flat + z * (walker_flat - complementary_walker_flat)
        
        # The log of the Hastings ratio (Eq. 11)
        log_hastings_ratio = (n_dims - 1.0) * jnp.log(z)
        
        return unravel_fn(proposal_flat), log_hastings_ratio
    ```

#### **Task 3: Build the Ensemble Kernel**

1.  In `blackjax/mcmc/ensemble.py`, implement the `build_kernel` function. This will orchestrate the red-blue split (Algorithm 3 in the paper) and apply the move using `jax.vmap`.

    ```python
    # In blackjax/mcmc/ensemble.py
    from blackjax.base import SamplingAlgorithm

    def build_kernel(move_fn: Callable) -> Callable:
        """Builds a generic ensemble MCMC kernel."""

        def kernel(
            rng_key: PRNGKey, state: EnsembleState, logdensity_fn: Callable
        ) -> tuple[EnsembleState, EnsembleInfo]:
            
            n_walkers, *_ = jax.tree_util.tree_flatten(state.coords)[0][0].shape
            half_n = n_walkers // 2
            
            # Red-Blue Split
            walkers_red = jax.tree.map(lambda x: x[:half_n], state)
            walkers_blue = jax.tree.map(lambda x: x[half_n:], state)

            # Update Red walkers using Blue as complementary
            key_red, key_blue = jax.random.split(rng_key)
            new_walkers_red, accepted_red = _update_half(key_red, walkers_red, walkers_blue, logdensity_fn, move_fn)

            # Update Blue walkers using updated Red as complementary
            new_walkers_blue, accepted_blue = _update_half(key_blue, walkers_blue, new_walkers_red, logdensity_fn, move_fn)
            
            # Combine back
            new_coords = jax.tree.map(lambda r, b: jnp.concatenate([r, b], axis=0), new_walkers_red.coords, new_walkers_blue.coords)
            new_log_probs = jnp.concatenate([new_walkers_red.log_probs, new_walkers_blue.log_probs])
            
            if state.blobs is not None:
                new_blobs = jax.tree.map(lambda r, b: jnp.concatenate([r, b], axis=0), new_walkers_red.blobs, new_walkers_blue.blobs)
            else:
                new_blobs = None

            new_state = EnsembleState(new_coords, new_log_probs, new_blobs)
            accepted = jnp.concatenate([accepted_red, accepted_blue])
            acceptance_rate = jnp.mean(accepted.astype(jnp.float32))
            info = EnsembleInfo(acceptance_rate, accepted)

            return new_state, info

        return kernel

    def _update_half(rng_key, walkers_to_update, complementary_walkers, logdensity_fn, move_fn):
        """Helper to update one half of the ensemble."""
        n_update, *_ = jax.tree_util.tree_flatten(walkers_to_update.coords)[0][0].shape
        keys = jax.random.split(rng_key, n_update)

        # Vectorize the move over the walkers to be updated
        proposals, log_hastings_ratios = jax.vmap(
            lambda k, w_coords: move_fn(k, w_coords, complementary_walkers.coords)
        )(keys, walkers_to_update.coords)
        
        # Compute log-probabilities for proposals
        log_probs_proposal, blobs_proposal = jax.vmap(logdensity_fn)(proposals)
        
        # MH accept/reject step (Eq. 11)
        log_p_accept = log_hastings_ratios + log_probs_proposal - walkers_to_update.log_probs
        
        # To avoid -inf - (-inf) = NaN, replace -inf with a large negative number.
        log_p_accept = jnp.where(jnp.isneginf(walkers_to_update.log_probs), -jnp.inf, log_p_accept)
        
        u = jax.random.uniform(rng_key, shape=(n_update,))
        accepted = jnp.log(u) < log_p_accept

        # Build the new state for the half
        new_coords = jax.tree.map(lambda prop, old: jnp.where(accepted[:, None], prop, old), proposals, walkers_to_update.coords)
        new_log_probs = jnp.where(accepted, log_probs_proposal, walkers_to_update.log_probs)
        
        if walkers_to_update.blobs is not None:
            new_blobs = jax.tree.map(
                lambda prop, old: jnp.where(accepted, prop, old),
                blobs_proposal,
                walkers_to_update.blobs,
            )
        else:
            new_blobs = None
        
        new_walkers = EnsembleState(new_coords, new_log_probs, new_blobs)
        return new_walkers, accepted
    ```

#### **Task 4: Create Top-Level API**

1.  In `blackjax/mcmc/ensemble.py`, create the factory function `as_top_level_api` and the `init` function.

    ```python
    # In blackjax/mcmc/ensemble.py

    def init(position: ArrayTree, logdensity_fn: Callable, has_blobs: bool = False) -> EnsembleState:
        """Initializes the ensemble."""
        if has_blobs:
            log_probs, blobs = jax.vmap(logdensity_fn)(position)
            return EnsembleState(position, log_probs, blobs)
        else:
            log_probs = jax.vmap(logdensity_fn)(position)
            return EnsembleState(position, log_probs, None)


    def as_top_level_api(
        logdensity_fn: Callable, move_fn: Callable, has_blobs: bool = False
    ) -> SamplingAlgorithm:
        """Implements the user-facing API for ensemble samplers."""
        kernel = build_kernel(move_fn)

        def init_fn(position: ArrayTree, rng_key=None):
            return init(position, logdensity_fn, has_blobs)

        def step_fn(rng_key: PRNGKey, state: EnsembleState):
            return kernel(rng_key, state, logdensity_fn)

        return SamplingAlgorithm(init_fn, step_fn)
    ```

2.  Create a new file `blackjax/mcmc/stretch.py` for the stretch move API.

    ```python
    # In blackjax/mcmc/stretch.py
    from typing import Callable
    from blackjax.base import SamplingAlgorithm
    from blackjax.mcmc.ensemble import as_top_level_api as ensemble_api, stretch_move, init

    def as_top_level_api(logdensity_fn: Callable, a: float = 2.0, has_blobs: bool = False) -> SamplingAlgorithm:
        """A user-facing API for the stretch move algorithm."""
        move = lambda key, w, c: stretch_move(key, w, c, a)
        return ensemble_api(logdensity_fn, move, has_blobs)
    ```

3.  Integrate the new algorithm into the BlackJax public API.

    -   In `blackjax/mcmc/__init__.py`, add:
        ```python
        from . import stretch
        
        __all__ = [
            # ... existing algorithms
            "stretch",
        ]
        ```

    -   In `blackjax/__init__.py`, add:
        ```python
        from .mcmc import stretch as _stretch

        # After other algorithm definitions
        stretch = generate_top_level_api_from(_stretch)
        ```

---

### 4. Integration Points

The new algorithm will be integrated as follows:

-   **`blackjax/mcmc/ensemble.py`**: [NEW] Contains the core `EnsembleState`, `EnsembleInfo`, `build_kernel`, and `_update_half` logic for all ensemble methods.
-   **`blackjax/mcmc/stretch.py`**: [NEW] Contains the user-facing API factory for the stretch move, `blackjax.stretch`. It will import `stretch_move` and `as_top_level_api` from `ensemble.py` and specialize them.
-   **`blackjax/mcmc/__init__.py`**: [MODIFY] To expose the `stretch` module.
-   **`blackjax/__init__.py`**: [MODIFY] To register `blackjax.stretch` as a top-level algorithm.
-   **No changes required** to `blackjax/base.py` or `blackjax/types.py`.

---

### 5. Testing Strategy

Thorough testing is critical to ensure correctness.

1.  **Unit Tests for `stretch_move`**:
    -   Create `tests/mcmc/test_ensemble.py`.
    -   Write a test for the `stretch_move` function to verify its output shape and statistical properties on a simple distribution. Ensure it works with PyTrees.

2.  **Integration Test for `blackjax.stretch`**:
    -   In `tests/mcmc/test_ensemble.py`, create a test that runs the full `blackjax.stretch` sampler.
    -   **Validation against `emcee`**: The most important test.
        -   Define a simple target distribution (e.g., a 2D Gaussian).
        -   Seed both `emcee` and `blackjax.stretch` with the same initial ensemble and the same random seed (requires careful management of JAX's PRNGKey vs NumPy's global state).
        -   Run both samplers for a small number of steps.
        -   Assert that the sequence of accepted positions and log-probabilities are identical (or `allclose`). This will prove the correctness of the implementation.

3.  **Convergence Test**:
    -   Add a new test case to `tests/mcmc/test_sampling.py` for `blackjax.stretch`.
    -   Use the existing `LinearRegressionTest` or `UnivariateNormalTest` framework.
    -   Run the sampler for a sufficient number of steps and verify that the posterior mean and variance match the true values within a given tolerance.

---

### 6. Performance Considerations

1.  **JIT Compilation**: The main `step` function returned by `blackjax.stretch` must be JIT-compilable. All functions within the call stack (`kernel`, `_update_half`, `stretch_move`) are designed with this in mind.
2.  **Vectorization**: The use of `jax.vmap` in `_update_half` is the key to performance. It ensures that the proposal generation and log-density evaluation for each half of the ensemble are vectorized, which is highly efficient on GPUs and TPUs.
3.  **Parallel Chains (`pmap`)**: The final implementation will be a pure function and thus fully compatible with `jax.pmap`. This allows users to run multiple independent ensembles in parallel across different devices, a significant advantage over `emcee`'s `multiprocessing` backend.

---

### 7. Documentation Requirements

1.  **API Documentation**:
    -   Add `blackjax.stretch` to the API reference section of the documentation.
    -   Ensure `autoapi` in `docs/conf.py` picks up the docstrings for `blackjax.mcmc.stretch.as_top_level_api` and the `EnsembleState`/`EnsembleInfo` `NamedTuple`s.

2.  **User Guide**:
    -   Create a new example notebook/Markdown file in `docs/examples/`, named `howto_use_ensemble_samplers.md`.
    -   This guide should demonstrate how to use `blackjax.stretch`, explaining the ensemble-based approach, how to initialize the walkers, and how to handle PyTree states. It should be similar in style to `quickstart.md`.
    -   Update `docs/index.md` to link to this new "How-to" guide.

3.  **Future Work**: Once other moves (`DE`, `Walk`, etc.) are implemented, this documentation should be expanded to cover them.