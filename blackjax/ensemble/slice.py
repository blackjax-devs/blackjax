# Copyright 2020- The Blackjax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Ensemble Slice Sampling (ESS) implementation."""
from typing import Callable, NamedTuple, Optional

import jax
import jax.numpy as jnp

from blackjax.base import SamplingAlgorithm
from blackjax.types import ArrayLikeTree, ArrayTree, PRNGKey

__all__ = [
    "SliceEnsembleState",
    "SliceEnsembleInfo",
    "init",
    "build_kernel",
    "as_top_level_api",
]


class SliceEnsembleState(NamedTuple):
    """State of the ensemble slice sampler.

    coords
        An array or PyTree of arrays of shape `(n_walkers, ...)` that
        stores the current position of the walkers.
    log_probs
        An array of shape `(n_walkers,)` that stores the log-probability of
        each walker.
    blobs
        An optional PyTree that stores metadata returned by the log-probability
        function.
    mu
        The current scale parameter for the slice sampling directions.
    tuning_active
        Whether adaptive tuning of mu is currently active.
    patience_count
        Counter for determining when to stop tuning.
    """

    coords: ArrayTree
    log_probs: jnp.ndarray
    blobs: Optional[ArrayTree] = None
    mu: float = 1.0
    tuning_active: bool = True
    patience_count: int = 0


class SliceEnsembleInfo(NamedTuple):
    """Additional information on the ensemble slice sampling transition.

    acceptance_rate
        Always 1.0 for slice sampling (no rejections).
    is_accepted
        A boolean array of shape `(n_walkers,)` - always True for slice sampling.
    expansions
        Total number of slice expansions performed.
    contractions
        Total number of slice contractions performed.
    nevals
        Total number of log-density evaluations performed.
    mu
        The current value of the scale parameter mu.
    """

    acceptance_rate: float
    is_accepted: jnp.ndarray
    expansions: int
    contractions: int
    nevals: int
    mu: float


def differential_direction(
    rng_key: PRNGKey,
    complementary_coords: ArrayTree,
    n_update: int,
    mu: float,
) -> tuple[ArrayTree, bool]:
    """Generate direction vectors using the differential move.

    Directions are defined by the difference between two randomly selected
    walkers from the complementary ensemble, scaled by 2*mu.

    Parameters
    ----------
    rng_key
        A PRNG key for random number generation.
    complementary_coords
        Coordinates of the complementary ensemble with shape (n_walkers, ...).
    n_update
        Number of walkers to update (number of directions needed).
    mu
        The scale parameter.

    Returns
    -------
    A tuple (directions, tune_once) where directions is a PyTree matching
    the structure of complementary_coords with leading dimension n_update,
    and tune_once is True (indicating mu should be tuned).
    """
    # Get the number of walkers in complementary ensemble
    comp_leaves, _ = jax.tree_util.tree_flatten(complementary_coords)
    n_comp = comp_leaves[0].shape[0]

    # Sample two different indices for each update walker
    key1, key2 = jax.random.split(rng_key)
    idx1 = jax.random.randint(key1, (n_update,), 0, n_comp)
    j = jax.random.randint(key2, (n_update,), 0, n_comp - 1)
    idx2 = j + (j >= idx1)  # Ensure idx2 != idx1

    # Compute directions as difference of pairs
    walker1 = jax.tree_util.tree_map(lambda x: x[idx1], complementary_coords)
    walker2 = jax.tree_util.tree_map(lambda x: x[idx2], complementary_coords)

    directions = jax.tree_util.tree_map(
        lambda w1, w2: 2.0 * mu * (w1 - w2), walker1, walker2
    )

    return directions, True


def random_direction(
    rng_key: PRNGKey,
    template_coords: ArrayTree,
    n_update: int,
    mu: float,
) -> tuple[ArrayTree, bool]:
    """Generate random isotropic direction vectors.

    Directions are sampled from a standard normal distribution and scaled by 2*mu.
    This corresponds to standard multivariate slice sampling without using
    ensemble information.

    Parameters
    ----------
    rng_key
        A PRNG key for random number generation.
    template_coords
        Template coordinates to match structure and shape (n_update, ...).
    n_update
        Number of walkers to update (number of directions needed).
    mu
        The scale parameter.

    Returns
    -------
    A tuple (directions, tune_once) where directions is a PyTree matching
    the structure of template_coords, and tune_once is True.
    """

    def sample_leaf(shape):
        return jax.random.normal(rng_key, shape)

    # Generate random directions with same structure as template
    directions = jax.tree_util.tree_map(
        lambda x: 2.0 * mu * sample_leaf(x.shape), template_coords
    )

    return directions, True


def slice_along_direction(
    rng_key: PRNGKey,
    x0: ArrayTree,
    logp0: float,
    direction: ArrayTree,
    logprob_fn: Callable,
    maxsteps: int = 10000,
    maxiter: int = 10000,
) -> tuple[ArrayTree, float, int, int, int]:
    """Perform slice sampling along a given direction.

    Implements the stepping-out and shrinking procedures for 1D slice sampling
    along the specified direction vector.

    Parameters
    ----------
    rng_key
        A PRNG key for random number generation.
    x0
        Current position (PyTree).
    logp0
        Log-probability at current position.
    direction
        Direction vector (PyTree with same structure as x0).
    logprob_fn
        Function that computes log-probability given a position.
    maxsteps
        Maximum number of steps for stepping-out procedure.
    maxiter
        Maximum total iterations to prevent infinite loops.

    Returns
    -------
    A tuple (x1, logp1, nexp, ncon, neval) where:
        x1: New position (PyTree)
        logp1: Log-probability at new position
        nexp: Number of expansions performed
        ncon: Number of contractions performed
        neval: Number of log-probability evaluations
    """
    key_z0, key_lr, key_j, key_shrink = jax.random.split(rng_key, 4)

    # Draw slice height: Z0 = logp0 - Exponential(1)
    z0 = logp0 - jax.random.exponential(key_z0)

    # Initialize interval [L, R] around 0
    l_init = -jax.random.uniform(key_lr)
    r_init = l_init + 1.0

    # Random allocation of expansion steps
    j = jax.random.randint(key_j, (), 0, maxsteps)
    k = maxsteps - 1 - j

    # Helper function to evaluate log-prob at x0 + t*direction
    def eval_at_t(t):
        xt = jax.tree_util.tree_map(lambda x, d: x + t * d, x0, direction)
        return logprob_fn(xt)

    # Stepping-out: expand left
    def left_expand_cond(carry):
        l, j_left, nexp, neval, iter_count = carry
        logp_l = eval_at_t(l)
        return (j_left > 0) & (iter_count < maxiter) & (logp_l > z0)

    def left_expand_body(carry):
        l, j_left, nexp, neval, iter_count = carry
        return l - 1.0, j_left - 1, nexp + 1, neval + 1, iter_count + 1

    l_final, _, nexp_left, neval_left, _ = jax.lax.while_loop(
        left_expand_cond, left_expand_body, (l_init, j, 0, 0, 0)
    )

    # Stepping-out: expand right
    def right_expand_cond(carry):
        r, k_right, nexp, neval, iter_count = carry
        logp_r = eval_at_t(r)
        return (k_right > 0) & (iter_count < maxiter) & (logp_r > z0)

    def right_expand_body(carry):
        r, k_right, nexp, neval, iter_count = carry
        return r + 1.0, k_right - 1, nexp + 1, neval + 1, iter_count + 1

    r_final, _, nexp_right, neval_right, _ = jax.lax.while_loop(
        right_expand_cond, right_expand_body, (r_init, k, 0, 0, 0)
    )

    nexp_total = nexp_left + nexp_right
    neval_after_expand = neval_left + neval_right

    # Shrinking: sample uniformly from [L, R] until inside slice
    def shrink_cond(carry):
        _, _, _, _, _, iter_count, accepted, _, _ = carry
        return (~accepted) & (iter_count < maxiter)

    def shrink_body(carry):
        key, l, r, neval, ncon, iter_count, accepted, t_acc, logp_acc = carry
        key, key_t = jax.random.split(key)
        t = jax.random.uniform(key_t, minval=l, maxval=r)
        logp_t = eval_at_t(t)
        neval_new = neval + 1

        # Check if inside slice
        inside_slice = logp_t >= z0
        accepted_new = accepted | inside_slice

        # Update interval or accept
        l_new = jnp.where(inside_slice, l, jnp.where(t < 0, t, l))
        r_new = jnp.where(inside_slice, r, jnp.where(t >= 0, t, r))
        ncon_new = jnp.where(inside_slice, ncon, ncon + 1)
        t_acc_new = jnp.where(inside_slice & ~accepted, t, t_acc)
        logp_acc_new = jnp.where(inside_slice & ~accepted, logp_t, logp_acc)

        return (
            key,
            l_new,
            r_new,
            neval_new,
            ncon_new,
            iter_count + 1,
            accepted_new,
            t_acc_new,
            logp_acc_new,
        )

    _, _, _, neval_shrink, ncon_total, _, _, t_final, logp_final = jax.lax.while_loop(
        shrink_cond,
        shrink_body,
        (key_shrink, l_final, r_final, 0, 0, 0, False, 0.0, logp0),
    )

    # Compute final position
    x1 = jax.tree_util.tree_map(lambda x, d: x + t_final * d, x0, direction)

    neval_total = neval_after_expand + neval_shrink

    return x1, logp_final, nexp_total, ncon_total, neval_total


def init(
    position: ArrayLikeTree,
    logdensity_fn: Callable,
    has_blobs: bool = False,
    mu: float = 1.0,
) -> SliceEnsembleState:
    """Initialize the ensemble slice sampling algorithm.

    Parameters
    ----------
    position
        Initial positions for all walkers, with shape (n_walkers, ...).
    logdensity_fn
        The log-density function to evaluate.
    has_blobs
        Whether the log-density function returns additional metadata (blobs).
    mu
        Initial value of the scale parameter.

    Returns
    -------
    Initial SliceEnsembleState.
    """
    logdensity_outputs = jax.vmap(logdensity_fn)(position)
    if isinstance(logdensity_outputs, tuple):
        log_probs, blobs = logdensity_outputs
        return SliceEnsembleState(position, log_probs, blobs, mu, True, 0)
    else:
        log_probs = logdensity_outputs
        return SliceEnsembleState(position, log_probs, None, mu, True, 0)


def build_kernel(
    move: str = "differential",
    move_fn: Optional[Callable] = None,
    randomize_split: bool = True,
    nsplits: int = 2,
    maxsteps: int = 10000,
    maxiter: int = 10000,
    tune: bool = True,
    patience: int = 5,
    tolerance: float = 0.05,
) -> Callable:
    """Build the ensemble slice sampling kernel.

    Parameters
    ----------
    move
        Type of move to use: "differential" or "random". Ignored if move_fn provided.
    move_fn
        Optional custom move function. If None, uses the specified move type.
    randomize_split
        If True, randomly shuffle walker indices before splitting into groups.
    nsplits
        Number of groups to split the ensemble into. Default is 2.
    maxsteps
        Maximum steps for slice stepping-out procedure.
    maxiter
        Maximum iterations to prevent infinite loops.
    tune
        Whether to enable adaptive tuning of mu.
    patience
        Number of steps within tolerance before stopping tuning.
    tolerance
        Tolerance for expansion/contraction ratio to stop tuning.

    Returns
    -------
    A kernel function that performs one step of ensemble slice sampling.
    """
    # Select move function
    if move_fn is None:
        if move == "differential":
            move_fn = differential_direction
        elif move == "random":
            move_fn = random_direction
        else:
            raise ValueError(f"Unknown move type: {move}")

    # At this point move_fn is guaranteed to be Callable
    selected_move_fn: Callable = move_fn

    def kernel(
        rng_key: PRNGKey, state: SliceEnsembleState, logdensity_fn: Callable
    ) -> tuple[SliceEnsembleState, SliceEnsembleInfo]:
        n_walkers, *_ = jax.tree_util.tree_flatten(state.coords)[0][0].shape

        # Shuffle walkers if requested
        if randomize_split:
            key_shuffle, key_update = jax.random.split(rng_key)
            indices = jax.random.permutation(key_shuffle, n_walkers)
            shuffled_coords = jax.tree_util.tree_map(lambda x: x[indices], state.coords)
            shuffled_log_probs = state.log_probs[indices]
            shuffled_blobs = (
                None
                if state.blobs is None
                else jax.tree_util.tree_map(lambda x: x[indices], state.blobs)
            )
            shuffled_state = SliceEnsembleState(
                shuffled_coords,
                shuffled_log_probs,
                shuffled_blobs,
                state.mu,
                state.tuning_active,
                state.patience_count,
            )
        else:
            key_update = rng_key
            shuffled_state = state
            indices = jnp.arange(n_walkers)

        # Split into groups
        group_size = n_walkers // nsplits
        groups = []
        for i in range(nsplits):
            start_idx = i * group_size
            end_idx = (i + 1) * group_size if i < nsplits - 1 else n_walkers

            group_coords = jax.tree_util.tree_map(
                lambda x: x[start_idx:end_idx], shuffled_state.coords
            )
            group_log_probs = shuffled_state.log_probs[start_idx:end_idx]
            group_blobs = (
                None
                if shuffled_state.blobs is None
                else jax.tree_util.tree_map(
                    lambda x: x[start_idx:end_idx], shuffled_state.blobs
                )
            )
            groups.append(
                SliceEnsembleState(
                    group_coords,
                    group_log_probs,
                    group_blobs,
                    state.mu,
                    state.tuning_active,
                    state.patience_count,
                )
            )

        # Update each group sequentially
        updated_groups = list(groups)
        total_nexp = 0
        total_ncon = 0
        total_neval = 0

        keys = jax.random.split(key_update, nsplits)
        for i in range(nsplits):
            # Build complementary ensemble from other groups
            other_indices = [j for j in range(nsplits) if j != i]
            comp_coords_list = [updated_groups[j].coords for j in other_indices]
            comp_log_probs_list = [updated_groups[j].log_probs for j in other_indices]
            comp_blobs_list = [updated_groups[j].blobs for j in other_indices]

            complementary_coords = jax.tree_util.tree_map(
                lambda *arrays: jnp.concatenate(arrays, axis=0), *comp_coords_list
            )
            complementary_log_probs = jnp.concatenate(comp_log_probs_list, axis=0)

            if state.blobs is not None:
                complementary_blobs = jax.tree_util.tree_map(
                    lambda *arrays: jnp.concatenate(arrays, axis=0), *comp_blobs_list
                )
            else:
                complementary_blobs = None

            complementary = SliceEnsembleState(
                complementary_coords,
                complementary_log_probs,
                complementary_blobs,
                state.mu,
                state.tuning_active,
                state.patience_count,
            )

            # Update this group
            updated_group, nexp, ncon, neval = _update_half_slice(
                keys[i],
                groups[i],
                complementary,
                logdensity_fn,
                selected_move_fn,
                maxsteps,
                maxiter,
            )
            updated_groups[i] = updated_group
            total_nexp += nexp
            total_ncon += ncon
            total_neval += neval

        # Concatenate updated groups
        shuffled_coords = jax.tree_util.tree_map(
            lambda *arrays: jnp.concatenate(arrays, axis=0),
            *[g.coords for g in updated_groups],
        )
        shuffled_log_probs = jnp.concatenate(
            [g.log_probs for g in updated_groups], axis=0
        )

        if state.blobs is not None:
            shuffled_blobs = jax.tree_util.tree_map(
                lambda *arrays: jnp.concatenate(arrays, axis=0),
                *[g.blobs for g in updated_groups],
            )
        else:
            shuffled_blobs = None

        # Unshuffle if needed
        if randomize_split:
            inverse_indices = jnp.argsort(indices)
            new_coords = jax.tree_util.tree_map(
                lambda x: x[inverse_indices], shuffled_coords
            )
            new_log_probs = shuffled_log_probs[inverse_indices]
            if shuffled_blobs is not None:
                new_blobs = jax.tree_util.tree_map(
                    lambda x: x[inverse_indices], shuffled_blobs
                )
            else:
                new_blobs = None
        else:
            new_coords = shuffled_coords
            new_log_probs = shuffled_log_probs
            new_blobs = shuffled_blobs

        # Adaptive tuning of mu
        should_tune = tune & state.tuning_active

        nexp_eff = jnp.maximum(total_nexp, 1)
        mu_tuned = state.mu * 2.0 * nexp_eff / (nexp_eff + total_ncon)

        # Check convergence of tuning
        exp_ratio = total_nexp / jnp.maximum(total_nexp + total_ncon, 1)
        within_tolerance = jnp.abs(exp_ratio - 0.5) < tolerance

        patience_count_updated = jnp.where(
            within_tolerance, state.patience_count + 1, 0
        )
        tuning_active_updated = patience_count_updated < patience

        # Apply tuning updates conditionally
        mu_new = jnp.where(should_tune, mu_tuned, state.mu)
        patience_count_new = jnp.where(
            should_tune, patience_count_updated, state.patience_count
        )
        tuning_active_new = jnp.where(
            should_tune, tuning_active_updated, state.tuning_active
        )

        new_state = SliceEnsembleState(
            new_coords,
            new_log_probs,
            new_blobs,
            mu_new,
            tuning_active_new,
            patience_count_new,
        )

        # Build info (acceptance always 1.0 for slice sampling)
        info = SliceEnsembleInfo(
            acceptance_rate=1.0,
            is_accepted=jnp.ones(n_walkers, dtype=bool),
            expansions=total_nexp,
            contractions=total_ncon,
            nevals=total_neval,
            mu=mu_new,
        )

        return new_state, info

    return kernel


def _update_half_slice(
    rng_key: PRNGKey,
    walkers_to_update: SliceEnsembleState,
    complementary_walkers: SliceEnsembleState,
    logdensity_fn: Callable,
    direction_fn: Callable,
    maxsteps: int,
    maxiter: int,
) -> tuple[SliceEnsembleState, int, int, int]:
    """Update a group of walkers using ensemble slice sampling.

    Parameters
    ----------
    rng_key
        PRNG key for random number generation.
    walkers_to_update
        Group of walkers to update.
    complementary_walkers
        Complementary ensemble used for generating directions.
    logdensity_fn
        Log-density function.
    direction_fn
        Function to generate direction vectors.
    maxsteps
        Maximum steps for stepping-out.
    maxiter
        Maximum iterations.

    Returns
    -------
    Tuple of (updated_group_state, total_expansions, total_contractions, total_evals).
    """
    n_update, *_ = jax.tree_util.tree_flatten(walkers_to_update.coords)[0][0].shape

    # Generate directions
    key_dir, key_slice = jax.random.split(rng_key)
    directions, _ = direction_fn(
        key_dir, complementary_walkers.coords, n_update, walkers_to_update.mu
    )

    # Define logprob-only function
    def logprob_only(x):
        out = logdensity_fn(x)
        return out[0] if isinstance(out, tuple) else out

    # Perform slice sampling for each walker
    keys = jax.random.split(key_slice, n_update)

    def slice_one_walker(key, x0, logp0, direction):
        return slice_along_direction(
            key, x0, logp0, direction, logprob_only, maxsteps, maxiter
        )

    results = jax.vmap(slice_one_walker)(
        keys, walkers_to_update.coords, walkers_to_update.log_probs, directions
    )

    new_coords, new_log_probs, nexp_array, ncon_array, neval_array = results

    # Sum statistics
    total_nexp = jnp.sum(nexp_array)
    total_ncon = jnp.sum(ncon_array)
    total_neval = jnp.sum(neval_array)

    # Handle blobs if needed
    if walkers_to_update.blobs is not None:
        # Re-evaluate at new positions to get blobs
        logdensity_outputs = jax.vmap(logdensity_fn)(new_coords)
        _, new_blobs = logdensity_outputs
    else:
        new_blobs = None

    updated_state = SliceEnsembleState(
        new_coords,
        new_log_probs,
        new_blobs,
        walkers_to_update.mu,
        walkers_to_update.tuning_active,
        walkers_to_update.patience_count,
    )

    return updated_state, total_nexp, total_ncon, total_neval


def as_top_level_api(
    logdensity_fn: Callable,
    move: str = "differential",
    mu: float = 1.0,
    has_blobs: bool = False,
    randomize_split: bool = True,
    nsplits: int = 2,
    maxsteps: int = 10000,
    maxiter: int = 10000,
    tune: bool = True,
    patience: int = 5,
    tolerance: float = 0.05,
) -> SamplingAlgorithm:
    """A user-facing API for the ensemble slice sampling algorithm.

    Parameters
    ----------
    logdensity_fn
        A function that returns the log density of the model at a given position.
    move
        Type of move: "differential" or "random".
    mu
        Initial value of the scale parameter.
    has_blobs
        Whether the logdensity function returns additional information (blobs).
    randomize_split
        If True, randomly shuffle walker indices before splitting into groups.
    nsplits
        Number of groups to split the ensemble into. Default is 2.
    maxsteps
        Maximum steps for slice stepping-out procedure.
    maxiter
        Maximum iterations to prevent infinite loops.
    tune
        Whether to enable adaptive tuning of mu.
    patience
        Number of steps within tolerance before stopping tuning.
    tolerance
        Tolerance for expansion/contraction ratio to stop tuning.

    Returns
    -------
    A `SamplingAlgorithm` that can be used to sample from the target distribution.
    """
    kernel = build_kernel(
        move=move,
        randomize_split=randomize_split,
        nsplits=nsplits,
        maxsteps=maxsteps,
        maxiter=maxiter,
        tune=tune,
        patience=patience,
        tolerance=tolerance,
    )

    def init_fn(position: ArrayTree, rng_key=None):
        return init(position, logdensity_fn, has_blobs, mu)

    def step_fn(
        rng_key: PRNGKey, state
    ) -> tuple[SliceEnsembleState, SliceEnsembleInfo]:
        return kernel(rng_key, state, logdensity_fn)

    return SamplingAlgorithm(init_fn, step_fn)
