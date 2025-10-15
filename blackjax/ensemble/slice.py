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
from blackjax.ensemble.base import (
    build_states_from_triples,
    complementary_triple,
    concat_triple_groups,
    get_nwalkers,
    prepare_split,
    unshuffle_1d,
    unshuffle_triple,
    vmapped_logdensity,
)
from blackjax.types import ArrayLikeTree, ArrayTree, PRNGKey

__all__ = [
    "SliceEnsembleState",
    "SliceEnsembleInfo",
    "init",
    "build_kernel",
    "as_top_level_api",
    "differential_direction",
    "random_direction",
    "gaussian_direction",
    "slice_along_direction",
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
        Fraction of walkers that found valid slice points.
    is_accepted
        Boolean array of shape `(n_walkers,)` indicating successful slice updates.
    expansions
        Total number of slice expansions performed.
    contractions
        Total number of slice contractions performed.
    nevals
        Total number of log-density evaluations performed.
    mu
        The current value of the scale parameter.
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
    comp_leaves, _ = jax.tree_util.tree_flatten(complementary_coords)
    n_comp = comp_leaves[0].shape[0]

    key1, key2 = jax.random.split(rng_key)
    idx1 = jax.random.randint(key1, (n_update,), 0, n_comp)
    j = jax.random.randint(key2, (n_update,), 0, n_comp - 1)
    idx2 = j + (j >= idx1)  # Ensure idx2 != idx1

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
        Template coordinates to match structure. Leading dimension will be n_update.
    n_update
        Number of walkers to update (number of directions needed).
    mu
        The scale parameter.

    Returns
    -------
    A tuple (directions, tune_once) where directions is a PyTree matching
    the structure of template_coords with leading dimension n_update,
    and tune_once is True.
    """
    leaves, treedef = jax.tree_util.tree_flatten(template_coords)
    n_leaves = len(leaves)

    keys = jax.random.split(rng_key, n_leaves)

    def sample_leaf(key, template_leaf):
        template_shape = template_leaf.shape
        if len(template_shape) > 0:
            new_shape = (n_update,) + template_shape[1:]
        else:
            new_shape = (n_update,)
        return jax.random.normal(key, new_shape)

    direction_leaves = [
        2.0 * mu * sample_leaf(key, leaf) for key, leaf in zip(keys, leaves)
    ]

    directions = jax.tree_util.tree_unflatten(treedef, direction_leaves)

    return directions, True


def gaussian_direction(
    rng_key: PRNGKey,
    complementary_coords: ArrayTree,
    n_update: int,
    mu: float,
) -> tuple[ArrayTree, bool]:
    """Generate direction vectors using the Gaussian move.

    Directions are sampled from a multivariate normal distribution with covariance
    estimated from the complementary ensemble. This move adapts to the local
    geometry of the target distribution.

    Parameters
    ----------
    rng_key
        A PRNG key for random number generation.
    complementary_coords
        Coordinates of the complementary ensemble.
    n_update
        Number of walkers to update.
    mu
        The scale parameter.

    Returns
    -------
    A tuple (directions, tune_once) where directions is a PyTree matching
    the structure of complementary_coords with leading dimension n_update,
    and tune_once is True.
    """
    leaves, treedef = jax.tree_util.tree_flatten(complementary_coords)
    n_leaves = len(leaves)

    keys = jax.random.split(rng_key, n_leaves)

    def sample_gaussian_leaf(key, leaf):
        n_comp = leaf.shape[0]
        leaf_flat = leaf.reshape(n_comp, -1)
        d = leaf_flat.shape[1]

        mean = jnp.mean(leaf_flat, axis=0)
        centered = leaf_flat - mean

        cov = jnp.dot(centered.T, centered) / (n_comp - 1)
        jitter = 1e-6 * jnp.eye(d)
        cov_reg = cov + jitter

        directions_flat = jax.random.multivariate_normal(
            key, jnp.zeros(d), cov_reg, (n_update,)
        )

        orig_shape = leaf.shape
        new_shape = (n_update,) + orig_shape[1:]
        return (2.0 * mu * directions_flat).reshape(new_shape)

    direction_leaves = [
        sample_gaussian_leaf(key, leaf) for key, leaf in zip(keys, leaves)
    ]

    directions = jax.tree_util.tree_unflatten(treedef, direction_leaves)

    return directions, True


def slice_along_direction(
    rng_key: PRNGKey,
    x0: ArrayTree,
    logp0: float,
    direction: ArrayTree,
    logprob_fn: Callable,
    maxsteps: int = 10000,
    maxiter: int = 10000,
) -> tuple[ArrayTree, float, bool, int, int, int]:
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
    A tuple (x1, logp1, accepted, nexp, ncon, neval) where:
        x1: New position (PyTree), or x0 if not accepted
        logp1: Log-probability at new position, or logp0 if not accepted
        accepted: Boolean indicating if a valid slice point was found
        nexp: Number of expansions performed
        ncon: Number of contractions performed
        neval: Number of log-probability evaluations
    """
    key_z0, key_lr, key_j, key_shrink = jax.random.split(rng_key, 4)

    z0 = logp0 - jax.random.exponential(key_z0)

    l_init = -jax.random.uniform(key_lr)
    r_init = l_init + 1.0

    j = jax.random.randint(key_j, (), 0, maxsteps)
    k = maxsteps - 1 - j

    def eval_at_t(t):
        xt = jax.tree_util.tree_map(lambda x, d: x + t * d, x0, direction)
        return logprob_fn(xt)

    logp_l_init = eval_at_t(l_init)

    def left_expand_cond(carry):
        l, logp_l, j_left, nexp, neval, iter_count = carry
        return (j_left > 0) & (iter_count < maxiter) & (logp_l > z0)

    def left_expand_body(carry):
        l, logp_l, j_left, nexp, neval, iter_count = carry
        l_new = l - 1.0
        logp_new = eval_at_t(l_new)
        return l_new, logp_new, j_left - 1, nexp + 1, neval + 1, iter_count + 1

    l_final, _, _, nexp_left, neval_left, _ = jax.lax.while_loop(
        left_expand_cond, left_expand_body, (l_init, logp_l_init, j, 0, 0, 0)
    )

    logp_r_init = eval_at_t(r_init)

    def right_expand_cond(carry):
        r, logp_r, k_right, nexp, neval, iter_count = carry
        return (k_right > 0) & (iter_count < maxiter) & (logp_r > z0)

    def right_expand_body(carry):
        r, logp_r, k_right, nexp, neval, iter_count = carry
        r_new = r + 1.0
        logp_new = eval_at_t(r_new)
        return r_new, logp_new, k_right - 1, nexp + 1, neval + 1, iter_count + 1

    r_final, _, _, nexp_right, neval_right, _ = jax.lax.while_loop(
        right_expand_cond, right_expand_body, (r_init, logp_r_init, k, 0, 0, 0)
    )

    nexp_total = nexp_left + nexp_right
    neval_after_expand = neval_left + neval_right + 2

    def shrink_cond(carry):
        _, _, _, _, _, iter_count, accepted, _, _ = carry
        return (~accepted) & (iter_count < maxiter)

    def shrink_body(carry):
        key, l, r, neval, ncon, iter_count, accepted, t_acc, logp_acc = carry
        key, key_t = jax.random.split(key)
        t = jax.random.uniform(key_t, minval=l, maxval=r)
        logp_t = eval_at_t(t)
        neval_new = neval + 1

        inside_slice = logp_t >= z0
        accepted_new = accepted | inside_slice

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

    (
        _,
        _,
        _,
        neval_shrink,
        ncon_total,
        _,
        accepted,
        t_final,
        logp_final,
    ) = jax.lax.while_loop(
        shrink_cond,
        shrink_body,
        (key_shrink, l_final, r_final, 0, 0, 0, False, 0.0, logp0),
    )

    x1 = jax.tree_util.tree_map(lambda x, d: x + t_final * d, x0, direction)

    neval_total = neval_after_expand + neval_shrink

    return x1, logp_final, accepted, nexp_total, ncon_total, neval_total


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
    log_probs, blobs = vmapped_logdensity(logdensity_fn, position)
    return SliceEnsembleState(position, log_probs, blobs, mu, True, 0)


def build_kernel(
    move: str = "differential",
    move_fn=None,
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
        Type of move: "differential", "random", or "gaussian". Ignored if move_fn provided.
    move_fn
        Optional custom move function. If None, uses the specified move type.
    randomize_split
        If True, randomly shuffle walker indices before splitting into groups.
    nsplits
        Number of groups to split the ensemble into.
    maxsteps
        Maximum steps for slice stepping-out procedure.
    maxiter
        Maximum iterations for shrinking procedure.
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
    if move_fn is None:
        if move == "differential":
            move_fn = differential_direction
        elif move == "random":
            move_fn = random_direction
        elif move == "gaussian":
            move_fn = gaussian_direction

    def kernel(
        rng_key: PRNGKey, state: SliceEnsembleState, logdensity_fn: Callable
    ) -> tuple[SliceEnsembleState, SliceEnsembleInfo]:
        key_update, group_triples, indices = prepare_split(
            rng_key,
            state.coords,
            state.log_probs,
            state.blobs,
            randomize_split,
            nsplits,
        )
        groups = build_states_from_triples(
            group_triples,
            SliceEnsembleState,
            (state.mu, state.tuning_active, state.patience_count),
        )

        updated_groups = list(groups)
        accepted_groups = []
        total_nexp = jnp.array(0, dtype=jnp.int32)
        total_ncon = jnp.array(0, dtype=jnp.int32)
        total_neval = jnp.array(0, dtype=jnp.int32)

        keys = jax.random.split(key_update, nsplits)
        for i in range(nsplits):
            comp_triple = complementary_triple(
                [(g.coords, g.log_probs, g.blobs) for g in updated_groups], i
            )
            complementary = SliceEnsembleState(
                comp_triple[0],
                comp_triple[1],
                comp_triple[2],
                state.mu,
                state.tuning_active,
                state.patience_count,
            )

            updated_group, accepted, nexp, ncon, neval = _update_half_slice(
                keys[i],
                groups[i],
                complementary,
                logdensity_fn,
                move_fn,
                maxsteps,
                maxiter,
            )
            updated_groups[i] = updated_group
            accepted_groups.append(accepted)
            total_nexp = total_nexp + nexp
            total_ncon = total_ncon + ncon
            total_neval = total_neval + neval

        coords_cat, logp_cat, blobs_cat = concat_triple_groups(
            [(g.coords, g.log_probs, g.blobs) for g in updated_groups]
        )
        shuffled_accepted = jnp.concatenate(accepted_groups, axis=0)

        if randomize_split:
            new_coords, new_log_probs, new_blobs = unshuffle_triple(
                coords_cat, logp_cat, blobs_cat, indices
            )
            accepted = unshuffle_1d(shuffled_accepted, indices)
        else:
            new_coords, new_log_probs, new_blobs = coords_cat, logp_cat, blobs_cat
            accepted = shuffled_accepted

        should_tune = tune & state.tuning_active

        nexp_eff = jnp.maximum(total_nexp, 1)
        mu_tuned = state.mu * 2.0 * nexp_eff / (nexp_eff + total_ncon)

        exp_ratio = total_nexp / jnp.maximum(total_nexp + total_ncon, 1)
        within_tolerance = jnp.abs(exp_ratio - 0.5) < tolerance

        patience_count_updated = jnp.where(
            within_tolerance, state.patience_count + 1, 0
        )
        tuning_active_updated = patience_count_updated < patience

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

        acceptance_rate = jnp.mean(accepted)
        info = SliceEnsembleInfo(
            acceptance_rate=acceptance_rate,
            is_accepted=accepted,
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
) -> tuple[SliceEnsembleState, jnp.ndarray, int, int, int]:
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
    Tuple of (updated_group_state, accepted_array, total_expansions, total_contractions, total_evals).
    """
    n_update = get_nwalkers(walkers_to_update.coords)

    key_dir, key_slice = jax.random.split(rng_key)
    directions, _ = direction_fn(
        key_dir, complementary_walkers.coords, n_update, walkers_to_update.mu
    )

    def logprob_only(x):
        out = logdensity_fn(x)
        return out[0] if isinstance(out, tuple) else out

    keys = jax.random.split(key_slice, n_update)

    def slice_one_walker(key, x0, logp0, direction):
        return slice_along_direction(
            key, x0, logp0, direction, logprob_only, maxsteps, maxiter
        )

    results = jax.vmap(slice_one_walker)(
        keys, walkers_to_update.coords, walkers_to_update.log_probs, directions
    )

    (
        new_coords,
        new_log_probs,
        accepted_array,
        nexp_array,
        ncon_array,
        neval_array,
    ) = results

    total_nexp = jnp.sum(nexp_array)
    total_ncon = jnp.sum(ncon_array)
    total_neval = jnp.sum(neval_array)

    if walkers_to_update.blobs is not None:
        _, new_blobs = vmapped_logdensity(logdensity_fn, new_coords)
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

    return updated_state, accepted_array, total_nexp, total_ncon, total_neval


def as_top_level_api(
    logdensity_fn: Callable,
    move: str = "differential",
    move_fn=None,
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
    """Ensemble slice sampling algorithm.

    Parameters
    ----------
    logdensity_fn
        Function that returns the log density at a given position.
    move
        Type of move: "differential", "random", or "gaussian". Ignored if move_fn provided.
    move_fn
        Optional custom move function. If None, uses the specified move type.
    mu
        Initial value of the scale parameter.
    has_blobs
        Whether the logdensity function returns additional information.
    randomize_split
        If True, randomly shuffle walker indices before splitting into groups.
    nsplits
        Number of groups to split the ensemble into.
    maxsteps
        Maximum steps for slice stepping-out procedure.
    maxiter
        Maximum iterations for shrinking procedure.
    tune
        Whether to enable adaptive tuning of mu.
    patience
        Number of steps within tolerance before stopping tuning.
    tolerance
        Tolerance for expansion/contraction ratio to stop tuning.

    Returns
    -------
    A `SamplingAlgorithm`.
    """
    kernel = build_kernel(
        move=move,
        move_fn=move_fn,
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
