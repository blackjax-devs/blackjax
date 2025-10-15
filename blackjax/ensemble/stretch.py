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
"""Stretch move ensemble sampler (affine-invariant MCMC)."""
from typing import Callable

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

from blackjax.base import SamplingAlgorithm
from blackjax.ensemble.base import EnsembleInfo, EnsembleState
from blackjax.types import ArrayLikeTree, ArrayTree, PRNGKey

__all__ = [
    "init",
    "build_kernel",
    "as_top_level_api",
    "stretch_move",
]


def stretch_move(
    rng_key: PRNGKey,
    walker_coords: ArrayTree,
    complementary_coords: ArrayTree,
    a: float = 2.0,
) -> tuple[ArrayTree, float]:
    """Generate a proposal using the affine-invariant stretch move.

    The stretch move selects a random walker from the complementary ensemble
    and proposes a new position along the line connecting the two walkers,
    scaled by a random factor z drawn from g(z) ∝ 1/√z on [1/a, a].

    Parameters
    ----------
    rng_key
        A PRNG key for random number generation.
    walker_coords
        The current walker's coordinates as an array or PyTree.
    complementary_coords
        The coordinates of the complementary ensemble with shape (n_walkers, ...)
        where the leading dimension indexes walkers.
    a
        The stretch scale parameter. Must be > 1. Default is 2.0.

    Returns
    -------
    A tuple (proposal, log_hastings_ratio) where proposal is the proposed
    position with the same structure as walker_coords, and log_hastings_ratio
    is (ndim - 1) * log(z).
    """
    key_select, key_stretch = jax.random.split(rng_key)

    walker_flat, unravel_fn = ravel_pytree(walker_coords)

    comp_leaves, _ = jax.tree_util.tree_flatten(complementary_coords)
    n_walkers_comp = comp_leaves[0].shape[0]

    idx = jax.random.randint(key_select, (), 0, n_walkers_comp)
    complementary_walker = jax.tree_util.tree_map(
        lambda x: x[idx], complementary_coords
    )

    complementary_walker_flat, _ = ravel_pytree(complementary_walker)

    z = ((a - 1.0) * jax.random.uniform(key_stretch) + 1) ** 2.0 / a

    proposal_flat = complementary_walker_flat + z * (
        walker_flat - complementary_walker_flat
    )

    n_dims = walker_flat.shape[0]
    log_hastings_ratio = (n_dims - 1.0) * jnp.log(z)

    return unravel_fn(proposal_flat), log_hastings_ratio


def build_kernel(
    move_fn=None, a: float = 2.0, randomize_split: bool = True, nsplits: int = 2
) -> Callable:
    """Build the stretch move kernel.

    Parameters
    ----------
    move_fn
        Optional custom move function. If None, uses stretch_move with parameter a.
    a
        The stretch parameter. Must be > 1. Default is 2.0.
    randomize_split
        If True, randomly shuffle walker indices before splitting into groups
        each iteration. This improves mixing and matches emcee's default behavior.
    nsplits
        Number of groups to split the ensemble into. Default is 2 (red-blue).
    """
    if move_fn is None:
        move_fn = lambda key, w, c: stretch_move(key, w, c, a)

    def kernel(
        rng_key: PRNGKey, state: EnsembleState, logdensity_fn: Callable
    ) -> tuple[EnsembleState, EnsembleInfo]:
        n_walkers, *_ = jax.tree_util.tree_flatten(state.coords)[0][0].shape

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
            shuffled_state = EnsembleState(
                shuffled_coords, shuffled_log_probs, shuffled_blobs
            )
        else:
            key_update = rng_key
            shuffled_state = state
            indices = jnp.arange(n_walkers)

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
            groups.append(EnsembleState(group_coords, group_log_probs, group_blobs))

        updated_groups = list(groups)
        accepted_groups = []

        keys = jax.random.split(key_update, nsplits)
        for i in range(nsplits):
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

            complementary = EnsembleState(
                complementary_coords, complementary_log_probs, complementary_blobs
            )

            updated_group, accepted = _update_half(
                keys[i], groups[i], complementary, logdensity_fn, move_fn
            )
            updated_groups[i] = updated_group
            accepted_groups.append(accepted)

        shuffled_coords = jax.tree_util.tree_map(
            lambda *arrays: jnp.concatenate(arrays, axis=0),
            *[g.coords for g in updated_groups],
        )
        shuffled_log_probs = jnp.concatenate(
            [g.log_probs for g in updated_groups], axis=0
        )
        shuffled_accepted = jnp.concatenate(accepted_groups, axis=0)

        if state.blobs is not None:
            shuffled_blobs = jax.tree_util.tree_map(
                lambda *arrays: jnp.concatenate(arrays, axis=0),
                *[g.blobs for g in updated_groups],
            )
        else:
            shuffled_blobs = None

        if randomize_split:
            inverse_indices = jnp.argsort(indices)
            new_coords = jax.tree_util.tree_map(
                lambda x: x[inverse_indices], shuffled_coords
            )
            new_log_probs = shuffled_log_probs[inverse_indices]
            accepted = shuffled_accepted[inverse_indices]
            if shuffled_blobs is not None:
                new_blobs = jax.tree_util.tree_map(
                    lambda x: x[inverse_indices], shuffled_blobs
                )
            else:
                new_blobs = None
        else:
            new_coords = shuffled_coords
            new_log_probs = shuffled_log_probs
            accepted = shuffled_accepted
            new_blobs = shuffled_blobs

        new_state = EnsembleState(new_coords, new_log_probs, new_blobs)
        acceptance_rate = jnp.mean(accepted)
        info = EnsembleInfo(acceptance_rate, accepted)

        return new_state, info

    return kernel


def _masked_select(mask, new_val, old_val):
    """Helper to broadcast mask to match array rank for jnp.where.

    Parameters
    ----------
    mask
        Boolean mask with shape (n_walkers,)
    new_val
        New values to select when mask is True
    old_val
        Old values to select when mask is False

    Returns
    -------
    Array with same shape as new_val/old_val, with values selected per mask
    """
    expand_dims = (1,) * (new_val.ndim - 1)
    mask_expanded = mask.reshape((mask.shape[0],) + expand_dims)
    return jnp.where(mask_expanded, new_val, old_val)


def _update_half(
    rng_key, walkers_to_update, complementary_walkers, logdensity_fn, move_fn
):
    """Helper to update one half of the ensemble."""
    n_update, *_ = jax.tree_util.tree_flatten(walkers_to_update.coords)[0][0].shape

    key_moves, key_accept = jax.random.split(rng_key)
    keys = jax.random.split(key_moves, n_update)

    proposals, log_hastings_ratios = jax.vmap(
        lambda k, w_coords: move_fn(k, w_coords, complementary_walkers.coords)
    )(keys, walkers_to_update.coords)

    logdensity_outputs = jax.vmap(logdensity_fn)(proposals)
    if isinstance(logdensity_outputs, tuple):
        log_probs_proposal, blobs_proposal = logdensity_outputs
    else:
        log_probs_proposal = logdensity_outputs
        blobs_proposal = None

    log_p_accept = (
        log_hastings_ratios + log_probs_proposal - walkers_to_update.log_probs
    )

    is_curr_fin = jnp.isfinite(walkers_to_update.log_probs)
    is_prop_fin = jnp.isfinite(log_probs_proposal)
    log_p_accept = jnp.where(
        ~is_curr_fin & is_prop_fin,
        jnp.inf,
        jnp.where(
            is_curr_fin & ~is_prop_fin,
            -jnp.inf,
            jnp.where(~is_curr_fin & ~is_prop_fin, -jnp.inf, log_p_accept),
        ),
    )

    u = jax.random.uniform(key_accept, shape=(n_update,))
    accepted = jnp.log(u) < log_p_accept

    new_coords = jax.tree_util.tree_map(
        lambda prop, old: _masked_select(accepted, prop, old),
        proposals,
        walkers_to_update.coords,
    )
    new_log_probs = jnp.where(accepted, log_probs_proposal, walkers_to_update.log_probs)

    if walkers_to_update.blobs is not None:
        new_blobs = jax.tree_util.tree_map(
            lambda prop, old: _masked_select(accepted, prop, old),
            blobs_proposal,
            walkers_to_update.blobs,
        )
    else:
        new_blobs = None

    new_walkers = EnsembleState(new_coords, new_log_probs, new_blobs)
    return new_walkers, accepted


def init(
    position: ArrayLikeTree,
    logdensity_fn: Callable,
    has_blobs: bool = False,
) -> EnsembleState:
    """Initialize the stretch move algorithm.

    Parameters
    ----------
    position
        Initial positions for all walkers, with shape (n_walkers, ...).
    logdensity_fn
        The log-density function to evaluate.
    has_blobs
        Whether the log-density function returns additional metadata (blobs).
    """
    logdensity_outputs = jax.vmap(logdensity_fn)(position)
    if isinstance(logdensity_outputs, tuple):
        log_probs, blobs = logdensity_outputs
        return EnsembleState(position, log_probs, blobs)
    else:
        log_probs = logdensity_outputs
        return EnsembleState(position, log_probs, None)


def as_top_level_api(
    logdensity_fn: Callable,
    a: float = 2.0,
    has_blobs: bool = False,
    randomize_split: bool = True,
    nsplits: int = 2,
) -> SamplingAlgorithm:
    """A user-facing API for the stretch move algorithm.

    Parameters
    ----------
    logdensity_fn
        A function that returns the log density of the model at a given position.
    a
        The stretch parameter. Must be > 1. Default is 2.0.
    has_blobs
        Whether the logdensity function returns additional information (blobs).
    randomize_split
        If True, randomly shuffle walker indices before splitting into groups
        each iteration. This improves mixing and matches emcee's default behavior.
    nsplits
        Number of groups to split the ensemble into. Default is 2 (red-blue).
        Each group is updated sequentially using all other groups as complementary.

    Returns
    -------
    A `SamplingAlgorithm` that can be used to sample from the target distribution.
    """
    move_fn = lambda key, w, c: stretch_move(key, w, c, a)
    kernel = build_kernel(move_fn, randomize_split=randomize_split, nsplits=nsplits)

    def init_fn(position: ArrayTree, rng_key=None):
        return init(position, logdensity_fn, has_blobs)

    def step_fn(rng_key: PRNGKey, state) -> tuple[EnsembleState, EnsembleInfo]:
        return kernel(rng_key, state, logdensity_fn)

    return SamplingAlgorithm(init_fn, step_fn)
