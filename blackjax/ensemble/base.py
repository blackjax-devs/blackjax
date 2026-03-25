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
"""Base classes and utilities for ensemble sampling methods."""
from typing import NamedTuple, Optional

import jax
import jax.numpy as jnp

from blackjax.types import Array, ArrayTree

__all__ = [
    "EnsembleState",
    "EnsembleInfo",
    "get_nwalkers",
    "tree_take",
    "shuffle_triple",
    "unshuffle_triple",
    "split_triple",
    "concat_triple_groups",
    "complementary_triple",
    "masked_select",
    "vmapped_logdensity",
    "prepare_split",
    "build_states_from_triples",
]


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
    is_accepted
        A boolean array of shape `(n_walkers,)` indicating whether each walker's
        proposal was accepted.
    """

    acceptance_rate: float
    is_accepted: Array


def get_nwalkers(coords: ArrayTree) -> int:
    """Get the number of walkers from ensemble coordinates."""
    return jax.tree_util.tree_flatten(coords)[0][0].shape[0]


def tree_take(tree: ArrayTree, idx: jnp.ndarray) -> ArrayTree:
    """Index into a PyTree along the leading dimension."""
    return jax.tree_util.tree_map(lambda a: a[idx], tree)


def shuffle_triple(key, coords, log_probs, blobs):
    """Shuffle ensemble coordinates, log_probs, and blobs."""
    n = get_nwalkers(coords)
    idx = jax.random.permutation(key, n)
    coords_s = tree_take(coords, idx)
    log_probs_s = log_probs[idx]
    blobs_s = None if blobs is None else tree_take(blobs, idx)
    return coords_s, log_probs_s, blobs_s, idx


def unshuffle_triple(coords, log_probs, blobs, indices):
    """Reverse a shuffle operation on ensemble coordinates, log_probs, and blobs."""
    inv = jnp.argsort(indices)
    coords_u = tree_take(coords, inv)
    log_probs_u = log_probs[inv]
    blobs_u = None if blobs is None else tree_take(blobs, inv)
    return coords_u, log_probs_u, blobs_u


def split_triple(coords, log_probs, blobs, nsplits):
    """Split ensemble into nsplits contiguous groups."""
    n = get_nwalkers(coords)
    group_size = n // nsplits
    groups = []
    for i in range(nsplits):
        s = i * group_size
        e = (i + 1) * group_size if i < nsplits - 1 else n
        coords_i = jax.tree_util.tree_map(lambda a: a[s:e], coords)
        log_probs_i = log_probs[s:e]
        blobs_i = (
            None if blobs is None else jax.tree_util.tree_map(lambda a: a[s:e], blobs)
        )
        groups.append((coords_i, log_probs_i, blobs_i))
    return groups


def concat_triple_groups(group_triples):
    """Concatenate groups of (coords, log_probs, blobs) triples."""
    coords_list, logp_list, blobs_list = zip(*group_triples)
    coords = jax.tree_util.tree_map(
        lambda *xs: jnp.concatenate(xs, axis=0), *coords_list
    )
    logp = jnp.concatenate(logp_list, axis=0)
    blobs = (
        None
        if all(b is None for b in blobs_list)
        else jax.tree_util.tree_map(
            lambda *xs: jnp.concatenate(xs, axis=0), *blobs_list
        )
    )
    return coords, logp, blobs


def complementary_triple(groups, i):
    """Build complementary ensemble from all groups except group i."""
    return concat_triple_groups([g for j, g in enumerate(groups) if j != i])


def masked_select(mask, new_val, old_val):
    """Select between new and old values based on mask."""
    expand_dims = (1,) * (new_val.ndim - 1)
    mask_expanded = mask.reshape((mask.shape[0],) + expand_dims)
    return jnp.where(mask_expanded, new_val, old_val)


def vmapped_logdensity(logdensity_fn, coords):
    """Evaluate logdensity function on ensemble coordinates with vmap."""
    outs = jax.vmap(logdensity_fn)(coords)
    return outs if isinstance(outs, tuple) else (outs, None)


def prepare_split(rng_key, coords, log_probs, blobs, randomize_split, nsplits):
    """Prepare ensemble for splitting into groups.

    Handles optional randomization, splitting, and returns components needed
    for group-wise updates and subsequent unshuffling.
    """
    if randomize_split:
        key_shuffle, key_update = jax.random.split(rng_key)
        coords_s, logp_s, blobs_s, indices = shuffle_triple(
            key_shuffle, coords, log_probs, blobs
        )
    else:
        key_update = rng_key
        coords_s, logp_s, blobs_s = coords, log_probs, blobs
        indices = jnp.arange(get_nwalkers(coords))
    group_triples = split_triple(coords_s, logp_s, blobs_s, nsplits)
    return key_update, group_triples, indices


def build_states_from_triples(group_triples, state_ctor, extra_fields=()):
    """Build state objects from triples with optional extra fields.

    Handles both base EnsembleState and algorithm-specific states like
    SliceEnsembleState that have additional fields.
    """
    return [state_ctor(t[0], t[1], t[2], *extra_fields) for t in group_triples]
