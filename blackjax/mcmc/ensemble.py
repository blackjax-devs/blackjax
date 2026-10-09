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
"""Public API for the ensemble sampler."""

from collections.abc import Callable
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp

import blackjax.mcmc.proposal as proposal
from blackjax.base import SamplingAlgorithm, build_sampling_algorithm
from blackjax.types import Array, ArrayLikeTree, PRNGKey
from blackjax.util import pytree_size

__all__ = [
    "EnsembleState",
    "EnsembleInfo",
    "init",
    "build_kernel",
    "as_top_level_api",
    "red_blue_update",
    "stretch_move",
]


class EnsembleState(NamedTuple):
    """State of an ensemble sampler.

    position
        Positions of the walkers, stacked along the leading axis.
    logdensity
        Log-density at each walker's position.

    """

    position: ArrayLikeTree
    logdensity: Array


class EnsembleInfo(NamedTuple):
    """Additional information on the ensemble transition.

    acceptance_rate
        The acceptance probability of each walker's proposal.
    is_accepted
        Whether each walker's proposal was accepted.

    """

    acceptance_rate: Array
    is_accepted: Array


def init(position: ArrayLikeTree, logdensity_fn: Callable) -> EnsembleState:
    """Create the state of an ensemble whose walkers are stacked along the
    leading axis of ``position``."""
    position = jax.tree.map(jnp.asarray, position)
    return EnsembleState(position, jax.vmap(logdensity_fn)(position))


def stretch_move(a: float = 2.0) -> Callable:
    """The stretch move :cite:p:`goodman2010ensemble`.

    Parameters
    ----------
    a
        The scale of the stretch move, greater than 1.

    Returns
    -------
    A move ``(rng_key, position, complementary_positions) -> (new_position,
    log_hastings_ratio)``.

    """

    def move(rng_key, position, complementary_positions):
        key_partner, key_stretch = jax.random.split(rng_key)
        partner = jax.tree.map(
            lambda x: jax.random.choice(key_partner, x), complementary_positions
        )
        z = ((a - 1.0) * jax.random.uniform(key_stretch) + 1.0) ** 2 / a
        new_position = jax.tree.map(
            lambda x, y: y + z.astype(x.dtype) * (x - y), position, partner
        )
        return new_position, (pytree_size(position) - 1) * jnp.log(z)

    return move


def build_kernel() -> Callable:
    """Build an ensemble kernel :cite:p:`goodman2010ensemble,foreman2013emcee`.

    The ensemble needs more walkers than the target has dimensions.

    Returns
    -------
    A kernel ``(rng_key, state, logdensity_fn, move) -> (EnsembleState,
    EnsembleInfo)``, where ``move(rng_key, position, complementary_positions)
    -> (new_position, log_hastings_ratio)`` proposes a new position for one
    walker.

    """

    def update_walker(rng_key, walker, complementary_positions, logdensity_fn, move):
        key_move, key_accept = jax.random.split(rng_key)
        position, log_hastings_ratio = move(
            key_move, walker.position, complementary_positions
        )
        new_walker = EnsembleState(position, logdensity_fn(position))

        log_p_accept = proposal.safe_energy_diff(
            -walker.logdensity, -new_walker.logdensity - log_hastings_ratio
        )
        accepted_walker, info = proposal.static_binomial_sampling(
            key_accept, log_p_accept, walker, new_walker
        )
        do_accept, p_accept, _ = info
        return accepted_walker, EnsembleInfo(p_accept, do_accept)

    def kernel(
        rng_key: PRNGKey,
        state: EnsembleState,
        logdensity_fn: Callable,
        move: Callable,
    ) -> tuple[EnsembleState, EnsembleInfo]:
        """Generate a new ensemble with the given move."""
        return red_blue_update(
            rng_key,
            state,
            partial(update_walker, logdensity_fn=logdensity_fn, move=move),
        )

    return kernel


def _update_half(
    rng_key: PRNGKey,
    state: EnsembleState,
    active: Array,
    complementary: Array,
    update_walker: Callable,
):
    """Update the ``active`` walkers given the ``complementary`` positions."""
    walkers = jax.tree.map(lambda x: x[active], state)
    complementary_positions = jax.tree.map(lambda x: x[complementary], state.position)
    update = jax.vmap(update_walker, in_axes=(0, 0, None))
    keys = jax.random.split(rng_key, active.shape[0])
    new_walkers, info = update(keys, walkers, complementary_positions)

    new_state = jax.tree.map(
        lambda x, new_x: x.at[active].set(new_x), state, new_walkers
    )
    return new_state, info


def red_blue_update(rng_key: PRNGKey, state: EnsembleState, update_walker: Callable):
    """Update a random half of the walkers given the other half, then the other
    half given the first.

    Parameters
    ----------
    rng_key
        The PRNG key.
    state
        The current state of the ensemble.
    update_walker
        A function ``(rng_key, walker, complementary_positions) -> (new_walker,
        info)`` that updates one walker.

    Returns
    -------
    The new state of the ensemble and the walkers' infos, in walker order.

    """
    num_walkers = state.logdensity.shape[0]
    key_split, key_first, key_second = jax.random.split(rng_key, 3)

    walkers = jax.random.permutation(key_split, num_walkers)
    half = (num_walkers + 1) // 2
    first, second = walkers[:half], walkers[half:]

    state, info_first = _update_half(key_first, state, first, second, update_walker)
    state, info_second = _update_half(key_second, state, second, first, update_walker)

    info = jax.tree.map(
        lambda x, y: (
            jnp.zeros_like(x, shape=(num_walkers,) + x.shape[1:])
            .at[walkers]
            .set(jnp.concatenate([x, y]))
        ),
        info_first,
        info_second,
    )
    return state, info


def as_top_level_api(
    logdensity_fn: Callable, move: Callable = stretch_move()
) -> SamplingAlgorithm:
    """Implements the user interface for the ensemble sampler.

    Examples
    --------

    A new kernel can be initialized and used with the following code:

    .. code::

        ensemble = blackjax.ensemble(logdensity_fn)
        state = ensemble.init(initial_positions)  # shape (num_walkers, dim)
        new_state, info = ensemble.step(rng_key, state)

    Kernels are not jit-compiled by default so you will need to do it manually:

    .. code::

       step = jax.jit(ensemble.step)
       new_state, info = step(rng_key, state)

    Parameters
    ----------
    logdensity_fn
        The log-density function of a single walker's position.
    move
        The move that proposes a new position for each walker,
        :func:`stretch_move` by default.

    Returns
    -------
    A ``SamplingAlgorithm``.

    """
    kernel = build_kernel()
    return build_sampling_algorithm(kernel, init, logdensity_fn, kernel_args=(move,))
