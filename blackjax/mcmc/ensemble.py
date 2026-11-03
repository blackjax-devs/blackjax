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
"""Public API for the ensemble sampler and its moves."""

from collections.abc import Callable
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

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
    "walk_move",
    "differential_evolution_move",
    "snooker_move",
    "kde_move",
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


def walk_move(num_helpers: int | None = None) -> Callable:
    """The walk move :cite:p:`goodman2010ensemble`.

    Parameters
    ----------
    num_helpers
        The number of complementary walkers, at least two, whose covariance is
        that of the proposal; all of them by default.

    Returns
    -------
    A move ``(rng_key, position, complementary_positions) -> (new_position,
    log_hastings_ratio)``.

    """

    def move(rng_key, position, complementary_positions):
        key_helpers, key_step = jax.random.split(rng_key)
        helpers = complementary_positions
        if num_helpers is not None:
            helpers = jax.tree.map(
                lambda x: jax.random.choice(
                    key_helpers, x, (num_helpers,), replace=False
                ),
                complementary_positions,
            )
        flat_helpers = jax.vmap(lambda x: ravel_pytree(x)[0])(helpers)
        flat_position, unravel_fn = ravel_pytree(position)
        covariance = jnp.atleast_2d(jnp.cov(flat_helpers, rowvar=False))
        step = jax.random.multivariate_normal(
            key_step, jnp.zeros_like(flat_position), covariance, method="svd"
        )
        return unravel_fn(flat_position + step), jnp.zeros(())

    return move


def differential_evolution_move(
    sigma: float = 1e-5, gamma0: float | None = None
) -> Callable:
    """The differential evolution move :cite:p:`terbraak2006markov,nelson2013run`.

    The ensemble needs at least two walkers in each half.

    Parameters
    ----------
    sigma
        The relative standard deviation of the scale of the difference vector.
    gamma0
        The mean scale of the difference vector; ``2.38 / sqrt(2 d)`` in ``d``
        dimensions by default.

    Returns
    -------
    A move ``(rng_key, position, complementary_positions) -> (new_position,
    log_hastings_ratio)``.

    """

    def move(rng_key, position, complementary_positions):
        key_pair, key_gamma = jax.random.split(rng_key)
        pair = jax.tree.map(
            lambda x: jax.random.choice(key_pair, x, (2,), replace=False),
            complementary_positions,
        )
        mean_gamma = gamma0
        if gamma0 is None:
            mean_gamma = 2.38 / jnp.sqrt(2 * pytree_size(position))
        gamma = mean_gamma * (1.0 + sigma * jax.random.normal(key_gamma))
        new_position = jax.tree.map(
            lambda x, y: x + gamma.astype(x.dtype) * (y[0] - y[1]), position, pair
        )
        return new_position, jnp.zeros(())

    return move


def snooker_move(gamma: float = 1.7) -> Callable:
    """The differential evolution snooker move :cite:p:`terbraak2008differential`.

    The ensemble needs at least three walkers in each half.

    Parameters
    ----------
    gamma
        The scale of the projected difference vector.

    Returns
    -------
    A move ``(rng_key, position, complementary_positions) -> (new_position,
    log_hastings_ratio)``.

    """

    def move(rng_key, position, complementary_positions):
        helpers = jax.tree.map(
            lambda x: jax.random.choice(rng_key, x, (3,), replace=False),
            complementary_positions,
        )
        z, z1, z2 = jax.vmap(lambda x: ravel_pytree(x)[0])(helpers)
        flat_position, unravel_fn = ravel_pytree(position)

        distance = jnp.linalg.norm(flat_position - z)
        u = (flat_position - z) / distance
        new_flat_position = flat_position + gamma * jnp.dot(u, z1 - z2) * u

        new_distance = jnp.linalg.norm(new_flat_position - z)
        log_hastings_ratio = (flat_position.shape[0] - 1) * (
            jnp.log(new_distance) - jnp.log(distance)
        )
        return unravel_fn(new_flat_position), log_hastings_ratio

    return move


def kde_move(bw_method: str | float | None = None) -> Callable:
    """An independent proposal from a Gaussian kernel density estimate of the
    complementary walkers.

    The complementary walkers must span the space: each half needs more
    walkers than the target has dimensions.

    Parameters
    ----------
    bw_method
        The bandwidth of the kernel density estimate, as for
        :class:`jax.scipy.stats.gaussian_kde`.

    Returns
    -------
    A move ``(rng_key, position, complementary_positions) -> (new_position,
    log_hastings_ratio)``.

    """

    def move(rng_key, position, complementary_positions):
        flat_complementary = jax.vmap(lambda x: ravel_pytree(x)[0])(
            complementary_positions
        )
        flat_position, unravel_fn = ravel_pytree(position)
        kde = jax.scipy.stats.gaussian_kde(flat_complementary.T, bw_method)

        new_flat_position = kde.resample(rng_key, (1,))[:, 0]
        log_hastings_ratio = (
            kde.logpdf(flat_position[:, None])[0]
            - kde.logpdf(new_flat_position[:, None])[0]
        )
        return unravel_fn(new_flat_position), log_hastings_ratio

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

    Moves can be mixed by choosing one at random at each step:

    .. code::

        moves = [
            blackjax.mcmc.ensemble.stretch_move(),
            blackjax.mcmc.ensemble.differential_evolution_move(),
        ]
        weights = jnp.array([0.8, 0.2])
        steps = [blackjax.ensemble(logdensity_fn, move).step for move in moves]

        def step(rng_key, state):
            key_choice, key_step = jax.random.split(rng_key)
            index = jax.random.choice(key_choice, len(steps), p=weights)
            return jax.lax.switch(index, steps, key_step, state)

    Parameters
    ----------
    logdensity_fn
        The log-density function of a single walker's position.
    move
        The move that proposes a new position for each walker:
        :func:`stretch_move` (the default), :func:`walk_move`,
        :func:`differential_evolution_move`, :func:`snooker_move` or
        :func:`kde_move`.

    Returns
    -------
    A ``SamplingAlgorithm``.

    """
    kernel = build_kernel()
    return build_sampling_algorithm(kernel, init, logdensity_fn, kernel_args=(move,))
