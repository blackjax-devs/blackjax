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
""""""
from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp

from blackjax.types import Array, ArrayLikeTree, PRNGKey

__all__ = ["init", "build_kernel", "NSState", "NSInfo", "delete_fn"]


class StateWithLogLikelihood(NamedTuple):
    """State of a particle in NS. Mostly dressing a conventional
    MCMC state with loglikelihood information. Positions are an ArrayTree
    where each leaf represents a variable from the posterior.

    Attributes
    ----------
    position
        The position of the particle (PyTree).
    logdensity
        The log-density of the particle under the prior (Array).
    loglikelihood
        The log-likelihood of the particle (Array).
    loglikelihood_birth
        The log-likelihood birth threshold for the particle (Array).
    """

    position: ArrayLikeTree
    logdensity: Array
    loglikelihood: Array
    loglikelihood_birth: Array


class NSState(NamedTuple):
    """State of the Nested Sampler.

    At the most basic level, this is just a wrapper around a StateWithLogLikelihood
    however it is extended in other NS implementations.
    """

    particles: StateWithLogLikelihood


class NSInfo(NamedTuple):
    """Additional information returned at each step of the Nested Sampling algorithm.

    Attributes
    ----------
    particles
        The StateWithLogLikelihood of particles that were marked as "dead" (replaced).
    update_info
        A NamedTuple (or any PyTree) containing information from the update step
        (inner kernel) used to generate new live particles.
    """

    particles: StateWithLogLikelihood
    update_info: NamedTuple


def init_state_strategy(
    position: ArrayLikeTree,
    logprior_fn: Callable,
    loglikelihood_fn: Callable,
    loglikelihood_birth: float = jnp.nan,
) -> StateWithLogLikelihood:
    """The default initialisation strategy for each state.

    Parameters
    ----------
    position
        A PyTree of arrays representing the initial positions of the particles.
        Each leaf array has a leading dimension corresponding to the number of particles.
    logprior
        A function that computes the log-prior density for a single particle.
    loglikelihood
        A function that computes the log-likelihood for a single particle.
    loglikelihood_birth
        The log-likelihood threshold that the particle must exceed. Defaults to NaN.

    Returns
    -------
    StateWithLogLikelihood
        The initialized state containing positions, log-prior, log-likelihood, and birth likelihood.
    """
    logprior_values = logprior_fn(position)
    loglikelihood_values = loglikelihood_fn(position)
    loglikelihood_birth_values = loglikelihood_birth * jnp.ones_like(
        loglikelihood_values
    )

    return StateWithLogLikelihood(
        position, logprior_values, loglikelihood_values, loglikelihood_birth_values
    )


def init(
    positions: ArrayLikeTree,
    init_state_fn: Callable,
    loglikelihood_birth: float = jnp.nan,
) -> NSState:
    """Initializes the Nested Sampler state.

    Parameters
    ----------
    positions
        An initial set of positions (PyTree of arrays) drawn from the prior
        distribution. The leading dimension of each leaf array must be equal to
        the number of positions.
    init_state_fn
        A function that initializes an NSState from positions.
    loglikelihood_birth
        The initial log-likelihood birth threshold. Defaults to NaN, which
        implies no initial likelihood constraint beyond the prior.

    Returns
    -------
    NSState
        The initial state of the Nested Sampler.
    """
    state_init = init_state_fn(positions)
    loglikelihood_birth_array = loglikelihood_birth * jnp.ones_like(
        state_init.loglikelihood_birth
    )
    return NSState(state_init._replace(loglikelihood_birth=loglikelihood_birth_array))


def build_kernel(
    delete_fn: Callable,
    inner_kernel: Callable,
) -> Callable:
    """Build a generic Nested Sampling kernel.

    This function creates a kernel for the Nested Sampling algorithm by combining
    a particle deletion function and an inner kernel for generating new particles.

    Parameters
    ----------
    delete_fn
        A deletion function, typically partially applied with ``num_delete``,
        with effective signature ``(state) -> (dead_idx, target_update_idx)``.
        Receives the full NS state (duck-typed) and identifies particles
        to be deleted and the indices to update.
    inner_kernel
        A kernel function with the signature
        ``(rng_key, state, loglikelihood_0) -> (new_particles, info)``
        that generates replacement particles. Receives the full NS state
        (duck-typed) and a single PRNG key; returns a
        ``StateWithLogLikelihood`` with leading dimension ``num_delete``.
        The number of particles to produce is known at construction time.

    Returns
    -------
    Callable
        A kernel function for Nested Sampling:
        ``(rng_key, state) -> (new_state, ns_info)``.
    """

    def kernel(rng_key: PRNGKey, state: NSState) -> tuple[NSState, NSInfo]:
        # Delete, and grab all the dead information
        dead_idx, target_update_idx = delete_fn(state)
        dead_particles = jax.tree.map(lambda x: x[dead_idx], state.particles)

        # Generate replacement particles
        rng_key, inner_key = jax.random.split(rng_key)
        loglikelihood_0 = dead_particles.loglikelihood.max()
        new_particles, inner_update_info = inner_kernel(
            inner_key, state, loglikelihood_0
        )

        # Update the particles
        state = state._replace(
            particles=jax.tree_util.tree_map(
                lambda p, n: p.at[target_update_idx].set(n),
                state.particles,
                new_particles,
            )
        )

        # Return updated state and info
        info = NSInfo(
            dead_particles,
            inner_update_info,
        )
        return state, info

    return kernel


def delete_fn(state: NSState, num_delete: int) -> tuple[Array, Array]:
    """Identifies particles to be deleted.

    Selects the ``num_delete`` particles with the lowest log-likelihoods
    and marks them as "dead".

    Parameters
    ----------
    state
        The current NS state (duck-typed; must have ``.particles.loglikelihood``).
    num_delete
        The number of particles to delete and subsequently replace.

    Returns
    -------
    tuple[Array, Array]
        A tuple containing:
        - ``dead_idx``: Indices of particles marked for deletion.
        - ``target_update_idx``: Indices of particles to be overwritten
          (same as ``dead_idx`` in this implementation).
    """
    loglikelihood = state.particles.loglikelihood
    _, dead_idx = jax.lax.top_k(-loglikelihood, num_delete)
    target_update_idx = dead_idx
    return dead_idx, target_update_idx
