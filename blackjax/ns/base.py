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
"""Base components for Nested Sampling algorithms in BlackJAX.

This module provides the fundamental data structures (`NSState`, `NSInfo`) and
a basic, non-adaptive kernel for Nested Sampling. Nested Sampling is a
Monte Carlo method primarily aimed at Bayesian evidence (marginal likelihood)
computation and posterior sampling, particularly effective for multi-modal
distributions.

The core idea is to transform the multi-dimensional evidence integral into a
one-dimensional integral over the prior volume, ordered by likelihood. This is
achieved by iteratively replacing the point with the lowest likelihood among a
set of "live" points with a new point sampled from the prior, subject to the
constraint that its likelihood must be higher than the one just discarded.

This base implementation uses a provided kernel to perform the constrained
sampling.
"""

from typing import Callable, NamedTuple, Tuple, Any, Dict
from functools import partial

import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp

from blackjax.types import Array, ArrayLikeTree, ArrayTree, PRNGKey

__all__ = [
    "init",
    "build_kernel",
    "NSState",
    "NSInfo",
    "delete_fn",
    "bi_directional_delete_fn",
]


class NSState(NamedTuple):
    """State of the Nested Sampler.

    Attributes
    ----------
    particles
        A PyTree of arrays, where each leaf array has a leading dimension
        equal to the number of live particles. Stores the current positions of
        the live particles.
    loglikelihood
        An array of log-likelihood values, one for each live particle,
        corresponding to `state.particles`.
    loglikelihood_birth
        An array storing the log-likelihood threshold that each current live
        particle was required to exceed when it was "born" (i.e., sampled).
        This is used for variance reduction techniques in evidence estimation.
    logprior
        An array of log-prior values, one for each live particle.
    pid
        Particle ID. An array of integers tracking the identity or lineage of
        particles, primarily for diagnostic purposes.
    logX
        The logarithm of the current prior volume estimate. This decreases as
        the algorithm progresses and likelihood contours shrink.
    logZ_live
        The current estimate of the evidence contribution from the live points.
    logZ
        The accumulated evidence estimate from the "dead" points (particles
        that have been replaced).
    """

    particles: ArrayLikeTree
    loglikelihood: Array  # The log-likelihood of the particles
    loglikelihood_birth: Array  # The log-likelihood threshold at particle birth
    logprior: Array  # The log-prior density of the particles
    pid: Array  # particle ID
    logX: float = 0.0  # The current log-volume estimate
    logZ_live: float = -jnp.inf  # The current evidence estimate from live points
    logZ: float = -jnp.inf  # The accumulated evidence estimate from dead points


class NSInfo(NamedTuple):
    """Additional information returned at each step of the Nested Sampling algorithm.

    Attributes
    ----------
    particles
        The PyTree of particles that were marked as "dead" (replaced) in the
        current step.
    loglikelihood
        The log-likelihood values of the dead particles.
    loglikelihood_birth
        The birth log-likelihood thresholds of the dead particles.
    logprior
        The log-prior values of the dead particles.
    inner_kernel_info
        A NamedTuple (or any PyTree) containing information from the update step
        (inner kernel) used to generate new live particles. The content
        depends on the specific inner kernel used.
    """

    particles: ArrayTree
    loglikelihood: Array  # The log-likelihood of the dead particles
    loglikelihood_birth: Array  # The log-likelihood threshold at particle birth
    logprior: Array  # The log-prior density of the dead particles
    inner_kernel_info: Any  # Typically a NamedTuple or PyTree from inner kernel


def init(
    particles: ArrayLikeTree,
    loglikelihood_fn: Callable[[ArrayTree], float],
    logprior_fn: Callable[[ArrayTree], float],
    loglikelihood_birth: Array = -jnp.nan,
) -> NSState:
    """Initializes the Nested Sampler state.

    Parameters
    ----------
    particles
        An initial set of particles (PyTree of arrays) drawn from the prior
        distribution. The leading dimension of each leaf array must be equal to
        the number of particles.
    loglikelihood_fn
        A function that computes the log-likelihood of a single particle.
    logprior_fn
        A function that computes the log-prior of a single particle.
    loglikelihood_birth
        The initial log-likelihood birth threshold for all particles.
        Defaults to -NaN, implying no initial likelihood constraint beyond the prior.
        If a scalar is provided, it's broadcast to all particles.

    Returns
    -------
    NSState
        The initial state of the Nested Sampler.
    """
    loglikelihood = jax.vmap(loglikelihood_fn)(particles)
    loglikelihood_birth = loglikelihood_birth * jnp.ones_like(loglikelihood)
    logprior = jax.vmap(logprior_fn)(particles)
    pid = jnp.arange(len(loglikelihood))
    return NSState(particles, loglikelihood, loglikelihood_birth, logprior, pid)


def build_kernel(
    logprior_fn: Callable[[ArrayTree], float],
    loglikelihood_fn: Callable[[ArrayTree], float],
    delete_fn: Callable[[PRNGKey, NSState], Tuple[Array, Array, Array]],
    inner_init_fn: Callable[[ArrayTree], Any],
    inner_kernel: Callable[[PRNGKey, Any, Callable[[ArrayTree], float]], Tuple[Any, Any]],
) -> Callable[[PRNGKey, NSState, Dict[str, Any]], Tuple[NSState, NSInfo]]:
    """Build a generic Nested Sampling kernel.

    This kernel implements one step of the Nested Sampling algorithm. In each step:
    1. A set of particles with the lowest log-likelihoods are identified and
       marked as "dead" using `delete_fn`. The log-likelihood of the "worst"
       of these dead particles (i.e., max among the lowest ones, or the lowest
       one if num_delete=1) defines the new likelihood constraint `loglikelihood_0`.
    2. Live particles are selected (determined by `delete_fn`, often by resampling
       from the remaining live particles) to act as starting points for generating
       new particles.
    3. These selected starting particles are evolved using an kernel
       (`inner_kernel`). The sampling is constrained to the region where
       `loglikelihood(new_particle) > loglikelihood_0` by modifying the
       target density for the inner kernel.
    4. The newly generated particles replace the dead ones in the `NSState`.
    5. The prior volume `logX` and evidence `logZ` are updated based on the
       number of deleted particles and their likelihoods.

    This base version does not adapt the inner kernel parameters itself;
    any parameters are passed via `inner_kernel_parameters`.

    Parameters
    ----------
    logprior_fn
        A function that computes the log-prior probability of a single particle.
    loglikelihood_fn
        A function that computes the log-likelihood of a single particle.
    delete_fn
        A function `(rng_key, current_ns_state) -> (dead_indices,
        target_update_indices, live_indices_for_resampling)` that identifies
        particles to be deleted (whose indices are `dead_indices`), determines
        which particle slots to update (`target_update_indices`), and selects
        live particles (`live_indices_for_resampling`) to be starting points for
        new particle generation.
    inner_init_fn
        A function `(initial_position: ArrayTree) -> inner_state` used to
        initialize the state for the inner kernel. The `logdensity_fn`
        for this inner kernel will be partially applied before this init function
        is called within the main NS loop.
    inner_kernel
        This kernel function has the signature
        `(rng_key, inner_state, constrained_logdensity_fn,
        **inner_kernel_parameters) -> (new_inner_state, inner_info)`.

    Returns
    -------
    Callable
        A kernel function for one step of Nested Sampling. It takes an `rng_key`,
        the current `NSState`, and a dictionary of `inner_kernel_parameters`,
        and returns a tuple containing the new `NSState` and `NSInfo` for the step.
    """

    def kernel(
        rng_key: PRNGKey,
        state: NSState,
        inner_kernel_parameters: Dict[str, Any],
    ) -> Tuple[NSState, NSInfo]:
        """Performs one step of the non-adaptive Nested Sampling algorithm.

        Parameters
        ----------
        rng_key
            A JAX PRNG key.
        state
            The current `NSState` of the sampler.
        inner_kernel_parameters
            A dictionary of parameters to be passed to the `inner_kernel`
            when constructing the transition kernel for sampling new live points.

        Returns
        -------
        tuple[NSState, NSInfo]
            A tuple containing the new `NSState` after this step and `NSInfo`
            detailing the dead particles and inner information.
        """
        # Delete, and grab all the dead information
        rng_key, delete_fn_key = jax.random.split(rng_key)
        dead_idx, target_update_idx, start_idx = delete_fn(delete_fn_key, state)

        dead_particles = jax.tree.map(lambda x: x[dead_idx], state.particles)
        dead_loglikelihood = state.loglikelihood[dead_idx]
        dead_loglikelihood_birth = state.loglikelihood_birth[dead_idx]
        dead_logprior = state.logprior[dead_idx]
        loglikelihood_0 = dead_loglikelihood.max()

        # Resample the live particles
        def logdensity_fn(x: ArrayTree) -> float:
            """Log-density for inner kernel: prior constrained by likelihood."""
            constraint = loglikelihood_fn(x) > loglikelihood_0
            return jnp.where(constraint, logprior_fn(x), -jnp.inf)

        step_fn = partial(
            inner_kernel, logdensity_fn=logdensity_fn, **inner_kernel_parameters
        )
        init_fn = partial(inner_init_fn, logdensity_fn=logdensity_fn)

        rng_key, sample_key = jax.random.split(rng_key)
        sample_keys = jax.random.split(sample_key, len(start_idx))
        particles = jax.tree.map(lambda x: x[start_idx], state.particles)
        inner_states = jax.vmap(init_fn)(particles)
        inner_states, inner_state_infos = jax.vmap(step_fn)(sample_keys, inner_states)

        # Update the particles
        particles = jax.tree_util.tree_map(
            lambda p, n: p.at[target_update_idx].set(n),
            state.particles,
            inner_states.position,
        )

        loglikelihood = state.loglikelihood.at[target_update_idx].set(
            jax.vmap(loglikelihood_fn)(inner_states.position)
        )
        loglikelihood_birth = state.loglikelihood_birth.at[target_update_idx].set(
            loglikelihood_0 * jnp.ones(len(target_update_idx))
        )
        logprior = state.logprior.at[target_update_idx].set(
            jax.vmap(logprior_fn)(inner_states.position)
        )
        pid = state.pid.at[target_update_idx].set(state.pid[start_idx])

        # Update the logX and logZ
        num_particles = len(state.loglikelihood)
        num_deleted = len(dead_idx)
        num_lives = jnp.arange(num_particles, num_particles - num_deleted, -1)
        delta_log_X = -1 / num_lives
        logX = state.logX + jnp.cumsum(delta_log_X)
        log_delta_X = logX + jnp.log(1 - jnp.exp(delta_log_X))
        log_delta_Z = dead_loglikelihood + log_delta_X

        delta_logZ = logsumexp(log_delta_Z)
        logZ = jnp.logaddexp(state.logZ, delta_logZ)
        logmeanlikelihood = logsumexp(loglikelihood) - jnp.log(num_particles)
        logZ_live = logmeanlikelihood + logX[-1]

        # Update the state
        state = NSState(
            particles,
            loglikelihood,
            loglikelihood_birth,
            logprior,
            pid,
            logX=logX[-1],
            logZ=logZ,
            logZ_live=logZ_live,
        )
        info = NSInfo(
            dead_particles,
            dead_loglikelihood,
            dead_loglikelihood_birth,
            dead_logprior,
            inner_state_infos,
        )
        return state, info

    return kernel


def delete_fn(
    rng_key: PRNGKey, state: NSState, num_delete: int
) -> Tuple[Array, Array, Array]:
    """Identifies particles to be deleted and selects live particles for resampling.

    This function implements a common strategy in Nested Sampling:
    1. Identify the `num_delete` particles with the lowest log-likelihoods. These
       are marked as "dead". Their indices are returned in `dead_idx`.
    2. The slots previously occupied by these `dead_idx` particles are designated
       to be filled by new particles. These indices are returned in `target_update_idx`.
       In this implementation, `target_update_idx` is the same as `dead_idx`.
    3. From the remaining live particles (those not marked as dead), `num_delete`
       particles are chosen uniformly with replacement to serve as starting points
       for generating new particles. Their indices are returned in `start_idx`.

    Parameters
    ----------
    rng_key
        A JAX PRNG key, used here for choosing live particles for resampling.
    state
        The current `NSState` of the Nested Sampler.
    num_delete
        The number of particles to delete and subsequently replace.

    Returns
    -------
    tuple[Array, Array, Array]
        A tuple containing:
        - `dead_idx`: An array of indices corresponding to the particles
          marked for deletion.
        - `target_update_idx`: An array of indices corresponding to the
          particle slots to be updated with new particles (same as `dead_idx`
          in this implementation).
        - `start_idx`: An array of indices corresponding to the live particles
          selected to initialize the generation of new particles.
    """
    loglikelihood = state.loglikelihood
    neg_dead_loglikelihood, dead_idx = jax.lax.top_k(-loglikelihood, num_delete)
    constraint = loglikelihood > -neg_dead_loglikelihood.min()
    weights = jnp.array(constraint, dtype=jnp.float32)
    start_idx = jax.random.choice(
        rng_key,
        len(weights),
        shape=(num_delete,),
        p=weights / weights.sum(),
        replace=True,
    )
    target_update_idx = dead_idx
    return dead_idx, target_update_idx, start_idx


def bi_directional_delete_fn(
    rng_key: PRNGKey, state: NSState, num_delete: int
) -> Tuple[Array, Array, Array]:
    """Selects particles for deletion and initialization for full state regeneration.

    This deletion strategy assumes the total number of particles (`N_total`) in the
    `NSState` is exactly twice `num_delete` (i.e., `N_total = 2 * num_delete`).
    It operates as follows:
    1. The `num_delete` particles with the lowest log-likelihoods are marked as 'dead'.
       Their indices are returned in `dead_idx`. These define the likelihood
       constraint `loglikelihood_0` for the next step.
    2. All `N_total` particle slots are targeted for replacement. The indices for
       these slots are returned in `target_update_idx` (i.e., `0` to `N_total-1`).
    3. The `num_delete` particles with the highest log-likelihoods (the 'surviving' set)
       are each duplicated. These `2 * num_delete` duplicated indices form the
       `start_idx`, serving as starting points for resampling all `N_total` particles.

    The `rng_key` is unused by this deterministic selection strategy but is
    included for interface compatibility.

    Parameters
    ----------
    rng_key
        A JAX PRNG key (unused in this function).
    state
        The current `NSState`. The total number of particles in `state`
        (i.e., `len(state.loglikelihood)`) must be `2 * num_delete`.
    num_delete
        The number of lowest-likelihood particles to mark as dead. This also
        implies the number of highest-likelihood particles to use for duplication.

    Returns
    -------
    tuple[Array, Array, Array]
        A tuple containing:
        - `dead_idx`: Indices of the `num_delete` lowest-likelihood particles.
        - `target_update_idx`: Indices of all `N_total` particles, for replacement.
        - `start_idx`: `N_total` starting indices, derived from duplicating
          the `num_delete` highest-likelihood particles.
    """
    del rng_key
    loglikelihood = state.loglikelihood
    sorted_indices = jnp.argsort(loglikelihood)

    dead_idx = sorted_indices[:num_delete]
    live_idx = sorted_indices[num_delete:]

    target_update_idx = jnp.arange(len(loglikelihood))
    start_idx = jnp.concatenate([live_idx, live_idx])

    return dead_idx, target_update_idx, start_idx
