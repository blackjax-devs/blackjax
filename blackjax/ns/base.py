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

This base implementation uses a provided MCMC kernel to perform the constrained
sampling.
"""
from functools import partial
from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp
from blackjax import SamplingAlgorithm
from blackjax.types import Array, ArrayLikeTree, ArrayTree, PRNGKey

__all__ = ["init", "as_top_level_api", "build_kernel"]


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
        This is used for variance reduction techniques in evidence estimation (see `blackjax.ns.utils.compute_nlive`).
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
    particles: ArrayTree
    loglikelihood: Array  # The log-likelihood of the particles
    loglikelihood_birth: Array  # The hard likelihood threshold of each particle at birth
    logprior: Array  # The log-prior density of the particles
    pid: Array = Array  # particle ID
    logX: float = 0.0  # The current log-volume estiamte
    logZ_live: float = -jnp.inf  # The current evidence estimate
    logZ: float = -jnp.inf  # The accumulated evidence estimate


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
    update_info
        A NamedTuple containing information from the MCMC update step used to
        generate new live particles. The content depends on the specific MCMC
        kernel used.
    """
    particles: ArrayTree
    loglikelihood: Array  # The log-likelihood of the particles
    loglikelihood_birth: (
        Array  # The hard likelihood threshold of each particle at birth
    )
    logprior: Array  # The log-prior density of the particles
    update_info: NamedTuple


def init(
    particles: ArrayLikeTree,
    loglikelihood_fn: Callable,
    logprior_fn: Callable,
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
        The initial log-likelihood birth threshold. Defaults to -NaN, which
        implies no initial likelihood constraint beyond the prior.

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
    logprior_fn: Callable,
    loglikelihood_fn: Callable,
    delete_fn: Callable,
    mcmc_build_kernel: Callable,
    mcmc_init_fn: Callable,
    num_mcmc_steps: int,
) -> Callable:
    """Build a generic Nested Sampling kernel.

    This kernel implements one step of the Nested Sampling algorithm. In each step:
    1. A set of particles with the lowest log-likelihoods are identified and
       marked as "dead" using `delete_fn`. The log-likelihood of the "worst"
       of these dead particles (i.e., max among the lowest ones) defines the new
       likelihood constraint `loglikelihood_0`.
    2. Live particles are selected (typically with replacement from the remaining
       live particles, determined by `delete_fn`) to act as starting points for
       the MCMC updates.
    3. These selected live particles are evolved using an MCMC kernel (`mcmc_build_kernel`,
       `mcmc_init_fn`) for `num_mcmc_steps`. The MCMC sampling is constrained
       to the region where `loglikelihood(new_particle) > loglikelihood_0`.
    4. The newly generated particles replace the dead ones.
    5. The prior volume `logX` and evidence `logZ` are updated based on the
       number of deleted particles and their likelihoods.

    This base version does not adapt the MCMC kernel parameters.

    Parameters
    ----------
    logprior_fn
        A function that computes the log-prior probability of a single particle.
    loglikelihood_fn
        A function that computes the log-likelihood of a single particle.
    delete_fn
        A function `(rng_key, current_ns_state) -> (dead_indices, live_indices_for_resampling)`
        that identifies particles to be deleted and selects live particles
        to be starting points for new particle generation.
    mcmc_build_kernel
        A function that, when called with MCMC parameters (e.g., step size),
        returns an MCMC kernel function `(rng_key, mcmc_state, logdensity_fn) -> (new_mcmc_state, info)`.
    mcmc_init_fn
        A function `(position, logdensity_fn) -> mcmc_state` that initializes
        the state for the MCMC kernel.
    num_mcmc_steps
        The number of MCMC steps to run for each new particle generation.
        The paper suggests 5 times the dimension of the parameter space.

    Returns
    -------
    Callable
        A kernel function for Nested Sampling:
        `(rng_key, ns_state, mcmc_parameters) -> (new_ns_state, ns_info)`.
    """

    def kernel(
        rng_key: PRNGKey,
        state: NSState,
        mcmc_parameters: dict,
    ) -> tuple[NSState, NSInfo]:
        # Delete, and grab all the dead information
        rng_key, delete_fn_key = jax.random.split(rng_key)
        dead_idx, target_update_idx, start_mcmc_idx = delete_fn(delete_fn_key, state)

        dead_particles = jax.tree.map(lambda x: x[dead_idx], state.particles)
        dead_loglikelihood = state.loglikelihood[dead_idx]
        dead_loglikelihood_birth = state.loglikelihood_birth[dead_idx]
        dead_logprior = state.logprior[dead_idx]
        loglikelihood_0 = dead_loglikelihood.max()

        num_deleted = len(dead_idx)
        num_updates = len(target_update_idx)
        num_evolve = len(start_mcmc_idx)

        # Resample the live particles
        kernel = mcmc_build_kernel(**mcmc_parameters)
        rng_key, sample_key = jax.random.split(rng_key)

        def logdensity_fn(x):
            return jnp.where(loglikelihood_fn(x) > loglikelihood_0, logprior_fn(x), -jnp.inf)

        def num_mcmc_steps_kernel(rng_key, particles):
            def body_fn(state, rng_key):
                new_state, info = kernel(rng_key, state, logdensity_fn)
                return new_state, info

            init = mcmc_init_fn(particles, logdensity_fn)
            keys = jax.random.split(rng_key, num_mcmc_steps)
            last_state, info = jax.lax.scan(body_fn, init, keys)
            return last_state, info

        new_particles = jax.tree.map(lambda x: x[start_mcmc_idx], state.particles)
        sample_keys = jax.random.split(sample_key, num_evolve)
        new_state, new_state_info = jax.vmap(num_mcmc_steps_kernel)(
            sample_keys, new_particles
        )

        # Update the particles
        particles = jax.tree_util.tree_map(
            lambda p, n: p.at[target_update_idx].set(n),
            state.particles,
            new_state.position,
        )
        new_state_loglikelihood = jax.vmap(loglikelihood_fn)(new_state.position)
        new_state_logprior = jax.vmap(logprior_fn)(new_state.position)
        loglikelihood = state.loglikelihood.at[target_update_idx].set(new_state_loglikelihood)

        # loglikelihood_births = loglikelihood_0 * jnp.ones(num_updates)
        # loglikelihood_birth = state.loglikelihood_birth.at[target_update_idx].set(loglikelihood_births)
        loglikelihood_births = loglikelihood_0 * jnp.ones(num_deleted)
        loglikelihood_birth = state.loglikelihood_birth.at[dead_idx].set(loglikelihood_births)

        logprior = state.logprior.at[target_update_idx].set(new_state_logprior)
        pid = state.pid.at[target_update_idx].set(state.pid[start_mcmc_idx])

        # Update the logX and logZ
        num_particles = len(state.loglikelihood)
        num_lives = jnp.arange(num_particles, num_particles - num_deleted, -1)
        delta_log_xi = -1 / num_lives
        log_delta_xi = (
            state.logX + jnp.cumsum(delta_log_xi) + jnp.log(1 - jnp.exp(delta_log_xi))
        )
        delta_logz_dead = dead_loglikelihood + log_delta_xi

        logX = state.logX + delta_log_xi.sum()
        logZ_dead = jnp.logaddexp(
            state.logZ, jax.scipy.special.logsumexp(delta_logz_dead)
        )
        logZ_live = (
            jax.scipy.special.logsumexp(loglikelihood) - jnp.log(num_particles) + logX
        )

        # Update the state
        new_state = NSState(
            particles,
            loglikelihood,
            loglikelihood_birth,
            logprior,
            pid,
            logX=logX,
            logZ=logZ_dead,
            logZ_live=logZ_live,
        )
        info = NSInfo(
            dead_particles,
            dead_loglikelihood,
            dead_loglikelihood_birth,
            dead_logprior,
            new_state_info,
        )
        return new_state, info

    return kernel


def delete_fn(
    rng_key: PRNGKey, state: NSState, n_delete: int
) -> tuple[Array, Array]:
    """Identifies particles to be deleted and selects live particles for resampling.

    This function implements a common strategy in Nested Sampling:
    1. Identify the `n_delete` particles with the lowest log-likelihoods. These
       are marked as "dead".
    2. From the remaining live particles (those not marked as dead), `n_delete`
       particles are chosen (typically with replacement, weighted by their
       current importance weights, here it is uniform from survivors)
       to serve as starting points for generating new particles via MCMC.

    Parameters
    ----------
    rng_key
        A JAX PRNG key, used here for choosing live particles.
    state
        The current state of the Nested Sampler.
    n_delete
        The number of particles to delete and subsequently replace.

    Returns:
    --------
    dead_idx : jnp.ndarray
        Indices of particles to be deleted.
    target_update_idx : jnp.ndarray
        Indices of particles to be updated (same as dead_idx in this implementation).
    start_mcmc_idx : jnp.ndarray
        Indices of particles to use as starting points for MCMC evolution.

    """
    loglikelihood = state.loglikelihood
    neg_dead_loglikelihood, dead_idx = jax.lax.top_k(-loglikelihood, n_delete)
    weights = jnp.array(
        loglikelihood > -neg_dead_loglikelihood.min(), dtype=jnp.float32
    )
    start_mcmc_idx = jax.random.choice(
        rng_key,
        len(weights),
        shape=(n_delete,),
        p=weights / weights.sum(),
        replace=True,
    )
    target_update_idx = dead_idx
    return dead_idx, target_update_idx, start_mcmc_idx

def bi_directional_delete_fn(key, state, n_delete):
    """Selects particles for deletion and MCMC initialization for full state regeneration.

    This deletion strategy assumes the total number of particles (`N_total`) in the
    `NSState` is exactly twice `n_delete` (i.e., `N_total = 2 * n_delete`).
    It operates as follows:
    1. The `n_delete` particles with the lowest log-likelihoods are marked as 'dead'.
       These define `loglikelihood_0` and are reported in `NSInfo`.
    2. All `N_total` particle slots are targeted for replacement.
    3. The `n_delete` particles with the highest log-likelihoods (the 'live' set)
       are each duplicated to serve as `N_total` starting points for MCMC evolution.

    The `key` (PRNGKey) is unused by this deterministic selection strategy but is
    included for interface compatibility.

    Parameters
    ----------
    key
        A JAX PRNG key (unused).
    state
        The current `NSState`. `len(state.loglikelihood)` must equal `2 * n_delete`.
    n_delete
        The number of lowest-likelihood particles to mark as dead.

    Returns
    -------
    tuple[Array, Array, Array]
        - dead_idx: Indices of the `n_delete` lowest-likelihood particles.
        - target_update_idx: Indices of all `N_total` particles, for replacement.
        - start_mcmc_idx: `N_total` MCMC starting indices, derived from duplicating
          the `n_delete` highest-likelihood particles.
    """
    loglikelihood = state.loglikelihood
    sorted_indices = jnp.argsort(loglikelihood)

    dead_idx = sorted_indices[:n_delete]
    live_idx = sorted_indices[n_delete:]

    return (
        dead_idx,
        jnp.arange(len(loglikelihood)),
        jnp.concatenate([live_idx, live_idx])
    )



def as_top_level_api(
    logprior_fn: Callable,
    loglikelihood_fn: Callable,
    mcmc_build_kernel: Callable,
    mcmc_init_fn: Callable,
    mcmc_parameters: dict,
    num_mcmc_steps: int,
    delete_fn: Callable = delete_fn,
    n_delete: int = 1,
) -> SamplingAlgorithm:
    """Creates a Nested Sampling algorithm with fixed MCMC parameters.

    This convenience function wraps the `build_kernel` and `init` functions
    into a `SamplingAlgorithm` object, making it easy to use with BlackJAX's
    inference loop. This version uses a fixed set of `mcmc_parameters` for
    the inner MCMC kernel throughout the Nested Sampling run.

    Parameters
    ----------
    logprior_fn
        A function that computes the log-prior probability of a single particle.
    loglikelihood_fn
        A function that computes the log-likelihood of a single particle.
    mcmc_build_kernel
        A function that, when called with MCMC parameters, returns an MCMC kernel.
    mcmc_init_fn
        A function that initializes the state for the MCMC kernel.
    mcmc_parameters
        A dictionary of fixed parameters for the MCMC kernel.
    num_mcmc_steps
        The number of MCMC steps to run for each new particle generation.
    n_delete
        The number of particles to delete and replace at each NS step.
        Defaults to 1.

    Returns
    -------
    SamplingAlgorithm
        A `SamplingAlgorithm` tuple containing `init` and `step` functions for
        the configured Nested Sampler.
    """
    delete_func = partial(delete_fn, n_delete=n_delete)

    kernel = build_kernel(
        logprior_fn,
        loglikelihood_fn,
        delete_func,
        mcmc_build_kernel,
        mcmc_init_fn,
        num_mcmc_steps,
    )

    def init_fn(particles: ArrayLikeTree, rng_key=None):
        del rng_key
        return init(particles, loglikelihood_fn, logprior_fn)

    def step_fn(rng_key: PRNGKey, state):
        return kernel(rng_key, state, mcmc_parameters)

    return SamplingAlgorithm(init_fn, step_fn)
