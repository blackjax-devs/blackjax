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

from typing import Callable, Dict, NamedTuple, Optional

import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp

from blackjax.types import Array, ArrayLikeTree, ArrayTree, PRNGKey

__all__ = ["init", "build_kernel", "NSState", "NSInfo", "delete_fn"]


class NSState(NamedTuple):
    """State of the Nested Sampler.

    Attributes
    ----------
    particles
        A PyTree of arrays, where each leaf array has a leading dimension
        equal to the number of live particles. Stores the current positions of
        the live particles.
    loglikelihood
        An array of log-likelihood values, one for each live particle.
    loglikelihood_birth
        An array storing the log-likelihood threshold that each current live
        particle was required to exceed when it was "born" (i.e., sampled).
        This is used for reconstructing the nested sampling path.
    logprior
        An array of log-prior values, one for each live particle.
    pid
        Particle ID. An array of integers tracking the identity or lineage of
        particles, primarily for diagnostic purposes.
    logX
        The log of the current prior volume estimate.
    logZ
        The accumulated log evidence estimate from the "dead" points .
    logZ_live
        The current estimate of the log evidence contribution from the live points.
    inner_kernel_params
        A dictionary of parameters for the inner kernel.
    """

    particles: ArrayLikeTree
    loglikelihood: Array  # The log-likelihood of the particles
    loglikelihood_birth: Array  # The log-likelihood threshold at particle birth
    logprior: Array  # The log-prior density of the particles
    pid: Array  # particle IDs
    logX: Array  # The current log-volume estimate
    logZ: Array  # The accumulated evidence estimate
    logZ_live: Array  # The current evidence estimate
    inner_kernel_params: Dict  # Parameters for the inner kernel


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
    loglikelihood: Array  # The log-likelihood of the particles
    loglikelihood_birth: Array  # The log-likelihood threshold at particle birth
    logprior: Array  # The log-prior density of the particles
    inner_kernel_info: NamedTuple  # Information from the inner kernel update step


class PartitionedState(NamedTuple):
    """State container that partitions out the loglikelihood and logprior.

    This intermediate construction wraps around the usual State of an MCMC chain
    so that the loglikelihood and logprior can be efficiently recorded, a
    necessary step for the Parition function reconstruction that Nested
    Sampling builds


    Attributes
    ----------
    position
        A PyTree of arrays representing the current positions of the particles.
        Each leaf array has a leading dimension corresponding to the number of particles.
    logprior
        An array of log-prior density values evaluated at the particle positions.
        Shape: (n_particles,)
    loglikelihood
        An array of log-likelihood values evaluated at the particle positions.
        Shape: (n_particles,)
    """

    position: ArrayLikeTree  # Current positions of particles in the inner kernel
    logprior: Array  # Log-prior values for particles in the inner kernel
    loglikelihood: Array  # Log-likelihood values for particles in the inner kernel


class PartitionedInfo(NamedTuple):
    """Transition information that additionally records a partitioned loglikelihood
    and logprior.

    See PartitionedState

    Attributes
    ----------
    position
        A PyTree of arrays representing the final positions after the transition step.
        Structure matches the input particle positions.
    logprior
        An array of log-prior density values at the final positions.
        Kept separate to support posterior repartitioning schemes.
        Shape: (n_particles,)
    loglikelihood
        An array of log-likelihood values at the final positions.
        Kept separate to support posterior repartitioning schemes.
        Shape: (n_particles,)
    info
        Additional transition-specific diagnostic information from the step.
        The content and structure depend on the specific transition implementation
        (e.g., acceptance rates, step sizes, number of evaluations, etc.).
    """

    position: ArrayTree
    logprior: ArrayTree
    loglikelihood: ArrayTree
    info: NamedTuple


def new_state_and_info(position, logprior, loglikelihood, info):
    """Create new PartitionedState and PartitionedInfo from transition results.

    This utility function packages the results of a transition into the standard
    partitioned state and info containers, maintaining the separation of logprior
    and loglikelihood components.

    Parameters
    ----------
    position
        The particle positions after the transition step.
    logprior
        The log-prior densities at the new positions.
    loglikelihood
        The log-likelihood values at the new positions.
    info
        Additional transition-specific information from the step.

    Returns
    -------
    tuple[PartitionedState, PartitionedInfo]
        A tuple containing the new partitioned state and associated information.
    """
    new_state = PartitionedState(
        position=position,
        logprior=logprior,
        loglikelihood=loglikelihood,
    )
    info = PartitionedInfo(
        position=position,
        logprior=logprior,
        loglikelihood=loglikelihood,
        info=info,
    )
    return new_state, info


def init(
    particles: ArrayLikeTree,
    logprior_fn: Callable,
    loglikelihood_fn: Callable,
    loglikelihood_birth: Array = -jnp.nan,
    logX: Optional[Array] = 0.0,
    logZ: Optional[Array] = -jnp.inf,
) -> NSState:
    """Initializes the Nested Sampler state.

    Parameters
    ----------
    particles
        An initial set of particles (PyTree of arrays) drawn from the prior
        distribution. The leading dimension of each leaf array must be equal to
        the number of particles.
    logprior_fn
        A function that computes the log-prior of a single particle.
    loglikelihood_fn
        A function that computes the log-likelihood of a single particle.
    loglikelihood_birth
        The initial log-likelihood birth threshold. Defaults to -NaN, which
        implies no initial likelihood constraint beyond the prior.
    logX
        The initial log prior volume estimate. Defaults to 0.0.
    logZ
        The initial log evidence estimate. Defaults to -inf.

    Returns
    -------
    NSState
        The initial state of the Nested Sampler.
    """
    loglikelihood = loglikelihood_fn(particles)
    loglikelihood_birth = loglikelihood_birth * jnp.ones_like(loglikelihood)
    logprior = logprior_fn(particles)
    pid = jnp.arange(len(loglikelihood))
    dtype = loglikelihood.dtype
    logX = jnp.array(logX, dtype=dtype)
    logZ = jnp.array(logZ, dtype=dtype)
    logZ_live = logmeanexp(loglikelihood) + logX
    inner_kernel_params: Dict = {}
    return NSState(
        particles,
        loglikelihood,
        loglikelihood_birth,
        logprior,
        pid,
        logX,
        logZ,
        logZ_live,
        inner_kernel_params,
    )


def build_kernel(
    logprior_fn: Callable,
    loglikelihood_fn: Callable,
    delete_fn: Callable,
    inner_kernel: Callable,
) -> Callable:
    """Build a generic Nested Sampling kernel.

    This kernel implements one step of the Nested Sampling algorithm. In each step:
    1. A set of particles with the lowest log-likelihoods are identified and
       marked as "dead" using `delete_fn`. The log-likelihood of the "worst"
       of these dead particles (i.e., max among the lowest ones) defines the new
       likelihood constraint `loglikelihood_0`.
    2. Live particles are selected (typically with replacement from the remaining
       live particles, determined by `delete_fn`) to act as starting points for
       the updates.
    3. These selected live particles are evolved using an kernel
       `inner_kernel`. The sampling is constrained to the region where
       `loglikelihood(new_particle) > loglikelihood_0`.
    4. The newly generated particles replace particles marked for replacement,
       (typically the ones that have just been deleted).
    5. The prior volume `logX` and evidence `logZ` are updated based on the
       number of deleted particles and their likelihoods.

    This base version does not adapt the kernel parameters.

    Parameters
    ----------
    logprior_fn
        A function that computes the log-prior probability of a single particle.
    loglikelihood_fn
        A function that computes the log-likelihood of a single particle.
    delete_fn
        this particle deletion function has the signature
        `(rng_key, current_state) -> (dead_idx, target_update_idx, start_idx)`
        and identifies particles to be deleted, particles to be updated, and
        selects live particles to be starting points for the inner kernel
        for new particle generation.
    inner_kernel
        This kernel function has the signature
        `(rng_key, inner_state, logprior_fn, loglikelihood_fn, loglikelihood_0, params) -> (new_inner_state, inner_info)`,
        and is used to generate new particles.

    Returns
    -------
    Callable
        A kernel function for Nested Sampling:
        `(rng_key, state) -> (new_state, ns_info)`.
    """

    def kernel(rng_key: PRNGKey, state: NSState) -> tuple[NSState, NSInfo]:
        # Delete, and grab all the dead information
        rng_key, delete_fn_key = jax.random.split(rng_key)
        dead_idx, target_update_idx, start_idx = delete_fn(delete_fn_key, state)
        dead_particles = jax.tree.map(lambda x: x[dead_idx], state.particles)
        dead_loglikelihood = state.loglikelihood[dead_idx]
        dead_loglikelihood_birth = state.loglikelihood_birth[dead_idx]
        dead_logprior = state.logprior[dead_idx]

        # Resample the live particles
        loglikelihood_0 = dead_loglikelihood.max()
        rng_key, sample_key = jax.random.split(rng_key)
        sample_keys = jax.random.split(sample_key, len(start_idx))
        particles = jax.tree.map(lambda x: x[start_idx], state.particles)
        logprior = state.logprior[start_idx]
        loglikelihood = state.loglikelihood[start_idx]
        inner_state = PartitionedState(particles, logprior, loglikelihood)
        new_inner_state, inner_info = inner_kernel(
            sample_keys,
            inner_state,
            logprior_fn,
            loglikelihood_fn,
            loglikelihood_0,
            state.inner_kernel_params,
        )

        # Update the particles
        particles = jax.tree_util.tree_map(
            lambda p, n: p.at[target_update_idx].set(n),
            state.particles,
            new_inner_state.position,
        )
        loglikelihood = state.loglikelihood.at[target_update_idx].set(
            new_inner_state.loglikelihood
        )
        loglikelihood_birth = state.loglikelihood_birth.at[target_update_idx].set(
            loglikelihood_0 * jnp.ones(len(target_update_idx))
        )
        logprior = state.logprior.at[target_update_idx].set(new_inner_state.logprior)
        pid = state.pid.at[target_update_idx].set(state.pid[start_idx])

        # Update the run-time information
        logX, logZ, logZ_live = update_ns_runtime_info(
            state.logX, state.logZ, loglikelihood, dead_loglikelihood
        )

        # Return updated state and info
        state = NSState(
            particles,
            loglikelihood,
            loglikelihood_birth,
            logprior,
            pid,
            logX,
            logZ,
            logZ_live,
            state.inner_kernel_params,
        )
        info = NSInfo(
            dead_particles,
            dead_loglikelihood,
            dead_loglikelihood_birth,
            dead_logprior,
            inner_info,
        )
        return state, info

    return kernel


def delete_fn(
    rng_key: PRNGKey, state: NSState, num_delete: int
) -> tuple[Array, Array, Array]:
    """Identifies particles to be deleted and selects live particles for resampling.

    This function implements a common strategy in Nested Sampling:
    1. Identify the `num_delete` particles with the lowest log-likelihoods. These
       are marked as "dead".
    2. From the remaining live particles (those not marked as dead), `num_delete`
       particles are chosen (typically with replacement, weighted by their
       current importance weights, here it is uniform from survivors)
       to serve as starting points for generating new particles.

    Parameters
    ----------
    rng_key
        A JAX PRNG key, used here for choosing live particles.
    state
        The current state of the Nested Sampler.
    num_delete
        The number of particles to delete and subsequently replace.

    Returns
    -------
    tuple[Array, Array, Array]
        A tuple containing:
        - `dead_idx`: An array of indices corresponding to the particles
          marked for deletion.
        - `target_update_idx`: An array of indices corresponding to the
          particles to be updated (same as dead_idx in this implementation).
        - `start_idx`: An array of indices corresponding to the particles
            selected for initialization.
    """
    loglikelihood = state.loglikelihood
    neg_dead_loglikelihood, dead_idx = jax.lax.top_k(-loglikelihood, num_delete)
    constraint_loglikelihood = loglikelihood > -neg_dead_loglikelihood.min()
    weights = jnp.array(constraint_loglikelihood, dtype=jnp.float32)
    weights = jnp.where(weights.sum() > 0., weights, jnp.ones_like(weights))
    start_idx = jax.random.choice(
        rng_key,
        len(weights),
        shape=(num_delete,),
        p=weights / weights.sum(),
        replace=True,
    )
    target_update_idx = dead_idx
    return dead_idx, target_update_idx, start_idx


def update_ns_runtime_info(
    logX: Array, logZ: Array, loglikelihood: Array, dead_loglikelihood: Array
) -> tuple[Array, Array, Array]:
    num_particles = len(loglikelihood)
    num_deleted = len(dead_loglikelihood)
    num_live = jnp.arange(num_particles, num_particles - num_deleted, -1)
    delta_logX = -1 / num_live
    logX = logX + jnp.cumsum(delta_logX)
    log_delta_X = logX + jnp.log(1 - jnp.exp(delta_logX))
    log_delta_Z = dead_loglikelihood + log_delta_X

    delta_logZ = logsumexp(log_delta_Z)
    logZ = jnp.logaddexp(logZ, delta_logZ)
    logZ_live = logmeanexp(loglikelihood) + logX[-1]
    return logX[-1], logZ, logZ_live


def logmeanexp(x: Array) -> Array:
    return logsumexp(x) - jnp.log(len(x))
