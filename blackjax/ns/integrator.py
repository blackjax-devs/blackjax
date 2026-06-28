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
"""Evidence integration for Nested Sampling.

This module provides utilities for tracking the evidence integral during
a Nested Sampling run. The NSIntegrator accumulates statistics as the algorithm
compresses the prior volume, computing the marginal likelihood (evidence),
the running prior-volume estimate, and the live-point evidence contribution.
"""

from typing import NamedTuple

import jax.numpy as jnp
from jax.scipy.special import logsumexp

from blackjax.ns.base import StateWithLogLikelihood
from blackjax.ns.utils import log1mexp
from blackjax.types import Array

__all__ = ["NSIntegrator", "init_integrator", "update_integrator"]


class NSIntegrator(NamedTuple):
    """Integrator for computing the evidence integral in Nested Sampling.

    This accumulates statistics over the course of a Nested Sampling run,
    computing the evidence (marginal likelihood) and related quantities
    from the history of dead particles. These are derived quantities that
    can be reconstructed from the dead particle history.

    Attributes
    ----------
    logX
        The log of the current prior volume estimate.
    logZ
        The accumulated log evidence estimate from the "dead" points.
    logZ_live
        The current estimate of the log evidence contribution from the live points.
    """

    logX: Array
    logZ: Array
    logZ_live: Array


def init_integrator(particle_state: StateWithLogLikelihood) -> NSIntegrator:
    """Initialize the evidence integrator from the initial live points.

    Parameters
    ----------
    particle_state
        The initial state containing the live particles.

    Returns
    -------
    NSIntegrator
        The initial integrator with logX=0, logZ=-inf, and logZ_live computed
        from the initial live points.
    """
    # Match dtypes to avoid weak_type recompiles on first step.
    ll_dtype = particle_state.loglikelihood.dtype
    logX = jnp.array(0.0, dtype=ll_dtype)
    logZ = jnp.array(-jnp.inf, dtype=ll_dtype)
    logZ_live = _logmeanexp(particle_state.loglikelihood) + logX
    return NSIntegrator(logX, logZ, logZ_live)


def update_integrator(
    integrator: NSIntegrator,
    particle_state: StateWithLogLikelihood,
    dead_particles: StateWithLogLikelihood,
) -> NSIntegrator:
    """Update the evidence integrator after a Nested Sampling step.

    Parameters
    ----------
    integrator
        The current integrator state.
    particle_state
        The updated live state after the NS step.
    dead_particles
        The particles that died in this step.

    Returns
    -------
    NSIntegrator
        The updated integrator. logX is the prior volume after all num_deleted
        deletions, logZ is the accumulated log evidence, and logZ_live is the
        log evidence contribution from the remaining live points.
    """
    loglikelihood = particle_state.loglikelihood
    dead_loglikelihood = dead_particles.loglikelihood

    num_particles = len(loglikelihood)
    num_deleted = len(dead_loglikelihood)
    ll_dtype = loglikelihood.dtype
    num_live = jnp.arange(num_particles, num_particles - num_deleted, -1)
    delta_logX = -1.0 / num_live.astype(ll_dtype)
    logX = integrator.logX + jnp.cumsum(delta_logX)  # X_i, volume after deleting i
    # Anchor each likelihood shell on the PRE-deletion volume X_{i-1}: shell i is
    # X_{i-1} - X_i = X_{i-1}(1 - exp(delta_logX_i)). Anchoring on the post-deletion
    # X_i instead underweights every shell by exp(delta_logX) and biases logZ low by
    # ~1/n (a constant-likelihood sweep then integrates to e^{-1/n} instead of 1).
    logX_prev = jnp.concatenate([integrator.logX[jnp.newaxis], logX[:-1]])  # X_{i-1}
    log_delta_X = logX_prev + log1mexp(delta_logX)
    log_delta_Z = dead_loglikelihood + log_delta_X

    delta_logZ = logsumexp(log_delta_Z)
    logZ = jnp.logaddexp(integrator.logZ, delta_logZ)
    logZ_live = _logmeanexp(loglikelihood) + logX[-1]
    return NSIntegrator(logX[-1], logZ, logZ_live)


def _logmeanexp(x: Array) -> Array:
    """Compute log(mean(exp(x))) in a numerically stable way."""
    n = jnp.array(x.shape[0])
    return logsumexp(x) - jnp.log(n)
