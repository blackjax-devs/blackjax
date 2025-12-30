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
"""All things related to SMC effective sample size"""
from typing import Callable

import jax.numpy as jnp
from jax.scipy.special import logsumexp

from blackjax.types import Array, ArrayLikeTree


def ess(log_weights: Array) -> float | Array:
    """Compute the effective sample size.

    Parameters
    ----------
    log_weights: Array
        Log-weights of the sample, shape (n_particles,).

    Returns
    -------
    ess: float | Array
        The effective sample size.
    """
    return jnp.exp(log_ess(log_weights))


def log_ess(log_weights: Array) -> float | Array:
    """Compute the logarithm of the effective sample size.

    Parameters
    ----------
    log_weights: Array
        Log-weights of the sample, shape (n_particles,).

    Returns
    -------
    log_ess: float | Array
        The logarithm of the effective sample size.
    """
    return 2 * logsumexp(log_weights) - logsumexp(2 * log_weights)


def ess_solver(
    logdensity_fn: Callable,
    particles: ArrayLikeTree,
    target_ess: float | Array,
    max_delta: float | Array,
    root_solver: Callable,
) -> float | Array:
    """ESS solver for computing the next increment of SMC tempering.

    Parameters
    ----------
    logdensity_fn: Callable
        The log probability function we wish to sample from.
    particles: ArrayLikeTree
        Current particles of the tempered SMC algorithm.
    target_ess: float | Array
        Target effective sample size (ESS) for the next increment of SMC tempering.
    max_delta: float | Array
        Maximum acceptable delta increment.
    root_solver: Callable
        A solver to find the root of a function. Signature is
        root_solver(fun, min_delta, max_delta). Use e.g. dichotomy from
        blackjax.smc.solver.

    Returns
    -------
    delta: float | Array
        The increment that solves for the target ESS.

    """
    logprob = logdensity_fn(particles)
    n_particles = logprob.shape[0]
    target_val = jnp.log(n_particles * target_ess)

    def fun_to_solve(delta: float | Array) -> Array:
        log_weights = jnp.nan_to_num(-delta * logprob)
        ess_val = log_ess(log_weights)

        return ess_val - target_val

    estimated_delta = root_solver(fun_to_solve, 0.0, max_delta)
    return estimated_delta
