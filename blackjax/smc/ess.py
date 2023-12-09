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
import jax.scipy as jsp

from blackjax.types import Array, ArrayLikeTree


def ess(log_weights: Array) -> float:
    return jnp.exp(log_ess(log_weights))


def log_ess(log_weights: Array) -> float:
    """Compute the effective sample size.

    Parameters
    ----------
    log_weights: 1D Array
        log-weights of the sample

    Returns
    -------
    log_ess: float
        The logarithm of the effective sample size

    """
    return 2 * jsp.special.logsumexp(log_weights) - jsp.special.logsumexp(
        2 * log_weights
    )


def ess_solver(
    logdensity_fn: Callable,
    particles: ArrayLikeTree,
    target_ess: float,
    max_delta: float,
    root_solver: Callable,
):
    """ESS solver for computing the next increment of SMC tempering.

    Parameters
    ----------
    logdensity_fn: Callable
        The log probability function we wish to sample from.
    particles: SMCState
        Current state of the tempered SMC algorithm
    target_ess: float
        The relative ESS targeted for the next increment of SMC tempering
    max_delta: float
        Max acceptable delta increment
    root_solver: Callable, optional
        A solver to find the root of a function, takes a function `f`, a starting point `delta0`,
        a min value `min_delta`, and a max value `max_delta`.
        Default is `BFGS` minimization of `f ** 2` and ignores `min_delta` and `max_delta`.

    Returns
    -------
    delta: float
        The increment that solves for the target ESS

    """
    logprob = logdensity_fn(particles)
    n_particles = logprob.shape[0]
    target_val = jnp.log(n_particles * target_ess)

    def fun_to_solve(delta):
        log_weights = jnp.nan_to_num(-delta * logprob)
        ess_val = log_ess(log_weights)

        return ess_val - target_val

    estimated_delta = root_solver(fun_to_solve, 0.0, max_delta)
    return estimated_delta
