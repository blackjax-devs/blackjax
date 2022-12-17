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

import jax
import jax.numpy as jnp

from blackjax.types import PyTree


def ess(log_weights: jnp.ndarray, log: bool = True) -> float:
    """Compute the effective sample size.

    Parameters
    ----------
    log_weights: np.ndarray
        log-weights of the sample
    log: bool
        Compute the log-ESS or the ESS

    Returns
    -------
    ess: float
        The effective sample size

    """
    log_weights = log_weights - jnp.max(log_weights)
    w = jnp.exp(log_weights)
    if log:
        w2: jnp.ndarray = jnp.exp(2 * log_weights)
        res = 2 * jnp.log(w.sum()) - jnp.log(w2.sum())
    else:
        res = jnp.sum(w) ** 2 / jnp.sum(w**2)
    return res


def ess_solver(
    logdensity_fn: Callable,
    particles: PyTree,
    target_ess: float,
    max_delta: float,
    root_solver: Callable,
    use_log_ess: bool = True,
):
    """Build a Tempered SMC step.

    Parameters
    ----------
    logdensity_fn: Callable
        The log probability function we wish to sample from.
    smc_state: SMCState
        Current state of the tempered SMC algorithm
    target_ess: float
        The relative ESS targeted for the next increment of SMC tempering
    max_delta: float
        Max acceptable delta increment
    root_solver: Callable, optional
        A solver to find the root of a function, takes a function `f`, a starting point `delta0`,
        a min value `min_delta`, and a max value `max_delta`.
        Default is `BFGS` minimization of `f ** 2` and ignores `min_delta` and `max_delta`.
    use_log_ess: bool
        Solve using the log ESS or the ESS directly. This may have different behaviours based on the potential function.

    Returns
    -------
    delta: float
        The increment that solves for the target ESS

    """
    n_particles = jax.tree_util.tree_flatten(particles)[0][0].shape[0]

    logdensity = logdensity_fn(particles)
    if use_log_ess:
        target_val = jnp.log(n_particles * target_ess)
    else:
        target_val = n_particles * target_ess

    def fun_to_solve(delta):
        log_weights = jnp.nan_to_num(-delta * logdensity)
        ess_val = ess(log_weights, log=use_log_ess)

        return ess_val - target_val

    estimated_delta = root_solver(fun_to_solve, 0.0, 0.0, max_delta)
    return estimated_delta
