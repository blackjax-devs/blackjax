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
from typing import Callable, Tuple

import jax
import jax.numpy as jnp

import blackjax.smc.base as base
import blackjax.smc.ess as ess
import blackjax.smc.solver as solver
import blackjax.smc.tempered as tempered
from blackjax.types import PRNGKey

__all__ = ["kernel"]


def kernel(
    logprior_fn: Callable,
    loglikelihood_fn: Callable,
    mcmc_step_fn: Callable,
    mcmc_init_fn: Callable,
    resampling_fn: Callable,
    target_ess: float,
    root_solver: Callable = solver.dichotomy,
) -> Callable:
    r"""Build a Tempered SMC step using an adaptive schedule.

    Parameters
    ----------
    logprior_fn: Callable
        A function that computes the log-prior density.
    loglikelihood_fn: Callable
        A function that returns the log-likelihood density.
    mcmc_kernel_factory: Callable
        A callable function that creates a mcmc kernel from a log-probability
        density function.
    make_mcmc_state: Callable
        A function that creates a new mcmc state from a position and a
        log-probability density function.
    resampling_fn: Callable
        A random function that resamples generated particles based of weights
    target_ess: float
        The target ESS for the adaptive MCMC tempering
    root_solver: Callable, optional
        A solver utility to find delta matching the target ESS. Signature is
        `root_solver(fun, delta_0, min_delta, max_delta)`, default is a dichotomy solver
    use_log_ess: bool, optional
        Use ESS in log space to solve for delta, default is `True`.
        This is usually more stable when using gradient based solvers.

    Returns
    -------
    A callable that takes a rng_key and a TemperedSMCState that contains the current state
    of the chain and that returns a new state of the chain along with
    information about the transition.

    """

    def compute_delta(state: tempered.TemperedSMCState) -> float:
        lmbda = state.lmbda
        max_delta = 1 - lmbda
        delta = ess.ess_solver(
            jax.vmap(loglikelihood_fn),
            state.particles,
            target_ess,
            max_delta,
            root_solver,
        )
        delta = jnp.clip(delta, 0.0, max_delta)

        return delta

    kernel = tempered.kernel(
        logprior_fn,
        loglikelihood_fn,
        mcmc_step_fn,
        mcmc_init_fn,
        resampling_fn,
    )

    def one_step(
        rng_key: PRNGKey,
        state: tempered.TemperedSMCState,
        num_mcmc_steps: int,
        mcmc_parameters: dict,
    ) -> Tuple[tempered.TemperedSMCState, base.SMCInfo]:
        delta = compute_delta(state)
        lmbda = delta + state.lmbda
        return kernel(rng_key, state, num_mcmc_steps, lmbda, mcmc_parameters)

    return one_step
