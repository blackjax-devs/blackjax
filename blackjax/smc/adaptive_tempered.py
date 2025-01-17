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
import functools
from typing import Callable

import jax
import jax.numpy as jnp

import blackjax.smc.base as base
import blackjax.smc.ess as ess
import blackjax.smc.solver as solver
import blackjax.smc.tempered as tempered
from blackjax.base import SamplingAlgorithm
from blackjax.types import ArrayLikeTree, PRNGKey
from blackjax.smc import from_mcmc as smc_from_mcmc

__all__ = ["build_kernel", "init", "as_top_level_api"]


def build_kernel(loglikelihood_fn, target_ess, root_solver, tempered_kernel):
    """
    target_ess: float
        The target ESS for the adaptive MCMC tempering
    root_solver: Callable, optional
        A solver utility to find delta matching the target ESS. Signature is
        `root_solver(fun, delta_0, min_delta, max_delta)`, default is a dichotomy solver
    use_log_ess: bool, optional
        Use ESS in log space to solve for delta, default is `True`.
        This is usually more stable when using gradient based solvers.
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

    def kernel(
        rng_key: PRNGKey,
        state: tempered.TemperedSMCState,
    ) -> tuple[tempered.TemperedSMCState, base.SMCInfo]:
        delta = compute_delta(state)
        lmbda = delta + state.lmbda
        return tempered_kernel(rng_key, state, lmbda)

    return kernel

init = tempered.init


def as_top_level_api(
    logprior_fn: Callable,
    loglikelihood_fn: Callable,
    mcmc_step_fn: Callable,
    mcmc_init_fn: Callable,
    mcmc_parameters: dict,
    resampling_fn: Callable,
    target_ess: float,
    root_solver: Callable = solver.dichotomy,
    num_mcmc_steps: int = 10,
    **extra_parameters,
) -> SamplingAlgorithm:
    """Implements the (basic) user interface for the Adaptive Tempered SMC kernel.

    Parameters
    ----------
    logprior_fn
        The log-prior function of the model we wish to draw samples from.
    loglikelihood_fn
        The log-likelihood function of the model we wish to draw samples from.
    mcmc_step_fn
        The MCMC step function used to update the particles.
    mcmc_init_fn
        The MCMC init function used to build a MCMC state from a particle position.
    mcmc_parameters
        The parameters of the MCMC step function.  Parameters with leading dimension
        length of 1 are shared amongst the particles.
    resampling_fn
        The function used to resample the particles.
    target_ess
        The number of effective sample size to aim for at each step.
    root_solver
        The solver used to adaptively compute the temperature given a target number
        of effective samples.
    num_mcmc_steps
        The number of times the MCMC kernel is applied to the particles per step.

    Returns
    -------
    A ``SamplingAlgorithm``.


    """

    # TODO THIS WILL BREAK!

    if num_mcmc_steps is not None:
        # for backwards compatibility
        update_strategy = functools.partial(base.update_and_take_last, num_mcmc_steps=num_mcmc_steps)

    update_particles = (
        smc_from_mcmc.build_kernel(
            mcmc_step_fn, mcmc_init_fn, resampling_fn, mcmc_parameters, update_strategy
        )
    )

    tempered_kernel = tempered.build_kernel(
        logprior_fn,
        loglikelihood_fn,
        update_particles,
    )

    kernel = build_kernel(
        loglikelihood_fn,
        target_ess,
        root_solver,
        tempered_kernel
    )

    def init_fn(position: ArrayLikeTree, rng_key=None):
        del rng_key
        return init(position)

    def step_fn(rng_key: PRNGKey, state):
        return kernel(
            rng_key,
            state
        )

    return SamplingAlgorithm(init_fn, step_fn)
