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
from typing import Any, Callable, Optional

import jax
import jax.numpy as jnp

import blackjax.smc.base as base
import blackjax.smc.ess as ess
import blackjax.smc.solver as solver
import blackjax.smc.tempered as tempered
from blackjax.base import SamplingAlgorithm
from blackjax.types import Array, ArrayLikeTree, PRNGKey

__all__ = ["build_kernel", "init", "as_top_level_api"]


def build_kernel(
    logprior_fn: Callable,
    loglikelihood_fn: Callable,
    mcmc_step_fn: Callable,
    mcmc_init_fn: Callable,
    resampling_fn: Callable,
    target_ess: float,
    root_solver: Callable = solver.dichotomy,
    **extra_parameters: dict[str, Any],
) -> Callable:
    """Build a Tempered SMC step using an adaptive schedule.

    Parameters
    ----------
    logprior_fn: Callable
        Log prior probability function.
    loglikelihood_fn: Callable
        Log likelihood function.
    mcmc_step_fn: Callable
        Function that creates MCMC step from log-probability density function.
    mcmc_init_fn: Callable
        A function that creates a new mcmc state from a position and a
        log-probability density function.
    resampling_fn: Callable
        Resampling function (from blackjax.smc.resampling).
    target_ess: float | Array
        Target effective sample size (ESS) to determine the next tempering
        parameter.
    root_solver: Callable, optional
        The solver used to adaptively compute the temperature given a target number
        of effective samples. By default, blackjax.smc.solver.dichotomy.
    **extra_parameters : dict[str, Any]
        Additional parameters to pass to tempered.build_kernel.

    Returns
    -------
    kernel: Callable
        A callable that takes a rng_key, a TemperedSMCState, num_mcmc_steps,
        and mcmc_parameters, and returns a new TemperedSMCState along with
        information about the transition.

    """

    def compute_delta(state: tempered.TemperedSMCState) -> float | Array:
        tempering_param = state.tempering_param
        max_delta = 1 - tempering_param
        delta = ess.ess_solver(
            jax.vmap(loglikelihood_fn),
            state.particles,
            target_ess,
            max_delta,
            root_solver,
        )
        delta = jnp.clip(delta, 0.0, max_delta)

        return delta

    tempered_kernel = tempered.build_kernel(
        logprior_fn,
        loglikelihood_fn,
        mcmc_step_fn,
        mcmc_init_fn,
        resampling_fn,
        **extra_parameters,  # type: ignore
    )

    def kernel(
        rng_key: PRNGKey,
        state: tempered.TemperedSMCState,
        num_mcmc_steps: int | Array,
        mcmc_parameters: dict,
    ) -> tuple[tempered.TemperedSMCState, base.SMCInfo]:
        delta = compute_delta(state)
        tempering_param = delta + state.tempering_param
        return tempered_kernel(
            rng_key, state, num_mcmc_steps, tempering_param, mcmc_parameters
        )

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
    **extra_parameters: dict[str, Any],
) -> SamplingAlgorithm:
    """Implements the user interface for the Adaptive Tempered SMC kernel.

    Parameters
    ----------
    logprior_fn: Callable
        The log-prior function of the model we wish to draw samples from.
    loglikelihood_fn: Callable
        The log-likelihood function of the model we wish to draw samples from.
    mcmc_step_fn: Callable
        The MCMC step function used to update the particles.
    mcmc_init_fn: Callable
        The MCMC init function used to build a MCMC state from a particle position.
    mcmc_parameters: dict
        The parameters of the MCMC step function. Parameters with leading dimension
        length of 1 are shared amongst the particles.
    resampling_fn: Callable
        The function used to resample the particles.
    target_ess: float | Array
        Target effective sample size (ESS) to determine the next tempering
        parameter.
    root_solver: Callable, optional
        The solver used to adaptively compute the temperature given a target number
        of effective samples. By default, blackjax.smc.solver.dichotomy.
    num_mcmc_steps: int, optional
        The number of times the MCMC kernel is applied to the particles per step,
        by default 10.
    **extra_parameters: dict [str, Any]
        Additional parameters to pass to the kernel.

    Returns
    -------
    SamplingAlgorithm
        A ``SamplingAlgorithm`` instance with init and step methods.

    """
    kernel = build_kernel(
        logprior_fn,
        loglikelihood_fn,
        mcmc_step_fn,
        mcmc_init_fn,
        resampling_fn,
        target_ess,
        root_solver,
        **extra_parameters,
    )

    def init_fn(
        position: ArrayLikeTree, rng_key: Optional[PRNGKey] = None
    ) -> tempered.TemperedSMCState:
        del rng_key
        return init(position)

    def step_fn(
        rng_key: PRNGKey, state: tempered.TemperedSMCState
    ) -> tuple[tempered.TemperedSMCState, base.SMCInfo]:
        return kernel(
            rng_key,
            state,
            num_mcmc_steps,
            mcmc_parameters,
        )

    return SamplingAlgorithm(init_fn, step_fn)  # type: ignore[arg-type]
