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
from typing import Callable

import jax.numpy as jnp

import blackjax.smc.persistent_sampling as persistent_sampling
import blackjax.smc.solver as solver
from blackjax.base import SamplingAlgorithm
from blackjax.smc.base import update_and_take_last
from blackjax.types import Array, ArrayLikeTree, PRNGKey

__all__ = ["build_kernel", "init", "as_top_level_api"]


def build_kernel(
    logprior_fn: Callable,
    loglikelihood_fn: Callable,
    mcmc_step_fn: Callable,
    mcmc_init_fn: Callable,
    resampling_fn: Callable,
    target_ess: float | Array,
    update_strategy: Callable = update_and_take_last,
    root_solver: Callable = solver.dichotomy,
) -> Callable:
    """Build an adaptive Persistent Sampling kernel, with signature
    (rng_key,
    state,
    num_mcmc_steps,
    mcmc_parameters,) -> (new_state, info).

    The function implements the Persistent Sampling algorithm as described
    in Karamanis et al. (2025), with an adaptive tempering schedule. See
    blackjax.smc.persistent_sampling.build_kernel for more details.

    Parameters
    ----------
    logprior_fn: Callable
        Log prior probability function.
        NOTE: This function must be normalized (:math:`Z_0 = 1`), in order
        for the weighting scheme to function correctly.
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
        NOTE: In persistent sampling, the ESS is computed over all particles
        from all previous iterations and can be > 1.
    update_strategy: Callable
        Strategy to update particles using MCMC kernels, by default
        'update_and_take_last' from blackjax.smc.base. The function signature must be
        (mcmc_init_fn,
        loggerposterior_fn,
        mcmc_step_fn,
        num_mcmc_steps,
        n_particles,) -> (mcmc_kernel, n_particles), like 'update_and_take_last'.
        The mcmc_kernel must have signature
        (rng_key, position, mcmc_parameters) -> (new_position, info).
    root_solver
        The solver used to adaptively compute the temperature given a target number
        of effective samples. By default, blackjax.smc.solver.dichotomy.

    Returns
    -------
    kernel: Callable
        A callable that takes a rng_key, a PersistentSMCState, and a dictionary of
        mcmc_parameters, and that returns a the PersistentSMCState after
        the step along with information about the transition.
    """

    def calculate_lambda(state: persistent_sampling.PersistentSMCState) -> Array:
        """Calculate the next tempering parameter based on the target ESS."""

        n_particles = state.persistent_weights.shape[1]
        target_val = jnp.log(n_particles * target_ess)
        max_delta = 1 - state.tempering_schedule[state.iteration]  # so that lambda <= 1

        def fun_to_solve(delta: Array) -> Array:
            """Function for which we want to find a root, i.e. the difference
            between the current ESS (as a function of the tempering parameter)
            and the target ESS, as a function of
            delta = lambda_proposal - lambda_current."""
            log_weights, _ = persistent_sampling.compute_log_persistent_weights(
                state.persistent_log_likelihoods,
                state.persistent_log_Z,
                state.tempering_schedule.at[state.iteration + 1].set(
                    state.tempering_schedule[state.iteration] + delta
                ),
                state.iteration + 1,  # because we are proposing the next lambda
                normalize_to_one=True,  # needed to compute ESS correctly
            )

            ess_val = persistent_sampling.compute_persistent_ess(log_weights)
            return ess_val - target_val

        # in case no solution is found, delta is set to 0 using nan_to_num,
        # this allows for new particles to be added to the persistent ensemble
        # before trying again to solve for delta
        delta = jnp.nan_to_num(root_solver(fun_to_solve, 0.0, max_delta))

        new_lmbda = state.tempering_schedule[state.iteration] + jnp.clip(
            delta,
            0.0,
            max_delta,
        )
        return new_lmbda

    ps_kernel = persistent_sampling.build_kernel(
        logprior_fn=logprior_fn,
        loglikelihood_fn=loglikelihood_fn,
        mcmc_step_fn=mcmc_step_fn,
        mcmc_init_fn=mcmc_init_fn,
        resampling_fn=resampling_fn,
        update_strategy=update_strategy,
    )

    def kernel(
        rng_key: PRNGKey,
        state: persistent_sampling.PersistentSMCState,
        num_mcmc_steps: int | Array,
        mcmc_parameters: dict,
    ) -> persistent_sampling.PersistentSMCState:
        """The adaptive Persistent Sampling kernel. See kernel function in
        blackjax.smc.persistent_sampling.build_kernel for more details."""

        lmbda = calculate_lambda(state)
        return ps_kernel(rng_key, state, num_mcmc_steps, lmbda, mcmc_parameters)

    return kernel


init = persistent_sampling.init


def as_top_level_api(
    logprior_fn: Callable,
    loglikelihood_fn: Callable,
    max_iterations: int | Array,
    mcmc_step_fn: Callable,
    mcmc_init_fn: Callable,
    mcmc_parameters: dict,
    resampling_fn: Callable,
    target_ess: float | Array = 3,
    num_mcmc_steps: int = 10,
    update_strategy: Callable = update_and_take_last,
    root_solver: Callable = solver.dichotomy,
) -> SamplingAlgorithm:
    """
    Implements the user interface for the adaptive Persistent Sampling
    kernel from Karamanis et al. 2025. See build_kernel and
    blackjax.smc.persistent_sampling for more details.

    NOTE: For this algorithm, we need to keep track of all particles
    from all previous iterations. Since the number of tempering steps (and
    therefore the number of particles) is not known in advance, we need to
    define a maximum number of iterations (max_iterations). The inference
    loop should be written in such a way that it breaks if this maximum
    number of iterations is exceeded, even if the algorithm has not yet
    converged to the final posterior (lambda = 1). There is no internal
    check for this.

    Also note that the arrays are preallocated to their maximum size, so
    higher max_iterations will lead to higher memory usage.

    Parameters
    ----------
    logprior_fn : Callable
        The log-prior function of the model we wish to draw samples from.
        NOTE: This function must be normalized (:math:`Z_0 = 1`), in order
        for the weighting scheme to function correctly.
    loglikelihood_fn : Callable
        The log-likelihood function of the model we wish to draw samples from.
    max_iterations : int | Array
        The maximum number of iterations (tempering steps) to perform.
    mcmc_step_fn : Callable
        The MCMC step function used to update the particles.
    mcmc_init_fn : Callable
        The MCMC initialization function used to initialize the MCMC state
        from a position.
    mcmc_parameters : dict
        The parameters for the MCMC kernel.
    resampling_fn : Callable
        Resampling function (from blackjax.smc.resampling).
    target_ess : float | Array, optional
        Target effective sample size (ESS) to determine the next tempering
        parameter, by default 3.
        NOTE: In persistent sampling, the ESS is computed over all
        particles from all previous iterations and can be > 1.
    num_mcmc_steps : int, optional
        Number of MCMC steps to apply to each particle at each iteration,
        by default 10.
    update_strategy : Callable, optional
        The strategy to update particles using MCMC kernels, by default
        'update_and_take_last' from blackjax.smc.base. See build_kernel for
        details.
    root_solver : Callable, optional
        The solver used to adaptively compute the temperature given a target
        number of effective samples. By default, blackjax.smc.solver.dichotomy.

    Returns
    -------
    SamplingAlgorithm
        A ``SamplingAlgorithm`` instance with init and step methods. See
        blackjax.base.SamplingAlgorithm for details.
        The init method has signature
        (position: ArrayLikeTree) -> PersistentSMCState
        The step method has signature
        (rng_key: PRNGKey, state: PersistentSMCState, lmbda: float | Array) ->
        (new_state: PersistentSMCState, info: PersistentStateInfo)
    """

    kernel = build_kernel(
        logprior_fn,
        loglikelihood_fn,
        mcmc_step_fn,
        mcmc_init_fn,
        resampling_fn,
        target_ess,
        update_strategy,
        root_solver,
    )

    def init_fn(position: ArrayLikeTree) -> persistent_sampling.PersistentSMCState:
        return init(position, loglikelihood_fn, max_iterations)

    def step_fn(
        rng_key: PRNGKey,
        state: persistent_sampling.PersistentSMCState,
    ) -> tuple[
        persistent_sampling.PersistentSMCState,
        persistent_sampling.PersistentStateInfo,
    ]:
        return kernel(
            rng_key,
            state,
            num_mcmc_steps,
            mcmc_parameters,
        )

    return SamplingAlgorithm(init_fn, step_fn)  # type: ignore[arg-type]
