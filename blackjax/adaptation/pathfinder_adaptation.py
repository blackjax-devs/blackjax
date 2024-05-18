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
"""Implementation of the Pathinder warmup for the HMC family of sampling algorithms."""
from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp

import blackjax.vi as vi
from blackjax.adaptation.base import AdaptationResults, return_all_adapt_info
from blackjax.adaptation.step_size import (
    DualAveragingAdaptationState,
    dual_averaging_adaptation,
)
from blackjax.base import AdaptationAlgorithm
from blackjax.optimizers.lbfgs import lbfgs_inverse_hessian_formula_1
from blackjax.types import Array, ArrayLikeTree, PRNGKey

__all__ = ["PathfinderAdaptationState", "base", "pathfinder_adaptation"]


class PathfinderAdaptationState(NamedTuple):
    ss_state: DualAveragingAdaptationState
    step_size: float
    inverse_mass_matrix: Array


def base(
    target_acceptance_rate: float = 0.80,
):
    """Warmup scheme for sampling procedures based on euclidean manifold HMC.

    This adaptation runs in two steps:

    1. The Pathfinder algorithm is ran and we subsequently compute an estimate
    for the value of the inverse mass matrix, as well as a new initialization
    point for the markov chain that is supposedly closer to the typical set.
    2. We then start sampling with the MCMC algorithm and use the samples to
    adapt the value of the step size using an optimization algorithm so that
    the mcmc algorithm reaches a given target acceptance rate.

    Parameters
    ----------
    target_acceptance_rate:
        The target acceptance rate for the step size adaptation.

    Returns
    -------
    init
        Function that initializes the warmup.
    update
        Function that moves the warmup one step.
    final
        Function that returns the step size and mass matrix given a warmup state.

    """
    da_init, da_update, da_final = dual_averaging_adaptation(target_acceptance_rate)

    def init(
        alpha,
        beta,
        gamma,
        initial_step_size: float,
    ) -> PathfinderAdaptationState:
        """Initialze the adaptation state and parameter values.

        We use the Pathfinder algorithm to compute an estimate of the inverse
        mass matrix that will stay constant throughout the rest of the
        adaptation.

        Parameters
        ----------
        alpha, beta, gamma
            Factored representation of the inverse Hessian computed by the
            Pathfinder algorithm.
        initial_step_size
            The initial value for the step size.

        """
        inverse_mass_matrix = lbfgs_inverse_hessian_formula_1(alpha, beta, gamma)
        da_state = da_init(initial_step_size)
        warmup_state = PathfinderAdaptationState(
            da_state, initial_step_size, inverse_mass_matrix
        )

        return warmup_state

    def update(
        adaptation_state: PathfinderAdaptationState,
        position: ArrayLikeTree,
        acceptance_rate: float,
    ) -> PathfinderAdaptationState:
        """Update the adaptation state and parameter values.

        Since the value of the inverse mass matrix is already known we only
        update the state of the step size adaptation algorithm.

        Parameters
        ----------
        adaptation_state
            Current adptation state.
        position
            Current value of the model parameters.
        acceptance_rate
            Value of the acceptance rate for the last MCMC step.

        Returns
        -------
        The updated states of the chain and the warmup.

        """
        new_ss_state = da_update(adaptation_state.ss_state, acceptance_rate)
        new_step_size = jnp.exp(new_ss_state.log_step_size)

        return PathfinderAdaptationState(
            new_ss_state, new_step_size, adaptation_state.inverse_mass_matrix
        )

    def final(warmup_state: PathfinderAdaptationState) -> tuple[float, Array]:
        """Return the final values for the step size and inverse mass matrix."""
        step_size = jnp.exp(warmup_state.ss_state.log_step_size_avg)
        inverse_mass_matrix = warmup_state.inverse_mass_matrix
        return step_size, inverse_mass_matrix

    return init, update, final


def pathfinder_adaptation(
    algorithm,
    logdensity_fn: Callable,
    initial_step_size: float = 1.0,
    target_acceptance_rate: float = 0.80,
    adaptation_info_fn: Callable = return_all_adapt_info,
    **extra_parameters,
) -> AdaptationAlgorithm:
    """Adapt the value of the inverse mass matrix and step size parameters of
    algorithms in the HMC fmaily.

    Parameters
    ----------
    algorithm
        The algorithm whose parameters are being tuned.
    logdensity_fn
        The log density probability density function from which we wish to sample.
    initial_step_size
        The initial step size used in the algorithm.
    target_acceptance_rate
        The acceptance rate that we target during step size adaptation.
    adaptation_info_fn
        Function to select the adaptation info returned. See return_all_adapt_info
        and get_filter_adapt_info_fn in blackjax.adaptation.base.  By default all
        information is saved - this can result in excessive memory usage if the
        information is unused.
    **extra_parameters
        The extra parameters to pass to the algorithm, e.g. the number of
        integration steps for HMC.

    Returns
    -------
    A function that returns the last chain state and a sampling kernel with the
    tuned parameter values from an initial state.

    """

    mcmc_kernel = algorithm.build_kernel()

    adapt_init, adapt_update, adapt_final = base(
        target_acceptance_rate,
    )

    def one_step(carry, rng_key):
        state, adaptation_state = carry
        new_state, info = mcmc_kernel(
            rng_key,
            state,
            logdensity_fn,
            adaptation_state.step_size,
            adaptation_state.inverse_mass_matrix,
            **extra_parameters,
        )
        new_adaptation_state = adapt_update(
            adaptation_state, new_state.position, info.acceptance_rate
        )
        return (
            (new_state, new_adaptation_state),
            adaptation_info_fn(new_state, info, new_adaptation_state),
        )

    def run(rng_key: PRNGKey, position: ArrayLikeTree, num_steps: int = 400):
        init_key, sample_key, rng_key = jax.random.split(rng_key, 3)

        pathfinder_state, _ = vi.pathfinder.approximate(
            init_key, logdensity_fn, position
        )
        init_warmup_state = adapt_init(
            pathfinder_state.alpha,
            pathfinder_state.beta,
            pathfinder_state.gamma,
            initial_step_size,
        )

        init_position, _ = vi.pathfinder.sample(sample_key, pathfinder_state)
        init_state = algorithm.init(init_position, logdensity_fn)

        keys = jax.random.split(rng_key, num_steps)
        last_state, info = jax.lax.scan(
            one_step,
            (init_state, init_warmup_state),
            keys,
        )
        last_chain_state, last_warmup_state = last_state

        step_size, inverse_mass_matrix = adapt_final(last_warmup_state)
        parameters = {
            "step_size": step_size,
            "inverse_mass_matrix": inverse_mass_matrix,
            **extra_parameters,
        }

        return AdaptationResults(last_chain_state, parameters), info

    return AdaptationAlgorithm(run)
