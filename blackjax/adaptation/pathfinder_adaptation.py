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
from typing import NamedTuple, Tuple

import jax.numpy as jnp

from blackjax.adaptation.step_size import (
    DualAveragingAdaptationState,
    dual_averaging_adaptation,
)
from blackjax.optimizers.lbfgs import lbfgs_inverse_hessian_formula_1
from blackjax.types import Array, PyTree

__all__ = ["base"]


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
        position: PyTree,
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

    def final(warmup_state: PathfinderAdaptationState) -> Tuple[float, Array]:
        """Return the final values for the step size and inverse mass matrix."""
        step_size = jnp.exp(warmup_state.ss_state.log_step_size_avg)
        inverse_mass_matrix = warmup_state.inverse_mass_matrix
        return step_size, inverse_mass_matrix

    return init, update, final
