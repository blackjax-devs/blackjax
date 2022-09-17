"""Implementation of the Pathinder warmup for the HMC family of sampling algorithms."""
from typing import Callable, NamedTuple, Tuple

import jax
import jax.numpy as jnp

from blackjax.adaptation.step_size import (
    DualAveragingAdaptationState,
    dual_averaging_adaptation,
)
from blackjax.optimizers.lbfgs import lbfgs_inverse_hessian_formula_1
from blackjax.types import Array, PRNGKey, PyTree
from blackjax.vi.pathfinder import init as pathfinder_init_fn
from blackjax.vi.pathfinder import sample_from_state

__all__ = ["base"]


class PathfinderAdaptationState(NamedTuple):
    ss_state: DualAveragingAdaptationState
    step_size: float
    inverse_mass_matrix: Array


def base(
    logprob_fn: Callable,
    target_acceptance_rate: float = 0.65,
):
    """Warmup scheme for sampling procedures based on euclidean manifold HMC.
     This function tunes the values of the step size and the mass matrix according
     to this schema:
         * pathfinder algorithm is run and an estimation of the inverse mass matrix
           is derived, as well as an initialization point for the markov chain
         * Nesterov's dual averaging adaptation is then run to tune the step size

     Parameters
     ----------
     kernel_factory
         A function which returns a transition kernel given a step size and a
         mass matrix.
    logprob_fn
         The log density probability density function from which we wish to sample.
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
        rng_key: PRNGKey, position: PyTree, initial_step_size: float
    ) -> Tuple[PathfinderAdaptationState, PyTree]:
        """Initialize the warmup.

        To initialize the warmup we use pathfinder to estimate the inverse mass matrix and
        then we set up the dual averaging adaptation algorithm
        """
        da_state = da_init(initial_step_size)

        pathfinder_rng_key, sample_rng_key = jax.random.split(rng_key, 2)
        pathfinder_state = pathfinder_init_fn(pathfinder_rng_key, logprob_fn, position)
        new_initial_position, _ = sample_from_state(sample_rng_key, pathfinder_state)
        inverse_mass_matrix = lbfgs_inverse_hessian_formula_1(
            pathfinder_state.alpha, pathfinder_state.beta, pathfinder_state.gamma
        )

        warmup_state = PathfinderAdaptationState(
            da_state, initial_step_size, inverse_mass_matrix
        )

        return warmup_state, new_initial_position

    def update(
        adaptation_state: PathfinderAdaptationState,
        position: PyTree,
        acceptance_rate: float,
    ) -> PathfinderAdaptationState:
        """Move the warmup by one step.

        We first create a new kernel with the current values of the step size
        and mass matrix and move the chain one step. Then, we update the dual
        averaging adaptation algorithm.

        Parameters
        ----------
        adaptation_state
            Current adptation state.
        adaptation_stage
            The current stage of the warmup: whether this is a slow window,
            a fast window and if we are at the last step of a slow window.
        position
            Current value of the model parameters.
        acceptance_rate
            Value of the acceptance rate for the last mcmc step.

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
        """Return the final values for the step size and mass matrix."""
        step_size = jnp.exp(warmup_state.ss_state.log_step_size_avg)
        inverse_mass_matrix = warmup_state.inverse_mass_matrix
        return step_size, inverse_mass_matrix

    return init, update, final
