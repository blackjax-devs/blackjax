"""Implementation of the Pathinder warmup for the HMC family of sampling algorithms."""
from typing import Callable, NamedTuple, Tuple

import jax
import jax.numpy as jnp

from blackjax.adaptation.step_size import (
    DualAveragingAdaptationState,
    dual_averaging_adaptation,
)
from blackjax.mcmc.hmc import HMCState
from blackjax.optimizers.lbfgs import lbfgs_inverse_hessian_formula_1
from blackjax.types import Array, PRNGKey, PyTree
from blackjax.vi.pathfinder import init as pathfinder_init_fn
from blackjax.vi.pathfinder import sample_from_state

__all__ = ["base"]


class PathfinderAdaptationState(NamedTuple):
    da_state: DualAveragingAdaptationState
    inverse_mass_matrix: Array


def base(
    kernel_factory: Callable,
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
    da_init, da_update, da_final = dual_averaging_adaptation(
        target=target_acceptance_rate
    )

    def init(
        rng_key: PRNGKey, initial_position: Array, initial_step_size: float
    ) -> Tuple[PathfinderAdaptationState, PyTree]:
        """Initialize the warmup.

        To initialize the warmup we use pathfinder to estimate the inverse mass matrix and
        then we set up the dual averaging adaptation algorithm
        """
        da_state = da_init(initial_step_size)

        pathfinder_rng_key, sample_rng_key = jax.random.split(rng_key, 2)
        pathfinder_state = pathfinder_init_fn(
            pathfinder_rng_key, logprob_fn, initial_position
        )
        new_initial_position, _ = sample_from_state(sample_rng_key, pathfinder_state)
        inverse_mass_matrix = lbfgs_inverse_hessian_formula_1(
            pathfinder_state.alpha, pathfinder_state.beta, pathfinder_state.gamma
        )

        warmup_state = PathfinderAdaptationState(da_state, inverse_mass_matrix)

        return warmup_state, new_initial_position

    def update(
        rng_key: PRNGKey,
        chain_state: HMCState,
        adaptation_state: PathfinderAdaptationState,
    ) -> Tuple[HMCState, PathfinderAdaptationState, NamedTuple]:
        """Move the warmup by one step.

        We first create a new kernel with the current values of the step size
        and mass matrix and move the chain one step. Then, we update the dual
        averaging adaptation algorithm.

        Parameters
        ----------
        rng_key
            The key used in JAX's random number generator.
        chain_state
            Current state of the chain.
        adaprtation_state
            Current warmup state.

        Returns
        -------
        The updated states of the chain and the warmup.

        """
        step_size = jnp.exp(adaptation_state.da_state.log_step_size)
        inverse_mass_matrix = adaptation_state.inverse_mass_matrix
        kernel = kernel_factory(step_size, inverse_mass_matrix)

        chain_state, chain_info = kernel(rng_key, chain_state)
        new_da_state = da_update(adaptation_state.da_state, chain_info)
        new_warmup_state = PathfinderAdaptationState(
            new_da_state, adaptation_state.inverse_mass_matrix
        )

        return chain_state, new_warmup_state, chain_info

    def final(warmup_state: PathfinderAdaptationState) -> Tuple[float, Array]:
        """Return the step size and mass matrix."""
        step_size = jnp.exp(warmup_state.da_state.log_step_size_avg)
        inverse_mass_matrix = warmup_state.inverse_mass_matrix
        return step_size, inverse_mass_matrix

    return init, update, final
