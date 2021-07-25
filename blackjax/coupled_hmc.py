"""Public API for the coupled HMC Kernel"""

from typing import Callable, Dict, List, NamedTuple, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np

import blackjax.inference.base as base
import blackjax.inference.integrators as integrators
import blackjax.inference.metrics as metrics
import blackjax.inference.proposal as proposal
import blackjax.inference.trajectory as trajectory
from blackjax.hmc import generate as hmc_generate

Array = Union[np.ndarray, jnp.DeviceArray]
PyTree = Union[Dict, List, Tuple]

__all__ = ["new_state", "kernel"]


class CoupledHMCState(NamedTuple):
    state_1: base.HMCState
    state_2: base.HMCState


class CoupledHMCInfo(NamedTuple):
    info_state_1: base.HMCState
    info_state_2: base.HMCState


def new_state(position_1: PyTree, position_2: PyTree, potential_fn: Callable) -> CoupledHMCState:
    """ Creates two independent chain states from a position by calling base.new_hcm_state twice. """
    return CoupledHMCState(base.new_hmc_state(position_1, potential_fn),
                           base.new_hmc_state(position_2, potential_fn))


def kernel(
        potential_fn: Callable,
        step_size: float,
        inverse_mass_matrix: Array,
        num_integration_steps: int,
        *,
        integrator: Callable = integrators.velocity_verlet,
        divergence_threshold: int = 1000,
):
    """Build a coupled HMC kernel.

    Parameters
    ----------
    potential_fn
        A function that returns the potential energy of a chain at a given position.
    **parameters
        A NamedTuple that contains the parameters of the kernel to be built.

    Returns
    -------
    A kernel that takes a rng_key and a Pytree that contains the current state
    of the chain and that returns a new state of the chain along with
    information about the transition.

    """
    momentum_generator, kinetic_energy_fn, _ = metrics.gaussian_euclidean(
        inverse_mass_matrix
    )
    symplectic_integrator = integrator(potential_fn, kinetic_energy_fn)
    proposal_generator = coupled_hmc_proposal(
        symplectic_integrator,
        kinetic_energy_fn,
        step_size,
        num_integration_steps,
        divergence_threshold,
    )

    def one_step(rng_key: jnp.ndarray, state: CoupledHMCState) -> Tuple[CoupledHMCState, CoupledHMCInfo]:
        """Moves the chain by one step using the Hamiltonian dynamics.

        Parameters
        ----------
        rng_key:
           The pseudo-random number generator key used to generate random numbers.
        state:
            The current state of the two chain: position, log-probability and gradient
            of the log-probability.

        Returns
        -------
        The next state of the chain and additional information about the current step.
        """
        key_momentum, key_integrator = jax.random.split(rng_key, 2)

        state_1, state_2 = state

        position_1, potential_energy_1, potential_energy_grad_1 = state_1
        position_2, potential_energy_2, potential_energy_grad_2 = state_2
        # TODO: atm position is only used for matching tree structure, but the line below could become problematic
        #       if it ended up being used to actually generate the momentum.
        momentum = momentum_generator(key_momentum, position_1)  # same momentum for both chains

        augmented_state_1 = integrators.IntegratorState(
            position_1, momentum, potential_energy_1, potential_energy_grad_1
        )
        augmented_state_2 = integrators.IntegratorState(
            position_2, momentum, potential_energy_2, potential_energy_grad_2
        )
        proposal_1, proposal_2, info = proposal_generator(key_integrator, augmented_state_1, augmented_state_2)
        proposal_1 = base.HMCState(
            proposal_1.position, proposal_1.potential_energy, proposal_1.potential_energy_grad
        )
        proposal_2 = base.HMCState(
            proposal_2.position, proposal_2.potential_energy, proposal_2.potential_energy_grad
        )
        proposal = CoupledHMCState(proposal_1, proposal_2)
        return proposal, info

    return one_step


def coupled_hmc_proposal(
        integrator: Callable,
        kinetic_energy: Callable,
        step_size: float,
        num_integration_steps: int = 1,
        divergence_threshold: float = 1000,
) -> Callable:
    """Coupled HMC algorithm.

    The algorithm integrates two trajectories with the same momentum perturbation, applying a symplectic integrator
    `num_integration_steps` times in one direction to get proposals and uses the same uniform sample in the
    Metropolis-Hastings acceptance step to couple rejection or acceptance of the proposals.

    Parameters
    ----------
    integrator
        Symplectic integrator used to build the trajectory step by step.
    kinetic_energy
        Function that computes the kinetic energy.
    step_size
        Size of the integration step.
    num_integration_steps
        Number of times we run the symplectic integrator to build the trajectory
    divergence_threshold
        Threshold above which we say that there is a divergence.

    Returns
    -------
    A kernel that generates a new chain state and information about the transition.

    References
    ----------
    [1] J. Heng, P. E. Jacob, Unbiased Hamiltonian Monte Carlo with couplings,
        Biometrika, Volume 106, Issue 2, June 2019, Pages 287–302, https://doi.org/10.1093/biomet/asy074
    """
    build_trajectory = trajectory.static_integration(
        integrator, step_size, num_integration_steps
    )
    init_proposal, generate_proposal = proposal.proposal_generator(
        kinetic_energy, divergence_threshold
    )
    sample_proposal = proposal.static_binomial_sampling

    def generate(
            rng_key, state_1: integrators.IntegratorState, state_2: integrators.IntegratorState
    ) -> Tuple[integrators.IntegratorState, integrators.IntegratorState, CoupledHMCInfo]:
        """Generate a new chain state."""

        state_1, info_1 = hmc_generate(rng_key, state_1, build_trajectory, init_proposal, generate_proposal,
                                       sample_proposal, num_integration_steps)
        state_2, info_2 = hmc_generate(rng_key, state_2, build_trajectory, init_proposal, generate_proposal,
                                       sample_proposal, num_integration_steps)

        info = CoupledHMCInfo(info_1, info_2)

        return state_1, state_2, info

    return generate
