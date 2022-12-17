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
"""Public API for the HMC Kernel"""
from typing import Callable, NamedTuple, Tuple, Union

import jax

import blackjax.mcmc.integrators as integrators
import blackjax.mcmc.metrics as metrics
import blackjax.mcmc.proposal as proposal
import blackjax.mcmc.trajectory as trajectory
from blackjax.types import Array, PRNGKey, PyTree

__all__ = ["HMCState", "HMCInfo", "init", "kernel"]


class HMCState(NamedTuple):
    """State of the HMC algorithm.

    The HMC algorithm takes one position of the chain and returns another
    position. In order to make computations more efficient, we also store
    the current potential energy as well as the current gradient of the
    potential energy.

    """

    position: PyTree
    potential_energy: float
    potential_energy_grad: PyTree


class HMCInfo(NamedTuple):
    """Additional information on the HMC transition.

    This additional information can be used for debugging or computing
    diagnostics.

    momentum:
        The momentum that was sampled and used to integrate the trajectory.
    acceptance_rate
        The acceptance probability of the transition, linked to the energy
        difference between the original and the proposed states.
    is_accepted
        Whether the proposed position was accepted or the original position
        was returned.
    is_divergent
        Whether the difference in energy between the original and the new state
        exceeded the divergence threshold.
    energy:
        Energy of the transition.
    proposal
        The state proposed by the proposal. Typically includes the position and
        momentum.
    step_size
        Size of the integration step.
    num_integration_steps
        Number of times we run the symplectic integrator to build the trajectory

    """

    momentum: PyTree
    acceptance_rate: float
    is_accepted: bool
    is_divergent: bool
    energy: float
    proposal: integrators.IntegratorState
    num_integration_steps: int


def init(position: PyTree, logdensity_fn: Callable):
    def potential_fn(x):
        return -logdensity_fn(x)

    potential_energy, potential_energy_grad = jax.value_and_grad(potential_fn)(position)
    return HMCState(position, potential_energy, potential_energy_grad)


def kernel(
    integrator: Callable = integrators.velocity_verlet,
    divergence_threshold: float = 1000,
):
    """Build a HMC kernel.

    Parameters
    ----------
    integrator
        The symplectic integrator to use to integrate the Hamiltonian dynamics.
    divergence_threshold
        Value of the difference in energy above which we consider that the transition is divergent.

    Returns
    -------
    A kernel that takes a rng_key and a Pytree that contains the current state
    of the chain and that returns a new state of the chain along with
    information about the transition.

    """

    def one_step(
        rng_key: PRNGKey,
        state: HMCState,
        logdensity_fn: Callable,
        step_size: float,
        inverse_mass_matrix: Array,
        num_integration_steps: int,
    ) -> Tuple[HMCState, HMCInfo]:
        """Generate a new sample with the HMC kernel."""

        def potential_fn(x):
            return -logdensity_fn(x)

        momentum_generator, kinetic_energy_fn, _ = metrics.gaussian_euclidean(
            inverse_mass_matrix
        )
        symplectic_integrator = integrator(potential_fn, kinetic_energy_fn)
        proposal_generator = hmc_proposal(
            symplectic_integrator,
            kinetic_energy_fn,
            step_size,
            num_integration_steps,
            divergence_threshold,
        )

        key_momentum, key_integrator = jax.random.split(rng_key, 2)

        position, potential_energy, potential_energy_grad = state
        momentum = momentum_generator(key_momentum, position)

        integrator_state = integrators.IntegratorState(
            position, momentum, potential_energy, potential_energy_grad
        )
        proposal, info = proposal_generator(key_integrator, integrator_state)
        proposal = HMCState(
            proposal.position, proposal.potential_energy, proposal.potential_energy_grad
        )

        return proposal, info

    return one_step


def hmc_proposal(
    integrator: Callable,
    kinetic_energy: Callable,
    step_size: Union[float, PyTree],
    num_integration_steps: int = 1,
    divergence_threshold: float = 1000,
    *,
    sample_proposal: Callable = proposal.static_binomial_sampling,
) -> Callable:
    """Vanilla HMC algorithm.

    The algorithm integrates the trajectory applying a symplectic integrator
    `num_integration_steps` times in one direction to get a proposal and uses a
    Metropolis-Hastings acceptance step to either reject or accept this
    proposal. This is what people usually refer to when they talk about "the
    HMC algorithm".

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

    """
    build_trajectory = trajectory.static_integration(integrator)
    init_proposal, generate_proposal = proposal.proposal_generator(
        kinetic_energy, divergence_threshold
    )

    def generate(
        rng_key, state: integrators.IntegratorState
    ) -> Tuple[integrators.IntegratorState, HMCInfo]:
        """Generate a new chain state."""
        end_state = build_trajectory(state, step_size, num_integration_steps)
        end_state = flip_momentum(end_state)
        proposal = init_proposal(state)
        new_proposal, is_diverging = generate_proposal(proposal.energy, end_state)
        sampled_proposal, *info = sample_proposal(rng_key, proposal, new_proposal)
        do_accept, p_accept = info

        info = HMCInfo(
            state.momentum,
            p_accept,
            do_accept,
            is_diverging,
            new_proposal.energy,
            new_proposal,
            num_integration_steps,
        )

        return sampled_proposal.state, info

    return generate


def flip_momentum(
    state: integrators.IntegratorState,
) -> integrators.IntegratorState:
    """Flip the momentum at the end of the trajectory.

    To guarantee time-reversibility (hence detailed balance) we
    need to flip the last state's momentum. If we run the hamiltonian
    dynamics starting from the last state with flipped momentum we
    should indeed retrieve the initial state (with flipped momentum).

    """
    flipped_momentum = jax.tree_util.tree_map(lambda m: -1.0 * m, state.momentum)
    return integrators.IntegratorState(
        state.position,
        flipped_momentum,
        state.potential_energy,
        state.potential_energy_grad,
    )
