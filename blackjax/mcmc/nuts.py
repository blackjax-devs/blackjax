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
"""Public API for the NUTS Kernel"""
from typing import Callable, NamedTuple, Tuple

import jax
import jax.numpy as jnp
import numpy as np

import blackjax.mcmc.hmc as hmc
import blackjax.mcmc.integrators as integrators
import blackjax.mcmc.metrics as metrics
import blackjax.mcmc.proposal as proposal
import blackjax.mcmc.termination as termination
import blackjax.mcmc.trajectory as trajectory
from blackjax.types import Array, PRNGKey, PyTree

__all__ = ["NUTSInfo", "init", "kernel"]


init = hmc.init


class NUTSInfo(NamedTuple):
    """Additional information on the NUTS transition.

    This additional information can be used for debugging or computing
    diagnostics.

    momentum:
        The momentum that was sampled and used to integrate the trajectory.
    is_divergent
        Whether the difference in energy between the original and the new state
        exceeded the divergence threshold.
    is_turning
        Whether the sampling returned because the trajectory started turning
        back on itself.
    energy:
        Energy of the transition.
    trajectory_leftmost_state
        The leftmost state of the full trajectory.
    trajectory_rightmost_state
        The rightmost state of the full trajectory.
    num_trajectory_expansions
        Number of subtrajectory samples that were taken.
    num_integration_steps
        Number of integration steps that were taken. This is also the number of
        states in the full trajectory.
    acceptance_rate
        average acceptance probabilty across entire trajectory

    """

    momentum: PyTree
    is_divergent: bool
    is_turning: bool
    energy: float
    trajectory_leftmost_state: integrators.IntegratorState
    trajectory_rightmost_state: integrators.IntegratorState
    num_trajectory_expansions: int
    num_integration_steps: int
    acceptance_rate: float


def kernel(
    integrator: Callable = integrators.velocity_verlet,
    divergence_threshold: int = 1000,
    max_num_doublings: int = 10,
):
    """Build an iterative NUTS kernel.

    This algorithm is an iteration on the original NUTS algorithm [Hoffman2014]_
    with two major differences:

    - We do not use slice samplig but multinomial sampling for the proposal
      [Betancourt2017]_;
    - The trajectory expansion is not recursive but iterative [Phan2019]_,
      [Lao2020]_.

    The implementation can seem unusual for those familiar with similar
    algorithms. Indeed, we do not conceptualize the trajectory construction as
    building a tree. We feel that the tree lingo, inherited from the recursive
    version, is unnecessarily complicated and hides the more general concepts
    upon which the NUTS algorithm is built.

    NUTS, in essence, consists in sampling a trajectory by iteratively choosing
    a direction at random and integrating in this direction a number of times
    that doubles at every step. From this trajectory we continuously sample a
    proposal. When the trajectory turns on itself or when we have reached the
    maximum trajectory length we return the current proposal.

    Parameters
    ----------
    integrator
        The simplectic integrator used to build trajectories.
    divergence_threshold
        The absolute difference in energy above which we consider
        a transition "divergent".
    max_num_doublings
        The maximum number of times we expand the trajectory by
        doubling the number of steps if the trajectory does not
        turn onto itself.

    References
    ----------
    .. [Hoffman2014]: Hoffman, Matthew D., and Andrew Gelman.
                      "The No-U-Turn sampler: adaptively setting path lengths in Hamiltonian Monte Carlo."
                      J. Mach. Learn. Res. 15.1 (2014): 1593-1623.
    .. [Betancourt2017]: Betancourt, Michael.
                         "A conceptual introduction to Hamiltonian Monte Carlo."
                         arXiv preprint arXiv:1701.02434 (2017).
    .. [Phan2019]: Phan Du, Neeraj Pradhan, and Martin Jankowiak.
                   "Composable effects for flexible and accelerated probabilistic programming in NumPyro."
                   arXiv preprint arXiv:1912.11554 (2019).
    .. [Lao2020]: Lao, Junpeng, et al.
                  "tfp. mcmc: Modern markov chain monte carlo tools built for modern hardware."
                  arXiv preprint arXiv:2002.01184 (2020).

    """

    def one_step(
        rng_key: PRNGKey,
        state: hmc.HMCState,
        logdensity_fn: Callable,
        step_size: float,
        inverse_mass_matrix: Array,
    ) -> Tuple[hmc.HMCState, NUTSInfo]:
        """Generate a new sample with the NUTS kernel."""

        (
            momentum_generator,
            kinetic_energy_fn,
            uturn_check_fn,
        ) = metrics.gaussian_euclidean(inverse_mass_matrix)
        symplectic_integrator = integrator(logdensity_fn, kinetic_energy_fn)
        proposal_generator = iterative_nuts_proposal(
            symplectic_integrator,
            kinetic_energy_fn,
            uturn_check_fn,
            max_num_doublings,
            divergence_threshold,
        )

        key_momentum, key_integrator = jax.random.split(rng_key, 2)

        position, logdensity, logdensity_grad = state
        momentum = momentum_generator(key_momentum, position)

        integrator_state = integrators.IntegratorState(
            position, momentum, logdensity, logdensity_grad
        )
        proposal, info = proposal_generator(key_integrator, integrator_state, step_size)
        proposal = hmc.HMCState(
            proposal.position, proposal.logdensity, proposal.logdensity_grad
        )
        return proposal, info

    return one_step


def iterative_nuts_proposal(
    integrator: Callable,
    kinetic_energy: Callable,
    uturn_check_fn: Callable,
    max_num_expansions: int = 10,
    divergence_threshold: float = 1000,
) -> Callable:
    """Iterative NUTS proposal.

    Parameters
    ----------
    integrator
        Symplectic integrator used to build the trajectory step by step.
    kinetic_energy
        Function that computes the kinetic energy.
    uturn_check_fn:
        Function that determines whether the trajectory is turning on itself
        (metric-dependant).
    step_size
        Size of the integration step.
    max_num_expansions
        The number of sub-trajectory samples we take to build the trajectory.
    divergence_threshold
        Threshold above which we say that there is a divergence.

    Returns
    -------
    A kernel that generates a new chain state and information about the
    transition.

    """
    (
        new_termination_state,
        update_termination_state,
        is_criterion_met,
    ) = termination.iterative_uturn_numpyro(uturn_check_fn)

    trajectory_integrator = trajectory.dynamic_progressive_integration(
        integrator,
        kinetic_energy,
        update_termination_state,
        is_criterion_met,
        divergence_threshold,
    )

    expand = trajectory.dynamic_multiplicative_expansion(
        trajectory_integrator,
        uturn_check_fn,
        max_num_expansions,
    )

    def _compute_energy(state: integrators.IntegratorState) -> float:
        energy = -state.logdensity + kinetic_energy(state.momentum)
        return energy

    def propose(rng_key, initial_state: integrators.IntegratorState, step_size):
        initial_termination_state = new_termination_state(
            initial_state, max_num_expansions
        )
        initial_energy = _compute_energy(initial_state)  # H0 of the HMC step
        initial_proposal = proposal.Proposal(
            initial_state, initial_energy, 0.0, -np.inf
        )
        initial_trajectory = trajectory.Trajectory(
            initial_state,
            initial_state,
            initial_state.momentum,
            0,
        )
        initial_expansion_state = trajectory.DynamicExpansionState(
            0, initial_proposal, initial_trajectory, initial_termination_state
        )

        expansion_state, info = expand(
            rng_key, initial_expansion_state, initial_energy, step_size
        )
        is_diverging, is_turning = info
        num_doublings, sampled_proposal, new_trajectory, _ = expansion_state
        # Compute average acceptance probabilty across entire trajectory,
        # even over subtrees that may have been rejected
        acceptance_rate = (
            jnp.exp(sampled_proposal.sum_log_p_accept) / new_trajectory.num_states
        )

        info = NUTSInfo(
            initial_state.momentum,
            is_diverging,
            is_turning,
            sampled_proposal.energy,
            new_trajectory.leftmost_state,
            new_trajectory.rightmost_state,
            num_doublings,
            new_trajectory.num_states,
            acceptance_rate,
        )

        return sampled_proposal.state, info

    return propose
