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
from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

import blackjax.mcmc.hmc as hmc
import blackjax.mcmc.integrators as integrators
import blackjax.mcmc.metrics as metrics
import blackjax.mcmc.proposal as proposal
import blackjax.mcmc.termination as termination
import blackjax.mcmc.trajectory as trajectory
from blackjax.base import SamplingAlgorithm
from blackjax.types import ArrayLikeTree, ArrayTree, PRNGKey

__all__ = ["NUTSInfo", "init", "build_kernel", "as_top_level_api"]


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

    momentum: ArrayTree
    is_divergent: bool
    is_turning: bool
    energy: float
    trajectory_leftmost_state: integrators.IntegratorState
    trajectory_rightmost_state: integrators.IntegratorState
    num_trajectory_expansions: int
    num_integration_steps: int
    acceptance_rate: float


def build_kernel(
    integrator: Callable = integrators.velocity_verlet,
    divergence_threshold: int = 1000,
):
    """Build an iterative NUTS kernel.

    This algorithm is an iteration on the original NUTS algorithm :cite:p:`hoffman2014no`
    with two major differences:

    - We do not use slice samplig but multinomial sampling for the proposal
      :cite:p:`betancourt2017conceptual`;
    - The trajectory expansion is not recursive but iterative :cite:p:`phan2019composable`,
      :cite:p:`lao2020tfp`.

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

    """

    def kernel(
        rng_key: PRNGKey,
        state: hmc.HMCState,
        logdensity_fn: Callable,
        step_size: float,
        inverse_mass_matrix: metrics.MetricTypes,
        max_num_doublings: int = 10,
    ) -> tuple[hmc.HMCState, NUTSInfo]:
        """Generate a new sample with the NUTS kernel."""

        metric = metrics.default_metric(inverse_mass_matrix)
        symplectic_integrator = integrator(logdensity_fn, metric.kinetic_energy)
        proposal_generator = iterative_nuts_proposal(
            symplectic_integrator,
            metric.kinetic_energy,
            metric.check_turning,
            max_num_doublings,
            divergence_threshold,
        )

        key_momentum, key_integrator = jax.random.split(rng_key, 2)

        position, logdensity, logdensity_grad = state
        momentum = metric.sample_momentum(key_momentum, position)

        integrator_state = integrators.IntegratorState(
            position, momentum, logdensity, logdensity_grad
        )
        proposal, info = proposal_generator(key_integrator, integrator_state, step_size)
        proposal = hmc.HMCState(
            proposal.position, proposal.logdensity, proposal.logdensity_grad
        )
        return proposal, info

    return kernel


def as_top_level_api(
    logdensity_fn: Callable,
    step_size: float,
    inverse_mass_matrix: metrics.MetricTypes,
    *,
    max_num_doublings: int = 10,
    divergence_threshold: int = 1000,
    integrator: Callable = integrators.velocity_verlet,
) -> SamplingAlgorithm:
    """Implements the (basic) user interface for the nuts kernel.

    Examples
    --------

    A new NUTS kernel can be initialized and used with the following code:

    .. code::

        nuts = blackjax.nuts(logdensity_fn, step_size, inverse_mass_matrix)
        state = nuts.init(position)
        new_state, info = nuts.step(rng_key, state)

    We can JIT-compile the step function for more speed:

    .. code::

        step = jax.jit(nuts.step)
        new_state, info = step(rng_key, state)

    You can always use the base kernel should you need to:

    .. code::

       import blackjax.mcmc.integrators as integrators

       kernel = blackjax.nuts.build_kernel(integrators.yoshida)
       state = blackjax.nuts.init(position, logdensity_fn)
       state, info = kernel(rng_key, state, logdensity_fn, step_size, inverse_mass_matrix)

    Parameters
    ----------
    logdensity_fn
        The log-density function we wish to draw samples from.
    step_size
        The value to use for the step size in the symplectic integrator.
    inverse_mass_matrix
        The value to use for the inverse mass matrix when drawing a value for
        the momentum and computing the kinetic energy.
    max_num_doublings
        The maximum number of times we double the length of the trajectory before
        returning if no U-turn has been obserbed or no divergence has occured.
    divergence_threshold
        The absolute value of the difference in energy between two states above
        which we say that the transition is divergent. The default value is
        commonly found in other libraries, and yet is arbitrary.
    integrator
        (algorithm parameter) The symplectic integrator to use to integrate the trajectory.

    Returns
    -------
    A ``SamplingAlgorithm``.

    """
    kernel = build_kernel(integrator, divergence_threshold)

    def init_fn(position: ArrayLikeTree, rng_key=None):
        del rng_key
        return init(position, logdensity_fn)

    def step_fn(rng_key: PRNGKey, state):
        return kernel(
            rng_key,
            state,
            logdensity_fn,
            step_size,
            inverse_mass_matrix,
            max_num_doublings,
        )

    return SamplingAlgorithm(init_fn, step_fn)


def iterative_nuts_proposal(
    integrator: Callable,
    kinetic_energy: metrics.KineticEnergy,
    uturn_check_fn: metrics.CheckTurning,
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
