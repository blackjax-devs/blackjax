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
"""Procedures to build trajectories for algorithms in the HMC family.

To propose a new state, algorithms in the HMC family generally proceed by [1]_:

1. Sampling a trajectory starting from the initial point;
2. Sampling a new state from this sampled trajectory.

Step (1) ensures that the process is reversible and thus that detailed balance
is respected. The traditional implementation of HMC does not sample a
trajectory, but instead takes a fixed number of steps in the same direction and
flips the momentum of the last state.

We distinguish here between two different methods to sample trajectories: static
and dynamic sampling. In the static setting we sample trajectories with a fixed
number of steps, while in the dynamic setting the total number of steps is
determined by a dynamic termination criterion. Traditional HMC falls in the
former category, NUTS in the latter.

There are also two methods to sample proposals from these trajectories. In the
static setting we first build the trajectory and then sample a proposal from
this trajectory. In the progressive setting we update the proposal as the
trajectory is being sampled. While the former is faster, we risk saturating the
memory by keeping states that will subsequently be discarded.


References
----------
.. [1]: Betancourt, Michael.
        "A conceptual introduction to Hamiltonian Monte Carlo."
        arXiv preprint arXiv:1701.02434 (2017).

"""
from typing import Callable, NamedTuple, Tuple

import jax
import jax.numpy as jnp

from blackjax.mcmc.integrators import IntegratorState
from blackjax.mcmc.proposal import (
    Proposal,
    progressive_biased_sampling,
    progressive_uniform_sampling,
    proposal_generator,
)
from blackjax.types import PRNGKey, PyTree


class Trajectory(NamedTuple):
    leftmost_state: IntegratorState
    rightmost_state: IntegratorState
    momentum_sum: PyTree
    num_states: int


def append_to_trajectory(trajectory: Trajectory, state: IntegratorState) -> Trajectory:
    """Append a state to the (right of the) trajectory to form a new trajectory."""
    momentum_sum = jax.tree_util.tree_map(
        jnp.add, trajectory.momentum_sum, state.momentum
    )
    return Trajectory(
        trajectory.leftmost_state, state, momentum_sum, trajectory.num_states + 1
    )


def reorder_trajectories(
    direction: int, trajectory: Trajectory, new_trajectory: Trajectory
) -> Tuple[Trajectory, Trajectory]:
    """Order the two trajectories depending on the direction."""
    return jax.lax.cond(
        direction > 0,
        lambda _: (
            trajectory,
            new_trajectory,
        ),
        lambda _: (
            new_trajectory,
            trajectory,
        ),
        operand=None,
    )


def merge_trajectories(left_trajectory: Trajectory, right_trajectory: Trajectory):
    momentum_sum = jax.tree_util.tree_map(
        jnp.add, left_trajectory.momentum_sum, right_trajectory.momentum_sum
    )
    return Trajectory(
        left_trajectory.leftmost_state,
        right_trajectory.rightmost_state,
        momentum_sum,
        left_trajectory.num_states + right_trajectory.num_states,
    )


# -------------------------------------------------------------------
#                             Integration
#
# Generating samples by choosing a direction and running the integrator
# several times along this direction. Distinct from sampling.
# -------------------------------------------------------------------


def static_integration(
    integrator: Callable,
    direction: int = 1,
) -> Callable:
    """Generate a trajectory by integrating several times in one direction."""

    def integrate(
        initial_state: IntegratorState, step_size, num_integration_steps
    ) -> IntegratorState:
        directed_step_size = jax.tree_map(
            lambda step_size: direction * step_size, step_size
        )

        def one_step(state, _):
            state = integrator(state, directed_step_size)
            return state, state

        last_state, _ = jax.lax.scan(
            one_step, initial_state, jnp.arange(num_integration_steps)
        )

        return last_state

    return integrate


class DynamicIntegrationState(NamedTuple):
    step: int
    proposal: Proposal
    trajectory: Trajectory
    termination_state: NamedTuple


def dynamic_progressive_integration(
    integrator: Callable,
    kinetic_energy: Callable,
    update_termination_state: Callable,
    is_criterion_met: Callable,
    divergence_threshold: float,
):
    """Integrate a trajectory and update the proposal sequentially in one direction
    until the termination criterion is met.

    Parameters
    ----------
    integrator
        The symplectic integrator used to integrate the hamiltonian trajectory.
    kinetic_energy
        Function to compute the current value of the kinetic energy.
    update_termination_state
        Updates the state of the termination mechanism.
    is_criterion_met
        Determines whether the termination criterion has been met.
    divergence_threshold
        Value of the difference of energy between two consecutive states above
        which we say a transition is divergent.

    """
    _, generate_proposal = proposal_generator(kinetic_energy, divergence_threshold)
    sample_proposal = progressive_uniform_sampling

    def integrate(
        rng_key: PRNGKey,
        initial_state: IntegratorState,
        direction: int,
        termination_state,
        max_num_steps: int,
        step_size,
        initial_energy,
    ):
        """Integrate the trajectory starting from `initial_state` and update the
        proposal sequentially (hence progressive) until the termination
        criterion is met (hence dynamic).

        Parameters
        ----------
        rng_key
            Key used by JAX's random number generator.
        initial_state
            The initial state from which we start expanding the trajectory.
        direction int in {-1, 1}
            The direction in which to expand the trajectory.
        termination_state
            The state that keeps track of the information needed for the
            termination criterion.
        max_num_steps
            The maximum number of integration steps. The expansion will stop
            when this number is reached if the termination criterion has not
            been met.
        step_size
            The step size of the symplectic integrator.
        initial_energy
            Initial energy H0 of the HMC step (not to confused with the initial
            energy of the subtree)

        """

        def do_keep_integrating(loop_state):
            """Decide whether we should continue integrating the trajectory"""
            _, integration_state, (is_diverging, has_terminated) = loop_state
            return (
                (integration_state.step < max_num_steps)
                & ~has_terminated
                & ~is_diverging
            )

        def add_one_state(loop_state):
            rng_key, integration_state, _ = loop_state
            step, proposal, trajectory, termination_state = integration_state
            rng_key, proposal_key = jax.random.split(rng_key)

            new_state = integrator(trajectory.rightmost_state, direction * step_size)
            new_proposal, is_diverging = generate_proposal(initial_energy, new_state)

            # At step 0, we always accept the proposal, since we
            # take one step to get the leftmost state of the tree.
            (new_trajectory, sampled_proposal) = jax.lax.cond(
                step == 0,
                lambda _: (
                    Trajectory(new_state, new_state, new_state.momentum, 1),
                    new_proposal,
                ),
                lambda _: (
                    append_to_trajectory(trajectory, new_state),
                    sample_proposal(proposal_key, proposal, new_proposal),
                ),
                operand=None,
            )

            new_termination_state = update_termination_state(
                termination_state, new_trajectory.momentum_sum, new_state.momentum, step
            )
            has_terminated = is_criterion_met(
                new_termination_state, new_trajectory.momentum_sum, new_state.momentum
            )

            new_integration_state = DynamicIntegrationState(
                step + 1,
                sampled_proposal,
                new_trajectory,
                new_termination_state,
            )

            return (rng_key, new_integration_state, (is_diverging, has_terminated))

        proposal_placeholder, _ = generate_proposal(initial_energy, initial_state)
        trajectory_placeholder = Trajectory(
            initial_state, initial_state, initial_state.momentum, 0
        )
        integration_state_placeholder = DynamicIntegrationState(
            0,
            proposal_placeholder,
            trajectory_placeholder,
            termination_state,
        )

        _, integration_state, (is_diverging, has_terminated) = jax.lax.while_loop(
            do_keep_integrating,
            add_one_state,
            (rng_key, integration_state_placeholder, (False, False)),
        )
        step, proposal, trajectory, termination_state = integration_state

        # In the while_loop we always extend on the right most direction.
        new_trajectory = jax.lax.cond(
            direction > 0,
            lambda _: trajectory,
            lambda _: Trajectory(
                trajectory.rightmost_state,
                trajectory.leftmost_state,
                trajectory.momentum_sum,
                trajectory.num_states,
            ),
            operand=None,
        )

        return (
            proposal,
            new_trajectory,
            termination_state,
            is_diverging,
            has_terminated,
        )

    return integrate


def dynamic_recursive_integration(
    integrator: Callable,
    kinetic_energy: Callable,
    uturn_check_fn: Callable,
    divergence_threshold: float,
    use_robust_uturn_check: bool = False,
):
    """Integrate a trajectory and update the proposal recursively in Python
    until the termination criterion is met.

    This is the implementation of Algorithm 6 from [1] with multinomial sampling.
    The implemenation here is mostly for validating the progressive implementation
    to make sure the two are equivalent. The recursive implementation should not
    be used for actually sampling as it cannot be jitted and thus likely slow.

    Parameters
    ----------
    integrator
        The symplectic integrator used to integrate the hamiltonian trajectory.
    kinetic_energy
        Function to compute the current value of the kinetic energy.
    uturn_check_fn
        Determines whether the termination criterion has been met.
    divergence_threshold
        Value of the difference of energy between two consecutive states above which we say a transition is divergent.
    use_robust_uturn_check
        Bool to indicate whether to perform additional U turn check between two trajectory.

    References
    ----------
    .. [1]: Hoffman, Matthew D., and Andrew Gelman. "The No-U-Turn sampler: adaptively setting path lengths in Hamiltonian Monte Carlo." J. Mach. Learn. Res. 15.1 (2014): 1593-1623.

    """
    _, generate_proposal = proposal_generator(kinetic_energy, divergence_threshold)
    sample_proposal = progressive_uniform_sampling

    def buildtree_integrate(
        rng_key: PRNGKey,
        initial_state: IntegratorState,
        direction: int,
        tree_depth: int,
        step_size,
        initial_energy: float,
    ):
        """Integrate the trajectory starting from `initial_state` and update
        the proposal recursively with tree doubling until the termination criterion is met.

        The function `buildtree_integrate` calls itself for tree_depth > 0, thus invokes
        the recursive scheme that builds a trajectory by doubling a binary tree.

        Parameters
        ----------
        rng_key
            Key used by JAX's random number generator.
        initial_state
            The initial state from which we start expanding the trajectory.
        direction int in {-1, 1}
            The direction in which to expand the trajectory.
        tree_depth
            The depth of the binary tree doubling.
        step_size
            The step size of the symplectic integrator.
        initial_energy
            Initial energy H0 of the HMC step (not to confused with the initial energy of the subtree)

        """
        if tree_depth == 0:
            # Base case - take one leapfrog step in the direction v.
            next_state = integrator(initial_state, direction * step_size)
            new_proposal, is_diverging = generate_proposal(initial_energy, next_state)
            trajectory = Trajectory(next_state, next_state, next_state.momentum, 1)
            return (
                rng_key,
                new_proposal,
                trajectory,
                is_diverging,
                False,
            )
        else:
            (
                rng_key,
                proposal,
                trajectory,
                is_diverging,
                is_turning,
            ) = buildtree_integrate(
                rng_key,
                initial_state,
                direction,
                tree_depth - 1,
                step_size,
                initial_energy,
            )
            # Note that is_diverging and is_turning is inplace updated
            if ~is_diverging & ~is_turning:
                start_state = jax.lax.cond(
                    direction > 0,
                    lambda _: trajectory.rightmost_state,
                    lambda _: trajectory.leftmost_state,
                    operand=None,
                )
                (
                    rng_key,
                    new_proposal,
                    new_trajectory,
                    is_diverging,
                    is_turning,
                ) = buildtree_integrate(
                    rng_key,
                    start_state,
                    direction,
                    tree_depth - 1,
                    step_size,
                    initial_energy,
                )
                left_trajectory, right_trajectory = reorder_trajectories(
                    direction, trajectory, new_trajectory
                )
                trajectory = merge_trajectories(left_trajectory, right_trajectory)

                if ~is_turning:
                    is_turning = uturn_check_fn(
                        trajectory.leftmost_state.momentum,
                        trajectory.rightmost_state.momentum,
                        trajectory.momentum_sum,
                    )
                    if use_robust_uturn_check & (tree_depth - 1 > 0):
                        momentum_sum_left = jax.tree_util.tree_map(
                            jnp.add,
                            left_trajectory.momentum_sum,
                            right_trajectory.leftmost_state.momentum,
                        )
                        is_turning_left = uturn_check_fn(
                            left_trajectory.leftmost_state.momentum,
                            right_trajectory.leftmost_state.momentum,
                            momentum_sum_left,
                        )
                        momentum_sum_right = jax.tree_util.tree_map(
                            jnp.add,
                            left_trajectory.rightmost_state.momentum,
                            right_trajectory.momentum_sum,
                        )
                        is_turning_right = uturn_check_fn(
                            left_trajectory.rightmost_state.momentum,
                            right_trajectory.rightmost_state.momentum,
                            momentum_sum_right,
                        )
                        is_turning = is_turning | is_turning_left | is_turning_right
                rng_key, proposal_key = jax.random.split(rng_key)
                proposal = sample_proposal(proposal_key, proposal, new_proposal)

        return (
            rng_key,
            proposal,
            trajectory,
            is_diverging,
            is_turning,
        )

    return buildtree_integrate


# -------------------------------------------------------------------
#                             Sampling
#
# Sampling a trajectory by choosing a direction at random and integrating
# the trajectory in this direction. In the simplest case we perform one
# integration step, but can also perform several as is the case in the
# NUTS algorithm.
# -------------------------------------------------------------------


class DynamicExpansionState(NamedTuple):
    step: int
    proposal: Proposal
    trajectory: Trajectory
    termination_state: NamedTuple


def dynamic_multiplicative_expansion(
    trajectory_integrator: Callable,
    uturn_check_fn: Callable,
    max_num_expansions: int = 10,
    rate: int = 2,
) -> Callable:
    """Sample a trajectory and update the proposal sequentially
    until the termination criterion is met.

    The trajectory is sampled with the following procedure:
    1. Pick a direction at random;
    2. Integrate `num_step` steps in this direction;
    3. If the integration has stopped prematurely, do not update the proposal;
    4. Else if the trajectory is performing a U-turn, return current proposal;
    5. Else update proposal, `num_steps = num_steps ** rate` and repeat from (1).

    Parameters
    ----------
    trajectory_integrator
        A function that runs the symplectic integrators and returns a new proposal
        and the integrated trajectory.
    uturn_check_fn
        Function used to check the U-Turn criterion.
    step_size
        The step size used by the symplectic integrator.
    max_num_expansions
        The maximum number of trajectory expansions until the proposal is
        returned.
    rate
        The rate of the geometrical expansion. Typically 2 in NUTS, this is why
        the literature often refers to "tree doubling".

    """
    proposal_sampler = progressive_biased_sampling

    def expand(
        rng_key: PRNGKey,
        initial_expansion_state: DynamicExpansionState,
        initial_energy: float,
        step_size: float,
    ):
        def do_keep_expanding(loop_state) -> bool:
            """Determine whether we need to keep expanding the trajectory."""
            _, expansion_state, (is_diverging, is_turning) = loop_state
            return (
                (expansion_state.step < max_num_expansions)
                & ~is_diverging
                & ~is_turning
            )

        def expand_once(loop_state):
            """Expand the current trajectory.

            At each step we draw a direction at random, build a subtrajectory
            starting from the leftmost or rightmost point of the current
            trajectory that is twice as long as the current trajectory.

            Once that is done, possibly update the current proposal with that of
            the subtrajectory.

            """
            rng_key, expansion_state, _ = loop_state
            step, proposal, trajectory, termination_state = expansion_state

            rng_key, direction_key, trajectory_key, proposal_key = jax.random.split(
                rng_key, 4
            )

            # create new subtrajectory that is twice as long as the current
            # trajectory.
            direction = jnp.where(jax.random.bernoulli(direction_key), 1, -1)
            start_state = jax.lax.cond(
                direction > 0,
                lambda _: trajectory.rightmost_state,
                lambda _: trajectory.leftmost_state,
                operand=None,
            )
            (
                new_proposal,
                new_trajectory,
                termination_state,
                is_diverging,
                is_turning_subtree,
            ) = trajectory_integrator(
                trajectory_key,
                start_state,
                direction,
                termination_state,
                rate**step,
                step_size,
                initial_energy,
            )

            # Update the proposal
            #
            # We do not accept proposals that come from diverging or turning
            # subtrajectories.  However the definition of the acceptance
            # probability is such that the acceptance probability needs to be
            # computed across the entire trajectory.
            def update_sum_log_p_accept(inputs):
                _, proposal, new_proposal = inputs
                return Proposal(
                    proposal.state,
                    proposal.energy,
                    proposal.weight,
                    jnp.logaddexp(
                        proposal.sum_log_p_accept, new_proposal.sum_log_p_accept
                    ),
                )

            updated_proposal = jax.lax.cond(
                is_diverging | is_turning_subtree,
                update_sum_log_p_accept,
                lambda x: proposal_sampler(*x),
                operand=(proposal_key, proposal, new_proposal),
            )

            # Is the full trajectory making a U-Turn?
            #
            # We first merge the subtrajectory that was just generated with the
            # trajectory and check the U-Turn criterior on the whole trajectory.
            left_trajectory, right_trajectory = reorder_trajectories(
                direction, trajectory, new_trajectory
            )

            merged_trajectory = merge_trajectories(left_trajectory, right_trajectory)

            is_turning = uturn_check_fn(
                merged_trajectory.leftmost_state.momentum,
                merged_trajectory.rightmost_state.momentum,
                merged_trajectory.momentum_sum,
            )

            new_state = DynamicExpansionState(
                step + 1, updated_proposal, merged_trajectory, termination_state
            )
            info = (is_diverging, is_turning_subtree | is_turning)

            return (rng_key, new_state, info)

        _, expansion_state, (is_diverging, is_turning) = jax.lax.while_loop(
            do_keep_expanding,
            expand_once,
            (rng_key, initial_expansion_state, (False, False)),
        )

        return expansion_state, (is_diverging, is_turning)

    return expand
