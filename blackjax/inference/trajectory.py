"""Procedures to build trajectories for algorithms in the HMC family.

While the traditional implementation of HMC generates proposals by running the
integrators in a given direction and flipping the momentum of the last state,
this is only one possible way to generate trajectories and thus proposals.

The next level of complexity would be to choose directions at random at each
step, thus *sampling* a trajectory (which ensures detailed balance). NUTS goes
even further by choosing the direction at random and runs the interator a
number of times that is a function of the current step.

In this file we implement various ways of sampling trajectories. As in [1]_ we
distinguish between:

    - static trajectory sampling: we first sample a trajectory and then
      generate a proposal from this trajectory; this requires to store the
      whole trajectory in memory. We can also update proposals as we sample the
      trajectory; in this progressive scheme we only need to store the
      endpoints of the trajectory and the current proposal.

    - dynamic trajectory sampling: we stop sampling when a certain critetion is
      met.

References
----------
.. [1]: Betancourt, Michael. "A conceptual introduction to Hamiltonian Monte Carlo." arXiv preprint arXiv:1701.02434 (2017).

"""
from typing import Callable, Dict, List, NamedTuple, Tuple, Union

import jax
import jax.numpy as jnp

import blackjax.inference.proposal as proposal
from blackjax.inference.integrators import IntegratorState

PyTree = Union[Dict, List, Tuple]


class Trajectory(NamedTuple):
    leftmost_state: IntegratorState
    rightmost_state: IntegratorState
    momentum_sum: PyTree


def append_to_trajectory(direction, trajectory, state):
    """Append a state to the trajectory to form a new trajectory."""
    leftmost_state, rightmost_state = jax.lax.cond(
        direction > 0,
        lambda _: (trajectory.leftmost_state, state),
        lambda _: (state, trajectory.rightmost_state),
        operand=None,
    )
    momentum_sum = jax.tree_util.tree_multimap(
        jnp.add, trajectory.momentum_sum, state.momentum
    )
    return Trajectory(leftmost_state, rightmost_state, momentum_sum)


def merge_trajectories(
    direction: int, trajectory: Trajectory, new_trajectory: Trajectory
) -> Trajectory:
    """Merge two trajectories to form a new trajectory."""
    leftmost_state, rightmost_state = jax.lax.cond(
        direction > 0,
        lambda _: (
            trajectory.leftmost_state,
            new_trajectory.rightmost_state,
        ),
        lambda _: (
            new_trajectory.leftmost_state,
            trajectory.rightmost_state,
        ),
        operand=None,
    )
    momentum_sum = jax.tree_util.tree_multimap(
        jnp.add, trajectory.momentum_sum, new_trajectory.momentum_sum
    )
    return Trajectory(leftmost_state, rightmost_state, momentum_sum)


# -------------------------------------------------------------------
#                             Integration
#
# Generating samples by choosing a direction and running the integrator
# several times along this direction. Distinct from sampling.
# -------------------------------------------------------------------


def static_integration(
    integrator: Callable,
    step_size: float,
    num_integration_steps: int,
    direction: int = 1,
) -> Callable:
    """Generate a trajectory by integrating several times in one direction."""

    directed_step_size = direction * step_size

    def integrate(initial_state: IntegratorState) -> IntegratorState:
        def one_step(state, _):
            state = integrator(state, directed_step_size)
            return state, state

        last_state, _ = jax.lax.scan(
            one_step, initial_state, jnp.arange(num_integration_steps)
        )

        return last_state

    return integrate


def static_progressive_integration(
    integrator_step: Callable,
    update_proposal: Callable,
    step_size: float,
    num_integration_steps: int,
    direction: int = 1,
) -> Callable:
    """Generate a trajectory by integrating in one direction and updating the
    proposal at each step.

    """

    directed_step_size = direction * step_size

    def integrate(rng_key, initial_state: IntegratorState) -> IntegratorState:
        def one_step(integration_step, _):
            rng_key, state, proposal = integration_step
            _, rng_key = jax.random.split(rng_key)
            new_state = integrator_step(state, directed_step_size)
            new_proposal = update_proposal(rng_key, new_state, proposal)
            return (rng_key, new_state, new_proposal), new_proposal

        last_state, _ = jax.lax.scan(
            one_step, (initial_state, initial_state), jnp.arange(num_integration_steps)
        )

        return last_state[-1]

    return integrate


def dynamic_progressive_integration(
    integrator: Callable,
    kinetic_energy: Callable,
    update_termination: Callable,
    is_criterion_met: Callable,
    divergence_threshold: float,
):
    """Integrate a trajectory and update the proposal sequentially
    until the termination criterion is met.

    Parameters
    ----------
    integrator
        The symplectic integrator used to integrate the hamiltonian trajectory.
    termination_criterion
        The criterion used to stop the sampling. This function generates a function that
        initializes the state, a function that updates it and a function that determines
        whether the criterion is met.
    progressive_sample
        The sampler used to update the proposal as the trajectory is being sampled. Generates
        a function that initializes the proposal state and one that updates it.

    """
    init_proposal, generate_proposal = proposal.proposal_generator(
        kinetic_energy, divergence_threshold
    )
    sample_proposal = proposal.progressive_uniform_sampling

    def integrate(
        rng_key: jax.numpy.DeviceArray,
        initial_state: IntegratorState,
        direction: int,
        termination_state,
        max_num_steps: int,
        step_size,
    ):
        """Integrate the trajectory starting from `initial_state` and update
        the proposal sequentially until the termination criterion is met.

        Parameters
        ----------
        rng_key
            Key used by JAX's random number generator.
        initial_state
            The initial state from which we start expanding the trajectory.
        direction int in {-1, 1}
            The direction in which to expand the trajectory.
        max_num_steps
            The maximum number of integration steps. The expansion will stop
            when this number is reached if the termination criterion has not
            been met.
        step_size
            The step size of the symplectic integrator.

        """

        def do_keep_integrating(expansion_state):
            """Decide whether we should continue integrating the trajectory"""
            _, _, _, _, is_diverging, has_terminated, step = expansion_state
            return (step < max_num_steps) & ~has_terminated & ~is_diverging

        def add_one_state(expansion_state):
            """We need to compute the delta energy between the last and the new state
            proposed. Tree weight is taken to be energy_current - energy_new
            """
            (
                rng_key,
                proposal,
                trajectory,
                termination_state,
                _,
                _,
                step,
            ) = expansion_state
            _, rng_key = jax.random.split(rng_key)

            new_state = integrator(proposal.state, direction * step_size)
            new_trajectory = append_to_trajectory(direction, trajectory, new_state)
            new_proposal, is_diverging = generate_proposal(proposal, new_state)
            sampled_proposal = sample_proposal(rng_key, proposal, new_proposal)
            new_termination_state = update_termination(
                termination_state, new_trajectory, new_state, step
            )
            has_terminated = is_criterion_met(
                new_termination_state, new_trajectory, new_state
            )

            return (
                rng_key,
                sampled_proposal,
                new_trajectory,
                new_termination_state,
                is_diverging,
                has_terminated,
                step + 1,
            )

        initial_proposal = init_proposal(initial_state)
        initial_integration_state = (
            rng_key,
            initial_proposal,
            Trajectory(initial_state, initial_state, initial_state.momentum),
            termination_state,
            False,
            False,
            0,
        )

        _, proposal, trajectory, termination_state, _, _, step = jax.lax.while_loop(
            do_keep_integrating, add_one_state, initial_integration_state
        )

        return proposal, trajectory, termination_state, (step < max_num_steps)

    return integrate


# -------------------------------------------------------------------
#                             Sampling
#
# Sampling a trajectory by choosing a direction at random and integrating
# the trajectory in this direction. In the simplest case we perform one
# integration step, but can also perform several as is the case in the
# NUTS algorithm.
# -------------------------------------------------------------------


def dynamic_multiplicative_expansion(
    trajectory_integrator: Callable,
    uturn_check_fn: Callable,
    step_size: float,
    max_num_doublings: int = 10,
    rate: int = 2,
) -> Tuple[Callable, Callable]:
    """Sample a trajectory and update the proposal sequentially
    until the termination criterion is met.

    The trajectory is sampled with the following procedure:
    1. Pick a direction at random
    2. Integrate `num_step` steps in this direction
    3. If integration stopped prematurely, do not update proposal
    4. Else if trajectory is performing a U-turn, return proposal
    5. Else update proposal, `num_steps = num_steps ** rate` and repeat from (1).

    Parameters
    ----------
    integrator
    uturn_check_fn
        Function used to check the U-Turn criterion.
    step_size
        The step size used by the trajectory integrator.
    max_tree_depth
        The maximum number of trajectory expansions until the proposal is
        returned.
    rate
        The rate of the geometrical expansion. Typically 2 in NUTS, hence
        the mentions to binary trees.

    """
    proposal_sampler = proposal.progressive_biased_sampling

    def do_keep_expanding(expansion_state) -> bool:
        """Determine whether we need to keep expanding the trajectory."""
        _, trajectory, _, _, terminated_early, is_turning, depth = expansion_state
        return (depth <= max_num_doublings) & ~terminated_early & ~is_turning

    def expand(expansion_state):
        """Expand the current trajectory.

        At each step we draw a direction at random, build a subtrajectory starting
        from the leftmost or rightmost point of the current trajectory that is
        twice as long as the current trajectory.

        Once that is done, possibly update the current proposal with that of
        the subtrajectory.

        """
        # Q: Should this function be aware of all the elements that need to
        # be passed downstream?
        rng_key, trajectory, proposal, termination_state, _, _, depth = expansion_state
        rng_key, direction_key = jax.random.split(rng_key, 2)

        # create new subtrajectory that is twice as long as the current
        # trajectory.
        direction = jnp.where(jax.random.bernoulli(direction_key), 1, -1)
        start_state = jax.lax.cond(
            direction > 0,
            trajectory,
            lambda trajectory: trajectory.rightmost_state,
            trajectory,
            lambda trajectory: trajectory.leftmost_state,
        )
        (
            new_proposal,
            new_trajectory,
            termination_state,
            terminated_early,
        ) = trajectory_integrator(
            rng_key, start_state, direction, termination_state, rate ** depth, step_size
        )

        # merge the freshly integrated trajectory to the current trajectory
        new_trajectory = merge_trajectories(direction, trajectory, new_trajectory)

        # update the proposal
        # we reject proposals coming from diverging or turning subtrajectories
        maybe_updated_proposal = jax.lax.cond(
            terminated_early,
            proposal,
            lambda x: x,
            (rng_key, proposal, new_proposal),
            lambda x: proposal_sampler(*x),
        )

        is_turning = uturn_check_fn(
            trajectory.leftmost_state.momentum,
            trajectory.rightmost_state.momentum,
            trajectory.momentum_sum,
        )

        return (
            rng_key,
            new_trajectory,
            maybe_updated_proposal,
            termination_state,
            terminated_early,
            is_turning,
            depth + 1,
        )

    return expand, do_keep_expanding
