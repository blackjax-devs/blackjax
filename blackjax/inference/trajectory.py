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


def append_to_trajectory(
    direction: int, trajectory: Trajectory, state: IntegratorState
) -> Trajectory:
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
    update_termination_state: Callable,
    is_criterion_met: Callable,
    divergence_threshold: float,
):
    """Integrate a trajectory and update the proposal sequentially
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
        Value of the difference of energy between two consecutive states above which we say a transition is divergent.

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
        termination_state
            The state that keeps track of the information needed for the termination criterion.
        max_num_steps
            The maximum number of integration steps. The expansion will stop
            when this number is reached if the termination criterion has not
            been met.
        step_size
            The step size of the symplectic integrator.

        """

        def do_keep_integrating(expansion_state):
            """Decide whether we should continue integrating the trajectory"""
            _, step, _, _, _, is_diverging, has_terminated = expansion_state
            return (step < max_num_steps) & ~has_terminated & ~is_diverging

        def add_one_state(expansion_state):
            (
                rng_key,
                step,
                proposal,
                trajectory,
                termination_state,
                _,
                _,
            ) = expansion_state
            _, rng_key = jax.random.split(rng_key)

            new_state = integrator(proposal.state, direction * step_size)
            new_trajectory = append_to_trajectory(direction, trajectory, new_state)
            new_proposal, is_diverging = generate_proposal(proposal, new_state)
            sampled_proposal = sample_proposal(rng_key, proposal, new_proposal)
            new_termination_state = update_termination_state(
                termination_state, new_trajectory.momentum_sum, new_state.momentum, step
            )
            has_terminated = is_criterion_met(
                new_termination_state, new_trajectory.momentum_sum, new_state.momentum
            )

            return (
                rng_key,
                step + 1,
                sampled_proposal,
                new_trajectory,
                new_termination_state,
                is_diverging,
                has_terminated,
            )

        initial_proposal = init_proposal(initial_state)
        initial_integration_state = (
            rng_key,
            0,
            initial_proposal,
            Trajectory(initial_state, initial_state, initial_state.momentum),
            termination_state,
            False,
            False,
        )

        (
            _,
            step,
            proposal,
            trajectory,
            termination_state,
            is_diverging,
            has_terminated,
        ) = jax.lax.while_loop(
            do_keep_integrating, add_one_state, initial_integration_state
        )

        return proposal, trajectory, termination_state, is_diverging, has_terminated

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
    proposal_sampler = proposal.progressive_biased_sampling

    def expand(
        rng_key,
        initial_proposal: proposal.Proposal,
        criterion_state,
    ):
        def do_keep_expanding(expansion_state) -> bool:
            """Determine whether we need to keep expanding the trajectory."""
            _, step, trajectory, _, _, is_diverging, _, is_turning = expansion_state
            return (step < max_num_expansions) & ~is_diverging & ~is_turning

        def expand_once(expansion_state):
            """Expand the current trajectory.

            At each step we draw a direction at random, build a subtrajectory starting
            from the leftmost or rightmost point of the current trajectory that is
            twice as long as the current trajectory.

            Once that is done, possibly update the current proposal with that of
            the subtrajectory.

            """
            # Q: Should this function be aware of all the elements that need to
            # be passed downstream?
            (
                rng_key,
                step,
                proposal,
                trajectory,
                termination_state,
                _,
                _,
                _,
            ) = expansion_state
            rng_key, direction_key = jax.random.split(rng_key, 2)

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
                has_terminated,
            ) = trajectory_integrator(
                rng_key,
                start_state,
                direction,
                termination_state,
                rate ** step,
                step_size,
            )

            left_trajectory, right_trajectory = reorder_trajectories(
                direction, trajectory, new_trajectory
            )

            # robust u-turn check when merging the two trajectory
            # note this is different from the robust u-turn check done during
            # trajectory building.
            # momentum_sum_left = jax.tree_util.tree_multimap(
            #     jnp.add,
            #     left_trajectory.momentum_sum,
            #     right_trajectory.leftmost_state.momentum,
            # )
            # is_turning_left = uturn_check_fn(
            #     left_trajectory.leftmost_state.momentum,
            #     right_trajectory.leftmost_state.momentum,
            #     momentum_sum_left,
            # )
            # momentum_sum_right = jax.tree_util.tree_multimap(
            #     jnp.add,
            #     left_trajectory.rightmost_state.momentum,
            #     right_trajectory.momentum_sum,
            # )
            # is_turning_right = uturn_check_fn(
            #     left_trajectory.rightmost_state.momentum,
            #     right_trajectory.rightmost_state.momentum,
            #     momentum_sum_right,
            # )

            # merge the freshly integrated trajectory to the current trajectory
            momentum_sum = jax.tree_util.tree_multimap(
                jnp.add, left_trajectory.momentum_sum, right_trajectory.momentum_sum
            )
            merged_trajectory = Trajectory(
                left_trajectory.leftmost_state,
                right_trajectory.rightmost_state,
                momentum_sum,
            )

            # update the proposal
            # we reject proposals coming from diverging or turning subtrajectories
            sampled_proposal = jax.lax.cond(
                is_diverging | has_terminated,
                proposal,
                lambda x: x,
                (rng_key, proposal, new_proposal),
                lambda x: proposal_sampler(*x),
            )

            is_turning = uturn_check_fn(
                merged_trajectory.leftmost_state.momentum,
                merged_trajectory.rightmost_state.momentum,
                merged_trajectory.momentum_sum,
            )

            return (
                rng_key,
                step + 1,
                sampled_proposal,
                merged_trajectory,
                termination_state,
                is_diverging,
                has_terminated,
                is_turning,  # | is_turning_left | is_turning_right,
            )

        initial_state = initial_proposal.state
        initial_trajectory = Trajectory(
            initial_state,
            initial_state,
            initial_state.momentum,
        )

        (
            _,
            step,
            new_proposal,
            trajectory,
            _,
            is_diverging,
            has_terminated,
            is_turning,
        ) = jax.lax.while_loop(
            do_keep_expanding,
            expand_once,
            (
                rng_key,
                0,
                initial_proposal,
                initial_trajectory,
                criterion_state,
                False,
                False,
                False,
            ),
        )

        return new_proposal, trajectory, step, is_diverging, has_terminated, is_turning

    return expand
