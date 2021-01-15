"""Proposals in the HMC family.

Proposals take the current state of the chain, transform it to another state
that is returned to the base kernel. Proposals can differ from one another on
two aspects: the way the step size is chosen and the way the number of
integration steps is chose.

The standard HMC algorithm integrates the same number of times with the same
step size [1]_. It is also common to draw at each step the number of
integration steps from a distribution [1,2]_ ; empirical HMC [2]_ for instance
learns this distribution during the adaptation phase. Other algorithms, like
NUTS [3, 4, 5]_, determine the number of integration steps dynamically at runtime.

References
----------
.. [1]: Duane, Simon, et al. "Hybrid monte carlo." Physics letters B 195.2 (1987): 216-222.
.. [2]: Wu, Changye, Julien Stoehr, and Christian P. Robert. "Faster
        Hamiltonian Monte Carlo by learning leapfrog scale." arXiv preprint
        arXiv:1810.04449 (2018).
.. [3]: Hoffman, Matthew D., and Andrew Gelman. "The No-U-Turn sampler:
        adaptively setting path lengths in Hamiltonian Monte Carlo." J. Mach. Learn.
        Res. 15.1 (2014): 1593-1623.
.. [4]: Phan, Du, Neeraj Pradhan, and Martin Jankowiak. "Composable effects for
        flexible and accelerated probabilistic programming in NumPyro." arXiv preprint
        arXiv:1912.11554 (2019).
.. [5]: Lao, Junpeng, et al. "tfp. mcmc: Modern Markov Chain Monte Carlo Tools
        Built for Modern Hardware." arXiv preprint arXiv:2002.01184 (2020).

"""
from typing import Callable, Dict, List, NamedTuple, Tuple, Union

import jax
import jax.numpy as jnp

from blackjax.inference.integrators import IntegratorState

__all__ = ["hmc"]


PyTree = Union[Dict, List, Tuple]


class HMCTrajectoryInfo(NamedTuple):
    step_size: float
    num_integration_steps: int


ProposalState = IntegratorState


class ProposalInfo(NamedTuple):
    """Additional information on the HMC transition.

    This additional information can be used for debugging or computing
    diagnostics.

    acceptance_probability
        The acceptance probability of the transition, linked to the energy
        difference between the original and the proposed states.
    is_accepted
        Whether the proposed position was accepted or the original position
        was returned.
    is_divergent
        Whether the difference in energy between the original and the new state
        exceeded the divergence threshold.
    energy
        The total energy that corresponds to the returned state.
    proposal
        The state proposed by the proposal. Typically includes the position and
        momentum.
    proposal_info
        Information returned by the proposal. Typically includes the step size,
        number of integration steps and intermediate states.
    """

    acceptance_probability: float
    is_accepted: bool
    is_divergent: bool
    energy: float
    proposal: IntegratorState
    trajectory_info: HMCTrajectoryInfo


def hmc(
    integrator: Callable,
    kinetic_energy: Callable,
    step_size: float,
    num_integration_steps: int = 1,
    divergence_threshold: float = 1000,
) -> Callable:
    """Vanilla HMC proposal running the integrator for a fixed number of steps"""

    def integrate_trajectory(
        initial_state: IntegratorState,
    ) -> IntegratorState:
        """Integrate the trajectory  `num_integration_steps` times starting
        from `initial_state` in the direction set by the momentum.
        """

        def one_step(state, _):
            state = integrator(state, step_size)
            return state, ()

        end_state, _ = jax.lax.scan(
            one_step, initial_state, jnp.arange(num_integration_steps)
        )

        # To guarantee time-reversibility (hence detailed balance) we
        # need to flip the last state's momentum. If we run the hamiltonian
        # dynamics starting from the last state with flipped momentum we
        # should indeed retrieve the initial state (with flipped momentum).
        flipped_momentum = jax.tree_util.tree_multimap(
            lambda m: -1.0 * m, end_state.momentum
        )
        end_state = IntegratorState(
            end_state.position,
            flipped_momentum,
            end_state.potential_energy,
            end_state.potential_energy_grad,
        )

        return end_state

    def accept_endpoint(
        rng_key: jax.random.PRNGKey,
        initial_state: IntegratorState,
        last_state: IntegratorState,
    ) -> Tuple[bool, float, bool, float]:
        initial_energy = initial_state.potential_energy + kinetic_energy(
            initial_state.momentum, initial_state.position
        )
        energy = last_state.potential_energy + kinetic_energy(
            last_state.momentum, last_state.position
        )

        delta_energy = initial_energy - energy
        delta_energy = jnp.where(jnp.isnan(delta_energy), -jnp.inf, delta_energy)
        is_diverging = jnp.abs(delta_energy) > divergence_threshold

        p_accept = jnp.clip(jnp.exp(delta_energy), a_max=1)
        do_accept = jax.random.bernoulli(rng_key, p_accept)

        return do_accept, p_accept, is_diverging, energy

    def propose(
        rng_key, initial_state: IntegratorState
    ) -> Tuple[IntegratorState, ProposalInfo]:

        end_state = integrate_trajectory(initial_state)
        do_accept, p_accept, is_diverging, energy = accept_endpoint(
            rng_key, initial_state, end_state
        )

        info = ProposalInfo(
            p_accept,
            do_accept,
            is_diverging,
            energy,
            end_state,
            HMCTrajectoryInfo(step_size, num_integration_steps),
        )

        return jax.lax.cond(
            do_accept,
            end_state,
            lambda state: (state, info),
            initial_state,
            lambda state: (state, info),
        )

    return propose


def iterative_nuts(
    integrator: Callable,
    kinetic_energy: Callable,
    u_turn_fn: Callable,
    termination_criterion: Callable,
    step_size: float,
    max_tree_depth: int = 10,
    divergence_threshold: float = 1000,
):
    proposal_fn = proposal_generator(kinetic_energy, divergence_threshold)
    propose = dynamic_multiplicative_expansion(
        integrator, termination_criterion, proposal_fn, step_size, max_tree_depth
    )
    """ Move `propose` from dynamic expansion to here."""
    return propose


class Trajectory(NamedTuple):
    leftmost_state: IntegratorState
    rightmost_state: IntegratorState
    momentum_sum: PyTree


# dynamic geometrical expansion
def dynamic_multiplicative_expansion(
    integrator: Callable,
    termination_criterion: Callable,
    proposal_generator: Callable,
    step_size: float,
    max_tree_depth: int = 10,
    rate: int = 2,
) -> Callable:

    expand_trajectory = dynamic_deterministic_expansion(
        integrator, proposal_generator, termination_criterion
    )

    def do_keep_expanding(expansion_state) -> bool:
        _, _, _, depth, _ = expansion_state
        return depth < max_tree_depth

    def merge_trajectories(
        direction, trajectory: Trajectory, new_trajectory: Trajectory
    ) -> Trajectory:
        leftmost_state, rightmost_state = jax.lax.cond(
            direction > 0,
            trajectory,
            lambda trajectory: (
                trajectory.leftmost_state,
                new_trajectory.rightmost_state,
            ),
            trajectory,
            lambda trajectory: (
                new_trajectory.leftmost_state,
                trajectory.rightmost_state,
            ),
        )
        momentum_sum = jax.tree_util.tree_multimap(
            jnp.add, trajectory.momentum_sum, new_trajectory.momentum_sum
        )
        return Trajectory(leftmost_state, rightmost_state, momentum_sum)

    def expand(expansion_state):
        """Dynamic expansionm, thus return a proposal along the last state"""
        rng_key, trajectory, proposal, depth = expansion_state
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
        subtrajectory, new_proposal, stopped_early = expand_trajectory(
            rng_key, start_state, direction, rate ** depth, step_size
        )

        # merge the subtrajectory to the trajectory
        # if subtrjectory is diverging or turning we may need to interupt the process.
        new_trajectory = merge_trajectories(direction, trajectory, subtrajectory)

        # update the proposal
        # if subtrajectory is diverging or turning we need to rejects its proposal.
        maybe_updated_proposal = jax.lax.cond(
            stopped_early,
            proposal,
            lambda x: x,
            (rng_key, proposal, new_proposal),
            lambda x: progressive_biaised_sampling(*x),
        )

        return rng_key, new_trajectory, maybe_updated_proposal, depth + 1

    def propose(rng_key, initial_state):
        """This may be moved to iterative_nuts."""
        proposal = Proposal(initial_state, 0, False)
        trajectory = Trajectory(initial_state, initial_state, 0)
        _, _, proposal, _ = jax.lax.while_loop(
            do_keep_expanding,
            expand,
            (rng_key, trajectory, proposal, 0),
        )

    return propose


def dynamic_deterministic_expansion(
    integrator, proposal_generator, termination_criterion
):
    """Sample a trajectory and update the proposal sequentially
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
    (
        new_termination_state,
        update_termination,
        is_criterion_met,
    ) = termination_criterion()

    def expand(
        rng_key: jax.random.PRNGKey,
        initial_state: IntegratorState,
        direction: int,
        max_num_steps: int,
        step_size,
    ):
        """Integrate a trajectory and update the proposal sequentially
        until the termination criterion is met.

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

        def do_keep_expanding(expansion_state):
            _, state, trajectory, proposal, checkpoints, step = expansion_state
            is_terminated = is_criterion_met(checkpoints, trajectory, state)
            is_diverging = proposal.is_diverging
            return (step < max_num_steps) & ~is_terminated & ~is_diverging

        def append_to_trajectory(direction, trajectory, state):
            """Append a state to the trajectory to form a new trajectory."""
            leftmost_state, rightmost_state = jax.lax.cond(
                direction > 0,
                trajectory,
                lambda trajectory: (trajectory.leftmost_state, state),
                trajectory,
                lambda trajectory: (state, trajectory.rightmost_state),
            )
            momentum_sum = jax.tree_util.tree_multimap(
                jnp.add, trajectory.momentum_sum, state.momentum
            )
            return Trajectory(leftmost_state, rightmost_state, momentum_sum)

        def add_one_state(expansion_state):
            """We need to compute the delta energy between the last and the new state
            proposed. Tree weight is taken to be energy_current - energy_new
            """
            (
                rng_key,
                state,
                trajectory,
                proposal,
                termination_state,
                step,
            ) = expansion_state
            _, rng_key = jax.random.split(rng_key)

            new_state = integrator(state, direction * step_size)
            new_proposal = proposal_generator(direction, state, trajectory)
            maybe_updated_proposal = progressive_uniform_sampling(
                rng_key, proposal, new_proposal
            )

            new_trajectory = append_to_trajectory(direction, trajectory, new_state)
            new_termination_state = update_termination(
                termination_state, new_trajectory, new_state, step
            )

            return (
                rng_key,
                new_state,
                new_trajectory,
                maybe_updated_proposal,
                new_termination_state,
                step + 1,
            )

        initial_expansion_state = (
            rng_key,
            initial_state,
            Trajectory(initial_state, initial_state, initial_state.momentum),
            Proposal(initial_state, 0, False),
            new_termination_state(),
            0,
        )

        _, trajectory, proposal, _, step = jax.while_cond(
            do_keep_expanding, add_one_state, initial_expansion_state
        )

        return proposal, trajectory, (step < max_num_steps)

    return expand


# -------------------
#     PROPOSAL
# -------------------


class Proposal(NamedTuple):
    """Proposal
    Specify how the weight is computed and updated.
    Also contains traditional ProposalInfo
    """

    state: IntegratorState
    weight: float
    is_diverging: bool


def proposal_generator(kinetic_energy: Callable, divergence_threshold: float):
    def generate(direction, state, trajectory):
        """Create a new proposal from a new trajectory state."""
        previous_state = jax.lax.cond(
            direction > 0,
            trajectory,
            lambda x: x.rightmost_state,
            trajectory,
            lambda x: x.leftmost_state,
        )

        energy = previous_state.potential_energy + kinetic_energy(
            previous_state.position, previous_state.momentum
        )
        new_energy = state.state.potential_energy + kinetic_energy(
            state.position, previous_state.momentum
        )

        delta_energy = new_energy - energy
        delta_energy = jnp.where(jnp.isnan(delta_energy), jnp.inf, delta_energy)
        is_diverging = delta_energy > divergence_threshold
        weight = -delta_energy

        return Proposal(
            state,
            weight,
            is_diverging,
        )

    return generate


def progressive_uniform_sampling(rng_key, proposal, new_proposal):
    """Update the proposal using biaised sampling."""
    p_accept = jax.scipy.special.expit(proposal.weight, new_proposal.weight)
    do_accept = jax.random.bernoulli(rng_key, p_accept)

    updated_proposal = Proposal(
        new_proposal.state,
        jnp.logaddexp(proposal.weight, new_proposal.weight),
        new_proposal.is_diverging,
    )

    return jax.lax.cond(
        do_accept,
        updated_proposal,
        lambda x: x,
        proposal,
        lambda x: x,
    )


def progressive_biaised_sampling(rng_key, proposal, new_proposal):
    """Update the proposal using biaised sampling."""
    p_accept = jnp.exp(proposal.weight - new_proposal.weight)
    p_accept = jnp.clip(p_accept, a_max=1.0)
    do_accept = jax.random.bernoulli(rng_key, p_accept)

    updated_proposal = Proposal(
        new_proposal.state,
        jnp.logaddexp(proposal.weight, new_proposal.weight),
        new_proposal.is_diverging,
    )

    return jax.lax.cond(
        do_accept,
        updated_proposal,
        lambda x: x,
        proposal,
        lambda x: x,
    )


# --------------------------
#   TERMINATION CRITERION
# --------------------------


class IterativeUTurnState(NamedTuple):
    momentum: jnp.ndarray
    momentum_sum: jnp.ndarray
    idx_min: int
    idx_max: int


def numpyro_uturn_criterion(is_turning):
    """Numpyro style dynamic U-Turn criterion."""

    def new_state():
        pass

    def update_criterion_state(
        checkpoints: IterativeUTurnState,
        trajectory: Trajectory,
        state: IntegratorState,
        step: int,
    ):
        r_ckpts, r_sum_ckpts, _, _ = checkpoints
        ckpt_idx_min, ckpt_idx_max = _leaf_idx_to_ckpt_idxs(step)
        r, _ = jax.flatten_util.ravel_pytree(state.momentum)
        r_sum, _ = jax.flatten_util.ravel_pytree(trajectory.momentum_sum)
        r_ckpts, r_sum_ckpts = jax.lax.cond(
            step % 2 == 0,
            (r_ckpts, r_sum_ckpts),
            lambda x: (
                jax.lax.index_update(x[0], ckpt_idx_max, r),
                jax.lax.index_update(x[1], ckpt_idx_max, r_sum),
            ),
            (r_ckpts, r_sum_ckpts),
            lambda x: x,
        )
        return IterativeUTurnState(r_ckpts, r_sum_ckpts, ckpt_idx_min, ckpt_idx_max)

    def _leaf_idx_to_ckpt_idxs(n):
        """Find the checkpoint id from a step number."""
        # computes the number of non-zero bits except the last bit
        # e.g. 6 -> 2, 7 -> 2, 13 -> 2
        _, idx_max = jax.lax.while_loop(
            lambda nc: nc[0] > 0,
            lambda nc: (nc[0] >> 1, nc[1] + (nc[0] & 1)),
            (n >> 1, 0),
        )
        # computes the number of contiguous last non-zero bits
        # e.g. 6 -> 0, 7 -> 3, 13 -> 1
        _, num_subtrees = jax.lax.while_loop(
            lambda nc: (nc[0] & 1) != 0, lambda nc: (nc[0] >> 1, nc[1] + 1), (n, 0)
        )
        # TODO: explore the potential of setting idx_min=0 to allow more turning checks
        # It will be useful in case: e.g. assume a tree 0 -> 7 is a circle,
        # subtrees 0 -> 3, 4 -> 7 are half-circles, which two leaves might not
        # satisfy turning condition;
        # the full tree 0 -> 7 is a circle, which two leaves might also not satisfy
        # turning condition;
        # however, we can check the turning condition of the subtree 0 -> 5, which
        # likely satisfies turning condition because its trajectory 3/4 of a circle.
        # XXX: make sure that detailed balance is satisfied if we follow this direction
        idx_min = idx_max - num_subtrees + 1
        return idx_min, idx_max

    def _is_iterative_turning(checkpoints, trajectory, state):
        """Checks whether there is a U-turn in the iteratively built expanded trajectory.
        These checks only need to be performed as specific points.

        Does that include the robust U-turn check?
        """

        r = state.momentum
        r_sum = trajectory.momentum_sum
        r_ckpts, r_sum_ckpts, idx_min, idx_max = checkpoints

        def _body_fn(state):
            i, _ = state
            subtree_r_sum = r_sum - r_sum_ckpts[i] + r_ckpts[i]
            return i - 1, is_turning(r_ckpts[i], r, subtree_r_sum)

        _, turning = jax.lax.while_loop(
            lambda it: (it[0] >= idx_min) & ~it[1], _body_fn, (idx_max, False)
        )
        return turning

    return new_state, update_criterion_state, _is_iterative_turning
