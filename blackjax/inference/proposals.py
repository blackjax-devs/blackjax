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

        # Problem with weak_type = True in second operand
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
    uturn_check_fn: Callable,
    step_size: float,
    max_tree_depth: int = 10,
    divergence_threshold: float = 1000,
):
    """Iterative NUTS proposal."""

    # function that generates proposals from a transition between two states
    proposal_fn = proposal_generator(kinetic_energy, divergence_threshold)

    # iterative uturn criterion (here Numpyro's)
    (
        new_criterion_state,
        update_criterion_state,
        is_criterion_met,
    ) = numpyro_uturn_criterion(uturn_check_fn)
    
    # function that integrates the trajectory in one direction
    trajectory_integrator = dynamic_integration(
        integrator, proposal_fn, update_criterion_state, is_criterion_met
    )
    expand, do_keep_expanding = dynamic_multiplicative_expansion(
        trajectory_integrator,
        uturn_check_fn,
        step_size,
        max_tree_depth,
    )

    def propose(rng_key, initial_state):
        flat, unravel_fn = jax.flatten_util.ravel_pytree(initial_state.position)
        num_dims = jnp.shape(flat)[0]
        initial_momentum = unravel_fn(jnp.zeros_like(flat))

        proposal = Proposal(initial_state, 0, False)
        criterion_state = new_criterion_state(num_dims, max_tree_depth)
        trajectory = Trajectory(initial_state, initial_state, initial_momentum)
        _, _, proposal, _ = jax.lax.while_loop(
            do_keep_expanding,
            expand,
            (rng_key, trajectory, proposal, criterion_state, False, 0),
        )

        # Don't forget the proposal info here!

    return propose


# -------------------------------------------------------------------
#                            TRAJECTORY
# -------------------------------------------------------------------

# Q: Can I just create a class with `append` and `__add__` methods?
class Trajectory(NamedTuple):
    leftmost_state: IntegratorState
    rightmost_state: IntegratorState
    momentum_sum: PyTree


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


def merge_trajectories(
    direction: int, trajectory: Trajectory, new_trajectory: Trajectory
) -> Trajectory:
    """Merge two trajectories to form a new trajectory."""
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


# -------------------------------------------------------------------
#                   TRAJECTORY EXPANSION
# -------------------------------------------------------------------


def dynamic_multiplicative_expansion(
    integrator: Callable,
    uturn_check_fn: Callable,
    step_size: float,
    max_tree_depth: int = 10,
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

    def do_keep_expanding(expansion_state) -> bool:
        """Determine whether we need to keep expanding the trajectory."""
        _, trajectory, _, _, terminated_early, depth = expansion_state
        is_turning = uturn_check_fn(
            trajectory.leftmost_state.momentum,
            trajectory.rightmost_state.momentum,
            trajectory.momentum_sum,
        )
        return (depth < max_tree_depth) & ~terminated_early & ~is_turning

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
        rng_key, trajectory, proposal, termination_state, _, depth = expansion_state
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
            new_trajectory,
            new_proposal,
            termination_state,
            terminated_early,
        ) = integrator(
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
            lambda x: progressive_biased_sampling(*x),
        )

        return (
            rng_key,
            new_trajectory,
            maybe_updated_proposal,
            termination_state,
            terminated_early,
            depth + 1,
        )

    return expand, do_keep_expanding


def dynamic_integration(
    integrator: Callable,
    proposal_generator: Callable,
    update_termination: Callable,
    is_criterion_met: Callable,
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

    def integrate(
        rng_key: jax.random.PRNGKey,
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

        # QUESTION: Do we only signal a divergence if the proposal is issued from
        # a divergent transition (current) or anytime a divergence is detected?
        def do_keep_integrating(expansion_state):
            """Decide whether we should continue integrating the trajectory"""
            _, state, trajectory, proposal, termination_state, step = expansion_state
            has_terminated = is_criterion_met(termination_state, trajectory, state)
            is_diverging = proposal.is_diverging
            return (step < max_num_steps) & ~has_terminated & ~is_diverging

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
            new_trajectory = append_to_trajectory(direction, trajectory, new_state)

            # Q: couldn't we pass `state` instead of `(trajectory, direction)`?
            new_proposal = proposal_generator(direction, new_state, trajectory)
            maybe_updated_proposal = progressive_uniform_sampling(
                rng_key, proposal, new_proposal
            )

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

        initial_integration_state = (
            rng_key,
            initial_state,
            Trajectory(initial_state, initial_state, initial_state.momentum),
            Proposal(initial_state, 0, False),
            termination_state,
            0,
        )

        _, _, trajectory, proposal, termination_state, step = jax.lax.while_loop(
            do_keep_integrating, add_one_state, initial_integration_state
        )

        return proposal, trajectory, termination_state, (step < max_num_steps)

    return integrate


# -------------------
#     PROPOSALS
# -------------------


class Proposal(NamedTuple):
    """Proposal for the next chain step.

    state:
        The trajectory state corresponding to this proposal.
    weight:
        The logarithm of the sum of the canonical densities of each state
        :math:`e^{-H(z)}` along the trajectory.
    is_diverging
        Whether a divergence was observed when making one step.
    """

    state: IntegratorState
    log_weight: float
    is_diverging: bool


def proposal_generator(kinetic_energy: Callable, divergence_threshold: float):
    def generate(direction, state, trajectory):
        """Generate a new proposal from a trajectory state.

        Note
        ----
        We can optimize things a bit by sacrificing in memory and code
        complexity: the kinetic energy and the previous states can be carried
        around instead of being recomputed each time.

        Parameters
        ----------
        direction:
            The direction in which the trajectory was integrated to obtain the
            new state.
        state:
            The new state.
        trajectory:
            The trajectory before the state was added.

        """
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
        new_energy = state.potential_energy + kinetic_energy(
            state.position, state.momentum
        )

        delta_energy = new_energy - energy
        delta_energy = jnp.where(jnp.isnan(delta_energy), jnp.inf, delta_energy)
        is_diverging = delta_energy > divergence_threshold

        # The log-weight of the new proposal is equal to H(z) - H(z_new)?
        weight = -delta_energy

        return Proposal(
            state,
            weight,
            is_diverging,
        )

    return generate


def progressive_uniform_sampling(rng_key, proposal, new_proposal):
    """Generate a new proposal.

    To avoid keeping the entire trajectory in memory, we only memorize the
    extreme points and the point that will currently be proposed as a sample.
    Progressive sampling updates this proposal as the trajectory is being
    built. This is scheme is equivalent to drawing a sample uniformly at random
    from the final trajectory.

    """
    p_accept = jax.scipy.special.expit(new_proposal.log_weight - proposal.log_weight)
    do_accept = jax.random.bernoulli(rng_key, p_accept)

    updated_proposal = Proposal(
        new_proposal.state,
        jnp.logaddexp(proposal.log_weight, new_proposal.log_weight),
        new_proposal.is_diverging,
    )

    return jax.lax.cond(
        do_accept,
        updated_proposal,
        lambda x: x,
        proposal,
        lambda x: x,
    )


def progressive_biased_sampling(rng_key, proposal, new_proposal):
    """Generate a new proposal.

    To avoid keeping the entire trajectory in memory, we only memorize the
    extreme points and the point that will currently be proposed as a sample.
    Progressive sampling updates this proposal as the trajectory is being
    built.

    Unlike uniform sampling, biased sampling favors new proposals. It thus
    biases the transition away from the trajectory's initial state.

    """
    p_accept = jnp.exp(new_proposal.log_weight - proposal.log_weight)
    p_accept = jnp.clip(p_accept, a_max=1.0)
    do_accept = jax.random.bernoulli(rng_key, p_accept)

    updated_proposal = Proposal(
        new_proposal.state,
        jnp.logaddexp(proposal.log_weight, new_proposal.log_weight),
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

    def new_state(num_dims, max_tree_depth):
        return IterativeUTurnState(
            jnp.zeros((max_tree_depth, num_dims)),
            jnp.zeros((max_tree_depth, num_dims)),
            0,
            0,
        )

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

        r, _ = jax.flatten_util.ravel_pytree(state.momentum)
        r_sum, _ = jax.flatten_util.ravel_pytree(trajectory.momentum_sum)
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
