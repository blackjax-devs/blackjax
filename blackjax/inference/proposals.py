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

import blackjax.inference.mh_step as mh_step
import blackjax.inference.termination as termination
import blackjax.inference.trajectory as trajectory
import jax
import jax.numpy as jnp
from blackjax.inference.integrators import IntegratorState
from blackjax.inference.mh_step import Proposal
from blackjax.inference.trajectory import Trajectory

__all__ = ["HMCState", "HMCInfo", "hmc", "iterative_nuts"]


PyTree = Union[Dict, List, Tuple]
ProposalState = IntegratorState


class HMCTrajectoryInfo(NamedTuple):
    step_size: float
    num_integration_steps: int


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

    acceptance_probability
        The acceptance probability of the transition, linked to the energy
        difference between the original and the proposed states.
    is_accepted
        Whether the proposed position was accepted or the original position
        was returned.
    is_divergent
        Whether the difference in energy between the original and the new state
        exceeded the divergence threshold.
    proposal
        The state proposed by the proposal. Typically includes the position and
        momentum.
    proposal_info
        Information returned by the proposal. Typically includes the step size,
        number of integration steps and intermediate states.
    """

    momentum: PyTree
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
    """Vanilla HMC algorithm.

    The algorithm integrates the trajectory applying a symplectic integrator
    `num_integration_steps` times in one direction to get a proposal and
    uses a Metropolis-Hastings acceptance step.

    """
    integrate_trajectory = trajectory.static_integration(
        integrator, step_size, num_integration_steps
    )
    maybe_accept = mh_step.uniform_sampling(kinetic_energy, divergence_threshold)

    def flip_momentum(state: IntegratorState) -> IntegratorState:
        """To guarantee time-reversibility (hence detailed balance) we
        need to flip the last state's momentum. If we run the hamiltonian
        dynamics starting from the last state with flipped momentum we
        should indeed retrieve the initial state (with flipped momentum).

        """
        flipped_momentum = jax.tree_util.tree_multimap(
            lambda m: -1.0 * m, state.momentum
        )
        return IntegratorState(
            state.position,
            flipped_momentum,
            state.potential_energy,
            state.potential_energy_grad,
        )

    def generate(rng_key, state: IntegratorState) -> Tuple[IntegratorState, HMCInfo]:
        """Generate a new chain state."""
        end_state = integrate_trajectory(state)
        proposal = flip_momentum(end_state)
        new_state, *info = maybe_accept(rng_key, state, proposal)
        do_accept, p_accept, transition_info = info

        info = HMCInfo(
            state.momentum,
            p_accept,
            do_accept,
            transition_info.is_diverging,
            transition_info.new_energy,
            proposal,
            HMCTrajectoryInfo(step_size, num_integration_steps),
        )

        return new_state, info

    return generate


def iterative_nuts(
    integrator: Callable,
    kinetic_energy: Callable,
    uturn_check_fn: Callable,
    step_size: float,
    max_tree_depth: int = 10,
    divergence_threshold: float = 1000,
):
    """Iterative NUTS proposal."""

    (
        new_criterion_state,
        update_criterion_state,
        is_criterion_met,
    ) = termination.iterative_uturn_numpyro(uturn_check_fn)

    trajectory_integrator = trajectory.dynamic_progressive_integration(
        integrator,
        mh_step.progressive_uniform_sampling(kinetic_energy, divergence_threshold),
        update_criterion_state,
        is_criterion_met,
    )

    expand, do_keep_expanding = trajectory.dynamic_multiplicative_expansion(
        trajectory_integrator,
        uturn_check_fn,
        step_size,
        max_tree_depth,
    )

    def propose(rng_key, initial_state):
        flat, _ = jax.flatten_util.ravel_pytree(initial_state.position)
        num_dims = jnp.shape(flat)[0]
        criterion_state = new_criterion_state(num_dims, max_tree_depth)

        proposal = Proposal(initial_state, 0., 0.)
        trajectory = Trajectory(initial_state, initial_state, initial_state.momentum)

        _, _, proposal, _, _, _, _ = jax.lax.while_loop(
            do_keep_expanding,
            expand,
            (rng_key, trajectory, proposal, criterion_state, False, False, 1),
        )

        # Don't forget the proposal info here!
        return proposal.state, None

    return propose
