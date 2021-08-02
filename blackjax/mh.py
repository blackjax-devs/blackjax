"""Public API for the Random Walk Metropolis-Hastings Kernel"""
from typing import Callable, NamedTuple, Tuple

import jax.numpy as jnp
import jax.random

from blackjax.types import PyTree, PRNGKey, Array

__all__ = ["new_state", "kernel", "MHState"]


class MHState(NamedTuple):
    """ State of the MH chain

    position
        Current position of the chain.
    potential_energy
        Current value of the potential energy
    """
    position: PyTree
    potential_energy: float


class MHInfo(NamedTuple):
    """Additional information on the MH chain.

    This additional information can be used for debugging or computing
    diagnostics.

    acceptance_probability
        The acceptance probability of the transition, linked to the energy
        difference between the original and the proposed states.
    is_accepted
        Whether the proposed position was accepted or the original position
        was returned.
    proposal
        The state proposed by the proposal.
    """

    acceptance_probability: float
    is_accepted: bool
    proposal: PyTree


def new_state(position: PyTree, potential_fn: Callable) -> MHState:
    """Create a chain state from a position.

    Parameters:
    -----------
    position: PyTree
        The initial position of the chain
    potential_fn: Callable
        Target potential function of the chain
    """
    return MHState(position, potential_fn(position))


def kernel(
        potential_fn: Callable,
        get_proposal_generator: Callable[[],
                                         Tuple[Callable[[PRNGKey, Array], Array],
                                               Callable[[Array, Array], float]]]
):
    """Build a MH kernel.

    Parameters
    ----------
    potential_fn
        A function that returns the potential energy of a chain at a given position.
    get_proposal_generator
        A function that takes no argument and returns two functions:
        proposal_generator
            A function that generates proposal sample from a PRNGKey and the current position.
        proposal_loglikelihood_fn
            A function that returns the two states that takes two states and computes the conditional
            loglikelihood between them.

    Returns
    -------
    A kernel that takes a rng_key and a Pytree that contains the current state
    of the chain and that returns a new state of the chain along with
    information about the transition.

    """

    proposal_generator, proposal_loglikelihood_fn = get_proposal_generator()

    def one_step(rng_key: jnp.ndarray, state: MHState):
        """Moves the chain by one step using the Hamiltonian dynamics.

        Parameters
        ----------
        rng_key:
           The pseudo-random number generator key used to generate random numbers.
        state:
            The current state of the chain.

        Returns
        -------
        The next state of the chain and additional information about the current step.
        """
        momentum_key, proposal_key = jax.random.split(rng_key, 2)
        proposal = proposal_generator(momentum_key, state.position)
        proposed_potential = potential_fn(proposal)

        proposal_conditional_on_state = proposal_loglikelihood_fn(proposal, state.position)
        state_conditional_on_proposal = proposal_loglikelihood_fn(state.position, proposal)

        u = jax.random.uniform(proposal_key)
        p_accept = jnp.exp(state.potential_energy - proposed_potential +
                           state_conditional_on_proposal - proposal_conditional_on_state)
        do_accept = u < jnp.clip(p_accept, 0., 1.)

        new_position, new_potential = jax.lax.cond(
            do_accept,
            lambda _: (proposal, proposed_potential),
            lambda _: (state.position, state.potential_energy),
            operand=None)
        return MHState(new_position, new_potential), MHInfo(p_accept, do_accept, proposal)

    return one_step
