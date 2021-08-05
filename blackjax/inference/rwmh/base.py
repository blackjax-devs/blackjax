"""Public API for the Metropolis-Hastings Kernel"""
from typing import Callable, NamedTuple, Optional, Tuple

import jax.numpy as jnp
import jax.random

from blackjax.types import PRNGKey, PyTree

__all__ = ["new_rwmh_state", "rwmh", "RWMHState"]


class RWMHState(NamedTuple):
    """State of the MH chain

    position
        Current position of the chain.
    potential_energy
        Current value of the potential energy
    """

    position: PyTree
    potential_energy: float


class RWMHInfo(NamedTuple):
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
    proposal: RWMHState


def new_rwmh_state(position: PyTree, potential_fn: Callable) -> RWMHState:
    """Create a chain state from a position.

    Parameters:
    -----------
    position: PyTree
        The initial position of the chain
    potential_fn: Callable
        Target potential function of the chain
    """
    return RWMHState(position, potential_fn(position))


def rwmh(
    logprob_fn: Callable,
    proposal_generator: Callable,
    proposal_logprob_fn: Optional[Callable] = None,
):
    """Build a Random Walk Metropolis Hastings kernel.

    Parameters
    ----------
    potential_fn
        A function that returns the potential energy of a chain at a given position.
    proposal_generator
        A function that generates a new proposal.
    proposal_logprob_fn:
        For non-symmetric proposals, a function that returns the logprobability
        to obtain a given proposal knowing the current state. If it is not
        provided we assume the proposal is symmetric.

    Returns
    -------
    A kernel that takes a rng_key and a Pytree that contains the current state
    of the chain and that returns a new state of the chain along with
    information about the transition.

    """

    if proposal_logprob_fn is None:

        def acceptance_probability(state: RWMHState, proposal: RWMHState):
            return state.potential_energy - proposal.potential_energy

    else:

        def acceptance_probability(state: RWMHState, proposal: RWMHState):
            return (
                state.potential_energy
                + proposal_logprob_fn(state.position, proposal.position)  # type: ignore
                - proposal.potential_energy
                - proposal_logprob_fn(proposal.position, state.position)  # type: ignore
            )

    def kernel(rng_key: PRNGKey, state: RWMHState) -> Tuple[RWMHState, RWMHInfo]:
        """Moves the chain by one step using the Random Walk Metropolis Hastings algorithm.

        We temporarilly assume that the proposal distribution is symmetric.

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
        key_proposal, key_accept = jax.random.split(rng_key, 2)

        move_proposal = proposal_generator(key_proposal, state.position)
        new_position = jax.tree_util.tree_multimap(
            jnp.add, state.position, move_proposal
        )
        new_potential_energy = logprob_fn(new_position)
        new_state = RWMHState(new_position, new_potential_energy)

        delta = acceptance_probability(state, new_state)
        delta = jnp.where(jnp.isnan(delta), -jnp.inf, delta)
        p_accept = jnp.clip(jnp.exp(delta), a_max=1.0)

        do_accept = jax.random.bernoulli(key_accept, p_accept)
        accept_state = (new_state, RWMHInfo(p_accept, True, new_state))
        reject_state = (state, RWMHInfo(p_accept, False, new_state))

        return jax.lax.cond(
            do_accept, lambda _: accept_state, lambda _: reject_state, operand=None
        )

    return kernel
