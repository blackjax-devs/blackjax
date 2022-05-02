"""Public API for Rosenbluth-Metropolis-Hastings kernels."""
from typing import Callable, NamedTuple, Optional, Tuple

import jax
import jax.numpy as jnp

import blackjax.mcmc.random_walk as random_walk
from blackjax.types import Array, PRNGKey, PyTree

__all__ = ["RMHState", "RMHInfo", "init", "kernel"]


class RMHState(NamedTuple):
    """State of the RMH chain.

    position
        Current position of the chain.
    log_probability
        Current value of the log-probability
    """

    position: PyTree
    log_probability: float


class RMHInfo(NamedTuple):
    """Additional information on the RMH chain.

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
    proposal: RMHState


def init(position: PyTree, logprob_fn: Callable) -> RMHState:
    """Create a chain state from a position.

    Parameters:
    -----------
    position: PyTree
        The initial position of the chain
    logprob_fn: Callable
        Log-probability density function of the distribution we wish to sample
        from.

    """
    return RMHState(position, logprob_fn(position))


def kernel():
    """Build a Random Walk Rosenbluth-Metropolis-Hastings kernel with a gaussian
    proposal distribution.

    Returns
    -------
    A kernel that takes a rng_key and a Pytree that contains the current state
    of the chain and that returns a new state of the chain along with
    information about the transition.

    """

    def one_step(
        rng_key: PRNGKey, state: RMHState, logprob_fn: Callable, sigma: Array
    ) -> Tuple[RMHState, RMHInfo]:

        proposal_generator = random_walk.normal(sigma)
        kernel = rmh(logprob_fn, proposal_generator)
        return kernel(rng_key, state)

    return one_step


# -----------------------------------------------------------------------------
# Rosenbluth-Metropolis-Hastings Step
#
# We keep this separate as the basis of a self-standing implementation of the
# RMH correction step that can be re-used across the library.
#
# (TODO) Separate the RMH step from the proposal.
# -----------------------------------------------------------------------------


def rmh(
    logprob_fn: Callable,
    proposal_generator: Callable,
    proposal_logprob_fn: Optional[Callable] = None,
):
    """Build a Rosenbluth-Metropolis-Hastings kernel.

    Parameters
    ----------
    logprob_fn
        A function that returns the log-probability at a given position.
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

        def acceptance_probability(state: RMHState, proposal: RMHState):
            return proposal.log_probability - state.log_probability

    else:

        def acceptance_probability(state: RMHState, proposal: RMHState):
            return (
                proposal.log_probability
                + proposal_logprob_fn(proposal.position, state.position)  # type: ignore
                - state.log_probability
                - proposal_logprob_fn(state.position, proposal.position)  # type: ignore
            )

    def kernel(rng_key: PRNGKey, state: RMHState) -> Tuple[RMHState, RMHInfo]:
        """Move the chain by one step using the Rosenbluth Metropolis Hastings
        algorithm.

        We temporarilly assume that the proposal distribution is symmetric.

        Parameters
        ----------
        rng_key:
           The pseudo-random number generator key used to generate random
           numbers.
        state:
            The current state of the chain.

        Returns
        -------
        The next state of the chain and additional information about the current
        step.
        """
        key_proposal, key_accept = jax.random.split(rng_key, 2)

        move_proposal = proposal_generator(key_proposal, state.position)
        new_position = jax.tree_util.tree_map(jnp.add, state.position, move_proposal)
        new_log_probability = logprob_fn(new_position)
        new_state = RMHState(new_position, new_log_probability)

        delta = acceptance_probability(state, new_state)
        delta = jnp.where(jnp.isnan(delta), -jnp.inf, delta)
        p_accept = jnp.clip(jnp.exp(delta), a_max=1.0)

        do_accept = jax.random.bernoulli(key_accept, p_accept)
        accept_state = (new_state, RMHInfo(p_accept, True, new_state))
        reject_state = (state, RMHInfo(p_accept, False, new_state))

        return jax.lax.cond(
            do_accept, lambda _: accept_state, lambda _: reject_state, operand=None
        )

    return kernel
