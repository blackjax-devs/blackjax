"""Public API for the Random Walk Metropolis-Hastings Kernel"""
from typing import Callable, NamedTuple

import jax.numpy as jnp
import jax.random
from jax.flatten_util import ravel_pytree

import blackjax.inference.metrics as metrics
from blackjax.common import Array, PyTree

__all__ = ["new_state", "kernel", "RWMHState"]


class RWMHState(NamedTuple):
    """ State of the RWMH chain

    position
        Current position of the chain.
    potential_energy
        Current value of the potential energy
    """
    position: PyTree
    potential_energy: float


class RWMHInfo(NamedTuple):
    """Additional information on the RWMH chain.

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


def new_state(position: PyTree, potential_fn: Callable) -> RWMHState:
    """Create a chain state from a position.

    Parameters:
    -----------
    position: PyTree
        The initial position of the chain
    potential_fn: Callable
        Target potential function of the chain
    """
    return RWMHState(position, potential_fn(position))


def kernel(
        potential_fn: Callable,
        inverse_mass_matrix: Array,
):
    """Build a RWMH kernel.

    Parameters
    ----------
    potential_fn
        A function that returns the potential energy of a chain at a given position.
    inverse_mass_matrix
        One or two-dimensional array corresponding respectively to a diagonal
        or dense mass matrix. The inverse mass matrix is multiplied to a
        flattened version of the Pytree in which the chain position is stored
        (the current value of the random variables). The order of the variables
        should thus match JAX's tree flattening order, and more specifically
        that of `ravel_pytree`.
        In particular, JAX sorts dictionaries by key when flattening them. The
        value of each variables will appear in the flattened Pytree following
        the order given by `sort(keys)`.

    Returns
    -------
    A kernel that takes a rng_key and a Pytree that contains the current state
    of the chain and that returns a new state of the chain along with
    information about the transition.

    """

    momentum_generator, *_ = metrics.gaussian_euclidean(inverse_mass_matrix)

    def one_step(rng_key: jnp.ndarray, state: RWMHState):
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
        momentum = momentum_generator(momentum_key, state.position, unravel=False)

        ravelled_position, unravel_fn = ravel_pytree(state.position)
        ravelled_proposed_position = ravelled_position + momentum
        proposed_position = unravel_fn(ravelled_proposed_position)
        proposed_potential = potential_fn(proposed_position)

        u = jax.random.uniform(proposal_key)
        p_accept = jnp.exp(state.potential_energy - proposed_potential)
        do_accept = u < p_accept

        new_position, new_potential = jax.lax.cond(
            do_accept,
            lambda _: (proposed_position, proposed_potential),
            lambda _: (state.position, state.potential_energy),
            operand=None)
        return RWMHState(new_position, new_potential), RWMHInfo(p_accept, do_accept, proposed_position)

    return one_step
