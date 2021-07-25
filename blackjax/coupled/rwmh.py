"""Public API for the coupled RWMH Kernel"""

from typing import Callable, Dict, List, NamedTuple, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental.host_callback import id_print
from jax.flatten_util import ravel_pytree

from blackjax.rwmh import RWMHInfo, RWMHState, new_state as new_rwmh_state

Array = Union[np.ndarray, jnp.DeviceArray]
PyTree = Union[Dict, List, Tuple]

__all__ = ["new_state", "kernel"]


class CoupledRWMHState(NamedTuple):
    state_1: RWMHState
    state_2: RWMHState


class CoupledRWMHInfo(NamedTuple):
    info_state_1: RWMHInfo
    info_state_2: RWMHInfo


def new_state(position_1: PyTree, position_2: PyTree, potential_fn: Callable) -> CoupledRWMHState:
    """ Creates two independent chain states from a position by calling base.new_hcm_state twice. """
    return CoupledRWMHState(new_rwmh_state(position_1, potential_fn),
                            new_rwmh_state(position_2, potential_fn))


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

    def one_step(rng_key: jnp.ndarray, state: CoupledRWMHState):
        """Moves the chain by one step using the Hamiltonian dynamics.

        Parameters
        ----------
        rng_key:
           The pseudo-random number generator key used to generate random numbers.
        state:
            The current state of the two coupled chains.

        Returns
        -------
        The next state of the chain and additional information about the current step.
        """
        momentum_key, proposal_key = jax.random.split(rng_key, 2)
        state_1 = state.state_1
        state_2 = state.state_2
        ravelled_position_1, unravel_fn = ravel_pytree(state_1.position)
        ravelled_position_2, _ = ravel_pytree(state_2.position)

        ravelled_proposal_1, ravelled_proposal_2 = reflected_gaussians(momentum_key, inverse_mass_matrix,
                                                                       ravelled_position_1, ravelled_position_2)
        proposal_1 = unravel_fn(ravelled_proposal_1)
        proposal_2 = unravel_fn(ravelled_proposal_2)

        u = jax.random.uniform(proposal_key)

        def get_next_state(prev_state, proposed_position):
            proposed_potential = potential_fn(proposed_position)
            p_accept = jnp.exp(prev_state.potential_energy - proposed_potential)

            do_accept = u < jnp.minimum(1, p_accept)
            proposal = RWMHState(proposed_position, proposed_potential)
            res = jax.lax.cond(do_accept, lambda _: proposal, lambda _: prev_state, operand=None)
            return res, RWMHInfo(p_accept, do_accept, proposal)

        state_1, info_1 = get_next_state(state_1, proposal_1)
        state_2, info_2 = get_next_state(state_2, proposal_2)
        state = CoupledRWMHState(state_1, state_2)
        info = CoupledRWMHState(info_1, info_2)
        return state, info

    return one_step


def _mvn_loglikelihood(x):
    return - 0.5 * jnp.sum(x ** 2)


def reflected_gaussians(rng_key, inverse_mass_matrix: jnp.DeviceArray,
                        position_1: jnp.ndarray, position_2: jnp.ndarray):
    """
    TODO: document this
    Example:
    --------
    >>> rng_key = jax.random.PRNGKey(42)
    >>> rng_keys = jax.random.split(rng_key, 10)
    >>> inverse_mass_matrix = jnp.ones((1,))
    >>> position_1 = -jnp.ones((1,))
    >>> position_2 = jnp.ones((1,))
    >>> res_1, res_2 = jax.vmap(reflected_gaussians, in_axes=[0, None, None, None])(rng_keys,
    ...                                                                             inverse_mass_matrix,
    ...                                                                             position_1,
    ...                                                                             position_2)
    >>> (jnp.corrcoef(res_1, res_2, rowvar=False)[0, 1] < 0.5).item()
    True
    """
    ndim = jnp.ndim(inverse_mass_matrix)

    if ndim != 1:  # diagonal mass matrix
        raise NotImplemented("Only diagonal matrices supported so far")

    shape = jnp.shape(inverse_mass_matrix)[:1]

    sqrt_inv_matrix = jnp.sqrt(inverse_mass_matrix)
    scaled_diff = jnp.multiply(sqrt_inv_matrix, position_1 - position_2)
    normed_scaled_diff = scaled_diff / jnp.sqrt(jnp.sum(scaled_diff ** 2))

    normal_key, uniform_key = jax.random.split(rng_key, 2)
    norm = jax.random.normal(normal_key, shape)
    log_u = jnp.log(jax.random.uniform(uniform_key, ()))

    do_accept = log_u + _mvn_loglikelihood(norm) < _mvn_loglikelihood(norm + scaled_diff)
    accept = lambda _: norm + scaled_diff
    reject = lambda _: norm - 2 * jnp.dot(norm, normed_scaled_diff) * normed_scaled_diff
    reflected_norm = jax.lax.cond(do_accept, accept, reject, operand=None)

    mass_matrix_sqrt = jnp.reciprocal(sqrt_inv_matrix)
    res_1 = position_1 + jnp.multiply(mass_matrix_sqrt, norm)
    res_2 = position_2 + jnp.multiply(mass_matrix_sqrt, reflected_norm)

    return res_1, res_2
