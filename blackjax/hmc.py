"""Public API for the HMC Kernel"""
from typing import Callable, Dict, List, NamedTuple, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np

import blackjax.inference.base as base
import blackjax.inference.integrators as integrators
import blackjax.inference.metrics as metrics
import blackjax.inference.proposals as proposals

Array = Union[np.ndarray, jnp.DeviceArray]
PyTree = Union[Dict, List, Tuple]


class HMCParameters(NamedTuple):
    step_size: float = 1e-3
    num_integration_steps: int = 30
    inv_mass_matrix: Array = None
    divergence_threshold: int = 1000


def new_state(position: PyTree, potential_fn: Callable) -> base.HMCState:
    """Create a chain state from a position.

    The HMC kernel works with states that contain the current chain position
    and its associated potential energy and derivative of the potential energy.
    This function computes initial states from an initial position.  While this
    function is intended to work for one chain, it is possible to use
    `jax.vmap` to compute the initial state of several chains.

    Example
    -------
    Let us assume a model with two random variables. We wish to sample from
    4 chains with the following initial positions:

        >>> import numpy as np
        >>> init_positions = (np.random.rand(4, 1000), np.random.rand(4, 300))

    We have a `logpdf` function that returns the log-probability associated with
    the chain at a given position:

        >>> potential_fn((np.random.rand(1000), np.random.rand(3000)))
        -3.4

    We can compute the initial state for each of the 4 chain as follows:

        >>> import jax
        >>> jax.vmap(new_state, in_axes=(0, None))(init_positions, potential_fn)

    Parameters
    ----------
    position
        The current values of the random variables whose posterior we want to
        sample from. Can be anything from a list, a (named) tuple or a dict of
        arrays. The arrays can either be Numpy arrays or JAX DeviceArrays.
    potential_fn
        A function that returns the value of the potential energy when called
        with a position.

    Returns
    -------
    A HMC state that contains the position, the associated potential energy and gradient of the
    potential energy.
    """
    potential_energy, potential_energy_grad = jax.value_and_grad(potential_fn)(position)
    return base.HMCState(position, potential_energy, potential_energy_grad)


def kernel(potential_fn: Callable, parameters: HMCParameters) -> Callable:
    """Build a HMC kernel.

    Parameters
    ----------
    potential_fn
        A function that returns the potential energy of a chain at a given position.
    parameters
        A NamedTuple that contains the parameters of the kernel to be built.
    """
    step_size, num_integration_steps, inv_mass_matrix, divergence_threshold = parameters

    if inv_mass_matrix is None:
        raise ValueError(
            "Expected a value for `inv_mass_matrix`,"
            " got None. Please specify a value when initializing"
            " the parameters or run the window adaptation."
        )

    momentum_generator, kinetic_energy_fn = metrics.gaussian_euclidean(inv_mass_matrix)
    integrator = integrators.velocity_verlet(potential_fn, kinetic_energy_fn)
    proposal = proposals.hmc(integrator, step_size, num_integration_steps)
    kernel = base.hmc(
        proposal,
        momentum_generator,
        kinetic_energy_fn,
        divergence_threshold,
    )

    return kernel
