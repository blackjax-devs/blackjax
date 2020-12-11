"""Public API for the HMC Kernel"""
from typing import Callable, NamedTuple, Optional, Union

import jax
import jax.numpy as jnp
import numpy as np

import blackjax.inference.base as base
import blackjax.inference.integrators as integrators
import blackjax.inference.metrics as metrics
import blackjax.inference.proposals as proposals

Array = Union[np.array, jnp.DeviceArray]


class HMCParameters(NamedTuple):
    step_size: float
    inv_mass_matrix: Array
    num_integration_steps: int


def new_states(position: jax.pytree, logpdf: Callable):
    """Create chain states from positions.

    The HMC kernel works with states that contain the current chain position
    and its associated potential energy and derivative of the potential energy.
    This function computes initial states from an initial position.

    While this function is intended to work for one chain, it is possible to
    use `jax.vmap` to compute the initial state of several chains.

    Example
    -------
    Let us assume a model with two random variables. We wish to sample from
    4 chains with the following initial positions:

        >>> import numpy as np
        >>> init_positions = (np.random.rand(4, 1000), np.random.rand(4, 300))

    We have a `logpdf` function that returns the log-probability associated with
    the chain at a given position:

        >>> logpdf((np.random.rand(1000), np.random.rand(3000)))
        -3.4

    We can compute the initial state for each of the 4 chain as follows:

        >>> import jax
        >>> jax.vmap(new_states, in_axes=(0, None))(init_positions, logpdf)

    Parameters
    ----------
    position
        The current values of the random variables whose posterior we want to
        sample from. Can be anything from a list, a (named) tuple or a dict of
        arrays. The arrays can either be Numpy arrays or JAX DeviceArrays.
    logpdf
        The joint log-probability function of the model.

    Returns
    -------
    A HMC state that contains the position, the associated potential energy and gradient of the
    potential energy.

    """
    potential_fn = lambda x: -logpdf(x)
    potential_energy, potential_energy_grad = jax.value_and_grad(potential_fn)(position)
    return base.HMCState(position, potential_energy, potential_energy_grad)


def new_parameters(
    step_size: float = 1e-3,
    num_integration_steps: int = 30,
    inv_mass_matrix: Optional[Array] = None,
) -> HMCParameters:
    """Create a new set of parameters.

    Wrapping this in a function is convenient. It gives us more leeway on the
    parameters' internal representation, allows us to provide defaults and to
    perform some checks.

    """
    step_size = float(step_size)
    num_integration_steps = int(num_integration_steps)

    return HMCParameters(step_size, inv_mass_matrix, num_integration_steps)


def kernel(logpdf: Callable, parameters: HMCParameters) -> Callable:
    """Build a HMC kernel.

    The details won't probably speak to you now, and we will dicuss the internals soon.
    The most important for now is that users do not need to be aware of these internals
    to have a good HMC or NUTS kernel.

    This clear separation means that we can provide easy-to-use tools at the same time as
    modular and re-usable building blocks.

    """
    if not parameters.inv_mass_matrix:
        raise ValueError(
            "Expected a value for `inv_mass_matrix`,"
            " got None. Please specify a value when initializing"
            " the parameters or run the window adaptation."
        )

    potential_fn = lambda x: -logpdf(x)
    step_size, inv_mass_matrix, num_integration_steps = parameters

    momentum_generator, kinetic_energy_fn = metrics.gaussian_euclidean(inv_mass_matrix)
    integrator = integrators.velocity_verlet(potential_fn, kinetic_energy_fn)
    proposal = proposals.hmc(integrator, step_size, num_integration_steps)
    kernel = base.hmc(proposal, momentum_generator, kinetic_energy_fn, potential_fn)

    return kernel
