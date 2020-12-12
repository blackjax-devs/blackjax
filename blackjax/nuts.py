"""Public API for the NUTS kernel

My only question here is: do we expose both recursive and iterative NUTS?

"""
from typing import Callable, NamedTuple, Union

import jax
import jax.numpy as jnp
import numpy as np

import blackjax.hmc
import blackjax.inference.base as base
import blackjax.inference.integrators as integrators
import blackjax.inference.metrics as metrics
import blackjax.inference.proposals as proposals

Array = Union[np.array, jnp.DeviceArray]


class NUTSParameters(NamedTuple):
    """This could also be a dataclass that checks type with __post_init__"""
    step_size: float = 1e-3
    max_tree_depth: int = 100
    inv_mass_matrix: Array = None


new_states = blackjax.hmc.new_states


def kernel(logpdf: Callable, parameters: NUTSParameters) -> Callable:
    """Build an iterative NUTS kernel.

    Note that only the line corresponding to the proposal differs from the HMC
    kernel defined in `hmc.py`.

    """
    potential_fn = lambda x: -logpdf(x)
    step_size, inv_mass_matrix, max_tree_depth = parameters

    if not inv_mass_matrix:
        raise ValueError(
            "Expected a value for `inv_mass_matrix`,"
            " got None. Please specify a value when initializing"
            " the parameters or run the window adaptation."
        )

    momentum_generator, kinetic_energy_fn, u_turn_fn = metrics.gaussian_euclidean(
        inv_mass_matrix
    )
    integrator = integrators.velocity_verlet(potential_fn, kinetic_energy_fn)
    proposal = proposals.nuts(integrator, u_turn_fn, step_size, max_tree_depth)
    kernel = base.hmc(proposal, momentum_generator, kinetic_energy_fn, potential_fn)

    return kernel
