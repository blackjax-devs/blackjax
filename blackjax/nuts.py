"""Public API for the NUTS Kernel"""
from typing import Callable, Dict, List, NamedTuple, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np

import blackjax.hmc
import blackjax.inference.base as base
import blackjax.inference.integrators as integrators
import blackjax.inference.metrics as metrics
import blackjax.inference.proposals as proposals

Array = Union[np.array, jnp.DeviceArray]
PyTree = Union[Dict, List, Tuple]


class NUTSParameters(NamedTuple):
    step_size: float = 1e-3
    max_tree_depth: int = 100
    inv_mass_matrix: Array = None
    divergence_threshold: int = 1000


new_state = blackjax.hmc.new_state


def kernel(potential_fn: Callable, parameters: NUTSParameters) -> Callable:
    """Build an iterative NUTS kernel.

    Parameters
    ----------
    potential_fn
        A function that returns the potential energy of a chain at a given position. The potential energy
        is defined as minus the log-probability.
    parameters
        A NamedTuple that contains the parameters of the kernel to be built.

    """
    step_size, max_tree_depth, inv_mass_matrix, divergence_threshold = parameters

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
    termination_criterion = proposals.numpyro_uturn_criterion
    proposal = proposals.iterative_nuts(integrator, kinetic_energy_fn, u_turn_fn, termination_criterion, step_size, max_tree_depth, divergence_threshold)

    kernel = base.hmc(momentum_generator, proposal)

    return kernel
