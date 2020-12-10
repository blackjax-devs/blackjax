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
    step_size: float
    inv_mass_matrix: Array
    max_tree_depth: int


new_states = blackjax.hmc.new_states


def new_parameters(
    inv_mass_matrix: Array, step_size: float = 1e-3, max_tree_depth=10
) -> NUTSParameters:
    try:
        step_size = float(step_size)
    except ValueError:
        raise ValueError(f"Could not convert `step_size` to float: {step_size}")

    try:
        max_tree_depth = int(max_tree_depth)
    except ValueError:
        raise ValueError(
            f"Could not convert `num_integration_steps` to int {max_tree_depth}"
        )

    return NUTSParameters(step_size, inv_mass_matrix, max_tree_depth)


def kernel(logpdf: Callable, parameters: NUTSParameters) -> Callable:
    """Build an iterative NUTS kernel.

    Note that only the line corresponding to the proposal differs from the HMC
    kernel defined in `hmc.py`.

    """
    potential_fn = lambda x: -logpdf(x)
    step_size, inv_mass_matrix, max_tree_depth = parameters

    momentum_generator, kinetic_energy_fn = metrics.gaussian_euclidean(inv_mass_matrix)
    integrator = integrators.velocity_verlet(potential_fn, kinetic_energy_fn)
    proposal = proposals.nuts(integrator, step_size, max_tree_depth)
    kernel = base.hmc(proposal, momentum_generator, kinetic_energy_fn, potential_fn)

    return kernel
