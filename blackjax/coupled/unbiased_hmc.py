"""Public API for the unbiased coupled HMC Kernel"""

from typing import Callable, Dict, List, NamedTuple, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np

import blackjax.inference.base as base
import blackjax.inference.integrators as integrators
from blackjax.coupled.rwmh import kernel as coupled_rwmh_kernel, new_state as new_coupled_rwmh_state
from blackjax.hmc import kernel as hmc_kernel
from blackjax.rwmh import kernel as rwmh_kernel, new_state as new_rwmh_state

Array = Union[np.ndarray, jnp.DeviceArray]
PyTree = Union[Dict, List, Tuple]

__all__ = ["new_state", "kernel"]


class CoupledHMCState(NamedTuple):
    state_1: base.HMCState
    state_2: base.HMCState


class CoupledHMCInfo(NamedTuple):
    did_rwmh: bool


def new_state(position_1: PyTree, position_2: PyTree, potential_fn: Callable) -> CoupledHMCState:
    """ Creates two independent chain states from a position by calling base.new_hcm_state twice. """
    return CoupledHMCState(base.new_hmc_state(position_1, potential_fn),
                           base.new_hmc_state(position_2, potential_fn))


def kernel(
        potential_fn: Callable,
        step_size: float,
        inverse_mass_matrix: Array,
        num_integration_steps: int,
        sigma: float,
        gamma: float,
        *,
        integrator: Callable = integrators.velocity_verlet,
        divergence_threshold: int = 1000,
):
    """Build a debiased coupled HMC kernel. The algorithm essentially combines an HMC chain and an RWMH chain
    into an unbiased scheme.

    Parameters
    ----------
    potential_fn
        A function that returns the potential energy of a chain at a given position.
    **parameters
        A NamedTuple that contains the parameters of the kernel to be built.
    Returns
    -------
    A kernel that takes a rng_key and a Pytree that contains the current state
    of the chain and that returns a new state of the chain along with
    information about the transition.

    """
    dim = inverse_mass_matrix.shape[0]

    rwmh_kernel_inst = rwmh_kernel(potential_fn, jnp.eye(dim) / (sigma ** 2))
    coupled_rwmh_kernel_inst = coupled_rwmh_kernel(potential_fn, jnp.eye(dim) / (sigma ** 2))
    hmc_kernel_inst = hmc_kernel(potential_fn, inverse_mass_matrix, step_size, num_integration_steps,
                                 integrator=integrator, divergence_threshold=divergence_threshold)

    def one_step(rng_key: jnp.ndarray, state: CoupledHMCState, initial_step: bool) -> Tuple[
        CoupledHMCState, CoupledHMCInfo]:
        """Moves the chain by one step using the Hamiltonian dynamics.

        Parameters
        ----------
        rng_key:
           The pseudo-random number generator key used to generate random numbers.
        state:
            The current state of the two chain: position, log-probability and gradient
            of the log-probability.
        initial_step:
            Flag to say it is the first step of the algorithm. If it is the case, then the chains need to be initialized
            differently.

        Returns
        -------
        The next state of the chain and additional information about the current step.
        """
        uniform_key, kernel_key = jax.random.split(rng_key, 2)
        uniform = jax.random.uniform(uniform_key)
        do_rwmh = uniform < gamma

        if not initial_step:
            def rwmh_fun():
                rwmh_state = new_coupled_rwmh_state(state.state_1.position, state.state_2.position, potential_fn)
                res, _ = coupled_rwmh_kernel_inst(kernel_key, rwmh_state)
                proposed_state = new_state(res.state_1.position, res.state_2.position, potential_fn)
                return proposed_state

            def hmc_fun():
                res_1, _ = hmc_kernel_inst(kernel_key, state.state_1)
                res_2, _ = hmc_kernel_inst(kernel_key, state.state_2)
                return CoupledHMCState(res_1, res_2)
        else:
            def rwmh_fun():
                rwmh_state = new_rwmh_state(state.state_1.position, potential_fn)
                res, _ = rwmh_kernel_inst(kernel_key, rwmh_state)
                proposed_state = new_state(res.state_1.position, state.state_2.position, potential_fn)
                return proposed_state

            def hmc_fun():
                res, _ = hmc_kernel_inst(kernel_key, state.state_1)
                return CoupledHMCState(res, state.state_2)

        proposal = jax.lax.cond(do_rwmh, rwmh_fun, hmc_fun, operand=None)

        return proposal, CoupledHMCInfo(do_rwmh)

    return one_step
