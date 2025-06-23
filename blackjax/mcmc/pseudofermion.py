from functools import partial
from typing import Any, Callable, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

import blackjax.mcmc.hmc as hmc
import blackjax.mcmc.integrators as integrators
import blackjax.mcmc.metrics as metrics
import blackjax.mcmc.proposal as proposal
import blackjax.mcmc.termination as termination
import blackjax.mcmc.trajectory as trajectory
from blackjax.base import SamplingAlgorithm
from blackjax.types import ArrayLikeTree, ArrayTree, PRNGKey


class GibbsState(NamedTuple):

    # pos : Any 
    # aux: ArrayTree
    position: ArrayTree
    logdensity: float
    logdensity_grad: ArrayTree
    momentum: ArrayTree
    temporary_state : Any
    fermion_matrix : Any
    count : int

def build_kernel():
    return ()

def init(position, logdensity_fn, fermion_matrix, temporary_state, init_main, rng_key  ):
    # state = hmc.state
    # state.boson_state = position
    # fermion_matrix = hmc.theory.get_fermion_matrix(hmc.state)
    # temporary_state = hmc.theory.sample_temporary_state(position,hmc.state,fermion_matrix)
    position, momentum, logdensity, logdensity_grad = init_main(position, logdensity_fn(fermion_matrix, temporary_state), rng_key )
    return GibbsState(
        position=position,
        logdensity=logdensity,
        logdensity_grad=logdensity_grad,
        momentum=momentum,
        temporary_state=temporary_state,
        fermion_matrix=fermion_matrix,
        count=0,
    )

def as_top_level_api(
    kernel_main,
    init_main,
    logdensity_fn: Callable,
    # step_size: float,
    # inverse_mass_matrix: metrics.MetricTypes,
    *,
    max_num_doublings: int = 10,
    divergence_threshold: int = 1000,
    # integrator: Callable = integrators.velocity_verlet,
    get_fermion_matrix_fn: Callable = None,
    sample_temporary_state_fn: Callable = None,
    # num_integration_steps: int = 1,
    # alg1,
) -> SamplingAlgorithm:
    """Implements the (basic) user interface for the nuts kernel.

    Examples
    --------

    A new NUTS kernel can be initialized and used with the following code:

    .. code::

        nuts = blackjax.nuts(logdensity_fn, step_size, inverse_mass_matrix)
        state = nuts.init(position)
        new_state, info = nuts.step(rng_key, state)

    We can JIT-compile the step function for more speed:

    .. code::

        step = jax.jit(nuts.step)
        new_state, info = step(rng_key, state)

    You can always use the base kernel should you need to:

    .. code::

       import blackjax.mcmc.integrators as integrators

       kernel = blackjax.nuts.build_kernel(integrators.yoshida)
       state = blackjax.nuts.init(position, logdensity_fn)
       state, info = kernel(rng_key, state, logdensity_fn, step_size, inverse_mass_matrix)

    Parameters
    ----------
    logdensity_fn
        The log-density function we wish to draw samples from.
    step_size
        The value to use for the step size in the symplectic integrator.
    inverse_mass_matrix
        The value to use for the inverse mass matrix when drawing a value for
        the momentum and computing the kinetic energy.
    max_num_doublings
        The maximum number of times we double the length of the trajectory before
        returning if no U-turn has been obserbed or no divergence has occured.
    divergence_threshold
        The absolute value of the difference in energy between two states above
        which we say that the transition is divergent. The default value is
        commonly found in other libraries, and yet is arbitrary.
    integrator
        (algorithm parameter) The symplectic integrator to use to integrate the trajectory.

    Returns
    -------
    A ``SamplingAlgorithm``.

    """
    # kernel = build_kernel(integrator, divergence_threshold)

    def init_fn(position: ArrayLikeTree, rng_key=None):
        del rng_key
        fermion_matrix = get_fermion_matrix_fn(position)
        temporary_state = sample_temporary_state_fn(position,fermion_matrix)
        return init(position, logdensity_fn, fermion_matrix, temporary_state, init_main, rng_key)

    def step_fn(rng_key: PRNGKey, state):
        next_state, info = kernel_main(
            rng_key,
            state,
            logdensity_fn(state.fermion_matrix, state.temporary_state),
            # step_size,
            # inverse_mass_matrix,
            # max_num_doublings,
            # num_integration_steps=num_integration_steps,
        )
        new_fermion_matrix = get_fermion_matrix_fn(next_state.position)
        new_temporary_state = sample_temporary_state_fn(next_state.position, new_fermion_matrix)
        full_state = GibbsState(
            position=next_state.position,
            momentum=None,
            # momentum=next_state.momentum,
            logdensity=next_state.logdensity,
            logdensity_grad=next_state.logdensity_grad,
            temporary_state=new_temporary_state,
            fermion_matrix=new_fermion_matrix,
            count=state.count + info.num_integration_steps,
        )
        jax.debug.print("count {x}", x=full_state.count)
        return full_state, info

    return SamplingAlgorithm(init_fn, step_fn)
