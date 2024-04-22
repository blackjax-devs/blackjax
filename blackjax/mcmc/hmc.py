# Copyright 2020- The Blackjax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Public API for the HMC Kernel"""
from typing import Callable, NamedTuple, Union

import jax

import blackjax.mcmc.integrators as integrators
import blackjax.mcmc.metrics as metrics
import blackjax.mcmc.trajectory as trajectory
from blackjax.base import SamplingAlgorithm
from blackjax.mcmc.proposal import safe_energy_diff, static_binomial_sampling
from blackjax.mcmc.trajectory import hmc_energy
from blackjax.types import ArrayLikeTree, ArrayTree, PRNGKey

__all__ = [
    "HMCState",
    "HMCInfo",
    "init",
    "build_kernel",
    "as_top_level_api",
]


class HMCState(NamedTuple):
    """State of the HMC algorithm.

    The HMC algorithm takes one position of the chain and returns another
    position. In order to make computations more efficient, we also store
    the current logdensity as well as the current gradient of the logdensity.

    """

    position: ArrayTree
    logdensity: float
    logdensity_grad: ArrayTree


class HMCInfo(NamedTuple):
    """Additional information on the HMC transition.

    This additional information can be used for debugging or computing
    diagnostics.

    momentum:
        The momentum that was sampled and used to integrate the trajectory.
    acceptance_rate
        The acceptance probability of the transition, linked to the energy
        difference between the original and the proposed states.
    is_accepted
        Whether the proposed position was accepted or the original position
        was returned.
    is_divergent
        Whether the difference in energy between the original and the new state
        exceeded the divergence threshold.
    energy:
        Total energy of the transition.
    proposal
        The state proposed by the proposal. Typically includes the position and
        momentum.
    step_size
        Size of the integration step.
    num_integration_steps
        Number of times we run the symplectic integrator to build the trajectory

    """

    momentum: ArrayTree
    acceptance_rate: float
    is_accepted: bool
    is_divergent: bool
    energy: float
    proposal: integrators.IntegratorState
    num_integration_steps: int


def init(position: ArrayLikeTree, logdensity_fn: Callable):
    logdensity, logdensity_grad = jax.value_and_grad(logdensity_fn)(position)
    return HMCState(position, logdensity, logdensity_grad)


def build_kernel(
    integrator: Callable = integrators.velocity_verlet,
    divergence_threshold: float = 1000,
):
    """Build a HMC kernel.

    Parameters
    ----------
    integrator
        The symplectic integrator to use to integrate the Hamiltonian dynamics.
    divergence_threshold
        Value of the difference in energy above which we consider that the transition is
        divergent.

    Returns
    -------
    A kernel that takes a rng_key and a Pytree that contains the current state
    of the chain and that returns a new state of the chain along with
    information about the transition.

    """

    def kernel(
        rng_key: PRNGKey,
        state: HMCState,
        logdensity_fn: Callable,
        step_size: float,
        inverse_mass_matrix: metrics.MetricTypes,
        num_integration_steps: int,
    ) -> tuple[HMCState, HMCInfo]:
        """Generate a new sample with the HMC kernel."""

        metric = metrics.default_metric(inverse_mass_matrix)
        symplectic_integrator = integrator(logdensity_fn, metric.kinetic_energy)
        proposal_generator = hmc_proposal(
            symplectic_integrator,
            metric.kinetic_energy,
            step_size,
            num_integration_steps,
            divergence_threshold,
        )

        key_momentum, key_integrator = jax.random.split(rng_key, 2)

        position, logdensity, logdensity_grad = state
        momentum = metric.sample_momentum(key_momentum, position)

        integrator_state = integrators.IntegratorState(
            position, momentum, logdensity, logdensity_grad
        )
        proposal, info, _ = proposal_generator(key_integrator, integrator_state)
        proposal = HMCState(
            proposal.position, proposal.logdensity, proposal.logdensity_grad
        )

        return proposal, info

    return kernel


def as_top_level_api(
    logdensity_fn: Callable,
    step_size: float,
    inverse_mass_matrix: metrics.MetricTypes,
    num_integration_steps: int,
    *,
    divergence_threshold: int = 1000,
    integrator: Callable = integrators.velocity_verlet,
) -> SamplingAlgorithm:
    """Implements the (basic) user interface for the HMC kernel.

    The general hmc kernel builder (:meth:`blackjax.mcmc.hmc.build_kernel`, alias
    `blackjax.hmc.build_kernel`) can be cumbersome to manipulate. Since most users only
    need to specify the kernel parameters at initialization time, we provide a helper
    function that specializes the general kernel.

    We also add the general kernel and state generator as an attribute to this class so
    users only need to pass `blackjax.hmc` to SMC, adaptation, etc. algorithms.

    Examples
    --------

    A new HMC kernel can be initialized and used with the following code:

    .. code::

        hmc = blackjax.hmc(
            logdensity_fn, step_size, inverse_mass_matrix, num_integration_steps
        )
        state = hmc.init(position)
        new_state, info = hmc.step(rng_key, state)

    Kernels are not jit-compiled by default so you will need to do it manually:

    .. code::

       step = jax.jit(hmc.step)
       new_state, info = step(rng_key, state)

    Should you need to you can always use the base kernel directly:

    .. code::

       import blackjax.mcmc.integrators as integrators

       kernel = blackjax.hmc.build_kernel(integrators.mclachlan)
       state = blackjax.hmc.init(position, logdensity_fn)
       state, info = kernel(
           rng_key,
           state,
           logdensity_fn,
           step_size,
           inverse_mass_matrix,
           num_integration_steps,
       )

    Parameters
    ----------
    logdensity_fn
        The log-density function we wish to draw samples from.
    step_size
        The value to use for the step size in the symplectic integrator.
    inverse_mass_matrix
        The value to use for the inverse mass matrix when drawing a value for
        the momentum and computing the kinetic energy. This argument will be
        passed to the ``metrics.default_metric`` function so it supports the
        full interface presented there.
    num_integration_steps
        The number of steps we take with the symplectic integrator at each
        sample step before returning a sample.
    divergence_threshold
        The absolute value of the difference in energy between two states above
        which we say that the transition is divergent. The default value is
        commonly found in other libraries, and yet is arbitrary.
    integrator
        (algorithm parameter) The symplectic integrator to use to integrate the
        trajectory.

    Returns
    -------
    A ``SamplingAlgorithm``.
    """

    kernel = build_kernel(integrator, divergence_threshold)

    def init_fn(position: ArrayLikeTree, rng_key=None):
        del rng_key
        return init(position, logdensity_fn)

    def step_fn(rng_key: PRNGKey, state):
        return kernel(
            rng_key,
            state,
            logdensity_fn,
            step_size,
            inverse_mass_matrix,
            num_integration_steps,
        )

    return SamplingAlgorithm(init_fn, step_fn)


def hmc_proposal(
    integrator: Callable,
    kinetic_energy: metrics.KineticEnergy,
    step_size: Union[float, ArrayLikeTree],
    num_integration_steps: int = 1,
    divergence_threshold: float = 1000,
    *,
    sample_proposal: Callable = static_binomial_sampling,
) -> Callable:
    """Vanilla HMC algorithm.

    The algorithm integrates the trajectory applying a symplectic integrator
    `num_integration_steps` times in one direction to get a proposal and uses a
    Metropolis-Hastings acceptance step to either reject or accept this
    proposal. This is what people usually refer to when they talk about "the
    HMC algorithm".

    Parameters
    ----------
    integrator
        Symplectic integrator used to build the trajectory step by step.
    kinetic_energy
        Function that computes the kinetic energy.
    step_size
        Size of the integration step.
    num_integration_steps
        Number of times we run the symplectic integrator to build the trajectory
    divergence_threshold
        Threshold above which we say that there is a divergence.

    Returns
    -------
    A kernel that generates a new chain state and information about the transition.

    """
    build_trajectory = trajectory.static_integration(integrator)
    hmc_energy_fn = hmc_energy(kinetic_energy)

    def generate(
        rng_key, state: integrators.IntegratorState
    ) -> tuple[integrators.IntegratorState, HMCInfo, ArrayTree]:
        """Generate a new chain state."""
        end_state = build_trajectory(state, step_size, num_integration_steps)
        end_state = flip_momentum(end_state)
        proposal_energy = hmc_energy_fn(state)
        new_energy = hmc_energy_fn(end_state)
        delta_energy = safe_energy_diff(proposal_energy, new_energy)
        is_diverging = -delta_energy > divergence_threshold
        sampled_state, info = sample_proposal(rng_key, delta_energy, state, end_state)
        do_accept, p_accept, other_proposal_info = info

        info = HMCInfo(
            state.momentum,
            p_accept,
            do_accept,
            is_diverging,
            new_energy,
            end_state,
            num_integration_steps,
        )

        return sampled_state, info, other_proposal_info

    return generate


def flip_momentum(
    state: integrators.IntegratorState,
) -> integrators.IntegratorState:
    """Flip the momentum at the end of the trajectory.

    To guarantee time-reversibility (hence detailed balance) we
    need to flip the last state's momentum. If we run the hamiltonian
    dynamics starting from the last state with flipped momentum we
    should indeed retrieve the initial state (with flipped momentum).

    """
    flipped_momentum = jax.tree_util.tree_map(lambda m: -1.0 * m, state.momentum)
    return integrators.IntegratorState(
        state.position,
        flipped_momentum,
        state.logdensity,
        state.logdensity_grad,
    )
