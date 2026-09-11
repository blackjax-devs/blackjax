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
"""Public API for the Dynamic HMC Kernel"""

from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp

import blackjax.mcmc.integrators as integrators
from blackjax.base import SamplingAlgorithm, build_sampling_algorithm
from blackjax.mcmc.adjusted_mclmc import rescale
from blackjax.mcmc.hmc import HMCInfo, HMCState
from blackjax.mcmc.hmc import build_kernel as build_static_hmc_kernel
from blackjax.mcmc.hmc import hmc_proposal
from blackjax.types import Array, ArrayLikeTree, ArrayTree, PRNGKey

__all__ = [
    "DynamicHMCState",
    "init",
    "build_kernel",
    "halton_sequence",
    "as_top_level_api",
    "hmc_proposal",
]


class DynamicHMCState(NamedTuple):
    """State of the dynamic HMC algorithm.

    Adds a utility array for generating a pseudo or quasi-random sequence of
    number of integration steps.

    """

    position: ArrayTree
    logdensity: float
    logdensity_grad: ArrayTree
    random_generator_arg: Array


def init(
    position: ArrayLikeTree,
    logdensity_fn: Callable,
    random_generator_arg: Array,
):
    logdensity, logdensity_grad = jax.value_and_grad(logdensity_fn)(position)
    return DynamicHMCState(position, logdensity, logdensity_grad, random_generator_arg)


def build_kernel(
    integrator: Callable = integrators.velocity_verlet,
    divergence_threshold: float = 1000,
    next_random_arg_fn: Callable = lambda key: jax.random.split(key)[1],
    integration_steps_fn: Callable = lambda key: jax.random.randint(key, (), 1, 10),
    build_proposal: Callable = hmc_proposal,
):
    """Build a Dynamic HMC kernel where the number of integration steps is chosen randomly.

    Parameters
    ----------
    integrator
        The symplectic integrator to use to integrate the Hamiltonian dynamics.
    divergence_threshold
        Value of the difference in energy above which we consider that the transition is divergent.
    next_random_arg_fn
        Function that generates the next `random_generator_arg` from its previous value.
    integration_steps_fn
        Callable with signature ``(random_generator_arg, *integration_steps_params) -> int``
        that draws the number of integration steps for a single transition.
        Extra positional arguments beyond ``random_generator_arg`` are supplied
        at call time via ``integration_steps_params`` on the inner kernel, so
        tunable parameters (e.g. average number of steps, distribution bounds)
        can be adapted without rebuilding the kernel.
    build_proposal
        A callable with signature
        ``(integrator, kinetic_energy, step_size, num_integration_steps,
        divergence_threshold) -> generate`` that builds the proposal function.
        Defaults to :func:`hmc_proposal` (standard endpoint HMC).

    Returns
    -------
    A kernel that takes a rng_key and a Pytree that contains the current state
    of the chain and that returns a new state of the chain along with
    information about the transition.

    """
    hmc_base = build_static_hmc_kernel(integrator, divergence_threshold, build_proposal)

    def kernel(
        rng_key: PRNGKey,
        state: DynamicHMCState,
        logdensity_fn: Callable,
        step_size: float,
        inverse_mass_matrix: Array,
        integration_steps_params: tuple = (),
    ) -> tuple[DynamicHMCState, HMCInfo]:
        """Generate a new sample with the HMC kernel."""
        num_integration_steps = integration_steps_fn(
            state.random_generator_arg, *integration_steps_params
        )
        hmc_state = HMCState(state.position, state.logdensity, state.logdensity_grad)
        hmc_proposal, info = hmc_base(
            rng_key,
            hmc_state,
            logdensity_fn,
            step_size,
            inverse_mass_matrix,
            num_integration_steps,
        )
        next_random_arg = next_random_arg_fn(state.random_generator_arg)
        return (
            DynamicHMCState(
                hmc_proposal.position,
                hmc_proposal.logdensity,
                hmc_proposal.logdensity_grad,
                next_random_arg,
            ),
            info,
        )

    return kernel


def as_top_level_api(
    logdensity_fn: Callable,
    step_size: float,
    inverse_mass_matrix: Array,
    *,
    divergence_threshold: int = 1000,
    integrator: Callable = integrators.velocity_verlet,
    next_random_arg_fn: Callable = lambda key: jax.random.split(key)[1],
    integration_steps_fn: Callable = lambda key: jax.random.randint(key, (), 1, 10),
    integration_steps_params: tuple = (),
    build_proposal: Callable = hmc_proposal,
) -> SamplingAlgorithm:
    """Implements the (basic) user interface for the dynamic HMC kernel.

    Parameters
    ----------
    logdensity_fn
        The log-density function we wish to draw samples from.
    step_size
        The value to use for the step size in the symplectic integrator.
    inverse_mass_matrix
        The value to use for the inverse mass matrix when drawing a value for
        the momentum and computing the kinetic energy.
    divergence_threshold
        The absolute value of the difference in energy between two states above
        which we say that the transition is divergent. The default value is
        commonly found in other libraries, and yet is arbitrary.
    integrator
        (algorithm parameter) The symplectic integrator to use to integrate the trajectory.
    next_random_arg_fn
        Function that generates the next `random_generator_arg` from its previous value.
    integration_steps_fn
        Callable with signature ``(random_generator_arg, *integration_steps_params) -> int``
        that draws the number of integration steps for a single transition.
    integration_steps_params
        Extra positional arguments unpacked into ``integration_steps_fn`` after
        ``random_generator_arg`` on every step.  Use this to pass tunable
        parameters (e.g. ``(avg_num_integration_steps,)`` or
        ``(lower_bound, upper_bound)``) without rebuilding the kernel.
        Defaults to ``()`` so that a plain 1-arg ``integration_steps_fn`` works
        unchanged.
    build_proposal
        A callable with signature
        ``(integrator, kinetic_energy, step_size, num_integration_steps,
        divergence_threshold) -> generate`` that builds the proposal function.
        Defaults to :func:`hmc_proposal` (standard endpoint HMC).  Pass
        :func:`multinomial_hmc_proposal` for multinomial trajectory sampling.

    Returns
    -------
    A ``SamplingAlgorithm``.
    """
    kernel = build_kernel(
        integrator,
        divergence_threshold,
        next_random_arg_fn,
        integration_steps_fn,
        build_proposal,
    )

    return build_sampling_algorithm(
        kernel,
        init,
        logdensity_fn,
        kernel_args=(step_size, inverse_mass_matrix, integration_steps_params),
        pass_rng_key_to_init=True,
    )


def halton_sequence(i: Array, max_bits: int = 10) -> float:
    """Generate the (i+1)-th element of the Halton sequence.

    Warning: max_bits should be less than the bit width of i.dtype to prevent integer overflow (e.g., max_bits <= 63 for int64).
    """
    if max_bits >= jnp.iinfo(i.dtype).bits:
        raise ValueError(
            f"max_bits ({max_bits}) must be less than bit width of dtype {i.dtype} ({jnp.iinfo(i.dtype).bits})"
        )
    bit_masks = 2 ** jnp.arange(max_bits, dtype=i.dtype)
    return jnp.einsum("i,i->", jnp.mod((i + 1) // bit_masks, 2), 0.5 / bit_masks)


def halton_trajectory_length(
    i: Array, trajectory_length_adjustment: float, max_bits: int = 10
) -> int:
    """Generate a quasi-random number of integration steps."""
    s = rescale(trajectory_length_adjustment)
    return jnp.asarray(jnp.rint(0.5 + halton_sequence(i, max_bits) * s), dtype=int)
