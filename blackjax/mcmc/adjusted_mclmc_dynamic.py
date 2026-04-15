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
"""Public API for the Metropolis Hastings Microcanonical Hamiltonian Monte Carlo (MHMCHMC) Kernel. This is closely related to the Microcanonical Langevin Monte Carlo (MCLMC) Kernel, which is an unadjusted method. This kernel adds a Metropolis-Hastings correction to the MCLMC kernel. It also only refreshes the momentum variable after each MH step, rather than during the integration of the trajectory. Hence "Hamiltonian" and not "Langevin"."""
from typing import Callable

import jax
import jax.numpy as jnp

import blackjax.mcmc.integrators as integrators
from blackjax.base import SamplingAlgorithm, build_sampling_algorithm
from blackjax.mcmc.adjusted_mclmc import adjusted_mclmc_proposal, rescale
from blackjax.mcmc.dynamic_hmc import DynamicHMCState, halton_sequence
from blackjax.mcmc.hmc import HMCInfo
from blackjax.types import Array, ArrayLikeTree, PRNGKey
from blackjax.util import generate_unit_vector

__all__ = ["init", "build_kernel", "as_top_level_api"]


def init(
    position: ArrayLikeTree, logdensity_fn: Callable, random_generator_arg: Array
) -> DynamicHMCState:
    """Create an initial state for the dynamic MHMCHMC kernel.

    Parameters
    ----------
    position
        Initial position of the chain.
    logdensity_fn
        Log-density function of the target distribution.
    random_generator_arg
        Argument passed to ``integration_steps_fn`` and ``next_random_arg_fn``
        to generate the number of integration steps.

    Returns
    -------
    The initial DynamicHMCState.
    """
    logdensity, logdensity_grad = jax.value_and_grad(logdensity_fn)(position)
    return DynamicHMCState(position, logdensity, logdensity_grad, random_generator_arg)


def build_kernel(
    integration_steps_fn: Callable = lambda key: jax.random.randint(key, (), 1, 10),
    integrator: Callable = integrators.isokinetic_mclachlan,
    divergence_threshold: float = 1000,
    next_random_arg_fn: Callable = lambda key: jax.random.split(key)[1],
):
    """Build a Dynamic MHMCHMC kernel where the number of integration steps is chosen randomly.

    Parameters
    ----------
    integration_steps_fn
        Callable with signature ``(random_generator_arg, *integration_steps_params) -> int``
        that draws the number of integration steps for a single transition.
        Extra positional arguments beyond ``random_generator_arg`` are supplied
        at call time via ``integration_steps_params`` on the inner kernel, so
        tunable parameters (e.g. average number of steps, distribution bounds)
        can be adapted without rebuilding the kernel.
    integrator
        The integrator to use to integrate the Hamiltonian dynamics.
    divergence_threshold
        Value of the difference in energy above which we consider that the transition is divergent.
    next_random_arg_fn
        Function that generates the next `random_generator_arg` from its previous value.

    Returns
    -------
    A kernel that takes a rng_key and a Pytree that contains the current state
    of the chain and that returns a new state of the chain along with
    information about the transition.
    """

    def kernel(
        rng_key: PRNGKey,
        state: DynamicHMCState,
        logdensity_fn: Callable,
        step_size: float,
        L_proposal_factor: float = jnp.inf,
        inverse_mass_matrix=1.0,
        integration_steps_params: tuple = (),
    ) -> tuple[DynamicHMCState, HMCInfo]:
        """Generate a new sample with the MHMCHMC kernel."""

        num_integration_steps = integration_steps_fn(
            state.random_generator_arg, *integration_steps_params
        )

        key_momentum, key_integrator = jax.random.split(rng_key, 2)
        momentum = generate_unit_vector(key_momentum, state.position)
        proposal, info, _ = adjusted_mclmc_proposal(
            integrator=integrators.with_isokinetic_maruyama(
                integrator(
                    logdensity_fn=logdensity_fn, inverse_mass_matrix=inverse_mass_matrix
                )
            ),
            step_size=step_size,
            L_proposal_factor=L_proposal_factor * (num_integration_steps * step_size),
            num_integration_steps=num_integration_steps,
            divergence_threshold=divergence_threshold,
        )(
            key_integrator,
            integrators.IntegratorState(
                state.position, momentum, state.logdensity, state.logdensity_grad
            ),
        )

        return (
            DynamicHMCState(
                proposal.position,
                proposal.logdensity,
                proposal.logdensity_grad,
                next_random_arg_fn(state.random_generator_arg),
            ),
            info,
        )

    return kernel


def as_top_level_api(
    logdensity_fn: Callable,
    step_size: float,
    L_proposal_factor: float = jnp.inf,
    inverse_mass_matrix=1.0,
    *,
    divergence_threshold: int = 1000,
    integrator: Callable = integrators.isokinetic_mclachlan,
    next_random_arg_fn: Callable = lambda key: jax.random.split(key)[1],
    integration_steps_fn: Callable = lambda key: jax.random.randint(key, (), 1, 10),
    integration_steps_params: tuple = (),
) -> SamplingAlgorithm:
    """Implements the (basic) user interface for the dynamic MHMCHMC kernel.

    Parameters
    ----------
    logdensity_fn
        The log-density function we wish to draw samples from.
    step_size
        The value to use for the step size in the symplectic integrator.
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

    Returns
    -------
    A ``SamplingAlgorithm``.
    """

    kernel = build_kernel(
        integration_steps_fn=integration_steps_fn,
        integrator=integrator,
        next_random_arg_fn=next_random_arg_fn,
        divergence_threshold=divergence_threshold,
    )

    return build_sampling_algorithm(
        kernel,
        init,
        logdensity_fn,
        kernel_args=(
            step_size,
            L_proposal_factor,
            inverse_mass_matrix,
            integration_steps_params,
        ),
        pass_rng_key_to_init=True,
    )


def trajectory_length(t: int, mu: float):
    """Quasi-random trajectory length using the Halton sequence.

    Parameters
    ----------
    t
        Step index used to index into the Halton sequence.
    mu
        Target average number of integration steps.

    Returns
    -------
    Number of integration steps as a rounded integer.
    """
    s = rescale(mu)
    return jnp.rint(0.5 + halton_sequence(t) * s)


def make_random_trajectory_length_fn(random_trajectory_length: bool) -> Callable:
    """Build an ``integration_steps_fn`` with signature ``(key, avg_num_integration_steps) -> int``.

    Parameters
    ----------
    random_trajectory_length
        If ``True``, returns a randomized trajectory length function (uniform
        draw scaled by ``rescale(avg)``); otherwise returns a deterministic one
        that always yields ``ceil(avg)``.

    Returns
    -------
    A callable ``(random_generator_arg, avg_num_integration_steps) -> int``
    suitable for use as ``integration_steps_fn`` in :func:`build_kernel`.
    Pass ``integration_steps_params=(avg_num_integration_steps,)`` to the
    kernel so the value is forwarded at each step without a closure.
    """
    if random_trajectory_length:

        def integration_steps_fn(key, avg_num_integration_steps):
            return jnp.clip(
                jnp.ceil(jax.random.uniform(key) * rescale(avg_num_integration_steps)),
                min=1,
            ).astype(jnp.int32)

    else:

        def integration_steps_fn(key, avg_num_integration_steps):
            return jnp.clip(jnp.ceil(avg_num_integration_steps), min=1).astype(
                jnp.int32
            )

    return integration_steps_fn
