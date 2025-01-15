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
"""Public API for the Metropolis Hastings Microcanonical Hamiltonian Monte Carlo (MHMCHMC) Kernel. This is closely related to the Microcanonical Langevin Monte Carlo (MCLMC) Kernel, which is an unadjusted method. This kernel adds a Metropolis-Hastings correction to the MCLMC kernel. It also only refreshes the momentum variable after each MH step, rather than during the integration of the trajectory. Hence "Hamiltonian" and not "Langevin".

NOTE: For best performance, we recommend using adjusted_mclmc_dynamic instead of this module, which is primarily intended for use in parallelized versions of the algorithm.

"""
from typing import Callable, Union

import jax
import jax.numpy as jnp

import blackjax.mcmc.integrators as integrators
from blackjax.base import SamplingAlgorithm
from blackjax.mcmc.hmc import HMCInfo, HMCState
from blackjax.mcmc.proposal import static_binomial_sampling
from blackjax.types import ArrayLikeTree, ArrayTree, PRNGKey
from blackjax.util import generate_unit_vector

__all__ = ["init", "build_kernel", "as_top_level_api"]


def init(position: ArrayLikeTree, logdensity_fn: Callable):
    logdensity, logdensity_grad = jax.value_and_grad(logdensity_fn)(position)
    return HMCState(position, logdensity, logdensity_grad)


def build_kernel(
    logdensity_fn: Callable,
    integrator: Callable = integrators.isokinetic_mclachlan,
    divergence_threshold: float = 1000,
    inverse_mass_matrix=1.0,
):
    """Build an MHMCHMC kernel where the number of integration steps is chosen randomly.

    Parameters
    ----------
    integrator
        The integrator to use to integrate the Hamiltonian dynamics.
    divergence_threshold
        Value of the difference in energy above which we consider that the transition is divergent.
    next_random_arg_fn
        Function that generates the next `random_generator_arg` from its previous value.
    integration_steps_fn
        Function that generates the next pseudo or quasi-random number of integration steps in the
        sequence, given the current `random_generator_arg`. Needs to return an `int`.

    Returns
    -------
    A kernel that takes a rng_key and a Pytree that contains the current state
    of the chain and that returns a new state of the chain along with
    information about the transition.
    """

    def kernel(
        rng_key: PRNGKey,
        state: HMCState,
        step_size: float,
        num_integration_steps: int,
        L_proposal_factor: float = jnp.inf,
    ) -> tuple[HMCState, HMCInfo]:
        """Generate a new sample with the MHMCHMC kernel."""

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
            HMCState(
                proposal.position,
                proposal.logdensity,
                proposal.logdensity_grad,
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
    num_integration_steps,
) -> SamplingAlgorithm:
    """Implements the (basic) user interface for the MHMCHMC kernel.

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
        Function that generates the next pseudo or quasi-random number of integration steps in the
        sequence, given the current `random_generator_arg`.


    Returns
    -------
    A ``SamplingAlgorithm``.
    """

    kernel = build_kernel(
        logdensity_fn=logdensity_fn,
        integrator=integrator,
        inverse_mass_matrix=inverse_mass_matrix,
        divergence_threshold=divergence_threshold,
    )

    def init_fn(position: ArrayLikeTree, rng_key=None):
        del rng_key
        return init(position, logdensity_fn)

    def update_fn(rng_key: PRNGKey, state):
        return kernel(
            rng_key=rng_key,
            state=state,
            step_size=step_size,
            num_integration_steps=num_integration_steps,
            L_proposal_factor=L_proposal_factor,
        )

    return SamplingAlgorithm(init_fn, update_fn)  # type: ignore[arg-type]


def adjusted_mclmc_proposal(
    integrator: Callable,
    step_size: Union[float, ArrayLikeTree],
    L_proposal_factor: float,
    num_integration_steps: int = 1,
    divergence_threshold: float = 1000,
    *,
    sample_proposal: Callable = static_binomial_sampling,
) -> Callable:
    """Vanilla MHMCHMC algorithm.

    The algorithm integrates the trajectory applying a integrator
    `num_integration_steps` times in one direction to get a proposal and uses a
    Metropolis-Hastings acceptance step to either reject or accept this
    proposal. This is what people usually refer to when they talk about "the
    HMC algorithm".

    Parameters
    ----------
    integrator
        integrator used to build the trajectory step by step.
    kinetic_energy
        Function that computes the kinetic energy.
    step_size
        Size of the integration step.
    num_integration_steps
        Number of times we run the integrator to build the trajectory
    divergence_threshold
        Threshold above which we say that there is a divergence.

    Returns
    -------
    A kernel that generates a new chain state and information about the transition.

    """

    def step(i, vars):
        state, kinetic_energy, rng_key = vars
        rng_key, next_rng_key = jax.random.split(rng_key)
        next_state, next_kinetic_energy = integrator(
            state, step_size, L_proposal_factor, rng_key
        )

        return next_state, kinetic_energy + next_kinetic_energy, next_rng_key

    def build_trajectory(state, num_integration_steps, rng_key):
        return jax.lax.fori_loop(
            0 * num_integration_steps, num_integration_steps, step, (state, 0, rng_key)
        )

    def generate(
        rng_key, state: integrators.IntegratorState
    ) -> tuple[integrators.IntegratorState, HMCInfo, ArrayTree]:
        """Generate a new chain state."""
        end_state, kinetic_energy, rng_key = build_trajectory(
            state, num_integration_steps, rng_key
        )

        new_energy = -end_state.logdensity
        delta_energy = -state.logdensity + end_state.logdensity - kinetic_energy
        delta_energy = jnp.where(jnp.isnan(delta_energy), -jnp.inf, delta_energy)
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
