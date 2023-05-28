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

from typing import Callable, Tuple

import jax

import blackjax.mcmc.integrators as integrators
import blackjax.mcmc.metrics as metrics
from blackjax.base import MCMCSamplingAlgorithm
from blackjax.mcmc import hmc
from blackjax.types import PRNGKey, PyTree

__all__ = ["init", "build_kernel", "rmhmc"]


init = hmc.init


def build_kernel(
    integrator: Callable = integrators.implicit_midpoint,
    divergence_threshold: float = 1000,
):
    """Build a HMC kernel.

    Parameters
    ----------
    integrator
        The symplectic integrator to use to integrate the Hamiltonian dynamics.
    divergence_threshold
        Value of the difference in energy above which we consider that the transition is divergent.

    Returns
    -------
    A kernel that takes a rng_key and a Pytree that contains the current state
    of the chain and that returns a new state of the chain along with
    information about the transition.

    """

    def kernel(
        rng_key: PRNGKey,
        state: hmc.HMCState,
        logdensity_fn: Callable,
        step_size: float,
        mass_matrix_fn: Callable,
        num_integration_steps: int,
    ) -> Tuple[hmc.HMCState, hmc.HMCInfo]:
        """Generate a new sample with the HMC kernel."""

        momentum_generator, kinetic_energy_fn, _ = metrics.gaussian_riemannian(
            mass_matrix_fn
        )
        symplectic_integrator = integrator(logdensity_fn, kinetic_energy_fn)
        proposal_generator = hmc.hmc_proposal(
            symplectic_integrator,
            kinetic_energy_fn,
            step_size,
            num_integration_steps,
            divergence_threshold,
        )

        key_momentum, key_integrator = jax.random.split(rng_key, 2)

        position, logdensity, logdensity_grad = state
        momentum = momentum_generator(key_momentum, position)

        integrator_state = integrators.IntegratorState(
            position, momentum, logdensity, logdensity_grad
        )
        proposal, info = proposal_generator(key_integrator, integrator_state)
        proposal = hmc.HMCState(
            proposal.position, proposal.logdensity, proposal.logdensity_grad
        )

        return proposal, info

    return kernel


class rmhmc:
    init = staticmethod(init)
    build_kernel = staticmethod(build_kernel)

    def __new__(  # type: ignore[misc]
        cls,
        logdensity_fn: Callable,
        step_size: float,
        mass_matrix: Callable,
        num_integration_steps: int,
        *,
        divergence_threshold: int = 1000,
        integrator: Callable = integrators.implicit_midpoint,
    ) -> MCMCSamplingAlgorithm:
        kernel = cls.build_kernel(integrator, divergence_threshold)

        def init_fn(position: PyTree):
            return cls.init(position, logdensity_fn)

        def step_fn(rng_key: PRNGKey, state):
            return kernel(
                rng_key,
                state,
                logdensity_fn,
                step_size,
                mass_matrix,
                num_integration_steps,
            )

        return MCMCSamplingAlgorithm(init_fn, step_fn)
