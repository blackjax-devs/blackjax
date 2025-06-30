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
"""Public API for the Underdamped Langevin Kernel"""
from typing import Callable, NamedTuple

import jax

from blackjax.base import SamplingAlgorithm
from blackjax.mcmc.integrators import (
    IntegratorState,
    with_maruyama,
    velocity_verlet,
)
from blackjax.types import ArrayLike, PRNGKey
import blackjax.mcmc.metrics as metrics
import jax.numpy as jnp
from blackjax.util import pytree_size, generate_unit_vector
from blackjax.adaptation.mclmc_adaptation import handle_high_energy
__all__ = ["LangevinInfo", "init", "build_kernel", "as_top_level_api"]

class UHMCState(NamedTuple):
    position: ArrayLike
    momentum: ArrayLike
    logdensity: float
    logdensity_grad: ArrayLike
    steps_until_refresh: int

def integrator_state(state: UHMCState) -> IntegratorState:
    return IntegratorState(
        position=state.position,
        momentum=state.momentum,
        logdensity=state.logdensity,
        logdensity_grad=state.logdensity_grad,
    )

class LangevinInfo(NamedTuple):
    """
    Additional information on the Langevin transition.

    logdensity
        The log-density of the distribution at the current step of the Langevin chain.
    kinetic_change
        The difference in kinetic energy between the current and previous step.
    energy_change
        The difference in energy between the current and previous step.
    """

    logdensity: float
    kinetic_change: float
    energy_change: float


def init(position: ArrayLike, logdensity_fn, random_generator_arg):
    
    l, g = jax.value_and_grad(logdensity_fn)(position)

    metric = metrics.default_metric(jnp.ones_like(position))

    return UHMCState(
        position=position,
        momentum = metric.sample_momentum(random_generator_arg, position),
        logdensity=l,
        logdensity_grad=g,
        steps_until_refresh=0,
    )


def build_kernel(
        logdensity_fn, 
        inverse_mass_matrix, 
        integrator,
        desired_energy_var_max_ratio=jnp.inf,
        desired_energy_var=5e-4,):
    """Build a HMC kernel.

    Parameters
    ----------
    integrator
        The symplectic integrator to use to integrate the Langevin dynamics.
    L
        the momentum decoherence rate.
    step_size
        step size of the integrator.

    Returns
    -------
    A kernel that takes a rng_key and a Pytree that contains the current state
    of the chain and that returns a new state of the chain along with
    information about the transition.

    """

    metric = metrics.default_metric(inverse_mass_matrix)
    step = with_maruyama(integrator(logdensity_fn, metric.kinetic_energy), metric.kinetic_energy,inverse_mass_matrix)

    def kernel(
        rng_key: PRNGKey, state: UHMCState, L: float, step_size: float
    ) -> tuple[UHMCState, LangevinInfo]:
        
        refresh_key, energy_key, run_key = jax.random.split(rng_key, 3)

        (position, momentum, logdensity, logdensitygrad), (kinetic_change, energy_error) = step(
            integrator_state(state), step_size, jnp.inf, run_key
        )
        

        num_steps_per_traj = jnp.ceil(L/step_size).astype(jnp.int64)
        momentum = (state.steps_until_refresh==0) * metric.sample_momentum(refresh_key, position) + (state.steps_until_refresh>0) * momentum
        steps_until_refresh = (state.steps_until_refresh==0) * num_steps_per_traj + (state.steps_until_refresh>0) * (state.steps_until_refresh - 1)

        eev_max_per_dim = desired_energy_var_max_ratio * desired_energy_var
        ndims = pytree_size(position)


        energy, new_integrator_state = handle_high_energy(
            previous_state=integrator_state(state),
            next_state=IntegratorState(position, momentum, logdensity, logdensitygrad),
            energy_change=energy_error,
            key=energy_key,
            inverse_mass_matrix=inverse_mass_matrix,
            cutoff=jnp.sqrt(ndims * eev_max_per_dim),
            euclidean=True
        )

        return UHMCState(new_integrator_state.position, new_integrator_state.momentum, new_integrator_state.logdensity, new_integrator_state.logdensity_grad, steps_until_refresh), LangevinInfo(
            logdensity=logdensity,
            energy_change=energy,
            kinetic_change=kinetic_change
        )


    return kernel


def as_top_level_api(
    logdensity_fn: Callable,
    L,
    step_size,
    integrator=velocity_verlet,
    inverse_mass_matrix=1.0,
    desired_energy_var_max_ratio=jnp.inf,
    desired_energy_var=5e-4,
) -> SamplingAlgorithm:
    """The general Langevin kernel builder (:meth:`blackjax.mcmc.langevin.build_kernel`, alias `blackjax.langevin.build_kernel`) can be
    cumbersome to manipulate. Since most users only need to specify the kernel
    parameters at initialization time, we provide a helper function that
    specializes the general kernel.

    We also add the general kernel and state generator as an attribute to this class so
    users only need to pass `blackjax.langevin` to SMC, adaptation, etc. algorithms.



    """

    kernel = build_kernel(
        logdensity_fn, 
        inverse_mass_matrix, 
        integrator,
        desired_energy_var_max_ratio=desired_energy_var_max_ratio,
        desired_energy_var=desired_energy_var,
        )
    # metric = metrics.default_metric(inverse_mass_matrix)

    def init_fn(position: ArrayLike, rng_key: PRNGKey):
        return init(position, logdensity_fn, rng_key)

    def update_fn(rng_key, state):
        return kernel(rng_key, state, L, step_size)

    return SamplingAlgorithm(init_fn, update_fn)
