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
from blackjax.mcmc.adjusted_mclmc_dynamic import make_random_trajectory_length_fn
from blackjax.base import SamplingAlgorithm
from blackjax.mcmc.integrators import (
    IntegratorState,
    velocity_verlet,
)
from blackjax.types import ArrayLike, PRNGKey
import blackjax.mcmc.metrics as metrics
import jax.numpy as jnp
from blackjax.util import pytree_size, generate_unit_vector
from blackjax.mcmc.underdamped_langevin import handle_high_energy, handle_nans, LangevinInfo
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
        integrator,
        desired_energy_var_max_ratio=1e3,
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


    def kernel(
        rng_key: PRNGKey, state: UHMCState, logdensity_fn, L: float, step_size: float, inverse_mass_matrix, 
    ) -> tuple[UHMCState, LangevinInfo]:
        metric = metrics.default_metric(inverse_mass_matrix)
        step = integrator(logdensity_fn, metric.kinetic_energy)
        
        refresh_key, energy_cutoff_key, nan_key, randomization_key = jax.random.split(rng_key, 4)

        (position, momentum, logdensity, logdensitygrad) = step(
            integrator_state(state), step_size
        )

        kinetic_change = - metric.kinetic_energy(state.momentum) + metric.kinetic_energy(momentum)
        energy_change = kinetic_change - logdensity + state.logdensity
        
        num_steps_per_traj = make_random_trajectory_length_fn(True)(L/step_size)(randomization_key).astype(jnp.int64)

        momentum = (state.steps_until_refresh==0) * metric.sample_momentum(refresh_key, position) + (state.steps_until_refresh>0) * momentum
        
        steps_until_refresh = (state.steps_until_refresh==0) * num_steps_per_traj + (state.steps_until_refresh>0) * (state.steps_until_refresh - 1)

        eev_max_per_dim = desired_energy_var_max_ratio * desired_energy_var
        ndims = pytree_size(position)



        new_state, info = handle_high_energy(state, UHMCState(position, momentum, logdensity, logdensitygrad, steps_until_refresh), LangevinInfo(
            logdensity=logdensity,
            energy_change=energy_change,
            kinetic_change=kinetic_change,
            nonans=True
        ), energy_cutoff_key, cutoff = jnp.sqrt(ndims * eev_max_per_dim), inverse_mass_matrix=inverse_mass_matrix)

        new_state, info = handle_nans(state, new_state, info, nan_key, inverse_mass_matrix)
        return new_state, info
    
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
        integrator,
        desired_energy_var_max_ratio=desired_energy_var_max_ratio,
        desired_energy_var=desired_energy_var,
        )
    # metric = metrics.default_metric(inverse_mass_matrix)

    def init_fn(position: ArrayLike, rng_key: PRNGKey):
        return init(position, logdensity_fn, rng_key)

    def update_fn(rng_key, state):
        return kernel(rng_key, state, logdensity_fn, L, step_size, inverse_mass_matrix)

    return SamplingAlgorithm(init_fn, update_fn)


