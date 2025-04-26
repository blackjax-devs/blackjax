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
from blackjax.util import pytree_size

__all__ = ["LangevinInfo", "init", "build_kernel", "as_top_level_api"]


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


def init(position: ArrayLike, logdensity_fn, metric, rng_key):
    
    l, g = jax.value_and_grad(logdensity_fn)(position)

    return IntegratorState(
        position=position,
        momentum = metric.sample_momentum(rng_key, position),
        logdensity=l,
        logdensity_grad=g,
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
        rng_key: PRNGKey, state: IntegratorState, L: float, step_size: float
    ) -> tuple[IntegratorState, LangevinInfo]:
        (position, momentum, logdensity, logdensitygrad), (kinetic_change, energy_error) = step(
            state, step_size, L, rng_key
        )

        # jax.debug.print("energy change {x}", x=energy_change)

        
        # kinetic_change = - momentum@momentum/2 + state.momentum@state.momentum/2
        

        # return IntegratorState(
        #     position, momentum, logdensity, logdensitygrad
        # ), LangevinInfo(
        #     logdensity=logdensity,
        #     energy_change=energy_change,
        #     kinetic_change=kinetic_change
        # )

        eev_max_per_dim = desired_energy_var_max_ratio * desired_energy_var
        ndims = pytree_size(position)
        # jax.debug.print("diagnostics {x}", x=(eev_max_per_dim, jnp.abs(energy_error), jnp.abs(energy_error) > jnp.sqrt(ndims * eev_max_per_dim)))

        new_state, new_info = jax.lax.cond(
            jnp.abs(energy_error) > jnp.sqrt(ndims * eev_max_per_dim),
            lambda: (
                state,
                LangevinInfo(
                    logdensity=state.logdensity,
                    energy_change=0.0,
                    kinetic_change=0.0,
                ),
            ),
            lambda: (
                IntegratorState(position, momentum, logdensity, logdensitygrad),
                LangevinInfo(
                    logdensity=logdensity,
                    energy_change=energy_error,
                    kinetic_change=kinetic_change,
                ),
            ),
        )

        return new_state, new_info

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
    metric = metrics.default_metric(inverse_mass_matrix)

    def init_fn(position: ArrayLike, rng_key: PRNGKey):
        return init(position, logdensity_fn, metric, rng_key)

    def update_fn(rng_key, state):
        return kernel(rng_key, state, L, step_size)

    return SamplingAlgorithm(init_fn, update_fn)
