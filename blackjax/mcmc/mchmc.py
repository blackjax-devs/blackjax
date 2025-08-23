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
"""Public API for the MCLMC Kernel"""
from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp

from blackjax.base import SamplingAlgorithm
from blackjax.mcmc.integrators import (
    IntegratorState,
    isokinetic_mclachlan,
)
from blackjax.types import ArrayLike, PRNGKey
from blackjax.util import generate_unit_vector, pytree_size
from blackjax.mcmc.mclmc import handle_high_energy, handle_nans, MCLMCInfo
from blackjax.mcmc.adjusted_mclmc_dynamic import make_random_trajectory_length_fn
__all__ = ["MCLMCInfo", "init", "build_kernel", "as_top_level_api"]

class MCHMCState(NamedTuple):
    position: ArrayLike
    momentum: ArrayLike
    logdensity: float
    logdensity_grad: ArrayLike
    steps_until_refresh: int



def init(position: ArrayLike, logdensity_fn, random_generator_arg):
    if pytree_size(position) < 2:
        raise ValueError(
            "The target distribution must have more than 1 dimension for MCLMC."
        )
    l, g = jax.value_and_grad(logdensity_fn)(position)

    return MCHMCState(
        position=position,
        momentum=generate_unit_vector(random_generator_arg, position),
        logdensity=l,
        logdensity_grad=g,
        steps_until_refresh=0,
    )
    
def integrator_state(state: MCHMCState) -> IntegratorState:
    return IntegratorState(
        position=state.position,
        momentum=state.momentum,
        logdensity=state.logdensity,
        logdensity_grad=state.logdensity_grad,
    )


def build_kernel(
    # integration_steps_fn,
    integrator,
    desired_energy_var_max_ratio=jnp.inf,
    desired_energy_var=5e-4,
):
    """
    """



    def kernel(
        rng_key: PRNGKey, state: MCHMCState, logdensity_fn, L: float, step_size: float, inverse_mass_matrix: ArrayLike
    ) -> tuple[MCHMCState, MCLMCInfo]:
        step = integrator(logdensity_fn=logdensity_fn, inverse_mass_matrix=inverse_mass_matrix)
        (position, momentum, logdensity, logdensitygrad), kinetic_change = step(
            integrator_state(state), step_size
        )

        randomization_key, refresh_key, energy_cutoff_key, nan_key = jax.random.split(rng_key, 4)

        num_steps_per_traj = make_random_trajectory_length_fn(True)(L/step_size)(randomization_key).astype(jnp.int64)
        


        energy_change = kinetic_change - logdensity + state.logdensity

        eev_max_per_dim = desired_energy_var_max_ratio * desired_energy_var
        ndims = pytree_size(position)

        momentum=(state.steps_until_refresh==0) * generate_unit_vector(refresh_key, state.position) + (state.steps_until_refresh>0) * momentum

        steps_until_refresh = (state.steps_until_refresh==0) * num_steps_per_traj + (state.steps_until_refresh>0) * (state.steps_until_refresh - 1)

        new_state, info = handle_high_energy(state, MCHMCState(position, momentum, logdensity, logdensitygrad, steps_until_refresh), MCLMCInfo(
            logdensity=logdensity,
            energy_change=energy_change,
            kinetic_change=kinetic_change,
            nonans=True
        ), energy_cutoff_key, cutoff = jnp.sqrt(ndims * eev_max_per_dim))

        new_state, info = handle_nans(state, new_state, info, nan_key)

        return new_state, info

    return kernel


def as_top_level_api(
    logdensity_fn: Callable,
    L,
    step_size,
    integrator=isokinetic_mclachlan,
    inverse_mass_matrix=1.0,
    desired_energy_var_max_ratio=jnp.inf,
) -> SamplingAlgorithm:
    """
    """

    kernel = build_kernel(
        integrator,
        desired_energy_var_max_ratio=desired_energy_var_max_ratio,
    )

    def init_fn(position: ArrayLike, rng_key: PRNGKey):
        return init(position, logdensity_fn, rng_key)

    def update_fn(rng_key, state):
        return kernel(rng_key, state, logdensity_fn, L, step_size, inverse_mass_matrix)

    return SamplingAlgorithm(init_fn, update_fn)
