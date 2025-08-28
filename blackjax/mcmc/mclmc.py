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
    with_isokinetic_maruyama,
)
from blackjax.types import ArrayLike, PRNGKey
from blackjax.util import generate_unit_vector, pytree_size
from jax.flatten_util import ravel_pytree

__all__ = ["MCLMCInfo", "init", "build_kernel", "as_top_level_api"]


class MCLMCInfo(NamedTuple):
    """
    Additional information on the MCLMC transition.

    logdensity
        The log-density of the distribution at the current step of the MCLMC chain.
    kinetic_change
        The difference in kinetic energy between the current and previous step.
    energy_change
        The difference in energy between the current and previous step.
    """

    logdensity: float
    kinetic_change: float
    energy_change: float
    nonans: bool


def init(position: ArrayLike, logdensity_fn, random_generator_arg):
    if pytree_size(position) < 2:
        raise ValueError(
            "The target distribution must have more than 1 dimension for MCLMC."
        )
    l, g = jax.value_and_grad(logdensity_fn)(position)

    return IntegratorState(
        position=position,
        momentum=generate_unit_vector(random_generator_arg, position),
        logdensity=l,
        logdensity_grad=g,
    )



def build_kernel(
    integrator,
    desired_energy_var_max_ratio=jnp.inf,
    desired_energy_var=5e-4,
):
    """Build a HMC kernel.

    Parameters
    ----------
    integrator
        The symplectic integrator to use to integrate the Hamiltonian dynamics.
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
        rng_key: PRNGKey, state: IntegratorState, logdensity_fn, L: float, step_size: float,
    inverse_mass_matrix,
    ) -> tuple[IntegratorState, MCLMCInfo]:
        step = with_isokinetic_maruyama(
            integrator(logdensity_fn=logdensity_fn, inverse_mass_matrix=inverse_mass_matrix)
        )

        kernel_key, energy_cutoff_key, nan_key = jax.random.split(rng_key, 3)
        

        (position, momentum, logdensity, logdensitygrad), kinetic_change = step(
            state, step_size, L, kernel_key
        )

        energy_change = kinetic_change - logdensity + state.logdensity
        eev_max_per_dim = desired_energy_var_max_ratio * desired_energy_var
        ndims = pytree_size(position)

        new_state, info = handle_high_energy(state, IntegratorState(position, momentum, logdensity, logdensitygrad), MCLMCInfo(
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
    """The general mclmc kernel builder (:meth:`blackjax.mcmc.mclmc.build_kernel`, alias `blackjax.mclmc.build_kernel`) can be
    cumbersome to manipulate. Since most users only need to specify the kernel
    parameters at initialization time, we provide a helper function that
    specializes the general kernel.

    We also add the general kernel and state generator as an attribute to this class so
    users only need to pass `blackjax.mclmc` to SMC, adaptation, etc. algorithms.

    Examples
    --------

    A new mclmc kernel can be initialized and used with the following code:

    .. code::

        mclmc = blackjax.mcmc.mclmc.mclmc(
            logdensity_fn=logdensity_fn,
            L=L,
            step_size=step_size
        )
        state = mclmc.init(position)
        new_state, info = mclmc.step(rng_key, state)

    Kernels are not jit-compiled by default so you will need to do it manually:

    .. code::

        step = jax.jit(mclmc.step)
        new_state, info = step(rng_key, state)

    Parameters
    ----------
    logdensity_fn
        The log-density function we wish to draw samples from.
    L
        the momentum decoherence rate
    step_size
        step size of the integrator
    integrator
        an integrator. We recommend using the default here.

    Returns
    -------
    A ``SamplingAlgorithm``.
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


def handle_nans(
    previous_state, next_state, info, key
):

    new_momentum = generate_unit_vector(key, previous_state.position)

    nonans = jnp.logical_and(jnp.all(jnp.isfinite(next_state.position)), jnp.all(jnp.isfinite(next_state.momentum)))

    state, info = jax.lax.cond(
        nonans,
        lambda: (next_state, info),
        lambda: (previous_state._replace(
            momentum=new_momentum,
        ), MCLMCInfo(
            logdensity=previous_state.logdensity,
            energy_change=0.0,
            kinetic_change=0.0,
            nonans=nonans
        )),
    )

    return state, info
def handle_high_energy(
    previous_state, next_state, info, key, cutoff
):

    new_momentum = generate_unit_vector(key, previous_state.position)

    state, info = jax.lax.cond(
        jnp.abs(info.energy_change) > cutoff,
        lambda: (previous_state._replace(
            momentum=new_momentum,
        ), MCLMCInfo(
            logdensity=previous_state.logdensity,
            energy_change=0.0,
            kinetic_change=0.0,
            nonans=info.nonans
        )),
        lambda: (next_state, info),
    )

    return state, info