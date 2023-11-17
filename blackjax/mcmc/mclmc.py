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

import blackjax.mcmc.integrators as integrators
from blackjax.base import SamplingAlgorithm
from blackjax.types import Array, ArrayLike, PRNGKey

__all__ = ["MCLMCState", "MCLMCInfo", "init", "build_kernel", "mclmc"]

MCLMCState = integrators.IntegratorState


class MCLMCInfo(NamedTuple):
    """Additional information on the MCLMC transition.

    transformed_x
      The value of the samples after a transformation (e.g. projection onto lower dim subspace)
    logdensity
      logdensity at given step
    dE
      energy difference

    """

    transformed_x: Array
    logdensity: float
    kinetic_change: float
    dE: float


def init(x_initial: ArrayLike, logdensity_fn, rng_key):
    l, g = jax.value_and_grad(logdensity_fn)(x_initial)
    # jax.debug.print("🤯 {x} initial momentum 🤯", x=random_unit_vector(rng_key, dim=x_initial.shape[0]))

    return MCLMCState(
        position=x_initial,
        momentum=random_unit_vector(rng_key, dim=x_initial.shape[0]),
        logdensity=l,
        logdensity_grad=g,
    )


def build_kernel(grad_logp, integrator, transform):
    """Build a HMC kernel.

    Parameters
    ----------
    integrator
        The symplectic integrator to use to integrate the Hamiltonian dynamics.
    transform
        Value of the difference in energy above which we consider that the transition is divergent.
    L
      the momentum decoherence rate
    step_size
      step size of the integrator

    Returns
    -------
    A kernel that takes a rng_key and a Pytree that contains the current state
    of the chain and that returns a new state of the chain along with
    information about the transition.

    """
    step = integrator(
        T=integrators.update_position_mclmc(grad_logp),
        V=integrators.update_momentum_mclmc,
    )

    def kernel(
        rng_key: PRNGKey, state: MCLMCState, L: float, step_size: float
    ) -> tuple[MCLMCState, MCLMCInfo]:
        xx, uu, ll, gg, kinetic_change = step(state, step_size)
        # jax.debug.print("🤯 {x} new 🤯", x=(kinetic_change, ll, state.logdensity))

        dim = xx.shape[0]
        # Langevin-like noise
        nu = jnp.sqrt((jnp.exp(2 * step_size / L) - 1.0) / dim)
        uu = partially_refresh_momentum(u=uu, rng_key=rng_key, nu=nu)

        return MCLMCState(xx, uu, ll, gg), MCLMCInfo(
            transformed_x=transform(xx),
            logdensity=ll,
            dE=kinetic_change - ll + state.logdensity,
            kinetic_change=kinetic_change,
        )

    return kernel


class mclmc:
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
            transform=lambda x: x,
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
    transform
        A function to perform on the samples drawn from the target distribution
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

    init = staticmethod(init)
    build_kernel = staticmethod(build_kernel)

    def __new__(  # type: ignore[misc]
        cls,
        logdensity_fn: Callable,
        transform: Callable,
        L,
        step_size,
        integrator=integrators.minimal_norm,
    ) -> SamplingAlgorithm:
        grad_logp = jax.value_and_grad(logdensity_fn)

        kernel = cls.build_kernel(grad_logp, integrator, transform)

        def update_fn(rng_key, state):
            return kernel(rng_key, state, L, step_size)

        def init_fn(position: ArrayLike):
            return cls.init(position, logdensity_fn, jax.random.PRNGKey(0))

        return SamplingAlgorithm(init_fn, update_fn)


###
# helper funcs
###


def random_unit_vector(rng_key, dim):
    u = jax.random.normal(rng_key, shape=(dim,))
    u /= jnp.sqrt(jnp.sum(jnp.square(u)))
    return u


def partially_refresh_momentum(u, rng_key, nu):
    """Adds a small noise to u and normalizes."""
    z = nu * jax.random.normal(rng_key, shape=(u.shape[0],))
    return (u + z) / jnp.sqrt(jnp.sum(jnp.square(u + z)))
