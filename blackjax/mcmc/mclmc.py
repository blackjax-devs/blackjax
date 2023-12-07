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
from jax.flatten_util import ravel_pytree
from jax.random import normal

from blackjax.base import SamplingAlgorithm
from blackjax.mcmc.integrators import IntegratorState, noneuclidean_mclachlan
from blackjax.types import Array, ArrayLike, PRNGKey
from blackjax.util import generate_unit_vector, pytree_size

__all__ = ["MCLMCInfo", "init", "build_kernel", "mclmc"]


class MCLMCInfo(NamedTuple):
    """
    Additional information on the MCLMC transition.

    transformed_position
        The value of the samples after a transformation. This is typically a projection onto a lower dimensional subspace.
    logdensity
        The log-density of the distribution at the current step of the MCLMC chain.
    kinetic_change
        The difference in kinetic energy between the current and previous step.
    energy_change
        The difference in energy between the current and previous step.
    """

    transformed_position: Array
    logdensity: float
    kinetic_change: float
    energy_change: float


def init(x_initial: ArrayLike, logdensity_fn, rng_key):
    l, g = jax.value_and_grad(logdensity_fn)(x_initial)

    return IntegratorState(
        position=x_initial,
        momentum=generate_unit_vector(rng_key, x_initial),
        logdensity=l,
        logdensity_grad=g,
    )


def build_kernel(logdensity_fn, integrator, transform):
    """Build a HMC kernel.

    Parameters
    ----------
    integrator
        The symplectic integrator to use to integrate the Hamiltonian dynamics.
    transform
        Value of the difference in energy above which we consider that the transition is divergent.
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
    step = integrator(logdensity_fn)

    def kernel(
        rng_key: PRNGKey, state: IntegratorState, L: float, step_size: float
    ) -> tuple[IntegratorState, MCLMCInfo]:
        (position, momentum, logdensity, logdensitygrad), kinetic_change = step(
            state, step_size
        )

        dim = pytree_size(position)

        # Langevin-like noise
        momentum, dim = partially_refresh_momentum(
            momentum=momentum, rng_key=rng_key, L=L, step_size=step_size
        )

        return IntegratorState(
            position, momentum, logdensity, logdensitygrad
        ), MCLMCInfo(
            transformed_position=transform(position),
            logdensity=logdensity,
            energy_change=kinetic_change - logdensity + state.logdensity,
            kinetic_change=kinetic_change * (dim - 1),
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
        L,
        step_size,
        transform: Callable = (lambda x: x),
        integrator=noneuclidean_mclachlan,
        seed=1,
    ) -> SamplingAlgorithm:
        kernel = cls.build_kernel(logdensity_fn, integrator, transform)

        def update_fn(rng_key, state):
            return kernel(rng_key, state, L, step_size)

        def init_fn(position: ArrayLike):
            return cls.init(position, logdensity_fn, jax.random.PRNGKey(seed))

        return SamplingAlgorithm(init_fn, update_fn)


def partially_refresh_momentum(momentum, rng_key, step_size, L):
    """Adds a small noise to momentum and normalizes.

    Parameters
    ----------
    rng_key
        The pseudo-random number generator key used to generate random numbers.
    momentum
        PyTree that the structure the output should to match.
    step_size
        Step size
    L
        controls rate of momentum change

    Returns
    -------
    momentum with random change in angle
    """
    m, unravel_fn = ravel_pytree(momentum)
    dim = m.shape[0]
    nu = jnp.sqrt((jnp.exp(2 * step_size / L) - 1.0) / dim)
    z = nu * normal(rng_key, shape=m.shape, dtype=m.dtype)
    return unravel_fn((m + z) / jnp.linalg.norm(m + z)), dim
