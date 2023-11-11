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

__all__ = ["MCLMCState", "MCLMCInfo", "init", "build_kernel", "mclmc", "Parameters"]

MCLMCState = integrators.IntegratorState


class MCLMCInfo(NamedTuple):
    """Additional information on the MCLMC transition."""

    transformed_x: Array
    logdensity: Array
    dE: float


class Parameters(NamedTuple):
    """Tunable parameters"""

    L: float
    step_size: float
    inverse_mass_matrix: Array


def init(x_initial: ArrayLike, logdensity_fn, rng_key):
    l, g = jax.value_and_grad(logdensity_fn)(x_initial)
    return MCLMCState(
        position=x_initial,
        momentum=random_unit_vector(rng_key, dim=x_initial.shape[0]),
        logdensity=l,
        logdensity_grad=g,
    )


def build_kernel(grad_logp, dim: int, integrator, transform, params: Parameters):
    step = integrator(T=update_position(grad_logp), V=update_momentum, dim=dim)

    def kernel(rng_key: PRNGKey, state: MCLMCState) -> tuple[MCLMCState, MCLMCInfo]:
        xx, uu, ll, gg, kinetic_change = step(state, params)
        # Langevin-like noise
        nu = jnp.sqrt((jnp.exp(2 * params.step_size / params.L) - 1.0) / dim)
        uu = partially_refresh_momentum(u=uu, rng_key=rng_key, nu=nu)

        return MCLMCState(xx, uu, ll, gg), MCLMCInfo(
            transformed_x=transform(xx),
            logdensity=ll,
            dE=kinetic_change + ll - state.logdensity,
        )

    return kernel


def minimal_norm(dim, T, V):
    lambda_c = 0.1931833275037836  # critical value of the lambda parameter for the minimal norm integrator

    def step(state: MCLMCState, params: Parameters):
        """Integrator from https://arxiv.org/pdf/hep-lat/0505020.pdf, see Equation 20."""

        # V T V T V
        dt = params.step_size
        sigma = jnp.sqrt(params.inverse_mass_matrix)
        uu, r1 = V(dt * lambda_c, state.momentum, state.logdensity_grad * sigma)
        xx, ll, gg = T(dt, state.position, 0.5 * uu * sigma)
        uu, r2 = V(dt * (1 - 2 * lambda_c), uu, gg * sigma)
        xx, ll, gg = T(dt, xx, 0.5 * uu * sigma)
        uu, r3 = V(dt * lambda_c, uu, gg * sigma)

        # kinetic energy change
        kinetic_change = (r1 + r2 + r3) * (dim - 1)

        return xx, uu, ll, gg, kinetic_change

    return step


class mclmc:
    """todo: add documentation"""

    init = staticmethod(init)
    build_kernel = staticmethod(build_kernel)

    def __new__(  # type: ignore[misc]
        cls,
        logdensity_fn: Callable,
        dim: int,
        transform: Callable,
        params: Parameters,
        integrator=minimal_norm,
    ) -> SamplingAlgorithm:
        grad_logp = jax.value_and_grad(logdensity_fn)

        kernel = cls.build_kernel(grad_logp, dim, integrator, transform, params)

        def init_fn(position: ArrayLike):
            return cls.init(position, logdensity_fn, jax.random.PRNGKey(0))

        return SamplingAlgorithm(init_fn, kernel)


###
# helper funcs
###


def random_unit_vector(rng_key, dim):
    u = jax.random.normal(rng_key, shape=(dim,))
    u /= jnp.sqrt(jnp.sum(jnp.square(u)))
    return u


def update_position(grad_logp):
    def update(step_size, x, u):
        xx = x + step_size * u
        ll, gg = grad_logp(xx)
        return xx, ll, gg

    return update


def partially_refresh_momentum(u, rng_key, nu):
    """Adds a small noise to u and normalizes."""
    z = nu * jax.random.normal(rng_key, shape=(u.shape[0],))
    return (u + z) / jnp.sqrt(jnp.sum(jnp.square(u + z)))


###
# integrator
###


def update_momentum(step_size, u, g):
    """The momentum updating map of the esh dynamics (see https://arxiv.org/pdf/2111.02434.pdf)
    similar to the implementation: https://github.com/gregversteeg/esh_dynamics
    There are no exponentials e^delta, which prevents overflows when the gradient norm is large.
    """
    g_norm = jnp.sqrt(jnp.sum(jnp.square(g)))
    e = g / g_norm
    ue = jnp.dot(u, e)
    dim = u.shape[0]
    delta = step_size * g_norm / (dim - 1)
    zeta = jnp.exp(-delta)
    uu = e * (1 - zeta) * (1 + zeta + ue * (1 - zeta)) + 2 * zeta * u
    delta_r = delta - jnp.log(2) + jnp.log(1 + ue + (1 - ue) * zeta**2)
    return uu / jnp.sqrt(jnp.sum(jnp.square(uu))), delta_r
