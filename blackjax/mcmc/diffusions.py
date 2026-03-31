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
"""Solvers for Langevin diffusions."""
from typing import NamedTuple

import jax
import jax.numpy as jnp

from blackjax.mcmc.metrics import _scale, _sq_scale
from blackjax.types import Array, ArrayTree
from blackjax.util import generate_gaussian_noise

__all__ = ["overdamped_langevin", "overdamped_manifold_langevin"]


class DiffusionState(NamedTuple):
    position: ArrayTree
    logdensity: float
    logdensity_grad: ArrayTree


def overdamped_langevin(logdensity_grad_fn):
    """Euler solver for overdamped Langevin diffusion.

    Parameters
    ----------
    logdensity_grad_fn
        A function that returns a ``(logdensity, logdensity_grad)`` tuple given
        a position and optional batch arguments.

    Returns
    -------
    A ``one_step`` function that advances the diffusion by one Euler step.
    """

    def one_step(rng_key, state: DiffusionState, step_size: float, batch: tuple = ()):
        position, _, logdensity_grad = state
        noise = generate_gaussian_noise(rng_key, position)
        position = jax.tree.map(
            lambda p, g, n: p + step_size * g + jnp.sqrt(2 * step_size) * n,
            position,
            logdensity_grad,
            noise,
        )

        logdensity, logdensity_grad = logdensity_grad_fn(position, *batch)
        return DiffusionState(position, logdensity, logdensity_grad)

    return one_step


sqrt_multiply = lambda metric, x: _scale(
    metric.mass_matrix_sqrt, metric.inv_mass_matrix_sqrt, x, inv=False, trans=True
)
sqrt_solve = lambda metric, x: _scale(
    metric.mass_matrix_sqrt, metric.inv_mass_matrix_sqrt, x, inv=True, trans=False
)
multiply = lambda metric, x: _sq_scale(
    metric.mass_matrix_sqrt, metric.inv_mass_matrix_sqrt, x, inv=False, trans=False
)
solve = lambda metric, x: _sq_scale(
    metric.mass_matrix_sqrt, metric.inv_mass_matrix_sqrt, x, inv=True, trans=False
)
logdet = lambda metric: 2 * jnp.sum(jnp.log(jnp.diag(metric.mass_matrix_sqrt)))


class DiffusionMetric(NamedTuple):
    mass_matrix_sqrt: Array
    inv_mass_matrix_sqrt: Array
    # inv: bool


class ManifoldDiffusionState(NamedTuple):
    position: ArrayTree
    logdensity: float
    logdensity_grad: ArrayTree
    metric: DiffusionMetric


def overdamped_manifold_langevin(logdensity_grad_fn, mass_matrix_fn):
    """Euler solver for overdamped Langevin diffusion."""

    def one_step(
        rng_key, state: ManifoldDiffusionState, step_size: float, batch: tuple = ()
    ):
        position, _, grad, metric = state
        noise = generate_gaussian_noise(rng_key, position)

        noise = sqrt_solve(metric, noise)

        position = jax.tree_util.tree_map(
            lambda p, g, n: p + step_size * g + jnp.sqrt(2 * step_size) * n,
            position,
            grad,
            noise,
        )

        logdensity, grad = logdensity_grad_fn(position, *batch)

        metric = mass_matrix_fn(position)
        grad = solve(metric, grad)  # natural gradient

        return ManifoldDiffusionState(position, logdensity, grad, metric)

    return one_step
