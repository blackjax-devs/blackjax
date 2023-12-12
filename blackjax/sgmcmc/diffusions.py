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
import operator

import jax
import jax.numpy as jnp

from blackjax.types import ArrayLikeTree, ArrayTree, PRNGKey
from blackjax.util import generate_gaussian_noise, pytree_size

__all__ = ["overdamped_langevin", "sghmc", "sgnht"]


def overdamped_langevin():
    """Euler solver for overdamped Langevin diffusion.

    This algorithm was ported from :cite:p:`coullon2022sgmcmcjax`.

    """

    def one_step(
        rng_key: PRNGKey,
        position: ArrayLikeTree,
        logdensity_grad: ArrayLikeTree,
        step_size: float,
        temperature: float = 1.0,
    ) -> ArrayTree:
        noise = generate_gaussian_noise(rng_key, position)
        position = jax.tree_util.tree_map(
            lambda p, g, n: p
            + step_size * g
            + jnp.sqrt(2 * temperature * step_size) * n,
            position,
            logdensity_grad,
            noise,
        )

        return position

    return one_step


def sghmc(alpha: float = 0.01, beta: float = 0):
    """Euler solver for the diffusion equation of the SGHMC algorithm :cite:p:`chen2014stochastic`,
    with parameters alpha and beta scaled according to :cite:p:`ma2015complete`.

    This algorithm was ported from :cite:p:`coullon2022sgmcmcjax`.

    """

    def one_step(
        rng_key: PRNGKey,
        position: ArrayLikeTree,
        momentum: ArrayLikeTree,
        logdensity_grad: ArrayLikeTree,
        step_size: float,
        temperature: float = 1.0,
    ):
        noise = generate_gaussian_noise(rng_key, position)
        position = jax.tree_util.tree_map(
            lambda x, p: x + step_size * p, position, momentum
        )
        momentum = jax.tree_util.tree_map(
            lambda p, g, n: (1.0 - alpha * step_size) * p
            + step_size * g
            + jnp.sqrt(
                step_size * temperature * (2 * alpha - step_size * temperature * beta)
            )
            * n,
            momentum,
            logdensity_grad,
            noise,
        )

        return position, momentum

    return one_step


def sgnht(alpha: float = 0.01, beta: float = 0):
    """Euler solver for the diffusion equation of the SGNHT algorithm :cite:p:`ding2014bayesian`.

    This algorithm was ported from :cite:p:`coullon2022sgmcmcjax`.

    """

    def one_step(
        rng_key: PRNGKey,
        position: ArrayLikeTree,
        momentum: ArrayLikeTree,
        xi: float,
        logdensity_grad: ArrayLikeTree,
        step_size: float,
        temperature: float = 1.0,
    ):
        noise = generate_gaussian_noise(rng_key, position)
        position = jax.tree_util.tree_map(
            lambda x, p: x + step_size * p, position, momentum
        )
        momentum = jax.tree_util.tree_map(
            lambda p, g, n: (1.0 - xi * step_size) * p
            + step_size * g
            + jnp.sqrt(
                step_size * temperature * (2 * alpha - step_size * temperature * beta)
            )
            * n,
            momentum,
            logdensity_grad,
            noise,
        )
        momentum_dot = jax.tree_util.tree_reduce(
            operator.add, jax.tree_util.tree_map(lambda x: jnp.sum(x * x), momentum)
        )
        d = pytree_size(momentum)
        xi = xi + step_size * (momentum_dot / d - temperature)
        return position, momentum, xi

    return one_step
