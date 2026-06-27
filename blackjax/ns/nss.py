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
"""Nested Slice Sampling (NSS) algorithm.

An example implementation of Nested Sampling with a hit-and-run slice sampler as
the inner MCMC kernel (Yallup, Kroupa & Handley, 2026, arXiv:2601.23252).
"""

from functools import partial
from typing import Callable

import jax
import jax.flatten_util
import jax.numpy as jnp

from blackjax import SamplingAlgorithm
from blackjax.mcmc.slice import build_kernel as build_slice_kernel
from blackjax.mcmc.slice import stepping_out
from blackjax.ns.adaptive import build_kernel as build_adaptive_kernel
from blackjax.ns.adaptive import init
from blackjax.ns.base import NSInfo, NSState
from blackjax.ns.base import delete_fn as default_delete_fn
from blackjax.ns.base import init_state_strategy
from blackjax.ns.from_mcmc import update_with_mcmc_take_last
from blackjax.smc.tuning.from_particles import particles_covariance_matrix
from blackjax.types import Array, ArrayTree, PRNGKey

__all__ = [
    "as_top_level_api",
    "build_kernel",
    "init",
]


def sample_direction_from_covariance(
    rng_key: PRNGKey, position: ArrayTree, cov: Array
) -> ArrayTree:
    """A random direction shaped by ``cov``, scaled to Mahalanobis norm 2.

    Samples ``d ~ N(0, cov)`` and normalizes it with ``inv(cov)`` to a length of
    2 in the covariance metric (~2 std devs, a step size that mixes well). Uses
    ``inv(cov)`` rather than an explicit Cholesky factor, which is less robust on
    GPU.
    """
    _, unravel_fn = jax.flatten_util.ravel_pytree(position)
    d = jax.random.multivariate_normal(rng_key, jnp.zeros(cov.shape[0]), cov)
    d = 2.0 * d / jnp.sqrt(d @ jnp.linalg.inv(cov) @ d)
    return unravel_fn(d)


def _inner_kernel_params(
    rng_key: PRNGKey,
    state: NSState,
    info: NSInfo,
    inner_kernel_params: dict[str, ArrayTree] | None = None,
) -> dict[str, ArrayTree]:
    """Live-point covariance, recomputed each step to shape the direction."""
    # rng_key, info and inner_kernel_params are unused here but required by the
    # adaptive-kernel callback protocol (update_inner_kernel_params_fn).
    return {
        "cov": jnp.atleast_2d(particles_covariance_matrix(state.particles.position))
    }


def build_kernel(
    init_state_fn: Callable,
    num_inner_steps: int,
    num_delete: int = 1,
    max_steps: int = 10,
    max_shrinkage: int = 100,
) -> Callable:
    """Build the Nested Slice Sampling kernel.

    Parameters
    ----------
    init_state_fn
        Builds a particle state from a position and birth log-likelihood.
    num_inner_steps
        Number of slice steps per new particle (typically a multiple of the
        dimension).
    num_delete
        Number of particles deleted and replaced per step (default 1).
    max_steps
        Cap on stepping-out expansions per slice (default 10).
    max_shrinkage
        Cap on shrinkage evaluations per slice (default 100).

    Returns
    -------
    A kernel ``kernel(rng_key, state)`` that returns ``(new_state, info)``.
    """
    slice_kernel = build_slice_kernel(
        interval=stepping_out,
        max_expansions=max_steps,
        max_shrinkage=max_shrinkage,
    )

    def constrained_mcmc_slice_fn(rng_key, state, loglikelihood_0, cov):
        def proposal_generator(prop_key, position, _logdensity_fn):
            direction = sample_direction_from_covariance(prop_key, position, cov)

            def slice_fn(t):
                x = jax.tree.map(lambda p, d: p + t * d, position, direction)
                new_state = init_state_fn(x, loglikelihood_birth=loglikelihood_0)
                return new_state, new_state.loglikelihood > loglikelihood_0

            return slice_fn

        return slice_kernel(rng_key, state, None, proposal_generator)

    inner_kernel = update_with_mcmc_take_last(
        constrained_mcmc_slice_fn, num_inner_steps, num_delete
    )
    delete_fn = partial(default_delete_fn, num_delete=num_delete)
    return build_adaptive_kernel(
        delete_fn,
        inner_kernel,
        update_inner_kernel_params_fn=_inner_kernel_params,
    )


def as_top_level_api(
    logprior_fn: Callable,
    loglikelihood_fn: Callable,
    num_inner_steps: int,
    num_delete: int = 1,
    max_steps: int = 10,
    max_shrinkage: int = 100,
) -> SamplingAlgorithm:
    """Creates a Nested Slice Sampling (NSS) algorithm.

    Nested Sampling with a hit-and-run slice inner kernel: each particle
    replacement runs ``num_inner_steps`` constrained slice moves along
    directions shaped by the live-point covariance.

    Parameters
    ----------
    logprior_fn
        Log-prior of a single particle.
    loglikelihood_fn
        Log-likelihood of a single particle.
    num_inner_steps
        Number of slice steps per new particle (typically a multiple of the
        dimension).
    num_delete
        Number of particles deleted and replaced per step (default 1).
    max_steps
        Cap on stepping-out expansions per slice (default 10).
    max_shrinkage
        Cap on shrinkage evaluations per slice (default 100).

    Returns
    -------
    A ``SamplingAlgorithm`` whose ``step(rng_key, state)`` returns
    ``(new_state, info)``.
    """
    init_state_fn = partial(
        init_state_strategy,
        logprior_fn=logprior_fn,
        loglikelihood_fn=loglikelihood_fn,
    )
    kernel = build_kernel(
        init_state_fn,
        num_inner_steps,
        num_delete,
        max_steps=max_steps,
        max_shrinkage=max_shrinkage,
    )

    def init_fn(position, rng_key=None):
        return init(
            position,
            init_state_fn=jax.vmap(init_state_fn),
            update_inner_kernel_params_fn=_inner_kernel_params,
            rng_key=rng_key,
        )

    def step_fn(rng_key, state):
        return kernel(rng_key, state)

    return SamplingAlgorithm(init_fn, step_fn)
