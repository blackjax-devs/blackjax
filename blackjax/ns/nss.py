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
from blackjax.ns.adaptive import init
from blackjax.ns.base import NSInfo, NSState, init_state_strategy
from blackjax.ns.from_mcmc import build_kernel as build_from_mcmc_kernel
from blackjax.smc.tuning.from_particles import particles_covariance_matrix
from blackjax.types import Array, ArrayTree, PRNGKey

__all__ = [
    "as_top_level_api",
    "build_kernel",
    "covariance_proposal",
    "init",
    "live_covariance",
    "slice_constrained_step",
]


def sample_direction_from_covariance(
    rng_key: PRNGKey, position: ArrayTree, cov: Array
) -> ArrayTree:
    """A random direction shaped by ``cov``, scaled to Mahalanobis norm 2.

    Samples ``d ~ N(0, cov)`` and normalizes it with ``inv(cov)`` to a length of
    2 in the covariance metric (~2 std devs, a step size that mixes well). Uses
    ``inv(cov)`` rather than an explicit Cholesky factor, has encountered differing
    behavior on GPU.
    """
    _, unravel_fn = jax.flatten_util.ravel_pytree(position)
    d = jax.random.multivariate_normal(rng_key, jnp.zeros(cov.shape[0]), cov)
    d = 2.0 * d / jnp.sqrt(d @ jnp.linalg.inv(cov) @ d)
    return unravel_fn(d)


def covariance_proposal(
    init_state_fn: Callable, loglikelihood_0: Array, cov: Array
) -> Callable:
    """Proposal generator for nested slice sampling.

    The nested-sampling analogue of
    :func:`~blackjax.mcmc.slice.direction_proposal`: it steps along a
    covariance-shaped direction and gates the hard likelihood constraint into
    ``is_valid``. The returned ``slice_fn`` builds the candidate particle
    (recording its log-likelihood, computed once) and reports it admissible only
    when ``loglikelihood > loglikelihood_0``. Override it to write a custom
    nested stepper.
    """

    def proposal_generator(rng_key, position, logdensity_fn):
        del logdensity_fn  # NS gates on the recorded loglikelihood, not logdensity
        direction = sample_direction_from_covariance(rng_key, position, cov)

        def slice_fn(t):
            x = jax.tree.map(lambda p, d: p + t * d, position, direction)
            new_state = init_state_fn(x, loglikelihood_birth=loglikelihood_0)
            return new_state, new_state.loglikelihood > loglikelihood_0

        return slice_fn

    return proposal_generator


def live_covariance(
    rng_key: PRNGKey,
    state: NSState,
    info: NSInfo,
    params: dict[str, ArrayTree] | None = None,
) -> dict[str, ArrayTree]:
    """Live-point covariance, recomputed each step to shape the direction."""
    # rng_key, info and params are unused here but required by the
    # adaptive-kernel callback protocol (update_inner_kernel_params_fn).
    return {
        "cov": jnp.atleast_2d(particles_covariance_matrix(state.particles.position))
    }


def slice_constrained_step(
    init_state_fn: Callable, slice_kernel: Callable, proposal: Callable
) -> Callable:
    """The slice-family constrained inner step for nested sampling.

    Runs ``slice_kernel`` with a constrained proposal generator built by
    ``proposal(init_state_fn, loglikelihood_0, **params)``; the proposal's
    ``slice_fn`` gates ``is_valid`` on the likelihood contour, so the slice
    shrinks until it lands inside it (no wasted steps). The slice counterpart to
    :func:`~blackjax.ns.from_mcmc.reject_constrained_step`, consumed by
    :func:`~blackjax.ns.from_mcmc.build_kernel`.
    """

    def step(rng_key, state, loglikelihood_0, **params):
        proposal_generator = proposal(init_state_fn, loglikelihood_0, **params)
        return slice_kernel(rng_key, state, None, proposal_generator)

    return step


def build_kernel(
    init_state_fn: Callable,
    num_inner_steps: int,
    num_delete: int = 1,
    max_steps: int = 10,
    max_shrinkage: int = 100,
    proposal: Callable = covariance_proposal,
    inner_kernel_params: Callable = live_covariance,
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
    proposal
        Proposal factory ``(init_state_fn, loglikelihood_0, cov) ->
        proposal_generator`` (:func:`covariance_proposal` by default). Override
        to write a custom nested stepper.
    inner_kernel_params
        Computes the inner-kernel parameters from the live points each step,
        ``(rng_key, state, info, params) -> params`` (:func:`live_covariance`
        by default, the live-point covariance).

    Returns
    -------
    A kernel ``kernel(rng_key, state)`` that returns ``(new_state, info)``.
    """
    slice_kernel = build_slice_kernel(
        interval=stepping_out,
        max_expansions=max_steps,
        max_shrinkage=max_shrinkage,
    )
    constrained_step_fn = slice_constrained_step(init_state_fn, slice_kernel, proposal)
    return build_from_mcmc_kernel(
        constrained_step_fn,
        num_inner_steps,
        inner_kernel_params,
        num_delete,
    )


def as_top_level_api(
    logprior_fn: Callable,
    loglikelihood_fn: Callable,
    num_inner_steps: int,
    num_delete: int = 1,
    max_steps: int = 10,
    max_shrinkage: int = 100,
    proposal: Callable = covariance_proposal,
    inner_kernel_params: Callable = live_covariance,
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
    proposal
        Proposal factory ``(init_state_fn, loglikelihood_0, cov) ->
        proposal_generator`` (:func:`covariance_proposal` by default). Override
        to write a custom nested stepper.
    inner_kernel_params
        Computes the inner-kernel parameters from the live points,
        ``(rng_key, state, info, params) -> params`` (:func:`live_covariance`
        by default). Used both to seed ``init`` and to update each step.

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
        proposal=proposal,
        inner_kernel_params=inner_kernel_params,
    )

    def init_fn(position, rng_key=None):
        return init(
            position,
            init_state_fn=jax.vmap(init_state_fn),
            update_inner_kernel_params_fn=inner_kernel_params,
            rng_key=rng_key,
        )

    def step_fn(rng_key, state):
        return kernel(rng_key, state)

    return SamplingAlgorithm(init_fn, step_fn)
