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

An example implementation of Nested Sampling with a slice sampler as the inner
MCMC kernel (Yallup, Kroupa & Handley, 2026, arXiv:2601.23252). The default
:func:`build_kernel` uses hit-and-run moves shaped by the live-point covariance;
:func:`build_swig_kernel` offers an axis-aligned slice-within-Gibbs alternative.
"""

from functools import partial
from typing import Callable

import jax
import jax.flatten_util
import jax.numpy as jnp

from blackjax import SamplingAlgorithm
from blackjax.mcmc.slice import SliceInfo
from blackjax.mcmc.slice import build_kernel as build_slice_kernel
from blackjax.mcmc.slice import random_order, stepping_out
from blackjax.ns.adaptive import init
from blackjax.ns.base import NSInfo, NSState, init_state_strategy
from blackjax.ns.from_mcmc import build_kernel as build_from_mcmc_kernel
from blackjax.smc.tuning.from_particles import (
    particles_covariance_matrix,
    particles_stds,
)
from blackjax.types import Array, ArrayTree, PRNGKey

__all__ = [
    "as_top_level_api",
    "build_kernel",
    "build_swig_kernel",
    "coordinate_constrained_step",
    "coordinate_proposal",
    "covariance_proposal",
    "init",
    "live_covariance",
    "live_widths",
    "slice_constrained_step",
    "swig_as_top_level_api",
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


def coordinate_proposal(
    init_state_fn: Callable, loglikelihood_0: Array, i: Array, width: Array
) -> Callable:
    """Per-axis proposal generator for nested slice-within-Gibbs (SwiG).

    The coordinate counterpart of :func:`covariance_proposal`: it steps along
    axis ``i`` scaled by ``width`` (the direction ``width * e_i``) and gates the
    hard likelihood constraint into ``is_valid``. Like :func:`covariance_proposal`,
    the move's scale lives in the direction, so the univariate slice always runs at
    unit width. The returned ``slice_fn`` builds the candidate particle (recording
    its log-likelihood) and reports it admissible only when
    ``loglikelihood > loglikelihood_0``; it threads the full particle, so the
    recorded loglikelihood survives the sweep. Override it to write a custom
    nested coordinate stepper.
    """

    def proposal_generator(rng_key, position, logdensity_fn):
        del rng_key, logdensity_fn  # axis move is deterministic; gate on loglik
        flat, unravel_fn = jax.flatten_util.ravel_pytree(position)

        def slice_fn(t):
            x = unravel_fn(flat.at[i].add(t * width))
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


def live_widths(
    rng_key: PRNGKey,
    state: NSState,
    info: NSInfo,
    params: dict[str, ArrayTree] | None = None,
) -> dict[str, ArrayTree]:
    """Per-axis live-point spread (std): the per-coordinate slice widths for SwiG.

    The coordinate counterpart of :func:`live_covariance`: only the marginal
    per-axis spread is used, so axis correlations are deliberately ignored -- the
    defining trait of a coordinate (slice-within-Gibbs) move. Overridable via the
    ``inner_kernel_params`` seam of :func:`build_swig_kernel` and
    :func:`swig_as_top_level_api`, mirroring :func:`live_covariance`.
    """
    # rng_key, info and params are unused here but required by the
    # adaptive-kernel callback protocol (update_inner_kernel_params_fn).
    return {"widths": particles_stds(state.particles.position)}


def slice_constrained_step(
    init_state_fn: Callable, slice_kernel: Callable, proposal: Callable
) -> Callable:
    """The slice-family constrained inner step for nested sampling.

    Runs ``slice_kernel`` with a constrained proposal generator built by
    ``proposal(init_state_fn, loglikelihood_0, **params)``; the proposal's
    ``slice_fn`` gates ``is_valid`` on the likelihood contour, so the slice
    shrinks until it lands inside it (no wasted steps). The slice counterpart to
    :func:`reject_constrained_step`, consumed by
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
        Number of slice steps per new particle. Prefer
        ``num_inner_steps >= max(5, 2 * dim)`` for reliable mixing (bare ``dim`` is
        the minimum; see :func:`as_top_level_api`).
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


def coordinate_constrained_step(
    init_state_fn: Callable,
    slice_kernel: Callable,
    proposal: Callable = coordinate_proposal,
    coordinate_order: Callable = random_order,
) -> Callable:
    """The coordinate-sweep constrained inner step for nested sampling (SwiG).

    The slice-within-Gibbs counterpart of :func:`slice_constrained_step`: one call
    sweeps every axis once -- in the order set by ``coordinate_order`` -- updating
    each by a univariate slice from the per-axis proposal generator
    ``proposal(init_state_fn, loglikelihood_0, i, width)`` (:func:`coordinate_proposal`
    by default, the axis analogue of :func:`covariance_proposal` passed to
    :func:`slice_constrained_step`), which steps along ``width * e_i`` and gates the
    likelihood contour into ``is_valid``. As with the hit-and-run path the scale
    lives in the direction, so the univariate slice runs at unit width. Consumed by
    :func:`~blackjax.ns.from_mcmc.build_kernel` exactly like the hit-and-run step.
    """

    def step(rng_key, state, loglikelihood_0, widths):
        order_key, sweep_key = jax.random.split(rng_key)
        flat, unravel_fn = jax.flatten_util.ravel_pytree(state.position)
        d = flat.shape[0]
        order = coordinate_order(order_key, d)

        def body(carry_state, inp):
            key, i, width = inp
            # Scale lives in the proposal's direction (width * e_i), so the slice
            # runs at unit width -- symmetric with the hit-and-run path.
            proposal_generator = proposal(init_state_fn, loglikelihood_0, i, width)
            return slice_kernel(key, carry_state, None, proposal_generator)

        keys = jax.random.split(sweep_key, order.shape[0])
        final_state, swept = jax.lax.scan(body, state, (keys, order, widths[order]))

        # Sum the per-axis counters and scatter the brackets back into the
        # position structure, matching :func:`~blackjax.mcmc.slice.coordinate_slice`.
        def stitch(v):
            return unravel_fn(jnp.zeros(d, v.dtype).at[order].set(v))

        info = SliceInfo(
            is_accepted=jnp.all(swept.is_accepted),
            num_expansions=jnp.sum(swept.num_expansions),
            num_shrink=jnp.sum(swept.num_shrink),
            bracket_left=stitch(swept.bracket_left),
            bracket_right=stitch(swept.bracket_right),
        )
        return final_state, info

    return step


def build_swig_kernel(
    init_state_fn: Callable,
    num_inner_steps: int,
    num_delete: int = 1,
    max_steps: int = 10,
    max_shrinkage: int = 100,
    proposal: Callable = coordinate_proposal,
    coordinate_order: Callable = random_order,
    inner_kernel_params: Callable = live_widths,
) -> Callable:
    """Build the Nested Slice-within-Gibbs (SwiG) kernel.

    The coordinate counterpart of :func:`build_kernel`: each inner step is a full
    coordinate *sweep* rather than a hit-and-run move. Axes are visited in the
    order set by ``coordinate_order`` (:func:`~blackjax.mcmc.slice.random_order`
    by default, or :func:`~blackjax.mcmc.slice.fixed_order`), and each is updated
    by a univariate slice gated on the likelihood contour and scaled by that
    axis's live width (the per-axis spread of the live points; correlations are
    ignored). Prefer this when the target is close to axis-aligned, or when its
    correlations are unreliable to estimate. Pair with
    :func:`swig_as_top_level_api` for the bundled (init, step) algorithm.

    Parameters
    ----------
    init_state_fn
        Builds a particle state from a position and birth log-likelihood.
    num_inner_steps
        Number of coordinate sweeps per new particle. Prefer
        ``num_inner_steps >= max(5, 2 * dim)`` for reliable mixing (bare ``dim`` is
        the minimum; see :func:`swig_as_top_level_api`).
    num_delete
        Number of particles deleted and replaced per step (default 1).
    max_steps
        Cap on stepping-out expansions per univariate slice (default 10).
    max_shrinkage
        Cap on shrinkage evaluations per univariate slice (default 100).
    proposal
        Per-axis proposal factory ``(init_state_fn, loglikelihood_0, i, width)
        -> proposal_generator`` (:func:`coordinate_proposal` by default). The
        coordinate analogue of the ``proposal`` seam on :func:`build_kernel`.
    coordinate_order
        Sweep-order primitive ``(rng_key, d) -> indices``
        (:func:`~blackjax.mcmc.slice.random_order` by default).
    inner_kernel_params
        Computes the inner-kernel parameters from the live points each step,
        ``(rng_key, state, info, params) -> params`` (:func:`live_widths` by
        default, the per-axis live-point spread).

    Returns
    -------
    A kernel ``kernel(rng_key, state)`` that returns ``(new_state, info)``.
    """
    slice_kernel = build_slice_kernel(
        interval=stepping_out,
        max_expansions=max_steps,
        max_shrinkage=max_shrinkage,
    )
    constrained_step_fn = coordinate_constrained_step(
        init_state_fn,
        slice_kernel,
        proposal=proposal,
        coordinate_order=coordinate_order,
    )
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
    """Creates a Nested Slice Sampling (NSS) algorithm, ``blackjax.nss``.

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
        Number of slice steps per new particle. Use
        ``num_inner_steps >= max(5, 2 * dim)`` for reliable mixing within the
        likelihood contour; bare ``dim`` is the minimum and can bias the evidence
        *upward* for ``dim > 10`` (the inner chain must decorrelate the new particle
        from the deleted one, not merely satisfy the constraint).
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

    Notes
    -----
    The live particles in the run state (``state.particles``) are **not** posterior
    samples: they are the current likelihood shell and at termination collapse to
    the highest-likelihood mode. For correctly-weighted posterior draws, pass the
    dead points through :func:`~blackjax.ns.utils.finalise` and resample with
    :func:`~blackjax.ns.utils.sample`.

    The covariance-shaped proposal bridges between modes only up to moderate
    separation. For strongly multimodal targets, ensure the initial live points
    span every mode (and consider a clustering inner kernel); minor modes are still
    weighted correctly in the resampled posterior, but may be absent from the final
    live set.
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


def swig_as_top_level_api(
    logprior_fn: Callable,
    loglikelihood_fn: Callable,
    num_inner_steps: int,
    num_delete: int = 1,
    max_steps: int = 10,
    max_shrinkage: int = 100,
    proposal: Callable = coordinate_proposal,
    coordinate_order: Callable = random_order,
    inner_kernel_params: Callable = live_widths,
) -> SamplingAlgorithm:
    """Creates a Nested Slice-within-Gibbs (SwiG) sampling algorithm, ``blackjax.nsswig``.

    Nested Sampling with an axis-aligned slice-within-Gibbs inner kernel: each
    particle replacement runs ``num_inner_steps`` constrained coordinate sweeps,
    each axis scaled by the live-point spread (correlations are ignored). The
    coordinate counterpart of :func:`as_top_level_api`; prefer it when the target
    is close to axis-aligned or its correlations are unreliable to estimate.

    Parameters
    ----------
    logprior_fn
        Log-prior of a single particle.
    loglikelihood_fn
        Log-likelihood of a single particle.
    num_inner_steps
        Number of coordinate sweeps per new particle. Use
        ``num_inner_steps >= max(5, 2 * dim)`` for reliable mixing within the
        likelihood contour; bare ``dim`` is the minimum and can bias the evidence
        *upward* for ``dim > 10`` (the inner chain must decorrelate the new particle
        from the deleted one, not merely satisfy the constraint).
    num_delete
        Number of particles deleted and replaced per step (default 1).
    max_steps
        Cap on stepping-out expansions per univariate slice (default 10).
    max_shrinkage
        Cap on shrinkage evaluations per univariate slice (default 100).
    proposal
        Per-axis proposal factory ``(init_state_fn, loglikelihood_0, i, width)
        -> proposal_generator`` (:func:`coordinate_proposal` by default), the
        coordinate analogue of the ``proposal`` seam on :func:`as_top_level_api`.
    coordinate_order
        Sweep-order primitive ``(rng_key, d) -> indices``
        (:func:`~blackjax.mcmc.slice.random_order` by default).
    inner_kernel_params
        Computes the inner-kernel parameters from the live points,
        ``(rng_key, state, info, params) -> params`` (:func:`live_widths` by
        default). Used both to seed ``init`` and to update each step.

    Returns
    -------
    A ``SamplingAlgorithm`` whose ``step(rng_key, state)`` returns
    ``(new_state, info)``.

    Notes
    -----
    The live particles in the run state (``state.particles``) are **not** posterior
    samples: they are the current likelihood shell and at termination collapse to
    the highest-likelihood mode. For correctly-weighted posterior draws, pass the
    dead points through :func:`~blackjax.ns.utils.finalise` and resample with
    :func:`~blackjax.ns.utils.sample`.

    For strongly multimodal targets, ensure the initial live points span every mode
    (the axis-aligned per-particle proposal does not bridge well-separated modes);
    minor modes are still weighted correctly in the resampled posterior, but may be
    absent from the final live set.
    """
    init_state_fn = partial(
        init_state_strategy,
        logprior_fn=logprior_fn,
        loglikelihood_fn=loglikelihood_fn,
    )
    kernel = build_swig_kernel(
        init_state_fn,
        num_inner_steps,
        num_delete,
        max_steps=max_steps,
        max_shrinkage=max_shrinkage,
        proposal=proposal,
        coordinate_order=coordinate_order,
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
