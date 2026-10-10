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
"""Public API for the ensemble slice sampler."""

from collections.abc import Callable
from functools import partial

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

from blackjax.base import SamplingAlgorithm, build_sampling_algorithm
from blackjax.mcmc.ensemble import EnsembleState, init, red_blue_update
from blackjax.mcmc.slice import SliceInfo, SliceState, stepping_out
from blackjax.mcmc.slice import build_kernel as build_slice_kernel
from blackjax.types import PRNGKey

__all__ = [
    "init",
    "build_kernel",
    "as_top_level_api",
    "differential_direction",
    "gaussian_direction",
    "kde_direction",
]


def differential_direction(rng_key, complementary_positions):
    """The difference of two complementary walkers :cite:p:`karamanis2021ensemble`."""
    pair = jax.tree.map(
        lambda x: jax.random.choice(rng_key, x, (2,), replace=False),
        complementary_positions,
    )
    return jax.tree.map(lambda x: x[0] - x[1], pair)


def gaussian_direction(rng_key, complementary_positions):
    """A draw from a Gaussian with the covariance of the complementary walkers
    :cite:p:`karamanis2021ensemble`."""
    flat_complementary = jax.vmap(lambda x: ravel_pytree(x)[0])(complementary_positions)
    _, unravel_fn = ravel_pytree(jax.tree.map(lambda x: x[0], complementary_positions))
    covariance = jnp.atleast_2d(jnp.cov(flat_complementary, rowvar=False))
    mean = jnp.zeros_like(flat_complementary[0])
    step = jax.random.multivariate_normal(rng_key, mean, covariance, method="svd")
    return unravel_fn(step)


def kde_direction(bw_method: str | float | None = None) -> Callable:
    """The difference of two draws from a Gaussian kernel density estimate of the
    complementary walkers :cite:p:`karamanis2021ensemble`.

    Parameters
    ----------
    bw_method
        The bandwidth of the kernel density estimate, as for
        :class:`jax.scipy.stats.gaussian_kde`.

    """

    def direction(rng_key, complementary_positions):
        flat_complementary = jax.vmap(lambda x: ravel_pytree(x)[0])(
            complementary_positions
        )
        _, unravel_fn = ravel_pytree(
            jax.tree.map(lambda x: x[0], complementary_positions)
        )
        kde = jax.scipy.stats.gaussian_kde(flat_complementary.T, bw_method)
        draws = kde.resample(rng_key, (2,))
        return unravel_fn(draws[:, 0] - draws[:, 1])

    return direction


def build_kernel(max_expansions: int = 10_000, max_shrinkage: int = 10_000):
    """Build an ensemble slice kernel :cite:p:`karamanis2021ensemble`.

    Each walker takes one univariate slice, with the stepping-out procedure,
    along a direction drawn from the complementary walkers.

    Parameters
    ----------
    max_expansions
        Cap on stepping-out steps.
    max_shrinkage
        Cap on shrinkage evaluations. Bounds the loop; on exhaustion the walker
        stays put.

    Returns
    -------
    A kernel ``(rng_key, state, logdensity_fn, direction, width) ->
    (EnsembleState, SliceInfo)``, where ``direction(rng_key,
    complementary_positions)`` draws the direction of one walker's slice.

    """
    slice_kernel = build_slice_kernel(stepping_out, max_expansions, max_shrinkage)

    def update_walker(
        rng_key, walker, complementary_positions, logdensity_fn, direction, width
    ):
        key_direction, key_slice = jax.random.split(rng_key)
        vector = direction(key_direction, complementary_positions)

        def proposal_generator(rng_key, position, logdensity_fn):
            def slice_fn(t):
                x = jax.tree.map(
                    lambda p, v: p + t.astype(p.dtype) * v, position, vector
                )
                return SliceState(x, logdensity_fn(x)), True

            return slice_fn

        new_walker, info = slice_kernel(
            key_slice,
            SliceState(walker.position, walker.logdensity),
            logdensity_fn,
            proposal_generator,
            width,
        )
        return EnsembleState(new_walker.position, new_walker.logdensity), info

    def kernel(
        rng_key: PRNGKey,
        state: EnsembleState,
        logdensity_fn: Callable,
        direction: Callable,
        width: float,
    ) -> tuple[EnsembleState, SliceInfo]:
        """Generate a new ensemble with ensemble slice sampling."""
        return red_blue_update(
            rng_key,
            state,
            partial(
                update_walker,
                logdensity_fn=logdensity_fn,
                direction=direction,
                width=width,
            ),
        )

    return kernel


def as_top_level_api(
    logdensity_fn: Callable,
    direction: Callable = differential_direction,
    width: float = 2.0,
    max_expansions: int = 10_000,
    max_shrinkage: int = 10_000,
) -> SamplingAlgorithm:
    """Implements the user interface for the ensemble slice sampler.

    Examples
    --------

    A new kernel can be initialized and used with the following code:

    .. code::

        ensemble_slice = blackjax.ensemble_slice(logdensity_fn)
        state = ensemble_slice.init(initial_positions)  # shape (num_walkers, dim)
        new_state, info = ensemble_slice.step(rng_key, state)

    Kernels are not jit-compiled by default so you will need to do it manually:

    .. code::

       step = jax.jit(ensemble_slice.step)
       new_state, info = step(rng_key, state)

    Parameters
    ----------
    logdensity_fn
        The log-density function of a single walker's position.
    direction
        The direction of each walker's slice: :func:`differential_direction`
        (the default), :func:`gaussian_direction` or :func:`kde_direction`.
    width
        The initial bracket width along the direction, twice the scale factor
        ``mu`` of :cite:p:`karamanis2021ensemble` (default 2.0).
    max_expansions, max_shrinkage
        Caps on stepping-out steps and shrinkage evaluations.

    Returns
    -------
    A ``SamplingAlgorithm``.

    """
    kernel = build_kernel(max_expansions, max_shrinkage)
    return build_sampling_algorithm(
        kernel, init, logdensity_fn, kernel_args=(direction, width)
    )
