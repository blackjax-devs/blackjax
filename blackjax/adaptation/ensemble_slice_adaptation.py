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
"""Adaptation of the bracket width of the ensemble slice sampler."""

from collections.abc import Callable
from typing import NamedTuple

import jax
import jax.numpy as jnp

import blackjax.mcmc.ensemble_slice as ensemble_slice
from blackjax.adaptation.base import AdaptationResults, return_all_adapt_info
from blackjax.base import AdaptationAlgorithm
from blackjax.mcmc.slice import SliceInfo
from blackjax.types import Array, ArrayLikeTree, PRNGKey

__all__ = ["EnsembleSliceAdaptationState", "base", "ensemble_slice_adaptation"]


class EnsembleSliceAdaptationState(NamedTuple):
    """State of the ensemble slice adaptation.

    width
        The current bracket width.
    num_within_tolerance
        The number of steps so far whose fraction of expansions was within the
        tolerance of one half.

    """

    width: Array
    num_within_tolerance: Array


def base(tolerance: float = 0.05, patience: int = 5):
    """Adapt the bracket width so that expansions and contractions balance
    :cite:p:`karamanis2021ensemble`.

    Parameters
    ----------
    tolerance
        The tolerance on the fraction of expansions around one half.
    patience
        The number of steps within the tolerance after which the width is fixed.

    Returns
    -------
    init
        Function that initializes the adaptation state.
    update
        Function that updates the adaptation state with the information of one
        ensemble slice step.

    """

    def init(width: float) -> EnsembleSliceAdaptationState:
        return EnsembleSliceAdaptationState(jnp.asarray(width), jnp.asarray(0))

    def update(
        adaptation_state: EnsembleSliceAdaptationState, info: SliceInfo
    ) -> EnsembleSliceAdaptationState:
        width, num_within_tolerance = adaptation_state
        is_tuning = num_within_tolerance <= patience

        num_expansions = jnp.maximum(jnp.sum(info.num_expansions), 1)
        num_contractions = jnp.sum(info.num_shrink - info.is_accepted)
        expansion_fraction = num_expansions / (num_expansions + num_contractions)

        width = jnp.where(is_tuning, 2.0 * expansion_fraction * width, width)
        is_within_tolerance = jnp.abs(expansion_fraction - 0.5) < tolerance
        num_within_tolerance = num_within_tolerance + (is_tuning & is_within_tolerance)
        return EnsembleSliceAdaptationState(width, num_within_tolerance)

    return init, update


def ensemble_slice_adaptation(
    logdensity_fn: Callable,
    direction: Callable = ensemble_slice.differential_direction,
    initial_width: float = 2.0,
    max_expansions: int = 10_000,
    max_shrinkage: int = 10_000,
    tolerance: float = 0.05,
    patience: int = 5,
    adaptation_info_fn: Callable = return_all_adapt_info,
) -> AdaptationAlgorithm:
    """Adapt the bracket width of the ensemble slice sampler.

    Parameters
    ----------
    logdensity_fn
        The log-density function of a single walker's position.
    direction
        The direction of each walker's slice, as for
        :func:`blackjax.mcmc.ensemble_slice.as_top_level_api`.
    initial_width
        The bracket width at the start of the adaptation (default 2.0).
    max_expansions, max_shrinkage
        Caps on stepping-out steps and shrinkage evaluations.
    tolerance
        The tolerance on the fraction of expansions around one half.
    patience
        The number of steps within the tolerance after which the width is fixed.
    adaptation_info_fn
        Function to select the adaptation info returned. See return_all_adapt_info
        and get_filter_adapt_info_fn in blackjax.adaptation.base.  By default all
        information is saved - this can result in excessive memory usage if the
        information is unused.

    Returns
    -------
    A function that runs the adaptation and returns an `AdaptationResult`
    object.

    """
    kernel = ensemble_slice.build_kernel(max_expansions, max_shrinkage)
    adapt_init, adapt_update = base(tolerance, patience)

    def one_step(carry, rng_key):
        state, adaptation_state = carry
        new_state, info = kernel(
            rng_key, state, logdensity_fn, direction, adaptation_state.width
        )
        new_adaptation_state = adapt_update(adaptation_state, info)
        return (
            (new_state, new_adaptation_state),
            adaptation_info_fn(new_state, info, new_adaptation_state),
        )

    def run(rng_key: PRNGKey, position: ArrayLikeTree, num_steps: int = 1000):
        state = ensemble_slice.init(position, logdensity_fn)
        keys = jax.random.split(rng_key, num_steps)
        (last_state, last_adaptation_state), info = jax.lax.scan(
            one_step, (state, adapt_init(initial_width)), keys
        )
        parameters = {"width": last_adaptation_state.width}
        return AdaptationResults(last_state, parameters), info

    return AdaptationAlgorithm(run)
