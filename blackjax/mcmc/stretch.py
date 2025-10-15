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
"""Public API for the Stretch Move ensemble sampler."""
from typing import Callable

from blackjax.base import SamplingAlgorithm
from blackjax.mcmc.ensemble import as_top_level_api as ensemble_api
from blackjax.mcmc.ensemble import build_kernel as ensemble_build_kernel
from blackjax.mcmc.ensemble import init as ensemble_init
from blackjax.mcmc.ensemble import stretch_move

__all__ = ["as_top_level_api", "init", "build_kernel"]


def as_top_level_api(
    logdensity_fn: Callable,
    a: float = 2.0,
    has_blobs: bool = False,
    randomize_split: bool = True,
    live_dangerously: bool = False,
) -> SamplingAlgorithm:
    """A user-facing API for the stretch move algorithm.

    Parameters
    ----------
    logdensity_fn
        A function that returns the log density of the model at a given position.
    a
        The stretch parameter. Must be > 1. Default is 2.0.
    has_blobs
        Whether the logdensity function returns additional information (blobs).
    randomize_split
        If True, randomly shuffle walker indices before splitting into red/blue sets
        each iteration. This improves mixing and matches emcee's default behavior.
    live_dangerously
        If False (default), warns when n_walkers < 2*ndim. Set to True to suppress.

    Returns
    -------
    A `SamplingAlgorithm` that can be used to sample from the target distribution.
    """
    move = lambda key, w, c: stretch_move(key, w, c, a)
    return ensemble_api(
        logdensity_fn,
        move,
        has_blobs,
        randomize_split=randomize_split,
        live_dangerously=live_dangerously,
    )


def init(
    position, logdensity_fn, has_blobs: bool = False, live_dangerously: bool = False
):
    """Initialize the stretch move algorithm.

    Parameters
    ----------
    position
        Initial positions for all walkers, with shape (n_walkers, ...).
    logdensity_fn
        The log-density function to evaluate.
    has_blobs
        Whether the log-density function returns additional metadata (blobs).
    live_dangerously
        If False (default), warns when n_walkers < 2*ndim. Set to True to suppress.
    """
    return ensemble_init(position, logdensity_fn, has_blobs, live_dangerously)


def build_kernel(move_fn=None, a: float = 2.0, randomize_split: bool = True):
    """Build the stretch move kernel.

    Parameters
    ----------
    move_fn
        Optional custom move function. If None, uses stretch_move with parameter a.
    a
        The stretch parameter. Must be > 1. Default is 2.0.
    randomize_split
        If True, randomly shuffle walker indices before splitting into red/blue sets
        each iteration. This improves mixing and matches emcee's default behavior.
    """
    if move_fn is None:
        move_fn = lambda key, w, c: stretch_move(key, w, c, a)
    return ensemble_build_kernel(move_fn, randomize_split=randomize_split)
