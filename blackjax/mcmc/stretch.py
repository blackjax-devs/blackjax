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
from blackjax.mcmc.ensemble import as_top_level_api as ensemble_api, stretch_move, init as ensemble_init, build_kernel as ensemble_build_kernel

__all__ = ["as_top_level_api", "init", "build_kernel"]


def as_top_level_api(logdensity_fn: Callable, a: float = 2.0, has_blobs: bool = False) -> SamplingAlgorithm:
    """A user-facing API for the stretch move algorithm.
    
    Parameters
    ----------
    logdensity_fn
        A function that returns the log density of the model at a given position.
    a
        The stretch parameter. Must be > 1. Default is 2.0.
    has_blobs
        Whether the logdensity function returns additional information (blobs).
        
    Returns
    -------
    A `SamplingAlgorithm` that can be used to sample from the target distribution.
    """
    move = lambda key, w, c: stretch_move(key, w, c, a)
    return ensemble_api(logdensity_fn, move, has_blobs)


def init(position, logdensity_fn, has_blobs: bool = False):
    """Initialize the stretch move algorithm."""
    return ensemble_init(position, logdensity_fn, has_blobs)


def build_kernel(move_fn=None, a: float = 2.0):
    """Build the stretch move kernel."""
    if move_fn is None:
        move_fn = lambda key, w, c: stretch_move(key, w, c, a)
    return ensemble_build_kernel(move_fn)