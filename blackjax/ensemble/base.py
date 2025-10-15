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
"""Base classes and utilities for ensemble sampling methods."""
from typing import NamedTuple, Optional

from blackjax.types import Array, ArrayTree

__all__ = ["EnsembleState", "EnsembleInfo"]


class EnsembleState(NamedTuple):
    """State of an ensemble sampler.

    coords
        An array or PyTree of arrays of shape `(n_walkers, ...)` that
        stores the current position of the walkers.
    log_probs
        An array of shape `(n_walkers,)` that stores the log-probability of
        each walker.
    blobs
        An optional PyTree that stores metadata returned by the log-probability
        function.
    """

    coords: ArrayTree
    log_probs: Array
    blobs: Optional[ArrayTree] = None


class EnsembleInfo(NamedTuple):
    """Additional information on the ensemble transition.

    acceptance_rate
        The acceptance rate of the ensemble.
    is_accepted
        A boolean array of shape `(n_walkers,)` indicating whether each walker's
        proposal was accepted.
    """

    acceptance_rate: float
    is_accepted: Array
