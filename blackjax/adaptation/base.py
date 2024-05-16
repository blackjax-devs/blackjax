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
from typing import NamedTuple, Set

import jax

from blackjax.types import ArrayTree


class AdaptationResults(NamedTuple):
    state: ArrayTree
    parameters: dict


class AdaptationInfo(NamedTuple):
    state: NamedTuple
    info: NamedTuple
    adaptation_state: NamedTuple


def return_all_adapt_info(state, info, adaptation_state):
    """Return fully populated AdaptationInfo.  Used for adaptation_info_fn
    parameters of the adaptation algorithms.
    """
    return AdaptationInfo(state, info, adaptation_state)


def get_filter_adapt_info_fn(
    state_keys: Set[str] = set(),
    info_keys: Set[str] = set(),
    adapt_state_keys: Set[str] = set(),
):
    """Generate a function to filter what is saved in AdaptationInfo.  Used
    for adptation_info_fn parameters of the adaptation algorithms.
    adaptation_info_fn=get_filter_adapt_info_fn() saves no auxiliary information
    """

    def filter_tuple(tup, key_set):
        mapfn = lambda key, val: None if key not in key_set else val
        return jax.tree.map(mapfn, type(tup)(*tup._fields), tup)

    def filter_fn(state, info, adaptation_state):
        sample_state = filter_tuple(state, state_keys)
        new_info = filter_tuple(info, info_keys)
        new_adapt_state = filter_tuple(adaptation_state, adapt_state_keys)

        return AdaptationInfo(sample_state, new_info, new_adapt_state)

    return filter_fn
