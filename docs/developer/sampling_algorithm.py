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
from typing import Callable, NamedTuple, Tuple

import jax

# import basic compoments that are already implemented
# or that you have implemented with a general structure
# for example, if you do a Metropolis-Hastings accept/reject step:
import blackjax.mcmc.proposal as proposal
from blackjax.base import MCMCSamplingAlgorithm
from blackjax.types import PRNGKey, PyTree

__all__ = [
    "SamplingAlgoState",
    "SamplingAlgoInfo",
    "init",
    "build_kernel",
    "sampling_algorithm",
]


class SamplingAlgoState(NamedTuple):
    """State of the sampling algorithm.

    Give an overview of the variables needed at each iteration of the model.
    """

    ...


class SamplingAlgoInfo(NamedTuple):
    """Additional information on the algorithm transition.

    Given an overview of the collected values at each iteration of the model.
    """

    ...


def init(position: PyTree, logdensity_fn: Callable, *args, **kwargs):
    # build an inital state
    state = SamplingAlgoState(...)
    return state


def build_kernel(*args, **kwargs):
    """Build a HMC kernel.

    Parameters
    ----------
    List and describe its parameters.

    Returns
    -------
    Describe the kernel that is returned.
    """

    def kernel(
        rng_key: PRNGKey,
        state: SamplingAlgoState,
        logdensity_fn: Callable,
        *args,
        **kwargs,
    ) -> Tuple[SamplingAlgoState, SamplingAlgoInfo]:
        """Generate a new sample with the sampling kernel."""

        # build everything you'll need
        proposal_generator = sampling_algorithm_proposal(...)

        # generate pseudorandom keys
        key_other, key_proposal = jax.random.split(rng_key, 2)

        # generate the proposal with all its parts
        proposal, info = proposal_generator(key_proposal, ...)
        proposal = SamplingAlgoState(...)

        return proposal, info

    return kernel


class sampling_algorithm:
    """Implements the (basic) user interface for the sampling kernel.

    Describe in detail the inner mechanism of the algorithm and its use.

    Example
    -------
    Illustrate the use of the algorithm.

    Parameters
    ----------
    List and describe its parameters.

    Returns
    -------
    A ``MCMCSamplingAlgorithm``.
    """

    init = staticmethod(init)
    build_kernel = staticmethod(build_kernel)

    def __new__(  # type: ignore[misc]
        cls,
        logdensity_fn: Callable,
        *args,
        **kwargs,
    ) -> MCMCSamplingAlgorithm:
        kernel = cls.build_kernel(...)

        def init_fn(position: PyTree):
            return cls.init(position, logdensity_fn, ...)

        def step_fn(rng_key: PRNGKey, state):
            return kernel(
                rng_key,
                state,
                logdensity_fn,
                ...,
            )

        return MCMCSamplingAlgorithm(init_fn, step_fn)


# and other functions that help make `init` and/or `build_kernel` easier to read and understand
def sampling_algorithm_proposal(*args, **kwags) -> Callable:
    """Title

    Description

    Parameters
    ----------
    List and describe its parameters.

    Returns
    -------
    Describe what is returned.
    """
    ...

    def generate(*args, **kwargs):
        """Generate a new chain state."""
        sampled_state, info = ...

        return sampled_state, info

    return generate