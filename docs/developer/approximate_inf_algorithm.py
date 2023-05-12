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
from optax import GradientTransformation

# import basic compoments that are already implemented
# or that you have implemented with a general structure
from blackjax.base import VIAlgorithm
from blackjax.types import PRNGKey, PyTree

__all__ = [
    "ApproxInfState",
    "ApproxInfInfo",
    "init",
    "sample",
    "step",
    "approx_inf_algorithm",
]


class ApproxInfState(NamedTuple):
    """State of your approximate inference algorithm.

    Give an overview of the variables needed at each step and for sampling.
    """

    ...


class ApproxInfInfo(NamedTuple):
    """Additional information on your algorithm transition.

    Give an overview of the collected values at each step of the approximation.
    """

    ...


def init(position: PyTree, logdensity_fn: Callable, *args, **kwargs):
    # build an inital state
    state = ApproxInfState(...)
    return state


def step(
    rng_key: PRNGKey,
    state: ApproxInfInfo,
    logdensity_fn: Callable,
    optimizer: GradientTransformation,
    *args,
    **kwargs,
) -> Tuple[ApproxInfState, ApproxInfInfo]:
    """Approximate the target density using your approximation.

    Parameters
    ----------
    List and describe its parameters.
    """
    # extract the previous parameters from the state
    params = ...
    # generate pseudorandom keys
    key_other, key_update = jax.random.split(rng_key, 2)
    # update the parameters and build a new state
    new_state = ApproxInfState(...)
    info = ApproxInfInfo(...)

    return new_state, info


def sample(rng_key: PRNGKey, state: ApproxInfState, num_samples: int = 1):
    """Sample from your approximation."""
    # the sample should be a PyTree of the same structure as the `position` in the init function
    samples = ...
    return samples


class approx_inf_algorithm:
    """Implements the (basic) user interface for your approximate inference method.

    Describe in detail the inner mechanism of the method and its use.

    Example
    -------
    Illustrate the use of the algorithm.

    Parameters
    ----------
    List and describe its parameters.

    Returns
    -------
    A ``VIAlgorithm``.
    """

    init = staticmethod(init)
    step = staticmethod(step)
    sample = staticmethod(sample)

    def __new__(  # type: ignore[misc]
        cls,
        logdensity_fn: Callable,
        optimizer: GradientTransformation,
        *args,
        **kwargs,
    ) -> VIAlgorithm:
        def init_fn(position: PyTree):
            return cls.init(position, optimizer, ...)

        def step_fn(rng_key: PRNGKey, state):
            return cls.step(
                rng_key,
                state,
                logdensity_fn,
                optimizer,
                ...,
            )

        def sample_fn(rng_key: PRNGKey, state, num_samples):
            return cls.sample(rng_key, state, num_samples)

        return VIAlgorithm(init_fn, step_fn, sample_fn)


# other functions that help make `init`,` `step` and/or `sample` easier to read and understand
