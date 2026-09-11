# Copyright 2020- The Blackjax Authors.
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
from typing import Callable, NamedTuple, TypeAlias

from typing_extensions import Protocol

from .types import ArrayLikeTree, PRNGKey

Position: TypeAlias = ArrayLikeTree
State = NamedTuple
Info = NamedTuple


class InitFn(Protocol):
    """A `Callable` used to initialize the kernel state.

    Sampling algorithms often need to carry over some information between
    steps, often to avoid computing the same quantity twice. Therefore the
    kernels do not operate on the chain positions themselves, but on states that
    contain this position and other information.

    The `InitFn` returns the state corresponding to a chain position. This state
    can then be passed to the `update` function of the `SamplingAlgorithm`.

    """

    def __call__(self, position: Position, rng_key: PRNGKey | None) -> State:
        """Initialize the algorithm's state.

        Parameters
        ----------
        position
           A chain position.

        Returns
        -------
        The kernel state that corresponds to the position.

        """


class UpdateFn(Protocol):
    """A transition kernel used as the `update` of a `SamplingAlgorithms`.

    Kernels are pure functions and are idempotent. They necessarily take a
    random state `rng_key` and the current kernel state (which contains the
    current position) as parameters, return a new state and some information
    about the transition.

    Update functions is a simplified yet universal interface with every sampling
    algorithm. In essence, what all these algorithms do is take a rng state, a
    chain state (possibly a batch of data) and return a new state and some
    information about the transition.

    """

    def __call__(self, rng_key: PRNGKey, state: State) -> tuple[State, Info]:
        """Update the current state using the sampling algorithm.

        Parameters
        ----------
        rng_key:
            The random state used by JAX's random numbers generator.
        state:
            The current kernel state. The kernel state contains the current
            chain position as well as other information the kernel needs to
            carry over from the previous step.

        Returns
        -------
        A new state, as well as a NamedTuple that contains extra information
        about the transition that does not need to be carried over to the next
        step.

        """


class SamplingAlgorithm(NamedTuple):
    """A pair of functions that represents a MCMC sampling algorithm.

    Blackjax sampling algorithms are implemented as a pair of pure functions: a
    kernel, that generates a new sample from the current state, and an
    initialization function that creates a kernel state from a chain position.

    As they represent Markov kernels, the kernel functions are pure functions
    and do not have internal state. To save computation time they also operate
    on states which contain the chain state and additional information that
    needs to be carried over for the next step.

    init:
        A pure function which when called with the initial position and the
        target density probability function will return the kernel's initial
        state.

    step:
        A pure function that takes a rng key, a state and possibly some
        parameters and returns a new state and some information about the
        transition.

    """

    init: InitFn
    step: UpdateFn


class VIAlgorithm(NamedTuple):
    """A trio of functions that represents a Variational Inference algorithm.

    Blackjax variational inference algorithms are implemented as three pure
    functions: an initializer, an update step, and a sampler.

    init
        A pure function which initializes the VI state from an initial position.
        For iterative algorithms (e.g. mean-field VI) this sets up the optimizer
        state. For one-shot algorithms (e.g. Pathfinder) this builds the full
        approximation.
    step
        A pure function that advances the VI approximation by one step, taking
        a rng key and the current state and returning an updated state and
        diagnostic info. For one-shot algorithms this is a no-op.
    sample
        A pure function that draws samples from the current approximation.

    """

    init: Callable
    step: Callable
    sample: Callable


class RunFn(Protocol):
    """A `Callable` used to run the adaptation procedure."""

    def __call__(self, rng_key: PRNGKey, position: ArrayLikeTree):
        """Run the compiled algorithm."""


class AdaptationAlgorithm(NamedTuple):
    """A function that implements an adaptation algorithm."""

    run: RunFn


def build_sampling_algorithm(
    kernel: Callable,
    init_state: Callable,
    logdensity_fn: Callable,
    init_args: tuple = (),
    kernel_args: tuple = (),
    *,
    pass_rng_key_to_init: bool = False,
) -> SamplingAlgorithm:
    """Build a ``SamplingAlgorithm`` from standard components.

    Most BlackJAX MCMC algorithms follow the same boilerplate in their
    ``as_top_level_api`` functions: create an ``init_fn`` that delegates to the
    module-level ``init`` and a ``step_fn`` that calls the built kernel with
    fixed parameters.  This helper eliminates that repetition.

    Parameters
    ----------
    kernel
        A kernel function with signature
        ``(rng_key, state, logdensity_fn, *kernel_args) -> (state, info)``.
    init_state
        An initialization function with signature
        ``(position, logdensity_fn, *init_args) -> state`` (or
        ``(position, logdensity_fn, *init_args, rng_key) -> state`` when
        ``pass_rng_key_to_init`` is True).
    logdensity_fn
        The log-density function of the target distribution.  Passed to both
        ``init_state`` and ``kernel``.
    init_args
        Extra positional arguments forwarded to ``init_state`` after
        ``logdensity_fn`` (e.g. extra state needed for initialization).
    kernel_args
        Extra positional arguments forwarded to ``kernel`` after
        ``logdensity_fn`` on every step (e.g. ``(step_size, metric)``).
    pass_rng_key_to_init
        If True, ``rng_key`` is appended to the ``init_state`` call
        (for algorithms like mclmc/ghmc that need it for initialization).

    Returns
    -------
    A ``SamplingAlgorithm``.
    """

    def init_fn(position: Position, rng_key: PRNGKey | None = None):
        if pass_rng_key_to_init:
            return init_state(position, logdensity_fn, *init_args, rng_key)
        return init_state(position, logdensity_fn, *init_args)

    def step_fn(rng_key: PRNGKey, state: State) -> tuple[State, Info]:
        return kernel(rng_key, state, logdensity_fn, *kernel_args)

    return SamplingAlgorithm(init_fn, step_fn)
