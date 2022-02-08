from typing import NamedTuple, Tuple

from typing_extensions import Protocol

from .types import PRNGKey, PyTree

Position = PyTree
State = NamedTuple
Info = NamedTuple


class InitFn(Protocol):
    """A `Callable` used to initialize the kernel state.

    Sampling algorithms often need to carry over some informations between
    steps, often to avoid computing the same quantity twice. Therefore the
    kernels do not operate on the chain positions themselves, but on states that
    contain this position and other information.

    The `InitFn` returns the state corresponding to a chain position. This state
    can then be passed to the `update` function of the `SamplingAlgorithm`.

    """

    def __call__(self, position: Position) -> State:
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
    about the transtion.

    Update functions is a simplified yet universal interface with every sampling
    algorithm. In essence, what all these algorithms do is take a rng state, a
    chain state (possibly a batch of data) and return a new state and some
    information about the transition.

    """

    def __call__(self, rng_key: PRNGKey, state: State) -> Tuple[State, Info]:
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
    """A pair of functions that implement a sampling algorithm.

    Blackjax sampling algorithms are implemented as a pair of pure functions: a
    kernel, that takes a new samples starting from the current state, and an
    initialization function that creates a kernel state from a chain position.

    As they represent Markov kernels, the kernel functions are pure functions
    and do not have internal state. To save computation time they also operate
    on states which contain the chain state and additional information that
    needs to be carried over for the next step.

    Attributes
    ---------
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


class RunFn(Protocol):
    def __call__(self, rng_key: PRNGKey, position: PyTree):
        """Run the compiled algorithm."""


class AdaptationAlgorithm(NamedTuple):
    """A function that implements an adaptation algorithm."""

    run: RunFn
