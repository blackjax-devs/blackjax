"""Public API for General Cross-Chain Adaptations"""

from typing import Callable, NamedTuple, Tuple

import jax

from blackjax.types import PRNGKey, PyTree


class ChainState(NamedTuple):
    """State of all chains in cross-chain adaptation.

    We store the current iteration for adaptive mechanisms that need this information.
    """

    states: NamedTuple
    current_iter: int


def cross_chain(
    kernel_factory: Callable,
    adaptation_fn: Callable,
):
    """Cross-chain adaptation scheme for general adaptation mechanisms.

    Uses a, preferably large, group of chains proposing new values independently
    but using the same kernel with fixed parameters set using a parameter generating
    mechanism that learns new parameters with information from the last iteration
    of each chain grouped together and taken as independent samples from the target
    density. Samples generated during this warm-up phase for each chain create a
    Markov Chain which is NOT ergodic and should NOT be used as working samples.
    Instead, their utility is in training the parameters of the Markov kernel that
    then should be fixed and used to generate working samples.

    Parameters
    ----------
    kernel_factory
        Function that takes as input the parameters that need to be learned and
        outputs a batched kernel that generates new samples for each chain.
    adaptation_fn
        Function that takes as input the last state of all chains stacked on the
        first dimension, the current iteration of the warm-up phase, and other
        optional inputs and outputs updated parameters.

    Returns
    -------
    init
        Function that initializes the warmup.
    update
        Function that moves the warmup one step.
    """

    def init(initial_states: NamedTuple) -> ChainState:
        return ChainState(initial_states, 0)

    def update(
        rng_key: PRNGKey, state: ChainState, *param
    ) -> Tuple[ChainState, PyTree, NamedTuple]:
        parameters = adaptation_fn(state.states, state.current_iter, *param)
        kernel = kernel_factory(*parameters)
        num_chain = jax.tree_util.tree_leaves(
            jax.tree_map(lambda s: s.shape[0], state.states)
        )[0]
        keys = jax.random.split(rng_key, num_chain)
        new_states, infos = kernel(keys, state.states)
        return ChainState(new_states, state.current_iter + 1), parameters, infos

    return init, update
