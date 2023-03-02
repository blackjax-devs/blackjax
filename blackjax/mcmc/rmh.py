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
"""Public API for Rosenbluth-Metropolis-Hastings kernels."""
from typing import Callable, NamedTuple, Optional, Tuple

import jax
import numpy as np

from blackjax.mcmc import proposal
from blackjax.types import PRNGKey, PyTree

__all__ = ["RWState", "RWInfo", "init"]


class RWState(NamedTuple):
    """State of the RW chain.

    position
        Current position of the chain.
    log_density
        Current value of the log-density

    """

    position: PyTree
    logdensity: float


class RWInfo(NamedTuple):
    """Additional information on the RW chain.

    This additional information can be used for debugging or computing
    diagnostics.

    acceptance_rate
        The acceptance probability of the transition, linked to the energy
        difference between the original and the proposed states.
    is_accepted
        Whether the proposed position was accepted or the original position
        was returned.
    proposal
        The state proposed by the proposal.

    """

    acceptance_rate: float
    is_accepted: bool
    proposal: RWState


def init(position: PyTree, logdensity_fn: Callable) -> RWState:
    """Create a chain state from a position.

    Parameters
    ----------
    position: PyTree
        The initial position of the chain
    logdensity_fn: Callable
        Log-probability density function of the distribution we wish to sample
        from.

    """
    return RWState(position, logdensity_fn(position))


def rmh_proposal(
    logdensity_fn,
    transition_distribution,
    init_proposal,
    generate_proposal,
    sample_proposal: Callable = proposal.static_binomial_sampling,
) -> Callable:
    def build_trajectory(rng_key, initial_state: RWState) -> RWState:
        position, logdensity = initial_state
        new_position = transition_distribution(rng_key, position)
        return RWState(new_position, logdensity_fn(new_position))

    def generate(rng_key, state: RWState) -> Tuple[RWState, bool, float]:
        key_proposal, key_accept = jax.random.split(rng_key, 2)
        end_state = build_trajectory(key_proposal, state)
        previous_proposal = init_proposal(state)
        new_proposal, _ = generate_proposal(previous_proposal.energy, end_state)
        sampled_proposal, do_accept, p_accept = sample_proposal(
            key_accept, previous_proposal, new_proposal
        )
        return sampled_proposal, do_accept, p_accept

    return generate


def rmh(
    logdensity_fn: Callable,
    transition_generator: Callable,
    proposal_logdensity_fn: Optional[Callable] = None,
):
    """Build a Rosenbluth-Metropolis-Hastings kernel.

    Parameters
    ----------
    logdensity_fn
        A function that returns the log-probability at a given position.
    transition_generator
        A function that generates a candidate transition for the markov chain.
    proposal_logdensity_fn:
        For non-symmetric proposals, a function that returns the log-density
        to obtain a given proposal knowing the current state. If it is not
        provided we assume the proposal is symmetric.

    Returns
    -------
    A kernel that takes a rng_key and a Pytree that contains the current state
    of the chain and that returns a new state of the chain along with
    information about the transition.

    """

    def energy(state: RWState):
        return -state.logdensity

    if proposal_logdensity_fn is None:
        init_proposal, generate_proposal = proposal.proposal_generator(energy, np.inf)
    else:
        init_proposal, generate_proposal = proposal.asymmetric_proposal_generator(
            lambda state: -state.logdensity, proposal_logdensity_fn, np.inf
        )

    proposal_generator = rmh_proposal(
        logdensity_fn, transition_generator, init_proposal, generate_proposal
    )

    def kernel(rng_key: PRNGKey, state: RWState) -> Tuple[RWState, RWInfo]:
        """Move the chain by one step using the Rosenbluth Metropolis Hastings
        algorithm.

        Parameters
        ----------
        rng_key:
           The pseudo-random number generator key used to generate random
           numbers.
        state:
            The current state of the chain.

        Returns
        -------
        The next state of the chain and additional information about the current
        step.

        """
        sampled_proposal, do_accept, p_accept = proposal_generator(rng_key, state)
        new_state = sampled_proposal.state
        return new_state, RWInfo(p_accept, do_accept, new_state)

    return kernel
