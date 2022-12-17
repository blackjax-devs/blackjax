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
from jax import numpy as jnp

from blackjax.types import Array, PRNGKey, PyTree
from blackjax.util import generate_gaussian_noise

__all__ = ["RMHState", "RMHInfo", "init", "kernel"]


class RMHState(NamedTuple):
    """State of the RMH chain.

    position
        Current position of the chain.
    log_density
        Current value of the log-density

    """

    position: PyTree
    log_density: float


class RMHInfo(NamedTuple):
    """Additional information on the RMH chain.

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
    proposal: RMHState


def init(position: PyTree, logdensity_fn: Callable) -> RMHState:
    """Create a chain state from a position.

    Parameters:
    -----------
    position: PyTree
        The initial position of the chain
    logdensity_fn: Callable
        Log-probability density function of the distribution we wish to sample
        from.

    """
    return RMHState(position, logdensity_fn(position))


def kernel():
    """Build a Random Walk Rosenbluth-Metropolis-Hastings kernel with a gaussian
    proposal distribution.

    Returns
    -------
    A kernel that takes a rng_key and a Pytree that contains the current state
    of the chain and that returns a new state of the chain along with
    information about the transition.

    """

    def one_step(
        rng_key: PRNGKey, state: RMHState, logdensity_fn: Callable, sigma: Array
    ) -> Tuple[RMHState, RMHInfo]:

        move_proposal_generator = normal(sigma)

        def proposal_generator(key_proposal, position):
            move_proposal = move_proposal_generator(key_proposal, position)
            new_position = jax.tree_util.tree_map(jnp.add, position, move_proposal)
            return new_position

        kernel = rmh(logdensity_fn, proposal_generator)
        return kernel(rng_key, state)

    return one_step


# -----------------------------------------------------------------------------
# Rosenbluth-Metropolis-Hastings Step
#
# We keep this separate as the basis of a self-standing implementation of the
# RMH correction step that can be re-used across the library.
#
# (TODO) Separate the RMH step from the proposal.
# -----------------------------------------------------------------------------


def rmh(
    logdensity_fn: Callable,
    proposal_generator: Callable,
    proposal_logdensity_fn: Optional[Callable] = None,
):
    """Build a Rosenbluth-Metropolis-Hastings kernel.

    Parameters
    ----------
    logdensity_fn
        A function that returns the log-probability at a given position.
    proposal_generator
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
    if proposal_logdensity_fn is None:

        def acceptance_rate(state: RMHState, proposal: RMHState):
            return proposal.log_density - state.log_density

    else:

        def acceptance_rate(state: RMHState, proposal: RMHState):
            return (
                proposal.log_density
                + proposal_logdensity_fn(proposal.position, state.position)  # type: ignore
                - state.log_density
                - proposal_logdensity_fn(state.position, proposal.position)  # type: ignore
            )

    def kernel(rng_key: PRNGKey, state: RMHState) -> Tuple[RMHState, RMHInfo]:
        """Move the chain by one step using the Rosenbluth Metropolis Hastings
        algorithm.

        We temporarily assume that the proposal distribution is symmetric.

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
        key_proposal, key_accept = jax.random.split(rng_key, 2)

        new_position = proposal_generator(rng_key, state.position)
        new_log_density = logdensity_fn(new_position)
        new_state = RMHState(new_position, new_log_density)

        delta = acceptance_rate(state, new_state)
        delta = jnp.where(jnp.isnan(delta), -jnp.inf, delta)
        p_accept = jnp.clip(jnp.exp(delta), a_max=1.0)

        do_accept = jax.random.bernoulli(key_accept, p_accept)
        accept_state = (new_state, RMHInfo(p_accept, True, new_state))
        reject_state = (state, RMHInfo(p_accept, False, new_state))

        return jax.lax.cond(
            do_accept, lambda _: accept_state, lambda _: reject_state, operand=None
        )

    return kernel


def normal(sigma: Array) -> Callable:
    """Normal Random Walk proposal.

    Propose a new position such that its distance to the current position is
    normally distributed. Suitable for continuous variables.

    Parameter
    ---------
    sigma:
        vector or matrix that contains the standard deviation of the centered
        normal distribution from which we draw the move proposals.

    """
    if jnp.ndim(sigma) > 2:
        raise ValueError("sigma must be a vector or a matrix.")

    def propose(rng_key: PRNGKey, position: PyTree) -> PyTree:
        return generate_gaussian_noise(rng_key, position, sigma=sigma)

    return propose
