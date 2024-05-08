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
from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp

from blackjax.types import Array, PRNGKey

TrajectoryState = NamedTuple


class Proposal(NamedTuple):
    """Proposal for the next chain step for MCMC with trajectory building (e.g., NUTS).

    state:
        The trajectory state that corresponds to this proposal.
    energy:
        The total energy that corresponds to this proposal.
    weight:
        Weight of the proposal. It is equal to the logarithm of the sum of the canonical
        densities of each state :math:`e^{-H(z)}` along the trajectory.
    sum_log_p_accept:
        cumulated Metropolis-Hastings acceptance probability across entire trajectory.

    """

    state: TrajectoryState
    energy: float
    weight: float
    sum_log_p_accept: float


def safe_energy_diff(initial_energy: float, new_energy: float) -> float:
    delta_energy = initial_energy - new_energy
    delta_energy = jnp.where(jnp.isnan(delta_energy), -jnp.inf, delta_energy)
    return delta_energy


def proposal_generator(energy_fn: Callable) -> tuple[Callable, Callable]:
    """

    Parameters
    ----------
    energy_fn
        A function that computes the energy associated to a given state

    Returns
    -------
    Two functions, one to generate an initial proposal when no step has been taken,
    another to generate proposals after each step.
    """

    def new(state: TrajectoryState) -> Proposal:
        return Proposal(state, energy_fn(state), 0.0, -jnp.inf)

    def update(initial_energy: float, new_state: TrajectoryState) -> Proposal:
        """Generate a new proposal from a trajectory state.

        The trajectory state records information about the position in the state
        space and corresponding logdensity. A proposal also carries a
        weight that is equal to the difference between the current energy and
        the previous one. It thus carries information about the previous states
        as well as the current state.

        Parameters
        ----------
        initial_energy:
            The initial energy.
        new_state:
            The new state.

        Returns
        -------
        A proposal

        """
        new_energy = energy_fn(new_state)
        delta_energy = safe_energy_diff(initial_energy, new_energy)

        # The weight of the new proposal is equal to H0 - H(z_new)
        weight = delta_energy

        # Acceptance statistic min(e^{H0 - H(z_new)}, 1)
        sum_log_p_accept = jnp.minimum(delta_energy, 0.0)

        return Proposal(
            new_state,
            new_energy,
            weight,
            sum_log_p_accept,
        )

    return new, update


# --------------------------------------------------------------------
#                        PROGRESSIVE SAMPLING
#
# To avoid keeping the entire trajectory in memory, we only memorize the
# extreme points of the trajectory and the current sample proposal.
# Progressive sampling updates this proposal as the trajectory is being sampled
# or built.
# --------------------------------------------------------------------


def progressive_uniform_sampling(
    rng_key: PRNGKey, proposal: Proposal, new_proposal: Proposal
) -> Proposal:
    # Using expit to compute exp(w1) / (exp(w0) + exp(w1))
    p_accept = jax.scipy.special.expit(new_proposal.weight - proposal.weight)
    do_accept = jax.random.bernoulli(rng_key, p_accept)
    new_weight = jnp.logaddexp(proposal.weight, new_proposal.weight)
    new_sum_log_p_accept = jnp.logaddexp(
        proposal.sum_log_p_accept, new_proposal.sum_log_p_accept
    )

    return jax.lax.cond(
        do_accept,
        lambda _: Proposal(
            new_proposal.state,
            new_proposal.energy,
            new_weight,
            new_sum_log_p_accept,
        ),
        lambda _: Proposal(
            proposal.state,
            proposal.energy,
            new_weight,
            new_sum_log_p_accept,
        ),
        operand=None,
    )


def progressive_biased_sampling(
    rng_key: PRNGKey, proposal: Proposal, new_proposal: Proposal
) -> Proposal:
    """Baised proposal sampling :cite:p:`betancourt2017conceptual`.

    Unlike uniform sampling, biased sampling favors new proposals. It thus
    biases the transition away from the trajectory's initial state.

    """
    p_accept = jnp.clip(jnp.exp(new_proposal.weight - proposal.weight), max=1)
    do_accept = jax.random.bernoulli(rng_key, p_accept)
    new_weight = jnp.logaddexp(proposal.weight, new_proposal.weight)
    new_sum_log_p_accept = jnp.logaddexp(
        proposal.sum_log_p_accept, new_proposal.sum_log_p_accept
    )

    return jax.lax.cond(
        do_accept,
        lambda _: Proposal(
            new_proposal.state,
            new_proposal.energy,
            new_weight,
            new_sum_log_p_accept,
        ),
        lambda _: Proposal(
            proposal.state,
            proposal.energy,
            new_weight,
            new_sum_log_p_accept,
        ),
        operand=None,
    )


# --------------------------------------------------------------------
#                        STATIC SAMPLING
# --------------------------------------------------------------------


def compute_asymmetric_acceptance_ratio(transition_energy_fn: Callable) -> Callable:
    """Generate a meta function to compute the transition between two states.

    In particular, both states are used to compute the energies to consider in weighting
    the proposal, to account for asymmetries.

    Parameters
    ----------
    transition_energy_fn
        A function that computes the energy of a transition from an initial state
        to a new state, given some optional keyword arguments.

    Returns
    -------
    A functions to compute the acceptance ratio .
    """

    def compute_acceptance_ratio(
        initial_state: TrajectoryState,
        state: TrajectoryState,
        **energy_params,
    ) -> float:
        new_energy = transition_energy_fn(initial_state, state, **energy_params)
        prev_energy = transition_energy_fn(state, initial_state, **energy_params)
        log_p_accept = safe_energy_diff(prev_energy, new_energy)
        return log_p_accept

    return compute_acceptance_ratio


def static_binomial_sampling(
    rng_key: PRNGKey, log_p_accept: float, proposal, new_proposal
):
    """Accept or reject a proposal.

    In the static setting, the probability with which the new proposal is
    accepted is a function of the difference in energy between the previous and
    the current states. If the current energy is lower than the previous one
    then the new proposal is accepted with probability 1.

    """
    p_accept = jnp.clip(jnp.exp(log_p_accept), max=1)
    do_accept = jax.random.bernoulli(rng_key, p_accept)
    info = do_accept, p_accept, None
    return (
        jax.lax.cond(
            do_accept,
            lambda _: new_proposal,
            lambda _: proposal,
            operand=None,
        ),
        info,
    )


# --------------------------------------------------------------------
#                   NON-REVERSIVLE SLICE SAMPLING
# --------------------------------------------------------------------


def nonreversible_slice_sampling(
    slice: Array, delta_energy: float, proposal, new_proposal
):
    """Slice sampling for non-reversible Metropolis-Hasting update.

    Performs a non-reversible update of a uniform [0, 1] value
    for Metropolis-Hastings accept/reject decisions :cite:p:`neal2020non`, in addition
    to the accept/reject step of a current state and new proposal.

    """
    p_accept = jnp.clip(jnp.exp(delta_energy), max=1)
    do_accept = jnp.log(jnp.abs(slice)) <= delta_energy
    slice_next = slice * (jnp.exp(-delta_energy) * do_accept + (1 - do_accept))
    info = do_accept, p_accept, slice_next
    return (
        jax.lax.cond(
            do_accept,
            lambda _: new_proposal,
            lambda _: proposal,
            operand=None,
        ),
        info,
    )
