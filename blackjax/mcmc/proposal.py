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
import jax.numpy as jnp
import numpy as np

from blackjax.mcmc.integrators import IntegratorState


class Proposal(NamedTuple):
    """Proposal for the next chain step.

    state:
        The trajectory state that corresponds to this proposal.
    energy:
        The total energy that corresponds to this proposal.
    weight:
        Weight of the proposal. It is equal to the logarithm of the sum of the canonical
        densities of each state :math:`e^{-H(z)}` along the trajectory.
    sum_log_p_accept:
        cumulated Metropolis-Hastings acceptance probabilty across entire trajectory.

    """

    state: IntegratorState
    energy: float
    weight: float
    sum_log_p_accept: float


def proposal_generator(
    kinetic_energy: Callable, divergence_threshold: float
) -> Tuple[Callable, Callable]:
    def new(state: IntegratorState) -> Proposal:
        energy = state.potential_energy + kinetic_energy(state.momentum)
        return Proposal(state, energy, 0.0, -np.inf)

    def update(initial_energy: float, state: IntegratorState) -> Tuple[Proposal, bool]:
        """Generate a new proposal from a trajectory state.

        The trajectory state records information about the position in the state
        space and corresponding potential energy. A proposal also carries a
        weight that is equal to the difference between the current energy and
        the previous one. It thus carries information about the previous states
        as well as the current state.

        Parameters
        ----------
        initial_energy:
            The initial energy.
        state:
            The new state.

        """
        new_energy = state.potential_energy + kinetic_energy(state.momentum)

        delta_energy = initial_energy - new_energy
        delta_energy = jnp.where(jnp.isnan(delta_energy), -jnp.inf, delta_energy)
        is_transition_divergent = jnp.abs(delta_energy) > divergence_threshold

        # The weight of the new proposal is equal to H0 - H(z_new)
        weight = delta_energy

        # Acceptance statistic min(e^{H0 - H(z_new)}, 1)
        sum_log_p_accept = jnp.minimum(delta_energy, 0.0)

        return (
            Proposal(
                state,
                new_energy,
                weight,
                sum_log_p_accept,
            ),
            is_transition_divergent,
        )

    return new, update


# --------------------------------------------------------------------
#                        STATIC SAMPLING
# --------------------------------------------------------------------


def static_binomial_sampling(rng_key, proposal, new_proposal):
    """Accept or reject a proposal.

    In the static setting, the probability with which the new proposal is
    accepted is a function of the difference in energy between the previous and
    the current states. If the current energy is lower than the previous one
    then the new proposal is accepted with probability 1.

    """
    p_accept = jnp.clip(jnp.exp(new_proposal.weight), a_max=1)
    do_accept = jax.random.bernoulli(rng_key, p_accept)

    return jax.lax.cond(
        do_accept,
        lambda _: (new_proposal, do_accept, p_accept),
        lambda _: (proposal, do_accept, p_accept),
        operand=None,
    )


# --------------------------------------------------------------------
#                        PROGRESSIVE SAMPLING
#
# To avoid keeping the entire trajectory in memory, we only memorize the
# extreme points of the trajectory and the current sample proposal.
# Progressive sampling updates this proposal as the trajectory is being sampled
# or built.
# --------------------------------------------------------------------


def progressive_uniform_sampling(rng_key, proposal, new_proposal):
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


def progressive_biased_sampling(rng_key, proposal, new_proposal):
    """Baised proposal sampling.

    Unlike uniform sampling, biased sampling favors new proposals. It thus
    biases the transition away from the trajectory's initial state.

    References
    ----------
    .. [1]: Betancourt, Michael.
            "A conceptual introduction to Hamiltonian Monte Carlo."
            arXiv preprint arXiv:1701.02434 (2017).

    """
    p_accept = jnp.clip(jnp.exp(new_proposal.weight - proposal.weight), a_max=1)
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
#                   NON-REVERSIVLE SLICE SAMPLING
# --------------------------------------------------------------------


def nonreversible_slice_sampling(slice, proposal, new_proposal):
    """Slice sampling for non-reversible Metropolis-Hasting update.

    Performs a non-reversible update of a uniform [0, 1] value
    for Metropolis-Hastings accept/reject decisions [1]_, in addition
    to the accept/reject step of a current state and new proposal.

    References
    ----------
    .. [1]: Neal, R. M. (2020).
            "Non-reversibly updating a uniform [0, 1] value for Metropolis accept/reject decisions."
            arXiv preprint arXiv:2001.11950.

    """
    delta_energy = new_proposal.weight
    do_accept = jnp.log(jnp.abs(slice)) <= delta_energy
    return jax.lax.cond(
        do_accept,
        lambda _: (new_proposal, do_accept, slice * jnp.exp(-delta_energy)),
        lambda _: (proposal, do_accept, slice),
        operand=None,
    )
