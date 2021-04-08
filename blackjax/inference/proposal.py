from typing import Callable, NamedTuple, Tuple

import jax
import jax.numpy as jnp

from blackjax.inference.integrators import IntegratorState


class Proposal(NamedTuple):
    """Proposal for the next chain step.

    state:
        The trajectory state corresponding to this proposal.
    energy:
        The potential energy corresponding to the state.
    weight:
        Weight of the proposal. The logarithm of the sum of the canonical
        densities of each state :math:`e^{-H(z)}` along the trajectory.
    is_diverging
        Whether a divergence was observed when making one step.
    """

    state: IntegratorState
    energy: float
    log_weight: float


def proposal_generator(
    kinetic_energy: Callable, divergence_threshold: float
) -> Tuple[Callable, Callable]:
    def init(state: IntegratorState) -> Proposal:
        energy = state.potential_energy + kinetic_energy(state.position, state.momentum)
        return Proposal(state, energy, 0.0)

    def update(
        previous_proposal: Proposal, state: IntegratorState
    ) -> Tuple[Proposal, bool]:
        """Generate a new proposal from a trajectory state.

        Parameters
        ----------
        previous_proposal:
            The previous proposal.
        state:
            The new state.

        """
        energy = previous_proposal.energy
        new_energy = state.potential_energy + kinetic_energy(
            state.position, state.momentum
        )

        delta_energy = energy - new_energy
        delta_energy = jnp.where(jnp.isnan(delta_energy), -jnp.inf, delta_energy)
        is_transition_divergent = jnp.abs(delta_energy) > divergence_threshold

        # The log-weight of the new proposal is equal to H(z) - H(z_new)?
        log_weight = delta_energy

        return (
            Proposal(
                state,
                new_energy,
                log_weight,
            ),
            is_transition_divergent,
        )

    return init, update


# --------------------------------------------------------------------
#                        STATIC SAMPLING
# --------------------------------------------------------------------


def static_binomial_sampling(rng_key, proposal, new_proposal):
    """Choose a state between two states."""
    p_accept = jnp.clip(jnp.exp(proposal.log_weight), a_max=1)
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
    """Uniform proposal sampling."""
    p_accept = jax.scipy.special.expit(new_proposal.log_weight - proposal.log_weight)
    do_accept = jax.random.bernoulli(rng_key, p_accept)

    updated_proposal = Proposal(
        new_proposal.state,
        new_proposal.energy,
        jnp.logaddexp(proposal.log_weight, new_proposal.log_weight),
    )

    return jax.lax.cond(
        do_accept, lambda _: updated_proposal, lambda _: proposal, operand=None
    )


def progressive_biased_sampling(rng_key, proposal, new_proposal):
    """Baised proposal sampling.

    Unlike uniform sampling, biased sampling favors new proposals. It thus
    biases the transition away from the trajectory's initial state.

    """
    p_accept = jnp.exp(new_proposal.log_weight - proposal.log_weight)
    p_accept = jnp.clip(p_accept, a_max=1.0)
    do_accept = jax.random.bernoulli(rng_key, p_accept)

    updated_proposal = Proposal(
        new_proposal.state,
        new_proposal.energy,
        jnp.logaddexp(proposal.log_weight, new_proposal.log_weight),
    )

    return jax.lax.cond(
        do_accept, lambda _: updated_proposal, lambda _: proposal, operand=None
    )
