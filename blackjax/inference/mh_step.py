"""Metropolis-Hasting Step"""
from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp
from blackjax.inference.integrators import IntegratorState


class Proposal(NamedTuple):
    """Proposal for the next chain step.

    state:
        The trajectory state corresponding to this proposal.
    weight:
        The logarithm of the sum of the canonical densities of each state
        :math:`e^{-H(z)}` along the trajectory.
    is_diverging
        Whether a divergence was observed when making one step.
    """

    state: IntegratorState
    new_energy: float
    log_weight: float


def proposal_generator(kinetic_energy: Callable, divergence_threshold: float):
    def generate(
        current_state: IntegratorState, new_state: IntegratorState
    ) -> Proposal:
        """Generate a new proposal from a trajectory state.

        Note
        ----
        We can optimize things a bit by sacrificing in memory and code
        complexity: the kinetic energy and the previous states can be carried
        around instead of being recomputed each time.

        Parameters
        ----------
        direction:
            The direction in which the trajectory was integrated to obtain the
            new state.
        state:
            The new state.
        trajectory:
            The trajectory before the state was added.

        """
        energy = current_state.potential_energy + kinetic_energy(
            current_state.position, current_state.momentum
        )
        new_energy = new_state.potential_energy + kinetic_energy(
            new_state.position, new_state.momentum
        )

        delta_energy = energy - new_energy
        delta_energy = jnp.where(jnp.isnan(delta_energy), -jnp.inf, delta_energy)
        is_transition_divergent = jnp.abs(delta_energy) > divergence_threshold

        # The log-weight of the new proposal is equal to H(z) - H(z_new)?
        log_weight = delta_energy

        return (
            Proposal(
                current_state,
                new_energy,
                log_weight,
            ),
            is_transition_divergent,
        )

    return generate


def uniform_sampling(kinetic_energy, divergence_threshold):
    transition = proposal_generator(kinetic_energy, divergence_threshold)

    def sample(rng_key, state, new_state):
        transition_info, _ = transition(state, new_state)

        p_accept = jnp.clip(jnp.exp(transition_info.log_weight), a_max=1)
        do_accept = jax.random.bernoulli(rng_key, p_accept)

        return jax.lax.cond(
            do_accept,
            lambda _: (new_state, do_accept, p_accept, transition_info),
            lambda _: (state, do_accept, p_accept, transition_info),
            operand=None,
        )

    return sample


def progressive_uniform_sampling(kinetic_energy, divergence_threshold):
    """Generate a new proposal.

    To avoid keeping the entire trajectory in memory, we only memorize the
    extreme points and the point that will currently be proposed as a sample.
    Progressive sampling updates this proposal as the trajectory is being
    built. This is scheme is equivalent to drawing a sample uniformly at random
    from the final trajectory.

    """
    transition = proposal_generator(kinetic_energy, divergence_threshold)

    def sample(rng_key, state, proposal, new_state):
        new_proposal, is_diverging = transition(state, new_state)

        p_accept = jax.scipy.special.expit(
            new_proposal.log_weight - proposal.log_weight
        )
        do_accept = jax.random.bernoulli(rng_key, p_accept)

        updated_proposal = Proposal(
            new_proposal.state,
            new_proposal.new_energy,
            jnp.logaddexp(proposal.log_weight, new_proposal.log_weight),
        )

        return jax.lax.cond(
            do_accept,
            lambda _: (updated_proposal, is_diverging),
            lambda _: (proposal, is_diverging),
            operand=None,
        )

    return sample


def progressive_biased_sampling(rng_key, proposal, new_proposal):
    """Generate a new proposal.

    To avoid keeping the entire trajectory in memory, we only memorize the
    extreme points and the point that will currently be proposed as a sample.
    Progressive sampling updates this proposal as the trajectory is being
    built.

    Unlike uniform sampling, biased sampling favors new proposals. It thus
    biases the transition away from the trajectory's initial state.

    """
    p_accept = jnp.exp(new_proposal.log_weight - proposal.log_weight)
    p_accept = jnp.clip(p_accept, a_max=1.0)
    do_accept = jax.random.bernoulli(rng_key, p_accept)

    updated_proposal = Proposal(
        new_proposal.state,
        new_proposal.new_energy,
        jnp.logaddexp(proposal.log_weight, new_proposal.log_weight),
    )

    return jax.lax.cond(
        do_accept, lambda _: updated_proposal, lambda _: proposal, operand=None
    )
