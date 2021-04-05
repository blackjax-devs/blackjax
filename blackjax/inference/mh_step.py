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
    log_weight: float
    is_diverging: bool


def proposal_generator(kinetic_energy: Callable, divergence_threshold: float):
    def generate(direction, state, trajectory):
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
        previous_state = jax.lax.cond(
            direction > 0,
            trajectory,
            lambda x: x.rightmost_state,
            trajectory,
            lambda x: x.leftmost_state,
        )

        energy = previous_state.potential_energy + kinetic_energy(
            previous_state.position, previous_state.momentum
        )
        new_energy = state.potential_energy + kinetic_energy(
            state.position, state.momentum
        )

        delta_energy = energy - new_energy
        delta_energy = jnp.where(jnp.isnan(delta_energy), -jnp.inf, delta_energy)
        is_diverging = jnp.abs(delta_energy) > divergence_threshold

        # The log-weight of the new proposal is equal to H(z) - H(z_new)?
        log_weight = delta_energy

        return Proposal(
            state,
            log_weight,
            is_diverging,
        )

    return generate


def progressive_uniform_sampling(rng_key, proposal, new_proposal):
    """Generate a new proposal.

    To avoid keeping the entire trajectory in memory, we only memorize the
    extreme points and the point that will currently be proposed as a sample.
    Progressive sampling updates this proposal as the trajectory is being
    built. This is scheme is equivalent to drawing a sample uniformly at random
    from the final trajectory.

    """
    p_accept = jax.scipy.special.expit(new_proposal.log_weight - proposal.log_weight)
    do_accept = jax.random.bernoulli(rng_key, p_accept)

    updated_proposal = Proposal(
        new_proposal.state,
        jnp.logaddexp(proposal.log_weight, new_proposal.log_weight),
        new_proposal.is_diverging,
    )

    return jax.lax.cond(
        do_accept, lambda _: updated_proposal, lambda _: proposal, operand=None
    )


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
        jnp.logaddexp(proposal.log_weight, new_proposal.log_weight),
        new_proposal.is_diverging,
    )

    return jax.lax.cond(
        do_accept, lambda _: updated_proposal, lambda _: proposal, operand=None
    )
