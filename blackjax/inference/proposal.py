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
        Weight of the proposal. It is equal to the logarithm of the sum of the canonical
        densities of each state :math:`e^{-H(z)}` along the trajectory.
    """

    state: IntegratorState
    energy: float
    weight: float


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

        The trajectory state records information about the position in the state
        space and corresponding potential energy. A proposal also carries a
        weight that is equal to the difference between the current energy and
        the previous one. It thus carries information about the previous state
        as well as the current state.

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

        # The weight of the new proposal is equal to H(z) - H(z_new)
        weight = delta_energy

        return (
            Proposal(
                state,
                new_energy,
                weight,
            ),
            is_transition_divergent,
        )

    return init, update


# --------------------------------------------------------------------
#                        STATIC SAMPLING
# --------------------------------------------------------------------


def static_binomial_sampling(rng_key, proposal, new_proposal):
    """Accept or reject a proposal based on its weight.

    In the static setting, the `log_weight` of the proposal will be equal to the
    difference of energy between the beginning and the end of the trajectory. It
    is implemented this way to keep a consistent API with progressive sampling.

    """
    p_accept = jnp.clip(jnp.exp(proposal.weight), a_max=1)
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
    p_accept = jax.scipy.special.expit(new_proposal.weight - proposal.weight)
    do_accept = jax.random.bernoulli(rng_key, p_accept)

    updated_proposal = Proposal(
        new_proposal.state,
        new_proposal.energy,
        jnp.logaddexp(proposal.weight, new_proposal.weight),
    )

    return jax.lax.cond(
        do_accept, lambda _: updated_proposal, lambda _: proposal, operand=None
    )


def progressive_biased_sampling(rng_key, proposal, new_proposal):
    """Baised proposal sampling.

    Unlike uniform sampling, biased sampling favors new proposals. It thus
    biases the transition away from the trajectory's initial state.

    """
    p_accept = jnp.exp(new_proposal.weight - proposal.weight)
    p_accept = jnp.clip(p_accept, a_max=1.0)
    do_accept = jax.random.bernoulli(rng_key, p_accept)

    updated_proposal = Proposal(
        new_proposal.state,
        new_proposal.energy,
        jnp.logaddexp(proposal.weight, new_proposal.weight),
    )

    return jax.lax.cond(
        do_accept, lambda _: updated_proposal, lambda _: proposal, operand=None
    )
