"""Proposals in the HMC family.

Proposals take the current state of the chain, transform it to another state
that is returned to the base kernel. Proposals can differ from one another on
two aspects: the way the step size is chosen and the way the number of
integration steps is chose.

The standard HMC algorithm integrates the same number of times with the same
step size [1]_. It is also common to draw at each step the number of
integration steps from a distribution [1,2]_ ; empirical HMC [2]_ for instance
learns this distribution during the adaptation phase. Other algorithms, like
NUTS [3, 4, 5]_, determine the number of integration steps dynamically at runtime.

References
----------
.. [1]: Duane, Simon, et al. "Hybrid monte carlo." Physics letters B 195.2 (1987): 216-222.
.. [2]: Wu, Changye, Julien Stoehr, and Christian P. Robert. "Faster
        Hamiltonian Monte Carlo by learning leapfrog scale." arXiv preprint
        arXiv:1810.04449 (2018).
.. [3]: Hoffman, Matthew D., and Andrew Gelman. "The No-U-Turn sampler:
        adaptively setting path lengths in Hamiltonian Monte Carlo." J. Mach. Learn.
        Res. 15.1 (2014): 1593-1623.
.. [4]: Phan, Du, Neeraj Pradhan, and Martin Jankowiak. "Composable effects for
        flexible and accelerated probabilistic programming in NumPyro." arXiv preprint
        arXiv:1912.11554 (2019).
.. [5]: Lao, Junpeng, et al. "tfp. mcmc: Modern Markov Chain Monte Carlo Tools
        Built for Modern Hardware." arXiv preprint arXiv:2002.01184 (2020).

"""
from typing import Callable, NamedTuple, Tuple

import jax
import jax.numpy as jnp

from blackjax.inference.integrators import IntegratorState

__all__ = ["hmc"]


class HMCTrajectoryInfo(NamedTuple):
    step_size: float
    num_integration_steps: int


ProposalState = IntegratorState


class ProposalInfo(NamedTuple):
    """Additional information on the HMC transition.

    This additional information can be used for debugging or computing
    diagnostics.

    acceptance_probability
        The acceptance probability of the transition, linked to the energy
        difference between the original and the proposed states.
    is_accepted
        Whether the proposed position was accepted or the original position
        was returned.
    is_divergent
        Whether the difference in energy between the original and the new state
        exceeded the divergence threshold.
    energy
        The total energy that corresponds to the returned state.
    proposal
        The state proposed by the proposal. Typically includes the position and
        momentum.
    proposal_info
        Information returned by the proposal. Typically includes the step size,
        number of integration steps and intermediate states.
    """

    acceptance_probability: float
    is_accepted: bool
    is_divergent: bool
    energy: float
    proposal: IntegratorState
    trajectory_info: HMCTrajectoryInfo


def hmc(
    integrator: Callable,
    kinetic_energy: Callable,
    step_size: float,
    num_integration_steps: int = 1,
    divergence_threshold: float = 1000,
) -> Callable:
    """Vanilla HMC proposal running the integrator for a fixed number of steps"""

    def trajectory_deterministic_expansion(
        initial_state: IntegratorState,
    ) -> IntegratorState:
        """Integrate the trajectory  `num_integration_steps` times starting
        from `initial_state` in the direction set by the momentum.
        """

        def one_step(state, _):
            state = integrator(state, step_size)
            return state, ()

        end_state, _ = jax.lax.scan(
            one_step, initial_state, jnp.arange(num_integration_steps)
        )

        return end_state

    def choose(
        rng_key: jax.random.PRNGKey,
        initial_state: IntegratorState,
        last_state: IntegratorState,
    ):
        initial_energy = initial_state.potential_energy + kinetic_energy(
            initial_state.momentum, initial_state.position
        )
        energy = last_state.potential_energy + kinetic_energy(
            last_state.momentum, last_state.position
        )

        delta_energy = initial_energy - energy
        delta_energy = jnp.where(jnp.isnan(delta_energy), -jnp.inf, delta_energy)
        is_diverging = jnp.abs(delta_energy) > divergence_threshold

        p_accept = jnp.clip(jnp.exp(delta_energy), a_max=1)
        do_accept = jax.random.bernoulli(rng_key, p_accept)

        return do_accept, p_accept, is_diverging, energy

    def propose(
        rng_key, initial_state: IntegratorState
    ) -> Tuple[IntegratorState, ProposalInfo]:

        end_state = trajectory_deterministic_expansion(initial_state)

        # To guarantee time-reversibility (hence detailed balance) we
        # need to flip the last state's momentum. If we run the hamiltonian
        # dynamics starting from the last state with flipped momentum we
        # should indeed retrieve the initial state (with flipped momentum).
        flipped_momentum = jax.tree_util.tree_multimap(
            lambda m: -1.0 * m, end_state.momentum
        )
        end_state = IntegratorState(
            end_state.position,
            flipped_momentum,
            end_state.potential_energy,
            end_state.potential_energy_grad,
        )

        do_accept, p_accept, is_diverging, energy = choose(
            rng_key, initial_state, end_state
        )

        info = ProposalInfo(
            p_accept,
            do_accept,
            is_diverging,
            energy,
            end_state,
            HMCTrajectoryInfo(step_size, num_integration_steps),
        )

        return jax.lax.cond(
            do_accept,
            end_state,
            lambda state: (state, info),
            initial_state,
            lambda state: (state, info),
        )

    return propose
