"""Base kernel for the HMC family."""
from typing import Callable, Dict, List, NamedTuple, Tuple, Union

import jax
import jax.numpy as jnp

from blackjax.inference.proposals import HMCProposalInfo, HMCProposalState

__all__ = ["HMCState", "HMCInfo", "hmc"]

PyTree = Union[Dict, List, Tuple]


class HMCState(NamedTuple):
    """State of the HMC algorithm.

    The HMC algorithm takes one position of the chain and returns another
    position. In order to make computations more efficient, we also store
    the current potential energy as well as the current gradient of the
    potential energy.
    """

    position: PyTree
    potential_energy: float
    potential_energy_grad: PyTree


class HMCInfo(NamedTuple):
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
    proposal: HMCProposalState
    proposal_info: HMCProposalInfo


def hmc(
    potential_fn: Callable,
    proposal_generator: Callable,
    momentum_generator: Callable,
    kinetic_energy: Callable,
    divergence_threshold: float = 1000.0,
) -> Callable:
    """Create a Hamiltonian Monte Carlo transition kernel.

    Hamiltonian Monte Carlo (HMC) is known to yield effective Markov
    transitions and has been a major empirical success, leading to an extensive
    use in probabilistic programming languages and libraries [1,2,3]_.

    HMC works by augmenting the state space in which the chain evolves with an
    auxiliary momentum :math:`p`. At each step of the chain we draw a momentum
    value from the `momentum_generator` function. We then use Hamilton's
    equations [4]_ to push the state forward; we then compute the new state's
    energy using the `kinetic_energy` function and `logpdf` (potential energy).
    While the hamiltonian dynamics is conservative, numerical integration can
    introduce some discrepancy; we perform a Metropolis acceptance test to
    compensate for integration errors after having flipped the new state's
    momentum to make the transition reversible.

    I encourage anyone interested in the theoretical underpinning of the method
    to read Michael Betancourts' excellent introduction [3]_ and his more
    technical paper [5]_ on the geometric foundations of the method.

    This implementation is very general and should accomodate most variations
    on the method.

    Parameters
    ----------
    proposal_generator:
        The function used to propose a new state for the chain. For vanilla HMC this
        function integrates the trajectory over many steps, but gets more involved
        with other algorithms such as empirical and dynamical HMC.
    momentum_generator:
        A function that returns a new value for the momentum when called.
    kinetic_energy:
        A function that computes the current state's kinetic energy.
    potential:
        The potential function that is being explored, equal to minus the likelihood.
    divergence_threshold:
        The maximum difference in energy between the initial and final state
        after which we consider the transition to be divergent.

    Returns
    -------
    A kernel that moves the chain by one step when called.

    References
    ----------
    .. [1]: Duane, Simon, et al. "Hybrid monte carlo." Physics letters B
            195.2 (1987): 216-222.
    .. [2]: Neal, Radford M. "An improved acceptance procedure for the
            hybrid Monte Carlo algorithm." Journal of Computational Physics 111.1
            (1994): 194-203.
    .. [3]: Betancourt, Michael. "A conceptual introduction to
            Hamiltonian Monte Carlo." arXiv preprint arXiv:1701.02434 (2018).
    .. [4]: "Hamiltonian Mechanics", Wikipedia.
            https://en.wikipedia.org/wiki/Hamiltonian_mechanics#Deriving_Hamilton's_equations
    .. [5]: Betancourt, Michael, et al. "The geometric foundations
            of hamiltonian monte carlo." Bernoulli 23.4A (2017): 2257-2298.

    """

    def kernel(
        rng_key: jax.random.PRNGKey, state: HMCState
    ) -> Tuple[HMCState, HMCInfo]:
        """Moves the chain by one step using the Hamiltonian dynamics.

        Parameters
        ----------
        rng_key:
           The pseudo-random number generator key used to generate random numbers.
        state:
            The current state of the chain: position, log-probability and gradient
            of the log-probability.

        Returns
        -------
        The next state of the chain and additional information about the current step.
        """
        key_momentum, key_integrator, key_accept = jax.random.split(rng_key, 3)

        position, potential_energy, potential_energy_grad = state
        momentum = momentum_generator(key_momentum, position)
        energy = potential_energy + kinetic_energy(momentum)

        proposal, proposal_info = proposal_generator(
            key_integrator, HMCProposalState(position, momentum, potential_energy_grad)
        )
        new_position, new_momentum, new_potential_energy_grad = proposal

        flipped_momentum = -1.0 * new_momentum
        new_potential_energy = potential_fn(new_position)
        new_energy = new_potential_energy + kinetic_energy(flipped_momentum)
        new_state = HMCState(
            new_position, new_potential_energy, new_potential_energy_grad
        )

        delta_energy = energy - new_energy
        delta_energy = jnp.where(jnp.isnan(delta_energy), -jnp.inf, delta_energy)
        is_divergent = jnp.abs(delta_energy) > divergence_threshold

        p_accept = jnp.clip(jnp.exp(delta_energy), a_max=1)
        do_accept = jax.random.bernoulli(key_accept, p_accept)
        accept_state = (
            new_state,
            HMCInfo(
                p_accept,
                True,
                is_divergent,
                new_energy,
                proposal,
                proposal_info,
            ),
        )
        reject_state = (
            state,
            HMCInfo(
                p_accept,
                False,
                is_divergent,
                energy,
                proposal,
                proposal_info,
            ),
        )
        return jax.lax.cond(
            do_accept,
            accept_state,
            lambda state: state,
            reject_state,
            lambda state: state,
        )

    return kernel
