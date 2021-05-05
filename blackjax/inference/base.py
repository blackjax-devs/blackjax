"""Base kernel for the HMC family."""
from typing import Callable, Dict, List, NamedTuple, Tuple, Union

import jax

from blackjax.inference.integrators import IntegratorState

__all__ = ["HMCState", "hmc"]

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


def new_hmc_state(position: PyTree, potential_fn: Callable) -> HMCState:
    """Create a chain state from a position.

    The HMC kernel works with states that contain the current chain position
    and its associated potential energy and derivative of the potential energy.
    This function computes initial states from an initial position.  While this
    function is intended to work for one chain, it is possible to use
    `jax.vmap` to compute the initial state of several chains.

    Note
    ----
    The potential energy is also known as the negative joint loglikelihood of
    the model, priors, and data without the normalizing constant, otherwise
    known as the negative log posterior density.

    Example
    -------
    Let us assume a model with two random variables. We wish to sample from
    4 chains with the following initial positions:

        >>> import numpy as np
        >>> init_positions = (np.random.rand(4, 1000), np.random.rand(4, 300))

    We have a `logpdf` function that returns the log-probability associated with
    the chain at a given position:

        >>> potential_fn((np.random.rand(1000), np.random.rand(3000)))
        -3.4

    We can compute the initial state for each of the 4 chain as follows:

        >>> import jax
        >>> jax.vmap(new_state, in_axes=(0, None))(init_positions, potential_fn)

    Parameters
    ----------
    position
        The current values of the random variables whose posterior we want to
        sample from. Can be anything from a list, a (named) tuple or a dict of
        arrays. The arrays can either be Numpy arrays or JAX DeviceArrays.
    potential_fn
        A function that returns the value of the potential energy when called
        with a position.

    Returns
    -------
    A HMC state that contains the position, the associated potential energy and gradient of the
    potential energy.
    """
    potential_energy, potential_energy_grad = jax.value_and_grad(potential_fn)(position)
    return HMCState(position, potential_energy, potential_energy_grad)


def hmc(
    momentum_generator: Callable,
    proposal_generator: Callable,
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
        rng_key: jax.numpy.DeviceArray, state: HMCState
    ) -> Tuple[HMCState, NamedTuple]:
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
        key_momentum, key_integrator = jax.random.split(rng_key, 2)

        position, potential_energy, potential_energy_grad = state
        momentum = momentum_generator(key_momentum, position)

        augmented_state = IntegratorState(
            position, momentum, potential_energy, potential_energy_grad
        )
        proposal, info = proposal_generator(key_integrator, augmented_state)
        proposal = HMCState(
            proposal.position, proposal.potential_energy, proposal.potential_energy_grad
        )

        return proposal, info

    return kernel
