"""Public API for the NUTS Kernel"""
from typing import Callable, Dict, List, NamedTuple, Tuple, Union

import jax.numpy as jnp
import numpy as np

import blackjax.hmc
import blackjax.inference.base as base
import blackjax.inference.integrators as integrators
import blackjax.inference.metrics as metrics
import blackjax.inference.proposal as proposal
import blackjax.inference.termination as termination
import blackjax.inference.trajectory as trajectory

Array = Union[np.ndarray, jnp.DeviceArray]
PyTree = Union[Dict, List, Tuple]


class NUTSParameters(NamedTuple):
    step_size: float = 1e-3
    max_tree_depth: int = 10
    inv_mass_matrix: Array = None
    divergence_threshold: int = 1000


class NUTSInfo(NamedTuple):
    """Additional information on the NUTS transition.

    This additional information can be used for debugging or computing
    diagnostics.

    momentum:
        The momentum that was sampled and used to integrate the trajectory.
    is_divergent
        Whether the difference in energy between the original and the new state
        exceeded the divergence threshold.
    is_turning
        Whether the sampling returned because the trajectory started turning back on itself.
    energy:
        Energy of the transition.
    trajectory_leftmost_state
        The leftmost state of the full trajectory.
    trajectory_rightmost_state
        The rightmost state of the full trajectory.
    num_trajectory_expansions
        Number of subtrajectory samples that were taken.
    """

    momentum: PyTree
    is_divergent: bool
    is_turning: bool
    energy: float
    trajectory_leftmost_state: integrators.IntegratorState
    trajectory_rightmost_state: integrators.IntegratorState
    num_trajectory_expansions: int


new_state = blackjax.hmc.new_state


def kernel(potential_fn: Callable, parameters: NUTSParameters) -> Callable:
    """Build an iterative NUTS kernel.

    Parameters
    ----------
    potential_fn
        A function that returns the potential energy of a chain at a given position. The potential energy
        is defined as minus the log-probability.
    parameters
        A NamedTuple that contains the parameters of the kernel to be built.

    """
    step_size, max_tree_depth, inv_mass_matrix, divergence_threshold = parameters

    if inv_mass_matrix is None:
        raise ValueError(
            "Expected a value for `inv_mass_matrix`,"
            " got None. Please specify a value when initializing"
            " the parameters or run the window adaptation."
        )

    momentum_generator, kinetic_energy_fn, uturn_check_fn = metrics.gaussian_euclidean(
        inv_mass_matrix
    )
    symplectic_integrator = integrators.velocity_verlet(potential_fn, kinetic_energy_fn)
    proposal_generator = iterative_nuts_proposal(
        symplectic_integrator,
        kinetic_energy_fn,
        uturn_check_fn,
        step_size,
        max_tree_depth,
        divergence_threshold,
    )

    kernel = base.hmc(momentum_generator, proposal_generator)

    return kernel


def iterative_nuts_proposal(
    integrator: Callable,
    kinetic_energy: Callable,
    uturn_check_fn: Callable,
    step_size: float,
    max_num_expansions: int = 10,
    divergence_threshold: float = 1000,
) -> Callable:
    """Iterative NUTS algorithm.

    This algorithm is an iteration on the original NUTS algorithm [1]_ with two major differences:
    - We do not use slice samplig but multinomial sampling for the proposal [2]_;
    - The trajectory expansion is not recursive but iterative [3,4]_.

    The implementation can seem unusual for those familiar with similar
    algorithms. Indeed, we do not conceptualize the trajectory construction as
    building a tree. We feel that the tree lingo, inherited from the recursive
    version, is unnecessarily complicated and hides the more general concepts
    on which the NUTS algorithm is built.

    NUTS, in essence, consists in sampling a trajectory by iteratively choosing
    a direction at random and integrating in this direction a number of times
    that doubles at every step. From this trajectory we continuously sample a
    proposal. When the trajectory turns on itself or when we have reached the
    maximum trajectory length we return the current proposal.

    Parameters
    ----------
    integrator
        Symplectic integrator used to build the trajectory step by step.
    kinetic_energy
        Function that computes the kinetic energy.
    uturn_check_fn:
        Function that determines whether the trajectory is turning on itself (metric-dependant).
    step_size
        Size of the integration step.
    max_num_expansions
        The number of sub-trajectory samples we take to build the trajectory.
    divergence_threshold
        Threshold above which we say that there is a divergence.

    Returns
    -------
    A kernel that generates a new chain state and information about the transition.

    References
    ----------
    .. [1]: Hoffman, Matthew D., and Andrew Gelman. "The No-U-Turn sampler: adaptively setting path lengths in Hamiltonian Monte Carlo." J. Mach. Learn. Res. 15.1 (2014): 1593-1623.
    .. [2]: Betancourt, Michael. "A conceptual introduction to Hamiltonian Monte Carlo." arXiv preprint arXiv:1701.02434 (2017).
    .. [3]: Phan, Du, Neeraj Pradhan, and Martin Jankowiak. "Composable effects for flexible and accelerated probabilistic programming in NumPyro." arXiv preprint arXiv:1912.11554 (2019).
    .. [4]: Lao, Junpeng, et al. "tfp. mcmc: Modern markov chain monte carlo tools built for modern hardware." arXiv preprint arXiv:2002.01184 (2020).
    """
    (
        new_criterion_state,
        update_criterion_state,
        is_criterion_met,
    ) = termination.iterative_uturn_numpyro(uturn_check_fn)

    trajectory_integrator = trajectory.dynamic_progressive_integration(
        integrator,
        kinetic_energy,
        update_criterion_state,
        is_criterion_met,
        divergence_threshold,
    )

    expand = trajectory.dynamic_multiplicative_expansion(
        trajectory_integrator,
        uturn_check_fn,
        step_size,
        max_num_expansions,
    )

    def _compute_energy(state: integrators.IntegratorState) -> float:
        energy = state.potential_energy + kinetic_energy(state.position, state.momentum)
        return energy

    def propose(rng_key, initial_state: integrators.IntegratorState):
        criterion_state = new_criterion_state(initial_state, max_num_expansions)
        initial_proposal = proposal.Proposal(
            initial_state, _compute_energy(initial_state), 0.0
        )

        sampled_proposal, *info = expand(
            rng_key,
            initial_proposal,
            criterion_state,
        )
        trajectory, num_doublings, is_diverging, has_terminated, is_turning = info

        info = NUTSInfo(
            initial_state.momentum,
            is_diverging,
            has_terminated | is_turning,
            sampled_proposal.energy,
            trajectory.leftmost_state,
            trajectory.rightmost_state,
            num_doublings,
        )

        return sampled_proposal.state, info

    return propose
