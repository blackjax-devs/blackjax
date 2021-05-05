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
from typing import Callable, Dict, List, NamedTuple, Tuple, Union

import jax

import blackjax.inference.proposal as proposal
import blackjax.inference.termination as termination
import blackjax.inference.trajectory as trajectory
from blackjax.inference.integrators import IntegratorState
from blackjax.inference.proposal import Proposal
from blackjax.inference.trajectory import Trajectory

__all__ = ["HMCState", "HMCInfo", "hmc", "iterative_nuts"]

PyTree = Union[Dict, List, Tuple]
ProposalState = IntegratorState


class NUTSInfo(NamedTuple):
    """In PyMC3

    "depth": self.depth,
    "mean_tree_accept": self.mean_tree_accept,
    "energy_error": self.proposal.energy - self.start.energy,
    "energy": self.proposal.energy,
    "tree_size": self.n_proposals,
    "max_energy_error": self.max_energy_change,
    "model_logp": self.proposal.logp,

    We need some info on the divergence. In particular we need the offending
    new state.
    """

    proposal: Proposal
    delta_energy: float
    max_delta_energy: float
    num_subtrajectories: int
    num_integration_steps: int


def iterative_nuts(
    integrator: Callable,
    kinetic_energy: Callable,
    uturn_check_fn: Callable,
    step_size: float,
    max_num_trajectory_samples: int = 10,
    divergence_threshold: float = 1000,
):
    """Iterative NUTS algorithm.

    This algorithm is an iteration of the original NUTS algorithm [1]_ with two major differences:
    - We do not use slice samplig but multinomial sampling instead [2]_;
    - The trajectory expansion is not recursive but iterative [3]_.

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
    max_num_trajectory_samples
        The number of sub-trajectory samples we take to build the trajectory.
    divergence_threshold
        Threshold above which we say that there is a divergence.

    Returns
    -------
    A kernel that generates a new chain state and information about the transition.

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

    expand, do_keep_expanding = trajectory.dynamic_multiplicative_expansion(
        trajectory_integrator,
        uturn_check_fn,
        step_size,
        max_num_trajectory_samples,
    )

    def _compute_energy(state: IntegratorState) -> float:
        energy = state.potential_energy + kinetic_energy(state.position, state.momentum)
        return energy

    def propose(rng_key, initial_state: IntegratorState):
        criterion_state = new_criterion_state(initial_state, max_num_trajectory_samples)
        proposal = Proposal(initial_state, _compute_energy(initial_state), 0.0)
        trajectory = Trajectory(initial_state, initial_state, initial_state.momentum)

        _, _, proposal, _, _, _, _ = jax.lax.while_loop(
            do_keep_expanding,
            expand,
            (rng_key, 1, proposal, trajectory, criterion_state, False, False),
        )

        # Don't forget the proposal info here!
        return proposal.state, None

    return propose
