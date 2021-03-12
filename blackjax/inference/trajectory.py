"""Procedures to build trajectories for algorithms in the HMC family.

While the traditional implementation of HMC generates proposals by running the
integrators in a given direction and flipping the momentum of the last state,
this is only one possible way to generate trajectories and thus proposals.

The next level of complexity would be to choose directions at random at each
step, thus *sampling* a trajectory (which ensures detailed balance). NUTS goes
even further by choosing the direction at random and runs the interator a
number of times that is a function of the current step.

In this file we implement various ways of sampling trajectories. As in [1]_ we
distinguish between:

    - static trajectory sampling: we first sample a trajectory and then
      generate a proposal from this trajectory; this requires to store the
      whole trajectory in memory. We can also update proposals as we sample the
      trajectory; in this progressive scheme we only need to store the
      endpoints of the trajectory and the current proposal.
    - dynamic trajectory sampling: we stop sampling when a certain critetion is
      met.

References
----------
.. [1]: Betancourt, Michael. "A conceptual introduction to Hamiltonian Monte Carlo." arXiv preprint arXiv:1701.02434 (2017).

"""
from typing import Callable

import jax
import jax.numpy as jnp

from blackjax.inference.integrators import IntegratorState


# -------------------------------------------------------------------
#                             Integration
#
# Generating samples by choosing a direction and running the integrator
# several times along this direction. Distinct from sampling.
# -------------------------------------------------------------------


def static_integration(
    integrator: Callable,
    step_size: float,
    num_integration_steps: int,
    direction: int = 1,
) -> Callable:
    """Generate a trajectory by integrating in one direction."""

    directed_step_size = direction * step_size

    def integrate(initial_state: IntegratorState):
        def one_step(state, _):
            state = integrator(state, directed_step_size)
            return state, state

        last_state, states = jax.lax.scan(
            one_step, initial_state, jnp.arange(num_integration_steps)
        )

        return last_state, states

    return integrate


def static_progressive_integration(
    integrator: Callable,
    update_proposal: Callable,
    step_size: float,
    num_integration_steps: int,
    direction: int = 1,
):
    """Generate a trajectory by integrating in one direction and updating the
    proposal at each step.

    Returns
    -------
    An array that contains all the intermediate proposals.

    """

    directed_step_size = direction * step_size

    def integrate(rng_key, initial_state: IntegratorState):

        def one_step(integration_step, _):
            rng_key, state, proposal = integration_step
            _, rng_key = jax.random.split(rng_key)
            new_state = integrator(state, directed_step_size)
            new_proposal = update_proposal(rng_key, new_state, proposal)
            return (rng_key, new_state, new_proposal), ()

        _, states = jax.lax.scan(
            one_step, (initial_state, initial_state), jnp.arange(num_integration_steps)
        )
        proposals = states[-1]

        return proposals
