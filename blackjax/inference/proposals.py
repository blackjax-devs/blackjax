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

HMCProposalState = IntegratorState


class HMCProposalInfo(NamedTuple):
    step_size: float
    num_integration_steps: int
    trajectory: IntegratorState  # intermediate + last state


def hmc(
    integrator: Callable, step_size: float, num_integration_steps: int = 1
) -> Callable:
    """Vanilla HMC proposal running the integrator for a fixed number of steps"""

    def propose(
        _, initial_state: HMCProposalState
    ) -> Tuple[HMCProposalState, HMCProposalInfo]:
        """Integrate the trajectory  `num_integration_steps` times starting from `initial_state`."""

        def one_step(state, _):
            state = integrator(state, step_size)
            return state, state

        new_state, trajectory = jax.lax.scan(
            one_step, initial_state, jnp.arange(num_integration_steps)
        )
        info = HMCProposalInfo(step_size, num_integration_steps, trajectory)

        return new_state, info

    return propose
