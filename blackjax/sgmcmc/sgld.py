"""Public API for the Stochastic gradient Langevin Dynamics kernel."""
from typing import Callable

from blackjax.sgmcmc.diffusion import overdamped_langevin
from blackjax.types import PRNGKey, PyTree

__all__ = ["kernel"]


def kernel() -> Callable:
    """Stochastic gradient Langevin Dynamics (SgLD) algorithm."""

    integrator = overdamped_langevin()

    def one_step(
        rng_key: PRNGKey,
        position: PyTree,
        grad_estimator: Callable,
        minibatch: PyTree,
        step_size: float,
    ):

        logprob_grad = grad_estimator(position, minibatch)
        new_position = integrator(rng_key, position, logprob_grad, step_size, minibatch)

        return new_position

    return one_step
