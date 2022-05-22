"""Public API for the Stochastic gradient Langevin Dynamics kernel."""
from typing import Callable, NamedTuple

from blackjax.sgmcmc.diffusion import overdamped_langevin
from blackjax.types import PRNGKey, PyTree

__all__ = ["SGLDState", "init", "kernel"]


class SGLDState(NamedTuple):
    step: int
    position: PyTree
    logprob_grad: PyTree


def init(position: PyTree, batch, grad_estimator_fn: Callable):
    logprob_grad = grad_estimator_fn(position, batch)
    return SGLDState(0, position, logprob_grad)


def kernel(grad_estimator_fn: Callable) -> Callable:
    integrator = overdamped_langevin(grad_estimator_fn)

    def one_step(
        rng_key: PRNGKey, state: SGLDState, data_batch: PyTree, step_size: float
    ) -> SGLDState:

        step, *diffusion_state = state
        new_state = integrator(rng_key, diffusion_state, step_size, data_batch)

        return SGLDState(step + 1, *new_state)

    return one_step
