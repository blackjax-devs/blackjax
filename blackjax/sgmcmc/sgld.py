"""Public API for the Stochastic gradient Langevin Dynamics kernel."""
from typing import Callable, NamedTuple

from blackjax.sgmcmc.diffusion import overdamped_langevin
from blackjax.sgmcmc.gradients import GradientEstimator, GradientState
from blackjax.types import PRNGKey, PyTree

__all__ = ["SGLDState", "init", "kernel"]


class SGLDState(NamedTuple):
    position: PyTree
    grad_estimator_state: GradientState


def init(position: PyTree, minibatch, gradient_estimator: GradientEstimator):
    grad_estimator_state = gradient_estimator.init(minibatch)
    return SGLDState(position, grad_estimator_state)


def kernel(gradient_estimator: GradientEstimator) -> Callable:

    integrator = overdamped_langevin()

    def one_step(
        rng_key: PRNGKey, state: SGLDState, minibatch: PyTree, step_size: float
    ):

        position, grad_estimator_state = state
        logprob_grad, grad_estimator_state = gradient_estimator.estimate(
            grad_estimator_state, position, minibatch
        )
        new_position = integrator(rng_key, position, logprob_grad, step_size, minibatch)

        return SGLDState(new_position, grad_estimator_state)

    return one_step
