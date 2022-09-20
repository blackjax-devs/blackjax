"""Public API for the Stochastic gradient Langevin Dynamics kernel."""
from typing import Callable, NamedTuple

from blackjax.sgmcmc.diffusion import overdamped_langevin
from blackjax.sgmcmc.gradients import GradientEstimator, GradientState
from blackjax.types import PRNGKey, PyTree

__all__ = ["SGLDState", "init", "kernel"]


class SGLDState(NamedTuple):
    step: int
    position: PyTree
    logprob_grad: PyTree
    grad_estimator_state: GradientState


# We can compute the gradient at the begining of the kernel step
# This allows to get rid of much of the init function, AND
# Prevents a last useless gradient computation at the last step


def init(position: PyTree, minibatch, gradient_estimator: GradientEstimator):
    grad_estimator_state = gradient_estimator.init(minibatch)
    logprob_grad, grad_estimator_state = gradient_estimator.estimate(
        grad_estimator_state, position, minibatch
    )
    return SGLDState(0, position, logprob_grad, grad_estimator_state)


def kernel(gradient_estimator: GradientEstimator) -> Callable:

    grad_estimator_fn = gradient_estimator.estimate
    integrator = overdamped_langevin(grad_estimator_fn)

    def one_step(
        rng_key: PRNGKey, state: SGLDState, minibatch: PyTree, step_size: float
    ):

        step, *diffusion_state = state
        new_state = integrator(rng_key, diffusion_state, step_size, minibatch)

        return SGLDState(step + 1, *new_state)

    return one_step
