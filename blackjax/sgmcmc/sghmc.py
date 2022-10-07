"""Public API for the Stochastic gradient Hamiltonian Monte Carlo kernel."""
from typing import Callable

import jax

from blackjax.sgmcmc.diffusion import sghmc
from blackjax.sgmcmc.gradients import GradientEstimator
from blackjax.sgmcmc.sgld import SGLDState
from blackjax.types import PRNGKey, PyTree
from blackjax.util import generate_gaussian_noise

__all__ = ["kernel"]


def kernel(
    gradient_estimator: GradientEstimator, alpha: float = 0.01, beta: float = 0
) -> Callable:

    integrator = sghmc(alpha, beta)

    def one_step(
        rng_key: PRNGKey, state: SGLDState, minibatch: PyTree, step_size: float, L: int
    ) -> SGLDState:
        def body_fn(state, rng_key):
            position, momentum, grad_estimator_state = state
            logprob_grad, grad_estimator_state = gradient_estimator.estimate(
                grad_estimator_state, position, minibatch
            )
            position, momentum = integrator(
                rng_key, position, momentum, logprob_grad, step_size, minibatch
            )
            return (
                (position, momentum, grad_estimator_state),
                (position, grad_estimator_state),
            )

        step, position, grad_estimator_state = state
        momentum = generate_gaussian_noise(rng_key, position, step_size)
        init_diffusion_state = (position, momentum, grad_estimator_state)

        keys = jax.random.split(rng_key, L)
        last_state, _ = jax.lax.scan(body_fn, init_diffusion_state, keys)
        position, _, grad_estimator_state = last_state

        return SGLDState(step + 1, position, grad_estimator_state)

    return one_step
