"""Public API for the Stochastic gradient Hamiltonian Monte Carlo kernel."""
from typing import Callable

import jax
import jax.numpy as jnp

from blackjax.sgmcmc.diffusion import SGHMCState, sghmc
from blackjax.sgmcmc.gradients import GradientEstimator
from blackjax.sgmcmc.sgld import SGLDState
from blackjax.sgmcmc.sgld import init as sgld_init
from blackjax.types import PRNGKey, PyTree
from blackjax.util import generate_gaussian_noise

__all__ = ["kernel"]


def kernel(
    gradient_estimator: GradientEstimator, alpha: float = 0.01, beta: float = 0
) -> Callable:

    grad_estimator_fn = gradient_estimator.estimate
    integrator = sghmc(grad_estimator_fn)

    def one_step(
        rng_key: PRNGKey, state: SGLDState, minibatch: PyTree, step_size: float, L: int
    ) -> SGLDState:

        step, position, logprob_grad, grad_estimator_state = state
        momentum = generate_gaussian_noise(rng_key, position, jnp.sqrt(step_size))
        diffusion_state = SGHMCState(
            position, momentum, logprob_grad, grad_estimator_state
        )

        def body_fn(state, rng_key):
            new_state = integrator(rng_key, state, step_size, minibatch)
            return new_state, new_state

        keys = jax.random.split(rng_key, L)
        last_state, _ = jax.lax.scan(body_fn, diffusion_state, keys)
        position, _, logprob_grad, grad_estimator_state = last_state

        return SGLDState(step + 1, position, logprob_grad, grad_estimator_state)

    return one_step
