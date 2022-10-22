"""Public API for the Stochastic gradient Hamiltonian Monte Carlo kernel."""
from typing import Callable

import jax

from blackjax.sgmcmc.diffusion import sghmc
from blackjax.types import PRNGKey, PyTree
from blackjax.util import generate_gaussian_noise

__all__ = ["kernel"]


def kernel(alpha: float = 0.01, beta: float = 0) -> Callable:
    """Stochastic gradient Hamiltonian Monte Carlo (SgHMC) algorithm."""

    integrator = sghmc(alpha, beta)

    def one_step(
        rng_key: PRNGKey,
        position: PyTree,
        grad_estimator: Callable,
        minibatch: PyTree,
        step_size: float,
        num_integration_steps: int,
    ) -> PyTree:
        def body_fn(state, rng_key):
            position, momentum = state
            logprob_grad = grad_estimator(position, minibatch)
            position, momentum = integrator(
                rng_key, position, momentum, logprob_grad, step_size, minibatch
            )
            return ((position, momentum), position)

        momentum = generate_gaussian_noise(rng_key, position, step_size)
        keys = jax.random.split(rng_key, num_integration_steps)
        position, _ = jax.lax.scan(body_fn, (position, momentum), keys)

        return position

    return one_step
