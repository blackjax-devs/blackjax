"""Public API for the Stochastic gradient Hamiltonian Monte Carlo kernel."""
from typing import Callable

import jax
import jax.numpy as jnp

from blackjax.sgmcmc.diffusion import SGHMCState, sghmc
from blackjax.sgmcmc.sgld import SGLDState
from blackjax.types import PRNGKey, PyTree

__all__ = ["kernel"]


def sample_momentum(rng_key: PRNGKey, position: PyTree, step_size: float):
    position_flat, unravel_fn = jax.flatten_util.ravel_pytree(position)
    noise_flat = jnp.sqrt(step_size) * jax.random.normal(
        rng_key, shape=jnp.shape(position_flat)
    )
    return unravel_fn(noise_flat)


def kernel(
    grad_estimator_fn: Callable, alpha: float = 0.01, beta: float = 0
) -> Callable:
    integrator = sghmc(grad_estimator_fn)

    def one_step(
        rng_key: PRNGKey, state: SGLDState, data_batch: PyTree, step_size: float, L: int
    ) -> SGLDState:

        step, position, logprob_grad = state
        momentum = sample_momentum(rng_key, position, step_size)
        diffusion_state = SGHMCState(position, momentum, logprob_grad)

        def body_fn(state, rng_key):
            new_state = integrator(rng_key, state, step_size, data_batch)
            return new_state, new_state

        keys = jax.random.split(rng_key, L)
        last_state, _ = jax.lax.scan(body_fn, diffusion_state, keys)

        return SGLDState(step + 1, last_state.position, last_state.logprob_grad)

    return one_step
