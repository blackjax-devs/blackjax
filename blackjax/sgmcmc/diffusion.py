"""Solvers for Langevin diffusions."""
from typing import NamedTuple

import jax
import jax.numpy as jnp

from blackjax.types import PRNGKey, PyTree

__all__ = ["overdamped_langevin"]


class DiffusionState(NamedTuple):
    position: PyTree
    logprob_grad: PyTree


def generate_gaussian_noise(rng_key: PRNGKey, position: PyTree):
    position_flat, unravel_fn = jax.flatten_util.ravel_pytree(position)
    noise_flat = jax.random.normal(rng_key, shape=jnp.shape(position_flat))
    return unravel_fn(noise_flat)


def overdamped_langevin(logprob_grad_fn):
    """Euler solver for overdamped Langevin diffusion."""

    def one_step(
        rng_key: PRNGKey, state: DiffusionState, step_size: float, batch: tuple = ()
    ):
        position, logprob_grad = state
        noise = generate_gaussian_noise(rng_key, position)
        position = jax.tree_util.tree_map(
            lambda p, g, n: p + step_size * g + jnp.sqrt(2 * step_size) * n,
            position,
            logprob_grad,
            noise,
        )

        logprob_grad = logprob_grad_fn(position, batch)
        return DiffusionState(position, logprob_grad)

    return one_step


class SGHMCState(NamedTuple):
    position: PyTree
    momentum: PyTree
    logprob_grad: PyTree


def sghmc(logprob_grad_fn, alpha: float = 0.01, beta: float = 0):
    """Solver for the diffusion equation of the SGHMC algorithm [0]_.

    References
    ----------
    .. [0]:  Chen, T., Fox, E., & Guestrin, C. (2014, June). Stochastic
             gradient hamiltonian monte carlo. In International conference on
             machine learning (pp. 1683-1691). PMLR.

    """

    def one_step(
        rng_key: PRNGKey, state: SGHMCState, step_size: float, batch: tuple = ()
    ) -> SGHMCState:
        position, momentum, logprob_grad = state
        noise = generate_gaussian_noise(rng_key, position)
        position = jax.tree_util.tree_map(lambda x, p: x + p, position, momentum)
        momentum = jax.tree_util.tree_map(
            lambda p, g, n: (1.0 - alpha) * p
            + step_size * g
            + jnp.sqrt(2 * step_size * (alpha - beta)) * n,
            momentum,
            logprob_grad,
            noise,
        )

        logprob_grad = logprob_grad_fn(position, batch)
        return SGHMCState(
            position,
            momentum,
            logprob_grad,
        )

    return one_step
