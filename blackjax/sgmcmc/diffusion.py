"""Solvers for Langevin diffusions."""
from typing import NamedTuple

import jax
import jax.numpy as jnp

from blackjax.sgmcmc.gradients import GradientState
from blackjax.types import PRNGKey, PyTree
from blackjax.util import generate_gaussian_noise

__all__ = ["overdamped_langevin"]


class DiffusionState(NamedTuple):
    position: PyTree
    logprob_grad: PyTree
    grad_estimator_state: GradientState


def overdamped_langevin(grad_estimator_fn):
    """Euler solver for overdamped Langevin diffusion."""

    def one_step(
        rng_key: PRNGKey, state: DiffusionState, step_size: float, minibatch: tuple = ()
    ):
        position, logprob_grad, grad_estimator_state = state
        noise = generate_gaussian_noise(rng_key, position)
        position = jax.tree_util.tree_map(
            lambda p, g, n: p + step_size * g + jnp.sqrt(2 * step_size) * n,
            position,
            logprob_grad,
            noise,
        )

        logprob_grad, gradient_estimator_state = grad_estimator_fn(
            grad_estimator_state, position, minibatch
        )
        return DiffusionState(position, logprob_grad, gradient_estimator_state)

    return one_step


class SGHMCState(NamedTuple):
    position: PyTree
    momentum: PyTree
    logprob_grad: PyTree
    grad_estimator_state: GradientState


def sghmc(grad_estimator_fn, alpha: float = 0.01, beta: float = 0):
    """Solver for the diffusion equation of the SGHMC algorithm [0]_.

    References
    ----------
    .. [0]:  Chen, T., Fox, E., & Guestrin, C. (2014, June). Stochastic
             gradient hamiltonian monte carlo. In International conference on
             machine learning (pp. 1683-1691). PMLR.

    """

    def one_step(
        rng_key: PRNGKey, state: SGHMCState, step_size: float, minibatch: tuple = ()
    ):
        position, momentum, logprob_grad, grad_estimator_state = state
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

        logprob_grad, gradient_estimator_state = grad_estimator_fn(
            grad_estimator_state, position, minibatch
        )
        return SGHMCState(position, momentum, logprob_grad, gradient_estimator_state)

    return one_step
