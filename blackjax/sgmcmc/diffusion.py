"""Solvers for Langevin diffusions."""
import jax
import jax.numpy as jnp

from blackjax.types import PRNGKey, PyTree
from blackjax.util import generate_gaussian_noise

__all__ = ["overdamped_langevin"]


def overdamped_langevin():
    """Euler solver for overdamped Langevin diffusion."""

    def one_step(
        rng_key: PRNGKey,
        position: PyTree,
        logprob_grad: PyTree,
        step_size: float,
        minibatch: tuple = (),
    ) -> PyTree:

        noise = generate_gaussian_noise(rng_key, position)
        position = jax.tree_util.tree_map(
            lambda p, g, n: p + step_size * g + jnp.sqrt(2 * step_size) * n,
            position,
            logprob_grad,
            noise,
        )

        return position

    return one_step


def sghmc(alpha: float = 0.01, beta: float = 0):
    """Solver for the diffusion equation of the SGHMC algorithm [0]_.

    References
    ----------
    .. [0]:  Chen, T., Fox, E., & Guestrin, C. (2014, June). Stochastic
             gradient hamiltonian monte carlo. In International conference on
             machine learning (pp. 1683-1691). PMLR.

    """

    def one_step(
        rng_key: PRNGKey,
        position: PyTree,
        momentum: PyTree,
        logprob_grad: PyTree,
        step_size: float,
        minibatch: tuple = (),
    ):
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

        return position, momentum

    return one_step
