"""Solvers for Langevin diffusions."""
from typing import NamedTuple

import jax
import jax.numpy as jnp

from blackjax.types import PyTree
from blackjax.util import generate_gaussian_noise

__all__ = ["overdamped_langevin"]


class DiffusionState(NamedTuple):
    position: PyTree
    logprob: float
    logprob_grad: PyTree


def overdamped_langevin(logprob_grad_fn):
    """Euler solver for overdamped Langevin diffusion."""

    def one_step(rng_key, state: DiffusionState, step_size: float, batch: tuple = ()):
        position, _, logprob_grad = state
        noise = generate_gaussian_noise(rng_key, position)
        position = jax.tree_util.tree_map(
            lambda p, g, n: p + step_size * g + jnp.sqrt(2 * step_size) * n,
            position,
            logprob_grad,
            noise,
        )

        logprob, logprob_grad = logprob_grad_fn(position, *batch)
        return DiffusionState(position, logprob, logprob_grad)

    return one_step
