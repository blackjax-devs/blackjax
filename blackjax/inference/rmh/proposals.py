from typing import Callable

import jax
import jax.numpy as jnp

from blackjax.types import Array, PRNGKey, PyTree


def normal(sigma: Array) -> Callable:
    """Normal Random Walk proposal.

    Propose a new position such that its distance to the current position is
    normally distributed. Suitable for continuous variables.

    Parameter
    ---------
    sigma:
        vector or matrix that contains the standard deviation of the centered
        normal distribution from which we draw the move proposals.

    """
    ndim = jnp.ndim(sigma)  # type: ignore[arg-type]
    shape = jnp.shape(jnp.atleast_1d(sigma))[:1]

    if ndim == 1:
        dot = jnp.multiply
    elif ndim == 2:
        dot = jnp.dot
    else:
        raise ValueError

    def propose(rng_key: PRNGKey, position: PyTree) -> PyTree:
        _, unravel_fn = jax.flatten_util.ravel_pytree(position)
        sample = jax.random.normal(rng_key, shape)
        move_sample = dot(sigma, sample)
        move_unravel = unravel_fn(move_sample)
        return move_unravel

    return propose
