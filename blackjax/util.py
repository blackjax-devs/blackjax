"""Utility functions for BlackJax."""
from functools import partial
from typing import Union

import jax.numpy as jnp
from jax import jit, lax
from jax._src.numpy.util import _promote_dtypes
from jax.flatten_util import ravel_pytree
from jax.random import normal
from jax.tree_util import tree_leaves

from blackjax.types import Array, PRNGKey, PyTree


@partial(jit, static_argnames=("precision",), inline=True)
def linear_map(diag_or_dense_a, b, *, precision="highest"):
    """Perform a linear map of the form y = Ax.

    Dispatch matrix multiplication to either jnp.dot or jnp.multiply.

    Unlike jax.numpy.dot, this function output an Array that match the dtype
    and shape of the 2nd input:
    - diag_or_dense_a is a scalar or 1d vector, `diag_or_dense_a * b` is returned
    - diag_or_dense_a is a 2d matrix, `diag_or_dense_a @ b` is returned

    Note that unlike jax.numpy.dot, here we defaults to full (highest)
    precision. This is more useful for numerical algorithms and will be the
    default for jax.numpy in the future:
    https://github.com/google/jax/pull/7859

    Parameters
    ----------
    diag_or_dense_a:
        A diagonal (1d vector) or dense matrix (2d square matrix).
    b:
        A vector.
    precision:
        The precision of the computation. See jax.lax.dot_general for
        more details.

    Returns
    -------
        The result vector of the matrix multiplication.
    """
    diag_or_dense_a, b = _promote_dtypes(diag_or_dense_a, b)
    ndim = jnp.ndim(diag_or_dense_a)

    if ndim <= 1:
        return lax.mul(diag_or_dense_a, b)
    else:
        return lax.dot(diag_or_dense_a, b, precision=precision)


# TODO(https://github.com/blackjax-devs/blackjax/issues/376)
# Refactor this function to not use ravel_pytree might be more performant.
def generate_gaussian_noise(
    rng_key: PRNGKey,
    position: PyTree,
    mu: Union[float, Array] = 0.0,
    sigma: Union[float, Array] = 1.0,
) -> PyTree:
    """Generate N(mu, sigma) noise with output structure that match a given PyTree.

    Parameters
    ----------
    rng_key:
        The pseudo-random number generator key used to generate random numbers.
    position:
        PyTree that the structure the output should to match.
    mu:
        The mean of the Gaussian distribution.
    sigma:
        The standard deviation of the Gaussian distribution.

    Returns
    -------
    Gaussian noise following N(mu, sigma) that match the structure of position.
    """
    p, unravel_fn = ravel_pytree(position)
    sample = normal(rng_key, shape=p.shape, dtype=p.dtype)
    return unravel_fn(mu + linear_map(sigma, sample))


def pytree_size(pytree: PyTree) -> int:
    """Return the dimension of the flatten PyTree."""
    return sum(jnp.size(value) for value in tree_leaves(pytree))
