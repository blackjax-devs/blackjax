"""Utility functions for BlackJax."""
from functools import partial
from typing import Union

import jax.numpy as jnp
from jax import jit, lax
from jax.flatten_util import ravel_pytree
from jax.random import normal
from jax.tree_util import tree_leaves, tree_map, tree_structure, tree_unflatten

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
    dtype = jnp.result_type(diag_or_dense_a.dtype, b.dtype)
    diag_or_dense_a = diag_or_dense_a.astype(dtype)
    b = b.astype(dtype)
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


def unflatten_array(array: Array, position: PyTree) -> PyTree:
    """Builds a PyTree from a 1 or 2 dimensional Array.

    Various algorithms in BlackJAX take as input a 1 or 2 dimensional array which somehow
    affects the sampling or approximation of a PyTree. For instance, in HMC a 1 or 2
    dimensional inverse mass matrix is used when simulating Hamilonian dynamics on
    PyTree position and momentum variables. It is usually unclear how the elements of the
    array interact with the PyTree. This function demonstrates how all algorithms map an
    array to a PyTree of equivalent dimension.

    Parameters
    ----------
    array:
        1 `(ndim,)` or 2 `(ndim, ndim)` dimensional array.
    position:
        PyTree which individual dimensions add up to `ndim`.

    Returns
    -------
    PyTree (1 dimensional array) or nested PyTree (2 dimensional array) mapping each individual
    element of the array to elements or interaction of elements in the PyTree.
    """
    flat_position, unravel_fn = ravel_pytree(position)
    (dim_position,) = flat_position.shape
    shape_array = array.shape

    dtype = jnp.result_type(array.dtype, flat_position.dtype)
    array = array.astype(dtype)
    position = tree_map(lambda p: jnp.atleast_1d(p).astype(dtype), position)
    ndim = jnp.ndim(array)

    if ndim <= 1:
        if dim_position != shape_array[0]:
            raise ValueError(
                "The array has the wrong shape:"
                f" expected {(dim_position,)}, got {shape_array}."
            )

        return unravel_fn(array)

    elif ndim == 2:
        if (dim_position, dim_position) != shape_array:
            raise ValueError(
                "The array has the wrong shape:"
                f" expected {(dim_position, dim_position)}, got {shape_array}."
            )

        first_unravel = [unravel_fn(value) for value in array]
        leaves = tree_leaves(position)
        pydef = tree_structure(position)
        indx = 0
        unravel_leaves = []
        shapes = [jnp.atleast_1d(leaf).shape for leaf in leaves]
        for leaf in leaves:
            sh = leaf.shape
            (dim,) = jnp.ravel(leaf, order="C").shape
            shape = [jnp.array(sh + s) for s in shapes]
            shape = tree_unflatten(pydef, shape)
            unravel_leaf = tree_map(
                lambda s, *l: jnp.vstack(l).reshape(s),
                shape,
                *first_unravel[indx : (indx + dim)],
            )
            indx += dim
            unravel_leaves.append(unravel_leaf)
        unravel_array = tree_unflatten(pydef, unravel_leaves)
        return unravel_array

    else:
        raise ValueError(
            "The array has the wrong number of dimensions:"
            f" expected 1 or 2, got {ndim}."
        )
