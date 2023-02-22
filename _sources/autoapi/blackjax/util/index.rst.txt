:orphan:

:py:mod:`blackjax.util`
=======================

.. py:module:: blackjax.util

.. autoapi-nested-parse::

   Utility functions for BlackJax.



Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   blackjax.util.linear_map
   blackjax.util.generate_gaussian_noise
   blackjax.util.pytree_size



.. py:function:: linear_map(diag_or_dense_a, b, *, precision='highest')

   Perform a linear map of the form y = Ax.

   Dispatch matrix multiplication to either jnp.dot or jnp.multiply.

   Unlike jax.numpy.dot, this function output an Array that match the dtype
   and shape of the 2nd input:
   - diag_or_dense_a is a scalar or 1d vector, `diag_or_dense_a * b` is returned
   - diag_or_dense_a is a 2d matrix, `diag_or_dense_a @ b` is returned

   Note that unlike jax.numpy.dot, here we defaults to full (highest)
   precision. This is more useful for numerical algorithms and will be the
   default for jax.numpy in the future:
   https://github.com/google/jax/pull/7859

   :param diag_or_dense_a: A diagonal (1d vector) or dense matrix (2d square matrix).
   :param b: A vector.
   :param precision: The precision of the computation. See jax.lax.dot_general for
                     more details.

   :rtype: The result vector of the matrix multiplication.


.. py:function:: generate_gaussian_noise(rng_key: blackjax.types.PRNGKey, position: blackjax.types.PyTree, mu: Union[float, blackjax.types.Array] = 0.0, sigma: Union[float, blackjax.types.Array] = 1.0) -> blackjax.types.PyTree

   Generate N(mu, sigma) noise with output structure that match a given PyTree.

   :param rng_key: The pseudo-random number generator key used to generate random numbers.
   :param position: PyTree that the structure the output should to match.
   :param mu: The mean of the Gaussian distribution.
   :param sigma: The standard deviation of the Gaussian distribution.

   :rtype: Gaussian noise following N(mu, sigma) that match the structure of position.


.. py:function:: pytree_size(pytree: blackjax.types.PyTree) -> int

   Return the dimension of the flatten PyTree.


