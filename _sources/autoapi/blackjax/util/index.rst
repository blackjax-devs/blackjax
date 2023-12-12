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
   blackjax.util.generate_unit_vector
   blackjax.util.pytree_size
   blackjax.util.index_pytree
   blackjax.util.run_inference_algorithm



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


.. py:function:: generate_gaussian_noise(rng_key: blackjax.types.PRNGKey, position: blackjax.types.ArrayLikeTree, mu: Union[float, blackjax.types.Array] = 0.0, sigma: Union[float, blackjax.types.Array] = 1.0) -> blackjax.types.ArrayTree

   Generate N(mu, sigma) noise with output structure that match a given PyTree.

   :param rng_key: The pseudo-random number generator key used to generate random numbers.
   :param position: PyTree that the structure the output should to match.
   :param mu: The mean of the Gaussian distribution.
   :param sigma: The standard deviation of the Gaussian distribution.

   :rtype: Gaussian noise following N(mu, sigma) that match the structure of position.


.. py:function:: generate_unit_vector(rng_key: blackjax.types.PRNGKey, position: blackjax.types.ArrayLikeTree) -> blackjax.types.Array

   Generate a random unit vector with output structure that match a given PyTree.

   :param rng_key: The pseudo-random number generator key used to generate random numbers.
   :param position: PyTree that the structure the output should to match.

   :rtype: Random unit vector that match the structure of position.


.. py:function:: pytree_size(pytree: blackjax.types.ArrayLikeTree) -> int

   Return the dimension of the flatten PyTree.


.. py:function:: index_pytree(input_pytree: blackjax.types.ArrayLikeTree) -> blackjax.types.ArrayTree

   Builds a PyTree with elements indicating its corresponding index on a flat array.

   Various algorithms in BlackJAX take as input a 1 or 2 dimensional array which somehow
   affects the sampling or approximation of a PyTree. For instance, in HMC a 1 or 2
   dimensional inverse mass matrix is used when simulating Hamilonian dynamics on
   PyTree position and momentum variables. It is usually unclear how the elements of the
   array interact with the PyTree. This function demonstrates how all algorithms map an
   array to a PyTree of equivalent dimension.

   The function returns the index of a 1 dimensional array corresponding to each element of
   the PyTree. This way the user can tell which element in the PyTree corresponds to which
   column (and row) of a 1 dimensional (or 2 dimensional) array.

   :param input_pytree: Example PyTree.

   :rtype: PyTree mapping each individual element of an arange array to elements in the PyTree.


.. py:function:: run_inference_algorithm(rng_key: blackjax.types.PRNGKey, initial_state_or_position: blackjax.types.ArrayLikeTree, inference_algorithm: Union[blackjax.base.SamplingAlgorithm, blackjax.base.VIAlgorithm], num_steps: int, progress_bar: bool = False, transform: Callable = lambda x: x) -> tuple[blackjax.base.State, blackjax.base.State, blackjax.base.Info]

   Wrapper to run an inference algorithm.

   Note that this utility function does not work for Stochastic Gradient MCMC samplers
   like sghmc, as SG-MCMC samplers require additional control flow for batches of data
   to be passed in during each sample.

   :param rng_key: The random state used by JAX's random numbers generator.
   :param initial_state_or_position: The initial state OR the initial position of the inference algorithm. If an initial position
                                     is passed in, the function will automatically convert it into an initial state.
   :param inference_algorithm: One of blackjax's sampling algorithms or variational inference algorithms.
   :param num_steps: Number of MCMC steps.
   :param progress_bar: Whether to display a progress bar.
   :param transform: A transformation of the trace of states to be returned. This is useful for
                     computing determinstic variables, or returning a subset of the states.
                     By default, the states are returned as is.

   :returns:

             1. The final state of the inference algorithm.
             2. The trace of states of the inference algorithm (contains the MCMC samples).
             3. The trace of the info of the inference algorithm for diagnostics.
   :rtype: Tuple[State, State, Info]


