blackjax.vi.fullrank_vi
=======================

.. py:module:: blackjax.vi.fullrank_vi


Classes
-------

.. autoapisummary::

   blackjax.vi.fullrank_vi.FRVIState
   blackjax.vi.fullrank_vi.FRVIInfo


Functions
---------

.. autoapisummary::

   blackjax.vi.fullrank_vi.step
   blackjax.vi.fullrank_vi.sample
   blackjax.vi.fullrank_vi.as_top_level_api
   blackjax.vi.fullrank_vi.generate_fullrank_logdensity


Module Contents
---------------

.. py:class:: FRVIState



   State of the full-rank VI algorithm.

   mu:
       Mean of the Gaussian approximation.
   chol_params:
       Flattened Cholesky factor of the Gaussian approximation, used to parameterize
       the full-rank covariance matrix. A vector of length d(d+1)/2 for a
       d-dimensional Gaussian, containing d diagonal elements (in log space) followed
       by lower triangular elements in row-major order.
   opt_state:
       Optax optimizer state.



   .. py:attribute:: mu
      :type:  blackjax.types.ArrayTree


   .. py:attribute:: chol_params
      :type:  blackjax.types.Array


   .. py:attribute:: opt_state
      :type:  optax.OptState


.. py:class:: FRVIInfo



   Extra information of the full-rank VI algorithm.

   elbo:
       ELBO of approximation wrt target distribution.



   .. py:attribute:: elbo
      :type:  float


.. py:function:: step(rng_key: blackjax.types.PRNGKey, state: FRVIState, logdensity_fn: Callable, optimizer: optax.GradientTransformation, num_samples: int = 5, objective: blackjax.vi._gaussian_vi.Objective = KL(), stl_estimator: bool = True) -> tuple[FRVIState, FRVIInfo]

   Approximate the target density using the full-rank Gaussian approximation.

   :param rng_key: Key for JAX's pseudo-random number generator.
   :param state: Current state of the full-rank approximation.
   :param logdensity_fn: Function that represents the target log-density to approximate.
   :param optimizer: Optax `GradientTransformation` to be used for optimization.
   :param num_samples: The number of samples that are taken from the approximation
                       at each step to compute the Kullback-Leibler divergence between
                       the approximation and the target log-density.
   :param objective: The variational objective to minimize. `KL()` by default or
                     `RenyiAlpha(alpha)`. For alpha = 1, Renyi reduces to KL.
   :param stl_estimator: Whether to use the stick-the-landing (STL) gradient estimator
                         :cite:p:`roeder2017sticking`. Reduces gradient variance by removing
                         the score function term. Recommended in :cite:p:`agrawal2020advances`.

   :returns: * *new_state* -- Updated ``FRVIState``.
             * *info* -- ``FRVIInfo`` containing the current ELBO value.


.. py:function:: sample(rng_key: blackjax.types.PRNGKey, state: FRVIState, num_samples: int = 1)

   Sample from the full-rank approximation.

   :param rng_key: Key for JAX's pseudo-random number generator.
   :param state: Current ``FRVIState``.
   :param num_samples: Number of samples to draw.

   :returns: * *Samples from the full-rank Gaussian approximation, as a PyTree with a*
             * leading axis of size ``num_samples``.


.. py:function:: as_top_level_api(logdensity_fn: Callable, optimizer: optax.GradientTransformation, num_samples: int = 100, objective: blackjax.vi._gaussian_vi.Objective = KL(), stl_estimator: bool = True)

   High-level implementation of Full-Rank Variational Inference.

   :param logdensity_fn: A function that represents the log-density function associated with
                         the distribution we want to sample from.
   :param optimizer: Optax optimizer to use to optimize the ELBO.
   :param num_samples: Number of samples to take at each step to optimize the ELBO.
   :param objective: The variational objective to minimize. `KL()` by default or
                     `RenyiAlpha(alpha)`. For alpha = 1, Renyi reduces to KL.
   :param stl_estimator: Whether to use STL gradient estimator.
                         Only supported when `objective` is `KL()` or `RenyiAlpha(alpha=1.0)`.

   :rtype: A ``VIAlgorithm``.


.. py:function:: generate_fullrank_logdensity(mu, chol_params)

   Generate the log-density function of a full-rank Gaussian distribution.

   :param mu: Mean of the Gaussian distribution.
   :param chol_params: Flattened Cholesky factor of the Gaussian distribution.

   :rtype: A function that computes the log-density of the full-rank Gaussian distribution.


