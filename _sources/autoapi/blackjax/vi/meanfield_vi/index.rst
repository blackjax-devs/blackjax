blackjax.vi.meanfield_vi
========================

.. py:module:: blackjax.vi.meanfield_vi


Classes
-------

.. autoapisummary::

   blackjax.vi.meanfield_vi.MFVIState
   blackjax.vi.meanfield_vi.MFVIInfo


Functions
---------

.. autoapisummary::

   blackjax.vi.meanfield_vi.step
   blackjax.vi.meanfield_vi.sample
   blackjax.vi.meanfield_vi.as_top_level_api
   blackjax.vi.meanfield_vi.generate_meanfield_logdensity


Module Contents
---------------

.. py:class:: MFVIState



   .. py:attribute:: mu
      :type:  blackjax.types.ArrayTree


   .. py:attribute:: rho
      :type:  blackjax.types.ArrayTree


   .. py:attribute:: opt_state
      :type:  optax.OptState


.. py:class:: MFVIInfo



   .. py:attribute:: elbo
      :type:  float


.. py:function:: step(rng_key: blackjax.types.PRNGKey, state: MFVIState, logdensity_fn: Callable, optimizer: optax.GradientTransformation, num_samples: int = 5, objective: blackjax.vi._gaussian_vi.Objective = KL(), stl_estimator: bool = True) -> tuple[MFVIState, MFVIInfo]

   Approximate the target density using the mean-field approximation.

   :param rng_key: Key for JAX's pseudo-random number generator.
   :param state: Current state of the mean-field approximation.
   :param logdensity_fn: Function that represents the target log-density to approximate.
   :param optimizer: Optax ``GradientTransformation`` to be used for optimization.
   :param num_samples: The number of samples that are taken from the approximation
                       at each step to compute the Kullback-Leibler divergence between
                       the approximation and the target log-density.
   :param objective: The variational objective to minimize. `KL()` by default or
                     `RenyiAlpha(alpha)`. For alpha = 1, Renyi reduces to KL.
   :param stl_estimator: Whether to use the stick-the-landing (STL) gradient estimator
                         :cite:p:`roeder2017sticking`. The STL estimator has lower gradient
                         variance by removing the score function term from the gradient.
                         :cite:p:`agrawal2020advances` recommend keeping it enabled.

   :rtype: Updated MFVIState and MFVIInfo containing the ELBO value.


.. py:function:: sample(rng_key: blackjax.types.PRNGKey, state: MFVIState, num_samples: int = 1)

   Sample from the mean-field approximation.

   :param rng_key: Key for JAX's pseudo-random number generator.
   :param state: Current MFVIState containing the variational parameters.
   :param num_samples: Number of samples to draw.

   :rtype: A PyTree of samples with leading dimension ``num_samples``


.. py:function:: as_top_level_api(logdensity_fn: Callable, optimizer: optax.GradientTransformation, num_samples: int = 100, objective: blackjax.vi._gaussian_vi.Objective = KL(), stl_estimator: bool = True)

   High-level implementation of Mean-Field Variational Inference

    Parameters
   ----------
   logdensity_fn
       A function that represents the log-density function associated with
       the distribution we want to sample from.
   optimizer
       Optax optimizer to use to optimize the variational objective.
   num_samples
       Number of samples to take at each step to optimize the ELBO.
   objective
       The variational objective to minimize. `KL()` by default or
       `RenyiAlpha(alpha)`. For a = 1, Renyi reduces to KL.
   stl_estimator
       Whether to use the STL gradient estimator.
       Only supported when `objective` is `KL()` or `RenyiAlpha(alpha=1.0)`.

   :rtype: A ``VIAlgorithm``.


.. py:function:: generate_meanfield_logdensity(mu, rho)

