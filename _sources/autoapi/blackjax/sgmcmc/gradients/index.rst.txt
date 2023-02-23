:py:mod:`blackjax.sgmcmc.gradients`
===================================

.. py:module:: blackjax.sgmcmc.gradients


Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   blackjax.sgmcmc.gradients.logdensity_estimator
   blackjax.sgmcmc.gradients.grad_estimator
   blackjax.sgmcmc.gradients.control_variates



.. py:function:: logdensity_estimator(logprior_fn: Callable, loglikelihood_fn: Callable, data_size: int) -> Callable

   Builds a simple estimator for the log-density.

   This estimator first appeared in :cite:p:`robbins1951stochastic`. The `logprior_fn` function has a
   single argument:  the current position (value of parameters). The
   `loglikelihood_fn` takes two arguments: the current position and a batch of
   data; if there are several variables (as, for instance, in a supervised
   learning contexts), they are passed in a tuple.

   This algorithm was ported from :cite:p:`coullon2022sgmcmcjax`.

   :param logprior_fn: The log-probability density function corresponding to the prior
                       distribution.
   :param loglikelihood_fn: The log-probability density function corresponding to the likelihood.
   :param data_size: The number of items in the full dataset.


.. py:function:: grad_estimator(logprior_fn: Callable, loglikelihood_fn: Callable, data_size: int) -> Callable

   Build a simple estimator for the gradient of the log-density.


.. py:function:: control_variates(logdensity_grad_estimator: Callable, centering_position: blackjax.types.PyTree, data: blackjax.types.PyTree) -> Callable

   Builds a control variate gradient estimator :cite:p:`baker2019control`.

   This algorithm was ported from :cite:p:`coullon2022sgmcmcjax`.

   :param logdensity_grad_estimator: A function that approximates the target's gradient function.
   :param data: The full dataset.
   :param centering_position: Centering position for the control variates (typically the MAP).


