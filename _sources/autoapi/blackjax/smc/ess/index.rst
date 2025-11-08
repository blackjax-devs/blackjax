blackjax.smc.ess
================

.. py:module:: blackjax.smc.ess

.. autoapi-nested-parse::

   All things related to SMC effective sample size



Functions
---------

.. autoapisummary::

   blackjax.smc.ess.ess
   blackjax.smc.ess.log_ess
   blackjax.smc.ess.ess_solver


Module Contents
---------------

.. py:function:: ess(log_weights: blackjax.types.Array) -> float | blackjax.types.Array

   Compute the effective sample size.

   :param log_weights: Log-weights of the sample, shape (n_particles,).
   :type log_weights: Array

   :returns: **ess** -- The effective sample size.
   :rtype: float | Array


.. py:function:: log_ess(log_weights: blackjax.types.Array) -> float | blackjax.types.Array

   Compute the logarithm of the effective sample size.

   :param log_weights: Log-weights of the sample, shape (n_particles,).
   :type log_weights: Array

   :returns: **log_ess** -- The logarithm of the effective sample size.
   :rtype: float | Array


.. py:function:: ess_solver(logdensity_fn: Callable, particles: blackjax.types.ArrayLikeTree, target_ess: float | blackjax.types.Array, max_delta: float | blackjax.types.Array, root_solver: Callable) -> float | blackjax.types.Array

   ESS solver for computing the next increment of SMC tempering.

   :param logdensity_fn: The log probability function we wish to sample from.
   :type logdensity_fn: Callable
   :param particles: Current particles of the tempered SMC algorithm.
   :type particles: ArrayLikeTree
   :param target_ess: Target effective sample size (ESS) for the next increment of SMC tempering.
   :type target_ess: float | Array
   :param max_delta: Maximum acceptable delta increment.
   :type max_delta: float | Array
   :param root_solver: A solver to find the root of a function. Signature is
                       root_solver(fun, min_delta, max_delta). Use e.g. dichotomy from
                       blackjax.smc.solver.
   :type root_solver: Callable

   :returns: **delta** -- The increment that solves for the target ESS.
   :rtype: float | Array


