:py:mod:`blackjax.smc.ess`
==========================

.. py:module:: blackjax.smc.ess

.. autoapi-nested-parse::

   All things related to SMC effective sample size



Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   blackjax.smc.ess.ess
   blackjax.smc.ess.log_ess
   blackjax.smc.ess.ess_solver



.. py:function:: ess(log_weights: jax.numpy.ndarray) -> float


.. py:function:: log_ess(log_weights: jax.numpy.ndarray) -> float

   Compute the effective sample size.

   :param log_weights: log-weights of the sample
   :type log_weights: np.ndarray

   :returns: **log_ess** -- The logarithm of the effective sample size
   :rtype: float


.. py:function:: ess_solver(logdensity_fn: Callable, particles: blackjax.types.PyTree, target_ess: float, max_delta: float, root_solver: Callable)

   Build a Tempered SMC step.

   :param logdensity_fn: The log probability function we wish to sample from.
   :type logdensity_fn: Callable
   :param smc_state: Current state of the tempered SMC algorithm
   :type smc_state: SMCState
   :param target_ess: The relative ESS targeted for the next increment of SMC tempering
   :type target_ess: float
   :param max_delta: Max acceptable delta increment
   :type max_delta: float
   :param root_solver: A solver to find the root of a function, takes a function `f`, a starting point `delta0`,
                       a min value `min_delta`, and a max value `max_delta`.
                       Default is `BFGS` minimization of `f ** 2` and ignores `min_delta` and `max_delta`.
   :type root_solver: Callable, optional

   :returns: **delta** -- The increment that solves for the target ESS
   :rtype: float


