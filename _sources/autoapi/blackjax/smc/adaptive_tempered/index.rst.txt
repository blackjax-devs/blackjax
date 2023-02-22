:py:mod:`blackjax.smc.adaptive_tempered`
========================================

.. py:module:: blackjax.smc.adaptive_tempered


Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   blackjax.smc.adaptive_tempered.kernel



.. py:function:: kernel(logprior_fn: Callable, loglikelihood_fn: Callable, mcmc_step_fn: Callable, mcmc_init_fn: Callable, resampling_fn: Callable, target_ess: float, root_solver: Callable = solver.dichotomy) -> Callable

   Build a Tempered SMC step using an adaptive schedule.

   :param logprior_fn: A function that computes the log-prior density.
   :type logprior_fn: Callable
   :param loglikelihood_fn: A function that returns the log-likelihood density.
   :type loglikelihood_fn: Callable
   :param mcmc_kernel_factory: A callable function that creates a mcmc kernel from a log-probability
                               density function.
   :type mcmc_kernel_factory: Callable
   :param make_mcmc_state: A function that creates a new mcmc state from a position and a
                           log-probability density function.
   :type make_mcmc_state: Callable
   :param resampling_fn: A random function that resamples generated particles based of weights
   :type resampling_fn: Callable
   :param target_ess: The target ESS for the adaptive MCMC tempering
   :type target_ess: float
   :param root_solver: A solver utility to find delta matching the target ESS. Signature is
                       `root_solver(fun, delta_0, min_delta, max_delta)`, default is a dichotomy solver
   :type root_solver: Callable, optional
   :param use_log_ess: Use ESS in log space to solve for delta, default is `True`.
                       This is usually more stable when using gradient based solvers.
   :type use_log_ess: bool, optional

   :returns: * *A callable that takes a rng_key and a TemperedSMCState that contains the current state*
             * *of the chain and that returns a new state of the chain along with*
             * *information about the transition.*


