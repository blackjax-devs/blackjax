:py:mod:`blackjax.smc.adaptive_tempered`
========================================

.. py:module:: blackjax.smc.adaptive_tempered


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   blackjax.smc.adaptive_tempered.adaptive_tempered_smc



Functions
~~~~~~~~~

.. autoapisummary::

   blackjax.smc.adaptive_tempered.build_kernel



.. py:function:: build_kernel(logprior_fn: Callable, loglikelihood_fn: Callable, mcmc_step_fn: Callable, mcmc_init_fn: Callable, resampling_fn: Callable, target_ess: float, root_solver: Callable = solver.dichotomy) -> Callable

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


.. py:class:: adaptive_tempered_smc


   Implements the (basic) user interface for the Adaptive Tempered SMC kernel.

   :param logprior_fn: The log-prior function of the model we wish to draw samples from.
   :param loglikelihood_fn: The log-likelihood function of the model we wish to draw samples from.
   :param mcmc_step_fn: The MCMC step function used to update the particles.
   :param mcmc_init_fn: The MCMC init function used to build a MCMC state from a particle position.
   :param mcmc_parameters: The parameters of the MCMC step function.
   :param resampling_fn: The function used to resample the particles.
   :param target_ess: The number of effective sample size to aim for at each step.
   :param root_solver: The solver used to adaptively compute the temperature given a target number
                       of effective samples.
   :param num_mcmc_steps: The number of times the MCMC kernel is applied to the particles per step.

   :rtype: A ``SamplingAlgorithm``.

   .. py:attribute:: init

      

   .. py:attribute:: build_kernel

      


