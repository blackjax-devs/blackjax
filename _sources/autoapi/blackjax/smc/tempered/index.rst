blackjax.smc.tempered
=====================

.. py:module:: blackjax.smc.tempered


Classes
-------

.. autoapisummary::

   blackjax.smc.tempered.TemperedSMCState


Functions
---------

.. autoapisummary::

   blackjax.smc.tempered.init
   blackjax.smc.tempered.build_kernel
   blackjax.smc.tempered.as_top_level_api


Module Contents
---------------

.. py:class:: TemperedSMCState



   Current state for the tempered SMC algorithm.

   particles: PyTree
       The particles' positions.
   lmbda: float
       Current value of the tempering parameter.



   .. py:attribute:: particles
      :type:  blackjax.types.ArrayTree


   .. py:attribute:: weights
      :type:  blackjax.types.Array


   .. py:attribute:: lmbda
      :type:  float


.. py:function:: init(particles: blackjax.types.ArrayLikeTree)

.. py:function:: build_kernel(logprior_fn: Callable, loglikelihood_fn: Callable, mcmc_step_fn: Callable, mcmc_init_fn: Callable, resampling_fn: Callable, update_strategy: Callable = update_and_take_last, update_particles_fn: Optional[Callable] = None) -> Callable

   Build the base Tempered SMC kernel.

   Tempered SMC uses tempering to sample from a distribution given by

   .. math::
       p(x) \propto p_0(x) \exp(-V(x)) \mathrm{d}x

   where :math:`p_0` is the prior distribution, typically easy to sample from
   and for which the density is easy to compute, and :math:`\exp(-V(x))` is an
   unnormalized likelihood term for which :math:`V(x)` is easy to compute
   pointwise.

   :param logprior_fn: A function that computes the log density of the prior distribution
   :param loglikelihood_fn: A function that returns the probability at a given
                            position.
   :param mcmc_step_fn: A function that creates a mcmc kernel from a log-probability density function.
   :param mcmc_init_fn: A function that creates a new mcmc state from a position and a
                        log-probability density function.
   :type mcmc_init_fn: Callable
   :param resampling_fn: A random function that resamples generated particles based of weights
   :param num_mcmc_iterations: Number of iterations in the MCMC chain.

   :returns: * *A callable that takes a rng_key and a TemperedSMCState that contains the current state*
             * *of the chain and that returns a new state of the chain along with*
             * *information about the transition.*


.. py:function:: as_top_level_api(logprior_fn: Callable, loglikelihood_fn: Callable, mcmc_step_fn: Callable, mcmc_init_fn: Callable, mcmc_parameters: dict, resampling_fn: Callable, num_mcmc_steps: Optional[int] = 10, update_strategy=update_and_take_last, update_particles_fn=None) -> blackjax.base.SamplingAlgorithm

   Implements the (basic) user interface for the Adaptive Tempered SMC kernel.

   :param logprior_fn: The log-prior function of the model we wish to draw samples from.
   :param loglikelihood_fn: The log-likelihood function of the model we wish to draw samples from.
   :param mcmc_step_fn: The MCMC step function used to update the particles.
   :param mcmc_init_fn: The MCMC init function used to build a MCMC state from a particle position.
   :param mcmc_parameters: The parameters of the MCMC step function.  Parameters with leading dimension
                           length of 1 are shared amongst the particles.
   :param resampling_fn: The function used to resample the particles.
   :param num_mcmc_steps: The number of times the MCMC kernel is applied to the particles per step.

   :rtype: A ``SamplingAlgorithm``.


