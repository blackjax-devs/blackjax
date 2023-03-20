:py:mod:`blackjax.smc.tempered`
===============================

.. py:module:: blackjax.smc.tempered


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   blackjax.smc.tempered.TemperedSMCState



Functions
~~~~~~~~~

.. autoapisummary::

   blackjax.smc.tempered.init
   blackjax.smc.tempered.kernel



.. py:class:: TemperedSMCState



   Current state for the tempered SMC algorithm.

   particles: PyTree
       The particles' positions.
   lmbda: float
       Current value of the tempering parameter.


   .. py:attribute:: particles
      :type: blackjax.types.PyTree

      

   .. py:attribute:: weights
      :type: jax.Array

      

   .. py:attribute:: lmbda
      :type: float

      


.. py:function:: init(particles: blackjax.types.PyTree)


.. py:function:: kernel(logprior_fn: Callable, loglikelihood_fn: Callable, mcmc_step_fn: Callable, mcmc_init_fn: Callable, resampling_fn: Callable) -> Callable

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


