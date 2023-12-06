:py:mod:`blackjax.smc.inner_kernel_tuning`
==========================================

.. py:module:: blackjax.smc.inner_kernel_tuning


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   blackjax.smc.inner_kernel_tuning.StateWithParameterOverride
   blackjax.smc.inner_kernel_tuning.inner_kernel_tuning



Functions
~~~~~~~~~

.. autoapisummary::

   blackjax.smc.inner_kernel_tuning.init
   blackjax.smc.inner_kernel_tuning.build_kernel



.. py:class:: StateWithParameterOverride




   .. py:attribute:: sampler_state
      :type: blackjax.types.ArrayTree

      

   .. py:attribute:: parameter_override
      :type: blackjax.types.ArrayTree

      


.. py:function:: init(alg_init_fn, position, initial_parameter_value)


.. py:function:: build_kernel(smc_algorithm, logprior_fn: Callable, loglikelihood_fn: Callable, mcmc_factory: Callable, mcmc_init_fn: Callable, mcmc_parameters: Dict, resampling_fn: Callable, mcmc_parameter_update_fn: Callable[[blackjax.smc.base.SMCState, blackjax.smc.base.SMCInfo], blackjax.types.ArrayTree], num_mcmc_steps: int = 10, **extra_parameters) -> Callable

   In the context of an SMC sampler (whose step_fn returning state has a .particles attribute), there's an inner
   MCMC that is used to perturbate/update each of the particles. This adaptation tunes some parameter of that MCMC,
   based on particles. The parameter type must be a valid JAX type.

   :param smc_algorithm: Either blackjax.adaptive_tempered_smc or blackjax.tempered_smc (or any other implementation of
                         a sampling algorithm that returns an SMCState and SMCInfo pair).
   :param logprior_fn: A function that computes the log density of the prior distribution
   :param loglikelihood_fn: A function that returns the probability at a given position.
   :param mcmc_factory: A callable that can construct an inner kernel out of the newly-computed parameter
   :param mcmc_init_fn: A callable that initializes the inner kernel
   :param mcmc_parameters: Other (fixed across SMC iterations) parameters for the inner kernel
   :param mcmc_parameter_update_fn: A callable that takes the SMCState and SMCInfo at step i and constructs a parameter to be used by the inner kernel in i+1 iteration.
   :param extra_parameters: parameters to be used for the creation of the smc_algorithm.


.. py:class:: inner_kernel_tuning


   In the context of an SMC sampler (whose step_fn returning state
   has a .particles attribute), there's an inner MCMC that is used
   to perturbate/update each of the particles. This adaptation tunes some
   parameter of that MCMC, based on particles.
   The parameter type must be a valid JAX type.

   :param smc_algorithm: Either blackjax.adaptive_tempered_smc or blackjax.tempered_smc (or any other implementation of
                         a sampling algorithm that returns an SMCState and SMCInfo pair).
   :param logprior_fn: A function that computes the log density of the prior distribution
   :param loglikelihood_fn: A function that returns the probability at a given position.
   :param mcmc_factory: A callable that can construct an inner kernel out of the newly-computed parameter
   :param mcmc_init_fn: A callable that initializes the inner kernel
   :param mcmc_parameters: Other (fixed across SMC iterations) parameters for the inner kernel step
   :param mcmc_parameter_update_fn: A callable that takes the SMCState and SMCInfo at step i and constructs a parameter to be used by the
                                    inner kernel in i+1 iteration.
   :param initial_parameter_value: Paramter to be used by the mcmc_factory before the first iteration.
   :param extra_parameters: parameters to be used for the creation of the smc_algorithm.

   :rtype: A ``SamplingAlgorithm``.

   .. py:attribute:: init

      

   .. py:attribute:: build_kernel

      


