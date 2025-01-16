blackjax.smc.from_mcmc
======================

.. py:module:: blackjax.smc.from_mcmc


Functions
---------

.. autoapisummary::

   blackjax.smc.from_mcmc.unshared_parameters_and_step_fn
   blackjax.smc.from_mcmc.build_kernel


Module Contents
---------------

.. py:function:: unshared_parameters_and_step_fn(mcmc_parameters, mcmc_step_fn)

   Splits MCMC parameters into two dictionaries. The shared dictionary
   represents the parameters common to all chains, and the unshared are
   different per chain.
   Binds the step fn using the shared parameters.


.. py:function:: build_kernel(mcmc_step_fn: Callable, mcmc_init_fn: Callable, resampling_fn: Callable, update_strategy: Callable = update_and_take_last)

   SMC step from MCMC kernels.
   Builds MCMC kernels from the input parameters, which may change across iterations.
   Moreover, it defines the way such kernels are used to update the particles. This layer
   adapts an API defined in terms of kernels (mcmc_step_fn and mcmc_init_fn) into an API
   that depends on an update function over the set of particles.
   :returns: * *A callable that takes a rng_key and a state with .particles and .weights and returns a base.SMCState*
             * *and base.SMCInfo pair.*


