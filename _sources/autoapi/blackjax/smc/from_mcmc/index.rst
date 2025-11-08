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

.. py:function:: unshared_parameters_and_step_fn(mcmc_parameters: dict, mcmc_step_fn: Callable) -> tuple[dict, Callable]

   Split MCMC parameters into shared and unshared parameters.

   The shared dictionary represents the parameters common to all chains, and
   the unshared are different per chain. Binds the step function using the
   shared parameters.

   :param mcmc_parameters: Dictionary of MCMC parameters. Parameters with shape[0] == 1 are
                           considered shared across all chains.
   :type mcmc_parameters: dict
   :param mcmc_step_fn: MCMC step function.
   :type mcmc_step_fn: Callable

   :returns: * **unshared_mcmc_parameters** (*dict*) -- Parameters that differ per chain.
             * **shared_mcmc_step_fn** (*Callable*) -- MCMC step function with shared parameters bound.


.. py:function:: build_kernel(mcmc_step_fn: Callable, mcmc_init_fn: Callable, resampling_fn: Callable, update_strategy: Callable = update_and_take_last) -> Callable

   Build an SMC step function from MCMC kernels.

   Builds MCMC kernels from the input parameters, which may change across iterations.
   Moreover, it defines the way such kernels are used to update the particles. This
   layer adapts an API defined in terms of kernels (mcmc_step_fn and mcmc_init_fn)
   into an API that depends on an update function over the set of particles.

   :param mcmc_step_fn: MCMC step function.
   :type mcmc_step_fn: Callable
   :param mcmc_init_fn: Function that initializes an MCMC state from a position.
   :type mcmc_init_fn: Callable
   :param resampling_fn: Resampling function (from blackjax.smc.resampling).
   :type resampling_fn: Callable
   :param update_strategy: Strategy to update particles using MCMC kernels, by default
                           'update_and_take_last' from blackjax.smc.base.
   :type update_strategy: Callable

   :returns: **step** -- A callable that takes a rng_key and a state with .particles and .weights
             and returns a base.SMCState and base.SMCInfo pair.
   :rtype: Callable


