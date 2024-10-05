blackjax.smc.partial_posteriors_path
====================================

.. py:module:: blackjax.smc.partial_posteriors_path


Classes
-------

.. autoapisummary::

   blackjax.smc.partial_posteriors_path.PartialPosteriorsSMCState


Functions
---------

.. autoapisummary::

   blackjax.smc.partial_posteriors_path.init
   blackjax.smc.partial_posteriors_path.build_kernel
   blackjax.smc.partial_posteriors_path.as_top_level_api


Module Contents
---------------

.. py:class:: PartialPosteriorsSMCState



   Current state for the tempered SMC algorithm.

   particles: PyTree
       The particles' positions.
   weights:
       Weights of the particles, so that they represent a probability distribution
   data_mask:
       A 1D boolean array to indicate which datapoints to include
       in the computation of the observed likelihood.


   .. py:attribute:: particles
      :type:  blackjax.types.ArrayTree


   .. py:attribute:: weights
      :type:  blackjax.types.Array


   .. py:attribute:: data_mask
      :type:  blackjax.types.Array


.. py:function:: init(particles: blackjax.types.ArrayLikeTree, num_datapoints: int) -> PartialPosteriorsSMCState

   num_datapoints are the number of observations that could potentially be
   used in a partial posterior. Since the initial data_mask is all 0s, it
   means that no likelihood term will be added (only prior).


.. py:function:: build_kernel(mcmc_step_fn: Callable, mcmc_init_fn: Callable, resampling_fn: Callable, num_mcmc_steps: Optional[int], mcmc_parameters: blackjax.types.ArrayTree, partial_logposterior_factory: Callable[[blackjax.types.Array], Callable], update_strategy=update_and_take_last) -> Callable

   Build the Partial Posteriors (data tempering) SMC kernel.
   The distribution's trajectory includes increasingly adding more
   datapoints to the likelihood. See Section 2.2 of https://arxiv.org/pdf/2007.11936
   :param mcmc_step_fn: A function that computes the log density of the prior distribution
   :param mcmc_init_fn: A function that returns the probability at a given position.
   :param resampling_fn: A random function that resamples generated particles based of weights
   :param num_mcmc_steps: Number of iterations in the MCMC chain.
   :param mcmc_parameters: A dictionary of parameters to be used by the inner MCMC kernels
   :param partial_logposterior_factory: A callable that given an array of 0 and 1, returns a function logposterior(x).
                                        The array represents which values to include in the logposterior calculation. The logposterior
                                        must be jax compilable.

   :returns: * *A callable that takes a rng_key and PartialPosteriorsSMCState and selectors for*
             * *the current and previous posteriors, and takes a data-tempered SMC state.*


.. py:function:: as_top_level_api(mcmc_step_fn: Callable, mcmc_init_fn: Callable, mcmc_parameters: dict, resampling_fn: Callable, num_mcmc_steps, partial_logposterior_factory: Callable, update_strategy=update_and_take_last) -> blackjax.SamplingAlgorithm

   A factory that wraps the kernel into a SamplingAlgorithm object.
   See build_kernel for full documentation on the parameters.


