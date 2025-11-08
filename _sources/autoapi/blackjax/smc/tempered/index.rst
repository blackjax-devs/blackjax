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

   :param particles: The particles' positions.
   :type particles: ArrayLikeTree
   :param weights: Normalized weights for the particles.
   :type weights: Array
   :param tempering_param: Current value of the tempering parameter.
   :type tempering_param: float | Array


   .. py:attribute:: particles
      :type:  blackjax.types.ArrayLikeTree


   .. py:attribute:: weights
      :type:  blackjax.types.Array


   .. py:attribute:: tempering_param
      :type:  float | blackjax.types.Array


.. py:function:: init(particles: blackjax.types.ArrayLikeTree) -> TemperedSMCState

   Initialize the Tempered SMC state.

   :param particles: Initial N particles (typically sampled from prior).
   :type particles: ArrayLikeTree

   :returns: Initial state with uniform weights and tempering_param set to 0.0.
   :rtype: TemperedSMCState


.. py:function:: build_kernel(logprior_fn: Callable, loglikelihood_fn: Callable, mcmc_step_fn: Callable, mcmc_init_fn: Callable, resampling_fn: Callable, update_strategy: Callable = update_and_take_last, update_particles_fn: Optional[Callable] = None) -> Callable

   Build the base Tempered SMC kernel.

   Tempered SMC uses tempering to sample from a distribution given by

   .. math::
       p(x) \propto p_0(x) \exp(-V(x)) \mathrm{d}x

   where :math:`p_0` is the prior distribution, typically easy to sample from
   and for which the density is easy to compute, and :math:`\exp(-V(x))` is an
   unnormalized likelihood term for which :math:`V(x)` is easy to compute
   pointwise.

   :param logprior_fn: Log prior probability function.
   :type logprior_fn: Callable
   :param loglikelihood_fn: Log likelihood function.
   :type loglikelihood_fn: Callable
   :param mcmc_step_fn: Function that creates MCMC step from log-probability density function.
   :type mcmc_step_fn: Callable
   :param mcmc_init_fn: A function that creates a new mcmc state from a position and a
                        log-probability density function.
   :type mcmc_init_fn: Callable
   :param resampling_fn: Resampling function (from blackjax.smc.resampling).
   :type resampling_fn: Callable
   :param update_strategy: Strategy to update particles using MCMC kernels, by default
                           'update_and_take_last' from blackjax.smc.base.
   :type update_strategy: Callable
   :param update_particles_fn: Optional custom function to update particles. If None, uses
                               smc_from_mcmc.build_kernel.
   :type update_particles_fn: Callable, optional

   :returns: **kernel** -- A callable that takes a rng_key, a TemperedSMCState, num_mcmc_steps,
             tempering_param, and mcmc_parameters, and returns a new
             TemperedSMCState along with information about the transition.
   :rtype: Callable


.. py:function:: as_top_level_api(logprior_fn: Callable, loglikelihood_fn: Callable, mcmc_step_fn: Callable, mcmc_init_fn: Callable, mcmc_parameters: dict, resampling_fn: Callable, num_mcmc_steps: Optional[int] = 10, update_strategy: Callable = update_and_take_last, update_particles_fn: Optional[Callable] = None) -> blackjax.base.SamplingAlgorithm

   Implements the user interface for the Tempered SMC kernel.

   :param logprior_fn: The log-prior function of the model we wish to draw samples from.
   :type logprior_fn: Callable
   :param loglikelihood_fn: The log-likelihood function of the model we wish to draw samples from.
   :type loglikelihood_fn: Callable
   :param mcmc_step_fn: The MCMC step function used to update the particles.
   :type mcmc_step_fn: Callable
   :param mcmc_init_fn: The MCMC init function used to build a MCMC state from a particle position.
   :type mcmc_init_fn: Callable
   :param mcmc_parameters: The parameters of the MCMC step function. Parameters with leading dimension
                           length of 1 are shared amongst the particles.
   :type mcmc_parameters: dict
   :param resampling_fn: The function used to resample the particles.
   :type resampling_fn: Callable
   :param num_mcmc_steps: The number of times the MCMC kernel is applied to the particles per step,
                          by default 10.
   :type num_mcmc_steps: int, optional
   :param update_strategy: Strategy to update particles using MCMC kernels, by default
                           'update_and_take_last' from blackjax.smc.base.
   :type update_strategy: Callable, optional
   :param update_particles_fn: Optional custom function to update particles. If None, uses
                               smc_from_mcmc.build_kernel.
   :type update_particles_fn: Callable, optional

   :returns: A ``SamplingAlgorithm`` instance with init and step methods.
   :rtype: SamplingAlgorithm


