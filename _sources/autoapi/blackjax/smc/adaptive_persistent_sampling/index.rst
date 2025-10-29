blackjax.smc.adaptive_persistent_sampling
=========================================

.. py:module:: blackjax.smc.adaptive_persistent_sampling


Attributes
----------

.. autoapisummary::

   blackjax.smc.adaptive_persistent_sampling.init


Functions
---------

.. autoapisummary::

   blackjax.smc.adaptive_persistent_sampling.build_kernel
   blackjax.smc.adaptive_persistent_sampling.as_top_level_api


Module Contents
---------------

.. py:function:: build_kernel(logprior_fn: Callable, loglikelihood_fn: Callable, mcmc_step_fn: Callable, mcmc_init_fn: Callable, resampling_fn: Callable, target_ess: float | blackjax.types.Array, update_strategy: Callable = update_and_take_last, root_solver: Callable = solver.dichotomy) -> Callable

   Build an adaptive Persistent Sampling kernel, with signature
   (rng_key,
   state,
   num_mcmc_steps,
   mcmc_parameters,) -> (new_state, info).

   The function implements the Persistent Sampling algorithm as described
   in Karamanis et al. (2025), with an adaptive tempering schedule. See
   blackjax.smc.persistent_sampling.build_kernel for more details.

   :param logprior_fn: Log prior probability function.
                       NOTE: This function must be normalized (:math:`Z_0 = 1`), in order
                       for the weighting scheme to function correctly.
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
   :param target_ess: Target effective sample size (ESS) to determine the next tempering
                      parameter.
                      NOTE: In persistent sampling, the ESS is computed over all particles
                      from all previous iterations and can be > 1.
   :type target_ess: float | Array
   :param update_strategy: Strategy to update particles using MCMC kernels, by default
                           'update_and_take_last' from blackjax.smc.base. The function signature must be
                           (mcmc_init_fn,
                           loggerposterior_fn,
                           mcmc_step_fn,
                           num_mcmc_steps,
                           n_particles,) -> (mcmc_kernel, n_particles), like 'update_and_take_last'.
                           The mcmc_kernel must have signature
                           (rng_key, position, mcmc_parameters) -> (new_position, info).
   :type update_strategy: Callable
   :param root_solver: The solver used to adaptively compute the temperature given a target number
                       of effective samples. By default, blackjax.smc.solver.dichotomy.

   :returns: **kernel** -- A callable that takes a rng_key, a PersistentSMCState, and a dictionary of
             mcmc_parameters, and that returns a the PersistentSMCState after
             the step along with information about the transition.
   :rtype: Callable


.. py:data:: init

.. py:function:: as_top_level_api(logprior_fn: Callable, loglikelihood_fn: Callable, max_iterations: int | blackjax.types.Array, mcmc_step_fn: Callable, mcmc_init_fn: Callable, mcmc_parameters: dict, resampling_fn: Callable, target_ess: float | blackjax.types.Array = 3, num_mcmc_steps: int = 10, update_strategy: Callable = update_and_take_last, root_solver: Callable = solver.dichotomy) -> blackjax.base.SamplingAlgorithm

   Implements the user interface for the adaptive Persistent Sampling
   kernel from Karamanis et al. 2025. See build_kernel and
   blackjax.smc.persistent_sampling for more details.

   NOTE: For this algorithm, we need to keep track of all particles
   from all previous iterations. Since the number of tempering steps (and
   therefore the number of particles) is not known in advance, we need to
   define a maximum number of iterations (max_iterations). The inference
   loop should be written in such a way that it breaks if this maximum
   number of iterations is exceeded, even if the algorithm has not yet
   converged to the final posterior (lambda = 1). There is no internal
   check for this.

   Also note that the arrays are preallocated to their maximum size, so
   higher max_iterations will lead to higher memory usage.

   :param logprior_fn: The log-prior function of the model we wish to draw samples from.
                       NOTE: This function must be normalized (:math:`Z_0 = 1`), in order
                       for the weighting scheme to function correctly.
   :type logprior_fn: Callable
   :param loglikelihood_fn: The log-likelihood function of the model we wish to draw samples from.
   :type loglikelihood_fn: Callable
   :param max_iterations: The maximum number of iterations (tempering steps) to perform.
   :type max_iterations: int | Array
   :param mcmc_step_fn: The MCMC step function used to update the particles.
   :type mcmc_step_fn: Callable
   :param mcmc_init_fn: The MCMC initialization function used to initialize the MCMC state
                        from a position.
   :type mcmc_init_fn: Callable
   :param mcmc_parameters: The parameters for the MCMC kernel.
   :type mcmc_parameters: dict
   :param resampling_fn: Resampling function (from blackjax.smc.resampling).
   :type resampling_fn: Callable
   :param target_ess: Target effective sample size (ESS) to determine the next tempering
                      parameter, by default 3.
                      NOTE: In persistent sampling, the ESS is computed over all
                      particles from all previous iterations and can be > 1.
   :type target_ess: float | Array, optional
   :param num_mcmc_steps: Number of MCMC steps to apply to each particle at each iteration,
                          by default 10.
   :type num_mcmc_steps: int, optional
   :param update_strategy: The strategy to update particles using MCMC kernels, by default
                           'update_and_take_last' from blackjax.smc.base. See build_kernel for
                           details.
   :type update_strategy: Callable, optional
   :param root_solver: The solver used to adaptively compute the temperature given a target
                       number of effective samples. By default, blackjax.smc.solver.dichotomy.
   :type root_solver: Callable, optional

   :returns: A ``SamplingAlgorithm`` instance with init and step methods. See
             blackjax.base.SamplingAlgorithm for details.
             The init method has signature
             (position: ArrayLikeTree) -> PersistentSMCState
             The step method has signature
             (rng_key: PRNGKey, state: PersistentSMCState, lmbda: float | Array) ->
             (new_state: PersistentSMCState, info: PersistentStateInfo)
   :rtype: SamplingAlgorithm


