blackjax.smc.persistent_sampling
================================

.. py:module:: blackjax.smc.persistent_sampling


Classes
-------

.. autoapisummary::

   blackjax.smc.persistent_sampling.PersistentSMCState
   blackjax.smc.persistent_sampling.PersistentStateInfo


Functions
---------

.. autoapisummary::

   blackjax.smc.persistent_sampling.init
   blackjax.smc.persistent_sampling.remove_padding
   blackjax.smc.persistent_sampling.compute_log_Z
   blackjax.smc.persistent_sampling.compute_log_persistent_weights
   blackjax.smc.persistent_sampling.resample_from_persistent
   blackjax.smc.persistent_sampling.compute_persistent_ess
   blackjax.smc.persistent_sampling.step
   blackjax.smc.persistent_sampling.build_kernel
   blackjax.smc.persistent_sampling.as_top_level_api


Module Contents
---------------

.. py:class:: PersistentSMCState



   State of the Persistent Sampling algorithm.

   Contains all particles from all iterations, their weights,
   log-likelihoods, log normalizing constants, tempering parameters and an
   index for the current iteration.
   Particles of the current iteration can be accessed via the `particles` property
   for convenience.

   NOTE: All arrays should be padded with zeros up the length of the
   tempering schedule + 1. This is to allow JIT compilation.

   :param persistent_particles: Particles from all iterations (padded with zeros to expected length of
                                tempering schedule + 1).
   :type persistent_particles: ArrayLikeTree
   :param persistent_log_likelihoods: Log-likelihoods for all persistent particles, updated for current iteration.
                                      Shape is (n_schedule + 1, n_particles).
   :type persistent_log_likelihoods: Array
   :param persistent_log_Z: History of (log of) normalizing constants :math:`[log(Z_0), \ldots, log(Z_t)]`,
                            zero-padded for all iterations.
   :type persistent_log_Z: Array
   :param tempering_schedule: History of tempering parameters :math:`[\lambda_0, \ldots, \lambda_t]`,
                              zero-padded.
   :type tempering_schedule: Array
   :param iteration: Current iteration index.
   :type iteration: Array
   :param Derived Properties:
   :param ------------------:
   :param particles: Particles in current iteration (i.e. at index `iteration`).
   :type particles: ArrayLikeTree
   :param tempering_param: Tempering parameter in current iteration.
   :type tempering_param: float | Array
   :param log_Z: Log normalizing constant in current iteration.
   :type log_Z: float | Array
   :param persistent_weights: Normalized weights for all persistent particles, updated for current iteration.
                              Shape is (n_schedule + 1, n_particles), where n_schedule is the number of
                              tempering steps. Normalized such that they sum to iteration * n_particles.
                              Calculated using persistent_log_likelihoods, persistent_log_Z,
                              tempering_schedule, and iteration.
                              NOTE: The weights are calculated on-the-fly, rather than than stored during
                              the sampling process, since the weights in the current iteration depend on
                              the particles sampled at that iteration, while in the algorithm the weights are
                              calculated before sampling the new particles.
   :type persistent_weights: Array
   :param num_particles: Number of particles.
   :type num_particles: int


   .. py:attribute:: persistent_particles
      :type:  blackjax.types.ArrayLikeTree


   .. py:attribute:: persistent_log_likelihoods
      :type:  blackjax.types.Array


   .. py:attribute:: persistent_log_Z
      :type:  blackjax.types.Array


   .. py:attribute:: tempering_schedule
      :type:  blackjax.types.Array


   .. py:attribute:: iteration
      :type:  int | blackjax.types.Array


   .. py:property:: particles
      :type: blackjax.types.ArrayLikeTree


      Particles in current iteration.


   .. py:property:: tempering_param
      :type: float | blackjax.types.Array


      Tempering parameter in current iteration.


   .. py:property:: log_Z
      :type: float | blackjax.types.Array


      Log normalizing constant in current iteration.


   .. py:property:: persistent_weights
      :type: blackjax.types.Array


      Weights for all persistent particles in current iteration, normalized
      to sum to iteration * n_particles.


   .. py:property:: num_particles
      :type: int


      Number of particles.


.. py:class:: PersistentStateInfo



   Information from one step of Persistent Sampling.

   :param ancestors: The index of the particles selected by the resampling step.
   :type ancestors: Array
   :param update_info: Additional information returned by the update function.
   :type update_info: NamedTuple


   .. py:attribute:: ancestors
      :type:  blackjax.types.Array


   .. py:attribute:: update_info
      :type:  NamedTuple


.. py:function:: init(particles: blackjax.types.ArrayLikeTree, loglikelihood_fn: Callable, n_schedule: int | blackjax.types.Array) -> PersistentSMCState

   Initialize the Persistent Sampling state.

   The arrays are padded with zeros to alow for JIT compilation.
   The dimension of the arrays is (n_schedule + 1, n_particles),
   where n_schedule is the number of tempering steps. The + 1 is to
   account for the initial prior distribution at iteration 0.

   :param particles: Initial N particles (typically sampled from prior).
   :type particles: PyTree
   :param loglikelihood_fn: Log likelihood function.
   :type loglikelihood_fn: Callable
   :param n_schedule: Number of steps in the tempering schedule.
   :type n_schedule: int | Array

   :returns: Initial state, with
             - particles set to input particles,
             - weights set to uniform weights,
             - log-likelihoods set to the log-likelihoods of the input particles,
             - normalizing constant set to 1.0 (assume prior is normalized, this is
               important),
             - tempering parameters set to 0.0 (initial distribution is prior).
             - set iteration to 0.

             NOTE: All arrays in the PersistentSMCState are padded with zeros up
             to the length of the tempering schedule.
   :rtype: PersistentSMCState


.. py:function:: remove_padding(state: PersistentSMCState) -> PersistentSMCState

   Remove padding from PersistentSMCState arrays up to current iteration.

   :param state: The PersistentSMCState with padded arrays.
   :type state: PersistentSMCState

   :returns: New PersistentSMCState with arrays trimmed to current iteration.
   :rtype: PersistentSMCState


.. py:function:: compute_log_Z(log_weights: blackjax.types.Array, iteration: int | blackjax.types.Array) -> blackjax.types.Array

   Compute log normalizing constant from log weights.

   Implements Equation 16 from the Karamanis2025.

   :param log_weights: Log of unnormalized weights for all persistent particles at current iteration.
   :type log_weights: Array
   :param iteration: Current iteration index.
   :type iteration: int | Array

   :returns: **log_Z** -- Estimate of log of normalizing constant :math:`\hat{Z}_{t}` at current
             iteration.
   :rtype: float | Array


.. py:function:: compute_log_persistent_weights(persistent_log_likelihoods: blackjax.types.Array, persistent_log_Z: blackjax.types.Array, tempering_schedule: blackjax.types.Array, iteration: int | blackjax.types.Array, include_current: bool = False, normalize_to_one: bool = False) -> tuple[blackjax.types.Array, blackjax.types.Array]

   Compute importance weights for all persistent particles for
   current iteration.

   Implements Equations 14 and 15 from the Karamanis2025.

   NOTE: The returned weights are normalized such that they sum to
   :math:`(i \times N)`, where i is the current iteration and N
   is the number of particles. They need to be renormalized to sum to 1.0
   before resampling, this can be done using the 'normalize_to_one' argument.

   :param persistent_log_likelihoods: Log-likelihoods for all persistent particles (for all previous
                                      current iteration).
   :type persistent_log_likelihoods: Array
   :param persistent_log_Z: Log normalizing constants for all previous iterations.
   :type persistent_log_Z: Array
   :param tempering_schedule: Tempering parameters up to current iteration.
   :type tempering_schedule: Array
   :param iteration: Current iteration index.
   :type iteration: int | Array
   :param include_current: If `True`, include the current iteration in the weight computation (i.e.
                           sum to t rather than t-1 in equations 14-16). This is useful when calculating
                           the weights after the resampling step, where the current iteration's particles
                           are already included in the persistent ensemble.
   :type include_current: bool, optional
   :param normalize_to_one: If `True`, normalize the weights to sum to 1.0. By default, the weights
                            sum to (iteration * n_particles), as described in the paper.
   :type normalize_to_one: bool, optional

   :returns: * **normalized_log_weights** (*Array*) -- Log of normalized weights :math:`W^i_{tt'}` for all :math:`i \times N`
               persistent particles at current iteration.
             * **new_log_Z** (*float*) -- Estimate of log of normalizing constant :math:`\hat{Z}_{t}` at current
               iteration.


.. py:function:: resample_from_persistent(rng_key: blackjax.types.PRNGKey, persistent_particles: blackjax.types.ArrayLikeTree, persistent_weights: blackjax.types.Array, resample_fn: Callable) -> tuple[blackjax.types.ArrayTree, blackjax.types.Array]

   Resample N particles from the :math:`i \times N`
   persistent ensemble, where i is the current iteration.

   :param rng_key: JAX random key.
   :type rng_key: PRNGKey
   :param persistent_particles: Historical particles of the i previous iterations.
   :type persistent_particles: ArrayLikeTree
   :param persistent_weights: Normalized weights for all :math:`i \times N` particles.
                              NOTE: The weights need to sum to 1, this is different from
                              the 'normalized' described by equation 14 in Karamanis2025
                              amd computed by _compute_log_persistent_weights. These sum
                              to :math:`(i \times N)`, i.e. the current iteration times
                              the number of particles (the current number of persistent
                              particles in the current iteration).
   :type persistent_weights: Array
   :param resample_fn: Resampling function (from blackjax.smc.resampling)
   :type resample_fn: Callable

   :returns: * **resampled_particles** (*ArrayTree*) -- N particles resampled from persistent ensemble.
             * **resample_idx** (*Array*) -- Indices of the selected particles.


.. py:function:: compute_persistent_ess(log_persistent_weights: blackjax.types.Array, normalize_weights: bool = False) -> float | blackjax.types.Array

   Calculate the effective sample size (ESS) of the persistent
   ensemble. Equation 17 from Karamanis2025.

   NOTE: For the second identity in equation 17 to hold, the
   weights must be normalized to sum to 1.0. This function normalizes
   the weights internally if `normalize_weights` is set to `True`.

   NOTE: The ESS can be > 1 for Persistent Sampling, unlike standard
   SMC.

   :param log_persistent_weights: Normalized log weights for all persistent particles.
   :type log_persistent_weights: Array
   :param normalize_weights: If `True`, normalize the weights to sum to 1.0 before computing
                             the ESS. By default, the weights are assumed to be normalized.
   :type normalize_weights: bool, optional

   :returns: **ess** -- Effective sample size of the persistent ensemble.
   :rtype: float | Array


.. py:function:: step(rng_key: blackjax.types.PRNGKey, state: PersistentSMCState, lmbda: float | blackjax.types.Array, loglikelihood_fn: Callable, update_fn: Callable, resample_fn: Callable, weight_fn: Callable = compute_log_persistent_weights) -> tuple[PersistentSMCState, PersistentStateInfo]

   One step of the Persistent Sampling algorithm, as
   described in algorithm 2 of Karamanis et al. (2025).

   :param rng_key: Key used for random number generation.
   :param state: Current state of the PS sampler described by a PersistentSMCState.
   :param lmbda: New tempering parameter :math:`\lambda_t` for current iteration.
   :type lmbda: float | Array
   :param loglikelihood_fn: Log likelihood function.
   :type loglikelihood_fn: Callable
   :param update_fn: MCMC kernel that takes in an array of keys and particles and returns
                     updated particles along with any extra information.
   :type update_fn: Callable
   :param resample_fn: Resampling function (from blackjax.smc.resampling). This function
                       is passed to _resample_from_persistent to resample from the
                       persistent ensemble.
   :type resample_fn: Callable
   :param weight_fn: Function that assigns a weight to the particles, by default
                     _compute_log_persistent_weights, which implements equation 14-16 from
                     Karamanis2025. Should return normalized log weights and log normalizing
                     constant.

   :returns: * **new_state** (*PersistentSMCState*) -- The updated PersistentSMCState. Updated fields are:
               - particles: particles from all iterations, with current iteration's
                 particles added.
               - weights: normalized weights for all persistent particles at current
                 iteration.
               - log_likelihoods: log-likelihoods for all persistent particles,
                 with current iteration's log-likelihoods added.
               - log_Z: log normalizing constants, with current iteration's
                 normalizing constant added.
               - tempering_schedule: tempering parameters, with current iteration's
                 parameter added.
               - iteration: incremented by 1.
             * **info** (*PersistentStateInfo*) -- An `PersistentStateInfo` object that contains extra information about the PS
               transition. Contains:
               - ancestors: indices of the particles selected by the resampling step.
               - ess: effective sample size of the persistent ensemble.
               - update_info: any extra information returned by the update function.


.. py:function:: build_kernel(logprior_fn: Callable, loglikelihood_fn: Callable, mcmc_step_fn: Callable, mcmc_init_fn: Callable, resampling_fn: Callable, update_strategy: Callable = update_and_take_last) -> Callable

   Build a Persistent Sampling kernel, with signature
   (rng_key,
   state,
   num_mcmc_steps,
   lmbda,
   mcmc_parameters,) -> (new_state, info).

   The function implements the Persistent Sampling algorithm as described
   in Karamanis et al. (2025), with a fixed tempering schedule. It
   functions similarly to tempered SMC (see blackjax.smc.tempered),
   but keeps track of all particles from all previous iterations. This
   can lead to a more stable posterior and marginal likelihood estimation
   at the cost of higher memory usage.

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

   :returns: **kernel** -- A callable that takes a rng_key, a PersistentSMCState, a tempering parameter
             lmbda, and a dictionary of mcmc_parameters, and that returns a the
             PersistentSMCState after the step along with information about the transition.
   :rtype: Callable


.. py:function:: as_top_level_api(logprior_fn: Callable, loglikelihood_fn: Callable, n_schedule: int | blackjax.types.Array, mcmc_step_fn: Callable, mcmc_init_fn: Callable, mcmc_parameters: dict, resampling_fn: Callable, num_mcmc_steps: int = 10, update_strategy: Callable = update_and_take_last) -> blackjax.base.SamplingAlgorithm

   Implements the user interface for the Persistent Sampling
   kernel. See build_kernel for details.

   NOTE: For this algorithm, we need to keep track of all particles
   from all previous iterations. To do so in a JIT-compatible way,
   we need to know the number of tempering steps in advance, to
   preallocate arrays of the correct size. Therefore, the user must
   provide the number of steps in the tempering schedule via the
   `n_schedule` argument.
   Since all arrays are preallocated to (n_schedule + 1, n_particles),
   where the + 1 accounts for the initial value at iteration 0. The user
   must ensure that the tempering schedule used in the actual sampling
   matches n_schedule.
   A tempering schedule with many steps may lead to high memory usage.

   NOTE: The algorithm enforces the tempering schedule to start at 0.0,
   if the supplied schedule also starts at 0.0, the first step will be
   done twice.

   :param logprior_fn: The log-prior function of the model we wish to draw samples from.
                       NOTE: This function must be normalized (:math:`Z_0 = 1`), in order
                       for the weighting scheme to function correctly.
   :type logprior_fn: Callable
   :param loglikelihood_fn: The log-likelihood function of the model we wish to draw samples from.
   :type loglikelihood_fn: Callable
   :param n_schedule: Number of steps in the tempering schedule.
   :type n_schedule: int | Array
   :param mcmc_step_fn: The MCMC step function used to update the particles.
   :type mcmc_step_fn: Callable
   :param mcmc_init_fn: The MCMC initialization function used to initialize the MCMC state
                        from a position.
   :type mcmc_init_fn: Callable
   :param mcmc_parameters: The parameters for the MCMC kernel.
   :type mcmc_parameters: dict
   :param resampling_fn: Resampling function (from blackjax.smc.resampling).
   :type resampling_fn: Callable
   :param num_mcmc_steps: Number of MCMC steps to apply to each particle at each iteration,
                          by default 10.
   :type num_mcmc_steps: int, optional
   :param update_strategy: The strategy to update particles using MCMC kernels, by default
                           'update_and_take_last' from blackjax.smc.base. See build_kernel for
                           details.
   :type update_strategy: Callable, optional

   :returns: A ``SamplingAlgorithm`` instance with init and step methods. See
             blackjax.base.SamplingAlgorithm for details.
             The init method has signature
             (position: ArrayLikeTree) -> PersistentSMCState
             The step method has signature
             (rng_key: PRNGKey, state: PersistentSMCState, lmbda: float | Array) ->
             (new_state: PersistentSMCState, info: PersistentStateInfo)
   :rtype: SamplingAlgorithm


