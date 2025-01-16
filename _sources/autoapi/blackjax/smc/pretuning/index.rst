blackjax.smc.pretuning
======================

.. py:module:: blackjax.smc.pretuning


Classes
-------

.. autoapisummary::

   blackjax.smc.pretuning.SMCInfoWithParameterDistribution


Functions
---------

.. autoapisummary::

   blackjax.smc.pretuning.esjd
   blackjax.smc.pretuning.update_parameter_distribution
   blackjax.smc.pretuning.build_pretune
   blackjax.smc.pretuning.build_kernel
   blackjax.smc.pretuning.init
   blackjax.smc.pretuning.as_top_level_api


Module Contents
---------------

.. py:class:: SMCInfoWithParameterDistribution



   Stores both the sampling status and also a dictionary
   with parameter names as keys and (n_particles, *) arrays as values.
   The latter represents a parameter per chain for the next mutation step.


   .. py:attribute:: smc_info
      :type:  blackjax.smc.base.SMCInfo


   .. py:attribute:: parameter_override
      :type:  Dict[str, blackjax.types.ArrayTree]


.. py:function:: esjd(m)

   Implements ESJD (expected squared jumping distance). Inner Mahalanobis distance
   is computed using the Cholesky decomposition of M=LLt, and then inverting L.
   Whenever M is symmetrical definite positive then it must exist a Cholesky Decomposition.
   For example, if M is the Covariance Matrix of Metropolis-Hastings or
   the Inverse Mass Matrix of Hamiltonian Monte Carlo.


.. py:function:: update_parameter_distribution(key: blackjax.types.PRNGKey, previous_param_samples: blackjax.types.ArrayLikeTree, previous_particles: blackjax.types.ArrayLikeTree, latest_particles: blackjax.types.ArrayLikeTree, measure_of_chain_mixing: Callable, alpha: float, sigma_parameters: blackjax.types.ArrayLikeTree, acceptance_probability: blackjax.types.Array)

   Given an existing parameter distribution that was used to mutate previous_particles
   into latest_particles, updates that parameter distribution by resampling from previous_param_samples after adding
   noise to those samples. The weights used are a linear function of the measure of chain mixing.
   Only works with float parameters, not integers.
   See Equation 4 in https://arxiv.org/pdf/1005.1193.pdf

   :param previous_param_samples: samples of the parameters of SMC inner MCMC chains. To be updated.
   :param previous_particles: particles from which the kernel step started
   :param latest_particles: particles after the step was performed
   :param measure_of_chain_mixing: a callable that can compute a performance measure per chain
   :type measure_of_chain_mixing: Callable
   :param alpha: a scalar to add to the weighting. See paper for details
   :param sigma_parameters: noise to add to the population of parameters to mutate them. must have the same shape of
                            previous_param_samples.
   :param acceptance_probability: the energy difference for each of the chains when taking a step from previous_particles
                                  into latest_particles.


.. py:function:: build_pretune(mcmc_init_fn: Callable, mcmc_step_fn: Callable, alpha: float, sigma_parameters: blackjax.types.ArrayLikeTree, n_particles: int, performance_of_chain_measure_factory: Callable = lambda state: esjd(state.parameter_override['inverse_mass_matrix']), natural_parameters: Optional[List[str]] = None, positive_parameters: Optional[List[str]] = None)

   Implements Buchholz et al https://arxiv.org/pdf/1808.07730 pretuning procedure.
   The goal is to maintain a probability distribution of parameters, in order
   to assign different values to each inner MCMC chain.
   To have performant parameters for the distribution at step t, it takes a single step, measures
   the chain mixing, and reweights the probability distribution of parameters accordingly.
   Note that although similar, this strategy is different than inner_kernel_tuning. The latter updates
   the parameters based on the particles and transition information after the SMC step is executed. This
   implementation runs a single MCMC step which gets discarded, to then proceed with the SMC step execution.


.. py:function:: build_kernel(smc_algorithm, logprior_fn: Callable, loglikelihood_fn: Callable, mcmc_step_fn: Callable, mcmc_init_fn: Callable, resampling_fn: Callable, pretune_fn: Callable, num_mcmc_steps: int = 10, update_strategy=update_and_take_last, **extra_parameters) -> Callable

   In the context of an SMC sampler (whose step_fn returning state has a .particles attribute), there's an inner
   MCMC that is used to perturbate/update each of the particles. This adaptation tunes some parameter of that MCMC,
   based on particles. The parameter type must be a valid JAX type.

   :param smc_algorithm: Either blackjax.adaptive_tempered_smc or blackjax.tempered_smc (or any other implementation of
                         a sampling algorithm that returns an SMCState and SMCInfo pair).
   :param logprior_fn: A function that computes the log density of the prior distribution
   :param loglikelihood_fn: A function that returns the probability at a given position.
   :param mcmc_step_fn: The transition kernel, should take as parameters the dictionary output of mcmc_parameter_update_fn.
                        mcmc_step_fn(rng_key, state, tempered_logposterior_fn, **mcmc_parameter_update_fn())
   :param mcmc_init_fn: A callable that initializes the inner kernel
   :param pretune_fn: A callable that can update the probability distribution of parameters.
   :param extra_parameters: parameters to be used for the creation of the smc_algorithm.


.. py:function:: init(alg_init_fn, position, initial_parameter_value)

.. py:function:: as_top_level_api(smc_algorithm, logprior_fn: Callable, loglikelihood_fn: Callable, mcmc_step_fn: Callable, mcmc_init_fn: Callable, resampling_fn: Callable, num_mcmc_steps: int, initial_parameter_value: blackjax.types.ArrayLikeTree, pretune_fn: Callable, **extra_parameters)

   In the context of an SMC sampler (whose step_fn returning state has a .particles attribute), there's an inner
   MCMC that is used to perturbate/update each of the particles. This adaptation tunes some parameter of that MCMC,
   based on particles. The parameter type must be a valid JAX type.

   :param smc_algorithm: Either blackjax.adaptive_tempered_smc or blackjax.tempered_smc (or any other implementation of
                         a sampling algorithm that returns an SMCState and SMCInfo pair).
   :param logprior_fn: A function that computes the log density of the prior distribution
   :param loglikelihood_fn: A function that returns the probability at a given position.
   :param mcmc_step_fn: The transition kernel, should take as parameters the dictionary output of mcmc_parameter_update_fn.
                        mcmc_step_fn(rng_key, state, tempered_logposterior_fn, **mcmc_parameter_update_fn())
   :param mcmc_init_fn: A callable that initializes the inner kernel
   :param pretune_fn: A callable that can update the probability distribution of parameters.
   :param extra_parameters: parameters to be used for the creation of the smc_algorithm.


