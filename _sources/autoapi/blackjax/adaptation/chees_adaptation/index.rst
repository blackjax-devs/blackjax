:py:mod:`blackjax.adaptation.chees_adaptation`
==============================================

.. py:module:: blackjax.adaptation.chees_adaptation

.. autoapi-nested-parse::

   Public API for ChEES-HMC



Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   blackjax.adaptation.chees_adaptation.ChEESAdaptationState



Functions
~~~~~~~~~

.. autoapisummary::

   blackjax.adaptation.chees_adaptation.base
   blackjax.adaptation.chees_adaptation.chees_adaptation



Attributes
~~~~~~~~~~

.. autoapisummary::

   blackjax.adaptation.chees_adaptation.OPTIMAL_TARGET_ACCEPTANCE_RATE


.. py:data:: OPTIMAL_TARGET_ACCEPTANCE_RATE
   :value: 0.651

   

.. py:class:: ChEESAdaptationState




   State of the ChEES-HMC adaptation scheme.

   step_size
       Value of the step_size parameter of the HMC algorithm.
   log_step_size_moving_average
       Running moving average of the log step_size parameter.
   trajectory_length
       Value of the num_integration_steps * step_size parameter of
       the HMC algorithm.
   log_trajectory_length_moving_average
       Running moving average of the log num_integration_steps / step_size
       parameter.
   optim_state
       Optax optimizing state for used to maximize the ChEES criterion.
   random_generator_arg
       Utility array for generating a pseudo or quasi-random sequence of
       numbers.
   step
       Current iteration number.


   .. py:attribute:: step_size
      :type: float

      

   .. py:attribute:: log_step_size_moving_average
      :type: float

      

   .. py:attribute:: trajectory_length
      :type: float

      

   .. py:attribute:: log_trajectory_length_moving_average
      :type: float

      

   .. py:attribute:: da_state
      :type: blackjax.optimizers.dual_averaging.DualAveragingState

      

   .. py:attribute:: optim_state
      :type: optax.OptState

      

   .. py:attribute:: random_generator_arg
      :type: blackjax.types.Array

      

   .. py:attribute:: step
      :type: int

      


.. py:function:: base(jitter_generator: Callable, next_random_arg_fn: Callable, optim: optax.GradientTransformation, target_acceptance_rate: float, decay_rate: float) -> Tuple[Callable, Callable]

   Maximizing the Change in the Estimator of the Expected Square criterion
   (trajectory length) and dual averaging procedure (step size) for the jittered
   Hamiltonian Monte Carlo kernel :cite:p:`hoffman2021adaptive`.

   This adaptation algorithm tunes the step size and trajectory length, i.e.
   number of integration steps / step size, of the jittered HMC algorithm based
   on statistics collected from a population of many chains. It maximizes the Change
   in the Estimator of the Expected Square (ChEES) criterion to tune the trajectory
   length and uses dual averaging targeting an acceptance rate of 0.651 of the harmonic
   mean of the chain's acceptance probabilities to tune the step size.

   :param jitter_generator: Optional function that generates a value in [0, 1] used to jitter the trajectory
                            lengths given a PRNGKey, used to propose the number of integration steps. If None,
                            then a quasi-random Halton is used to jitter the trajectory length.
   :param next_random_arg_fn: Function that generates the next `random_generator_arg` from its previous value.
   :param optim: Optax compatible optimizer, which conforms to the `optax.GradientTransformation` protocol.
   :param target_acceptance_rate: Average acceptance rate to target with dual averaging.
   :param decay_rate: Float representing how much to favor recent iterations over earlier ones in the optimization
                      of step size and trajectory length.

   :returns: * *init* -- Function that initializes the warmup.
             * *update* -- Function that moves the warmup one step.


.. py:function:: chees_adaptation(logdensity_fn: Callable, num_chains: int, *, jitter_generator: Optional[Callable] = None, jitter_amount: float = 1.0, target_acceptance_rate: float = OPTIMAL_TARGET_ACCEPTANCE_RATE, decay_rate: float = 0.5) -> blackjax.base.AdaptationAlgorithm

   Adapt the step size and trajectory length (number of integration steps / step size)
   parameters of the jittered HMC algorthm.

   The jittered HMC algorithm depends on the value of a step size, controlling
   the discretization step of the integrator, and a trajectory length, given by the
   number of integration steps / step size, jittered by using only a random percentage
   of this trajectory length.

   This adaptation algorithm tunes the trajectory length by heuristically maximizing
   the Change in the Estimator of the Expected Square (ChEES) criterion over
   an ensamble of parallel chains. At equilibrium, the algorithm aims at eliminating
   correlations between target dimensions, making the HMC algorithm efficient.

   Jittering requires generating a random sequence of uniform variables in [0, 1].
   However, this adds another source of variance to the sampling procedure,
   which may slow adaptation or lead to suboptimal mixing. To alleviate this,
   rather than use uniform random noise to jitter the trajectory lengths, we use a
   quasi-random Halton sequence, which ensures a more even distribution of trajectory
   lengths.

   .. rubric:: Examples

   An HMC adapted kernel can be learned and used with the following code:

   .. code::

       warmup = blackjax.chees_adaptation(logdensity_fn, num_chains)
       key_warmup, key_sample = jax.random.split(rng_key)
       optim = optax.adam(learning_rate)
       (last_states, parameters), _ = warmup.run(
           key_warmup,
           positions, #PyTree where each leaf has shape (num_chains, ...)
           initial_step_size,
           optim,
           num_warmup_steps,
       )
       kernel = blackjax.dynamic_hmc(logdensity_fn, **parameters).step
       new_states, info = jax.vmap(kernel)(key_sample, last_states)

   :param logdensity_fn: The log density probability density function from which we wish to sample.
   :param num_chains: Number of chains used for cross-chain warm-up training.
   :param jitter_generator: Optional function that generates a value in [0, 1] used to jitter the trajectory
                            lengths given a PRNGKey, used to propose the number of integration steps. If None,
                            then a quasi-random Halton is used to jitter the trajectory length.
   :param jitter_value: A percentage in [0, 1] representing how much of the calculated trajectory should be jitted.
   :param target_acceptance_rate: Average acceptance rate to target with dual averaging. Defaults to optimal tuning for HMC.
   :param decay_rate: Float representing how much to favor recent iterations over earlier ones in the optimization
                      of step size and trajectory length. A value of 1 gives equal weight to all history. A value
                      of 0 gives weight only to the most recent iteration.

   :returns: * *A function that returns the last cross-chain state, a sampling kernel with the*
             * *tuned parameter values, and all the warm-up states for diagnostics.*


