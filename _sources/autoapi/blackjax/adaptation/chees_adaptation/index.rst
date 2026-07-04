blackjax.adaptation.chees_adaptation
====================================

.. py:module:: blackjax.adaptation.chees_adaptation

.. autoapi-nested-parse::

   Public API for ChEES-HMC



Attributes
----------

.. autoapisummary::

   blackjax.adaptation.chees_adaptation.OPTIMAL_TARGET_ACCEPTANCE_RATE
   blackjax.adaptation.chees_adaptation.LOG_UPDATE_CLIP
   blackjax.adaptation.chees_adaptation.EPS_FLOAT
   blackjax.adaptation.chees_adaptation.CHEES_LENGTH_FLOOR_FACTOR


Classes
-------

.. autoapisummary::

   blackjax.adaptation.chees_adaptation.ChEESAdaptationState


Functions
---------

.. autoapisummary::

   blackjax.adaptation.chees_adaptation.weighted_empirical_mean
   blackjax.adaptation.chees_adaptation.base
   blackjax.adaptation.chees_adaptation.chees_adaptation


Module Contents
---------------

.. py:data:: OPTIMAL_TARGET_ACCEPTANCE_RATE
   :value: 0.651


.. py:data:: LOG_UPDATE_CLIP
   :value: 0.35


.. py:data:: EPS_FLOAT
   :value: 1e-20


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
      :type:  float


   .. py:attribute:: log_step_size_moving_average
      :type:  float


   .. py:attribute:: trajectory_length
      :type:  float


   .. py:attribute:: log_trajectory_length_moving_average
      :type:  float


   .. py:attribute:: da_state
      :type:  blackjax.optimizers.dual_averaging.DualAveragingState


   .. py:attribute:: optim_state
      :type:  optax.OptState


   .. py:attribute:: random_generator_arg
      :type:  blackjax.types.Array


   .. py:attribute:: step
      :type:  int


.. py:data:: CHEES_LENGTH_FLOOR_FACTOR
   :type:  float

.. py:function:: weighted_empirical_mean(x, w)

.. py:function:: base(jitter_generator: Callable, next_random_arg_fn: Callable, optim: optax.GradientTransformation, target_acceptance_rate: float, decay_rate: float, max_leapfrog_steps: int, _whiten_criterion: bool = True) -> tuple[Callable, Callable]

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
   :param _whiten_criterion: Private, undocumented ablation seam (not part of the public API). When
                             True (default) the ChEES criterion accounts for a non-identity
                             ``inverse_mass_matrix`` passed to ``update`` (see the derivation in
                             ``compute_parameters``). When False, the criterion is computed exactly
                             as the original identity-mass-matrix expression regardless of the
                             ``inverse_mass_matrix`` passed in -- used in validation studies to
                             isolate the whitening correction's contribution relative to the mass
                             matrix alone.

   :returns: * *init* -- Function that initializes the warmup.
             * *update* -- Function that moves the warmup one step.


.. py:function:: chees_adaptation(logdensity_fn: Callable, num_chains: int, *, jitter_generator: Callable | None = None, jitter_amount: float = 1.0, target_acceptance_rate: float = OPTIMAL_TARGET_ACCEPTANCE_RATE, decay_rate: float = 0.5, max_leapfrog_steps: int = 1000, adaptation_info_fn: Callable = return_all_adapt_info, mass_matrix_estimation: str | None = None, mass_matrix_window_fraction: float = 0.5, _whiten_criterion: bool = True, _length_floor: bool = True) -> blackjax.base.AdaptationAlgorithm

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
       kernel = blackjax.dhmc(logdensity_fn, **parameters).step
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
   :param adaptation_info_fn: Function to select the adaptation info returned. See return_all_adapt_info
                              and get_filter_adapt_info_fn in blackjax.adaptation.base.  By default all
                              information is saved - this can result in excessive memory usage if the
                              information is unused.
   :param mass_matrix_estimation: Opt-in ensemble-estimated *diagonal* ``inverse_mass_matrix`` (opt-in,
                                  default ``None``). ``None`` keeps the original behavior, bit-for-bit:
                                  the kernel always uses ``inverse_mass_matrix=jnp.ones(num_dim)``.
                                  ``"diagonal"`` instead estimates a per-dimension variance from the
                                  ensemble of all ``num_chains`` chains via a running Welford
                                  accumulator (:func:`~blackjax.adaptation.mass_matrix.welford_algorithm`),
                                  accumulated over the *last* ``mass_matrix_window_fraction`` of warmup
                                  (see that parameter), and uses it as the kernel's diagonal
                                  ``inverse_mass_matrix`` once enough samples have accumulated (see
                                  "Engagement gate" below). ChEES's own trajectory-length criterion is
                                  also whitened by this estimate so it stays metric-consistent with the
                                  kernel it is tuning (see the derivation in
                                  :func:`base`'s ``compute_parameters``) -- this whitening keeps the
                                  criterion metric-consistent and responsive to the preconditioned geometry;
                                  measured effect is small (≲12%) on NCP-standardized targets where most
                                  dimensions are near unit scale. The feature's main value is delivered by
                                  the slow-direction *length floor* (when enabled), which recovers large
                                  ESS-per-gradient wins on targets with a residual slow correlation
                                  direction the diagonal metric cannot remove.

                                  **Scope: validated under located (typical-set) initializations.** On
                                  funnel-like targets the diagonal metric can *reduce* per-draw ESS
                                  relative to the identity metric's fine-step regime (an efficiency caveat,
                                  not a bias—posterior means are unaffected); the length floor recovers most
                                  of it. Cold or dispersed initialization on hard geometry is a separate
                                  warmup-robustness limitation, out of scope for this feature.

                                  Engagement gate: before the pooled accumulator (effective sample
                                  size ``num_chains * window_steps``) exceeds ``max(64, 2*sqrt(num_dim))``
                                  -- a modest floor chosen because a per-dimension variance estimate,
                                  unlike a joint eigenbasis, does not need ``O(d)`` samples to escape
                                  noise domination, so a small constant plus a mild ``d``-scaling for
                                  very high-dimensional targets is enough -- the kernel and criterion
                                  both use ``jnp.ones(num_dim)`` (identical to ``None``). The final
                                  ``inverse_mass_matrix`` returned by ``run()`` is the engaged-or-
                                  fallback estimate at the end of the accumulation window.
   :param mass_matrix_window_fraction: Only used when ``mass_matrix_estimation="diagonal"``. Fraction of
                                       warmup steps, counted from the end, over which the diagonal
                                       ``inverse_mass_matrix``'s Welford accumulator collects samples
                                       (default ``0.5``: the last half of warmup, mirroring
                                       ``meads_adaptation``'s ``low_rank_window_fraction`` and Stan's
                                       practice of excluding window adaptation's own early/transient
                                       fraction). Must be in ``[0.0, 1.0]``; ``0.0`` accumulates from step
                                       0, ``1.0`` disables accumulation entirely (falls back to
                                       ``jnp.ones(num_dim)`` throughout the run).

   :returns: * *A function that returns the last cross-chain state, a sampling kernel with the*
             * *tuned parameter values, and all the warm-up states for diagnostics.*


