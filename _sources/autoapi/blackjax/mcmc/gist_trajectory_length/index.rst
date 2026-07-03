blackjax.mcmc.gist_trajectory_length
====================================

.. py:module:: blackjax.mcmc.gist_trajectory_length

.. autoapi-nested-parse::

   GIST instance (b): self-tuning trajectory length (no-U-turn, GIST paper
   section 5) -- NOT NUTS's recursive doubling.

   The tuning parameter is ``alpha = L``, the number of leapfrog steps at a
   *fixed* step size ``h``. ``L`` is drawn uniformly from ``[Lo(theta, rho) :
   U(theta, rho)]`` where ``U(theta, rho)`` is the number of leapfrog steps
   until the no-U-turn condition first fires (a single forward ``while_loop``
   rollout, materially simpler than NUTS's recursive-doubling tree, section
   2.2.2) and ``Lo`` shifts the lower end of that range by a fixed path
   fraction ``psi``. Because ``g = identity`` on ``L``, the reversibility of the
   resulting kernel is automatic (Corollary 4); the only thing the acceptance
   ratio has to account for is that the forward and reverse draws are uniform
   over *different-width* intervals containing the same ``L`` (section 2.2.4).

   .. rubric:: References

   .. [1] Bou-Rabee, Carpenter, Marsden, "GIST: Gibbs self-tuning for locally
      adaptive Hamiltonian Monte Carlo", arXiv:2404.15253, section 5 (p.20-25),
      Algorithm 2 (p.21), eq. 33 (the no-U-turn condition), eq. 34-35 (the step
      distributions).



Attributes
----------

.. autoapisummary::

   blackjax.mcmc.gist_trajectory_length.init


Classes
-------

.. autoapisummary::

   blackjax.mcmc.gist_trajectory_length.GISTTrajectoryLengthInfo


Functions
---------

.. autoapisummary::

   blackjax.mcmc.gist_trajectory_length.num_steps_to_uturn
   blackjax.mcmc.gist_trajectory_length.build_kernel
   blackjax.mcmc.gist_trajectory_length.as_top_level_api


Module Contents
---------------

.. py:data:: init

.. py:class:: GISTTrajectoryLengthInfo



   Additional information for a ``gist_trajectory_length`` transition.

   momentum, tuning_parameter, is_accepted, is_divergent, acceptance_rate,
   energy, num_integration_steps
       Same convention as :class:`~blackjax.mcmc.gist_step_size.GISTStepSizeInfo`
       (flat extension, not nesting). ``tuning_parameter`` is ``L`` itself
       (an integer, not a compound PyTree, since ``g = identity`` is the
       only tuning-parameter component here).
   num_steps_to_uturn_forward, num_steps_to_uturn_reverse
       ``U(theta, rho) = M`` and ``U(theta', rho') = N``, section 2.2.2 --
       the (possibly capped) leapfrog-step counts to the no-return
       condition, forward from the current state and from the proposal.
   is_no_return_rejected
       True when ``L`` fell outside ``[Lo(theta', rho'), N]`` -- the
       paper's own "no-return" rejection category (Fig. 5, section 2.2.4),
       tracked separately from an ordinary energy-based Metropolis
       rejection.


   .. py:attribute:: momentum
      :type:  blackjax.types.Array


   .. py:attribute:: tuning_parameter
      :type:  blackjax.types.Array


   .. py:attribute:: is_accepted
      :type:  blackjax.types.Array


   .. py:attribute:: is_divergent
      :type:  blackjax.types.Array


   .. py:attribute:: acceptance_rate
      :type:  blackjax.types.Array


   .. py:attribute:: energy
      :type:  float


   .. py:attribute:: num_integration_steps
      :type:  blackjax.types.Array


   .. py:attribute:: num_steps_to_uturn_forward
      :type:  blackjax.types.Array


   .. py:attribute:: num_steps_to_uturn_reverse
      :type:  blackjax.types.Array


   .. py:attribute:: is_no_return_rejected
      :type:  blackjax.types.Array


.. py:function:: num_steps_to_uturn(integrator: Callable, step_size: float, metric: blackjax.mcmc.metrics.Metric, max_num_steps: int) -> Callable

   Build the ``U(theta, rho)`` forward-rollout function, section 2.2.2.

   A single forward ``while_loop``, one leapfrog step at a time, checking
   the sign of the (metric-corrected) no-return condition after every step
   -- no tree/doubling data structure, no recursion, no sub-U-turn
   bookkeeping, unlike NUTS's ``1sub-U-turn`` criterion (section 3.5).

   The dot product uses the **metric-corrected velocity**
   ``M^{-1} rho`` (the gradient of the kinetic energy) rather than the raw
   momentum, so the criterion generalizes correctly to a non-identity
   ``inverse_mass_matrix`` (diagonal / dense / low-rank). This is eq. 33's
   position-displacement-times-velocity condition, a *different* turning
   criterion from NUTS's momentum-sum ``check_turning`` -- the analogy to
   ``metrics.gaussian_euclidean``'s ``check_turning`` is only in *how* it
   substitutes ``M^{-1} rho`` for raw momentum in a dot product, not that
   the two criteria coincide. This changes no formula for the
   identity-metric case the paper's own experiments use.

   ``max_num_steps`` is a hard cap, exactly analogous to NUTS's
   ``max_num_doublings`` (except here it bounds a *linear* rollout, not
   ``2**max`` leapfrog steps). Capping is not an approximation: as long as
   ``p(L|theta,rho)`` and the acceptance ratio consistently use this capped
   ``U``, the resulting ``p(L|theta,rho)`` is still an exact, well-defined,
   strictly-positive conditional density (the GIST reversibility guarantee,
   Theorem 3, does not care that ``U`` is a capped version of the "true"
   U-turn step count).

   :param integrator: Symplectic integrator used for the one-leapfrog-at-a-time rollout.
   :param step_size: ``h``, the fixed step size (not GIST-adapted in this instance).
   :param metric: The (already-resolved) :class:`~blackjax.mcmc.metrics.Metric`.
   :param max_num_steps: Hard cap on the rollout length.

   :returns: * ``uturn_fn(state: IntegratorState, logdensity_fn) -> Array``, the
             * *(possibly capped) number of leapfrog steps to the no-return condition.*


.. py:function:: build_kernel(integrator: Callable = integrators.velocity_verlet, divergence_threshold: float = 1000, path_fraction: float = 0.5, max_num_steps: int = 1024) -> Callable

   Build a ``gist_trajectory_length`` kernel.

   :param integrator: The symplectic integrator to use to integrate the Hamiltonian
                      dynamics.
   :param divergence_threshold: Value of the difference in energy above which we consider that the
                                transition is divergent.
   :param path_fraction: ``psi`` in ``[0, 1]``, section 2.2.3. Default 0.5 per the paper's
                         own recommendation (comparable leapfrog-step counts to NUTS;
                         ``psi=0`` is the simpler eq. 34 special case).
   :param max_num_steps: Hard cap on each U-turn rollout (forward and reverse); analogous to
                         NUTS's ``max_num_doublings``, but bounds a *linear* rollout here,
                         not ``2**max`` leapfrog steps.

   :returns: * *A kernel that takes a rng_key and a Pytree that contains the current*
             * *state of the chain and that returns a new state of the chain along with*
             * *information about the transition.*


.. py:function:: as_top_level_api(logdensity_fn: Callable, inverse_mass_matrix: blackjax.mcmc.metrics.MetricTypes, step_size: float, *, path_fraction: float = 0.5, max_num_steps: int = 1024, divergence_threshold: float = 1000, integrator: Callable = integrators.velocity_verlet) -> blackjax.base.SamplingAlgorithm

   ``blackjax.gist_trajectory_length`` -- GIST self-tuning path length
   (no-U-turn condition, section 2.2; NOT NUTS's recursive doubling).

   .. rubric:: Examples

   A new ``gist_trajectory_length`` kernel can be initialized and used with
   the following code:

   .. code::

       gist_trajectory_length = blackjax.gist_trajectory_length(
           logdensity_fn, inverse_mass_matrix, step_size=0.1
       )
       state = gist_trajectory_length.init(position)
       new_state, info = gist_trajectory_length.step(rng_key, state)

   :param logdensity_fn: The log-density function we wish to draw samples from.
   :param inverse_mass_matrix: The value to use for the inverse mass matrix when drawing a value
                               for the momentum and computing the kinetic energy.
   :param step_size: ``h``, fixed (not GIST-adapted in this instance; see the module
                     docstring for composing with ``gist_step_size``).
   :param path_fraction: ``psi`` in ``[0, 1]``, section 2.2.3. Default 0.5 per the paper's
                         own recommendation (comparable leapfrog-step counts to NUTS;
                         ``psi=0`` is the simpler eq. 34 special case).
   :param max_num_steps: Hard cap on each U-turn rollout (forward and reverse); analogous to
                         NUTS's ``max_num_doublings``, but bounds a *linear* rollout here,
                         not ``2**max`` leapfrog steps -- size accordingly (NUTS's default of
                         10 doublings caps at 1023 steps; a directly-comparable cap here is
                         ``max_num_steps ~= 1024``).
   :param divergence_threshold: The absolute value of the difference in energy between two states
                                above which we say that the transition is divergent.
   :param integrator: (algorithm parameter) The symplectic integrator to use to integrate
                      the trajectory.

   :rtype: A ``SamplingAlgorithm``.


