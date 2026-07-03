blackjax.mcmc.gist
==================

.. py:module:: blackjax.mcmc.gist

.. autoapi-nested-parse::

   General spine for the GIST (Gibbs self-tuning) sampler family.

   GIST (:cite:p:`bou2024gist`) augments the Hamiltonian phase space ``(theta,
   rho)`` with a tuning parameter ``alpha`` (a step size, a trajectory length,
   ...) drawn from a user-supplied conditional density ``p(alpha | theta, rho)``,
   then applies a measure-preserving involution ``G(theta, rho, alpha) = (S o
   F(alpha)(theta, rho), g(theta, rho)(alpha))`` where ``S`` is the momentum-flip
   map and ``F(alpha)`` a reversible, volume-preserving map indexed by ``alpha``
   (typically a fixed number of leapfrog steps, or a variable number of steps at
   a fixed step size). Provided ``G`` is measure preserving, the resulting
   Metropolis-Hastings chain is reversible w.r.t. the target and the
   ``theta``-marginal recovers the original target exactly.

   This module implements the *general* kernel (Gibbs-refresh momentum,
   Gibbs-refresh tuning parameter, apply the involution, one Metropolis test) --
   it is not user-facing on its own. Concrete instances plug in
   ``tuning_parameter_fn``/``apply_fn`` and are exposed as
   ``blackjax.gist_step_size`` (:mod:`blackjax.mcmc.gist_step_size`) and
   ``blackjax.gist_trajectory_length`` (:mod:`blackjax.mcmc.gist_trajectory_length`).

   Both shipped instances take ``g(theta, rho) = identity`` on the tuning
   parameter, which is a sufficient condition for ``G`` to be a
   measure-preserving involution (Corollary 4): since ``S o F(alpha)`` is a
   volume-preserving involution for every fixed ``alpha``, Fubini's theorem lifts
   this to the product space ``R^{2d} x A`` without needing a nontrivial ``g``.

   .. rubric:: References

   .. [1] Bou-Rabee, Carpenter, Marsden, "GIST: Gibbs self-tuning for locally
      adaptive Hamiltonian Monte Carlo", arXiv:2404.15253, Statistical Surveys
      2026, Vol. 20, pp. 135-179. Algorithm 1 (p.6), eq. 9 (p.7).



Classes
-------

.. autoapisummary::

   blackjax.mcmc.gist.GISTState
   blackjax.mcmc.gist.GISTInfo


Functions
---------

.. autoapisummary::

   blackjax.mcmc.gist.init
   blackjax.mcmc.gist.build_kernel
   blackjax.mcmc.gist.as_top_level_api


Module Contents
---------------

.. py:class:: GISTState



   State of the GIST sampling chain.

   position
       Current position of the chain.
   logdensity
       Log-density of the target at ``position``.
   logdensity_grad
       Gradient of the log-density at ``position``.


   .. py:attribute:: position
      :type:  blackjax.types.ArrayTree


   .. py:attribute:: logdensity
      :type:  float


   .. py:attribute:: logdensity_grad
      :type:  blackjax.types.ArrayTree


.. py:class:: GISTInfo



   Additional information on a GIST transition.

   momentum
       The momentum sampled at the start of the transition (before the
       involution is applied) -- the Gibbs draw ``rho ~ N(0, Sigma)``.
   tuning_parameter
       The tuning parameter ``alpha`` drawn from ``p(. | position,
       momentum)`` (a PyTree; e.g. an integer step-count for
       ``gist_trajectory_length``, or ``(a, b, j)`` for ``gist_step_size``).
   is_accepted
       Whether the involution's output was accepted.
   is_divergent
       Whether the *energy* term of the acceptance ratio (``-delta_energy``)
       exceeded ``divergence_threshold``. This is distinct from a structural
       rejection (e.g. the step-size reversibility check failing, or the
       tuning parameter falling outside the reverse U-turn interval) --
       those are reported via instance-specific ``Info`` fields, not here.
   acceptance_rate
       The realized GIST acceptance probability actually used -- already
       includes any tuning-density-ratio factor (the reversibility-check
       indicator for ``gist_step_size``, or the interval-width ratio for
       ``gist_trajectory_length``), not just the bare energy-based
       Metropolis term.
   energy
       Total Hamiltonian energy of the proposal state.
   num_integration_steps
       Number of leapfrog evaluations spent on the accepted/rejected
       proposal's own trajectory (NOT counting any extra evaluations spent
       inside the tuning-parameter sampler itself, e.g. the step-size
       doubling/halving search or the forward/reverse U-turn rollouts --
       see the instance-specific ``Info`` fields for those separately, since
       they matter for tuning-inclusive cost accounting).

   .. rubric:: References

   Bou-Rabee, Carpenter, Marsden, "GIST: Gibbs self-tuning for locally
   adaptive Hamiltonian Monte Carlo", arXiv:2404.15253, Algorithm 1 (p.6),
   eq. 9 (p.7).


   .. py:attribute:: momentum
      :type:  blackjax.types.ArrayTree


   .. py:attribute:: tuning_parameter
      :type:  blackjax.types.ArrayTree


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


.. py:function:: init(position: blackjax.types.ArrayLikeTree, logdensity_fn: Callable) -> GISTState

   Create the initial GIST state.

   Numerically identical to :func:`blackjax.mcmc.hmc.init` (same
   ``value_and_grad`` computation); re-wrapped into :class:`GISTState`
   rather than aliased directly (``init = hmc.init``) so that every
   ``GISTState`` produced by this module is actually an instance of
   ``GISTState``, not ``HMCState``. This matters because ``jax.lax.cond`` in
   :func:`build_kernel` branches between a freshly built ``GISTState`` and
   the incoming ``state`` -- both branches must share the exact same pytree
   treedef, which a bare alias would violate on the very first call.


.. py:function:: build_kernel(divergence_threshold: float = 1000) -> Callable

   Build the general GIST kernel.

   :param divergence_threshold: Threshold on the energy term of the acceptance ratio above which a
                                transition is flagged ``is_divergent`` (same convention as
                                ``hmc``/``nuts``).

   :returns: * A kernel with signature ``kernel(rng_key, state, logdensity_fn,
             * *tuning_parameter_fn, apply_fn, inverse_mass_matrix) -> (GISTState,*
             * GISTInfo)`` where
             * **``tuning_parameter_fn(rng_key, state** (*IntegratorState, logdensity_fn,*)
             * metric) -> (alpha, aux)`` -- The GIBBS step: draws ``alpha ~ p(. | theta, rho)``. ``aux`` is an
               opaque PyTree threaded to ``apply_fn`` for anything computed here
               that ``apply_fn`` would otherwise have to recompute (e.g. the
               forward U-turn count ``U(theta, rho)``, or the ``(a, b)``
               thresholds).
             * **``apply_fn(state** (*IntegratorState, alpha, aux, logdensity_fn, metric) ->*)
             * **(proposal_state** (*IntegratorState, log_tuning_density_ratio: Array,*)
             * **extra_info** (ArrayTree)``) -- Computes the involution ``G(theta, rho, alpha) = (S o
               F(alpha)(theta, rho), g(theta, rho)(alpha))`` and returns ``log[
               p(g(theta, rho)(alpha) | S o F(alpha)(theta, rho)) / p(alpha |
               theta, rho) ]`` directly as a scalar log-ratio -- NOT two separate
               density evaluations (a Dirac-measure ``p``, as in ``gist_step_size``,
               has no well-defined value away from its atom, so the ratio must be
               computed directly rather than as two independent evaluations).
               ``extra_info`` is threaded straight into the instance's own ``Info``
               NamedTuple; it must carry (at least) a ``num_integration_steps``
               field.
             * *Note on tracing cost*
             * *--------------------*
             * Because ``tuning_parameter_fn`` and ``apply_fn`` each receive
             * ``logdensity_fn``/``metric`` fresh from this seam (not a shared
             * pre-built trajectory function), an instance whose ``apply_fn`` re-runs
             * *the tuning-parameter search at the proposal (as both shipped instances*
             * *do, for the reversibility/no-return check) pays one extra*
             * ``logdensity_fn`` trace for that re-check, on top of the forward
             * *search and the accepted-move build -- three per kernel call, not the*
             * *two of a single-forward-pass sampler like hmc/nuts. This is a*
             * **deliberate seam-simplicity tradeoff, not a retracing bug** (*threading a*)
             * *shared pre-built trajectory function through this seam could bring it*
             * *down to two, at the cost of a more complex seam contract. See*
             * ``gist_step_size``/``gist_trajectory_length``'s own
             * ``chex.assert_max_traces(n=4)`` tests (1 at ``init`` + 3 per kernel
             * *trace) for the empirically-verified count.*


.. py:function:: as_top_level_api(logdensity_fn: Callable, inverse_mass_matrix: blackjax.mcmc.metrics.MetricTypes, tuning_parameter_fn: Callable, apply_fn: Callable, *, divergence_threshold: float = 1000) -> blackjax.base.SamplingAlgorithm

   Build a ``SamplingAlgorithm`` from the general GIST kernel.

   This is mainly an internal building block for the two instance modules
   (:mod:`blackjax.mcmc.gist_step_size`,
   :mod:`blackjax.mcmc.gist_trajectory_length`) -- end users call
   ``blackjax.gist_step_size(...)`` / ``blackjax.gist_trajectory_length(...)``,
   not this function directly (mirroring how ``nuts.as_top_level_api``
   doesn't route end users through ``hmc.as_top_level_api``, even though it
   reuses ``hmc.py``'s pieces internally). There is deliberately no
   top-level ``blackjax.gist``; reach this via
   ``blackjax.mcmc.gist.as_top_level_api``.

   :param logdensity_fn: The log-density function we wish to draw samples from.
   :param inverse_mass_matrix: The value to use for the inverse mass matrix when drawing a value
                               for the momentum and computing the kinetic energy.
   :param tuning_parameter_fn: The GIBBS seam, see :func:`build_kernel`.
   :param apply_fn: The involution seam, see :func:`build_kernel`.
   :param divergence_threshold: The absolute value of the difference in energy between two states
                                above which we say that the transition is divergent.

   :rtype: A ``SamplingAlgorithm``.


