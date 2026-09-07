blackjax.mcmc.coupled_hmc
=========================

.. py:module:: blackjax.mcmc.coupled_hmc

.. autoapi-nested-parse::

   Two HMC chains advanced from a shared source of randomness.

   This module runs two ordinary HMC marginals side by side and couples them
   *only through their random inputs*: the standard normal that becomes each
   momentum, and the single uniform that drives each Metropolis test.

   .. warning::

       The scope of what is implemented here is deliberately narrow.  This
       module provides a **coupling of the random inputs**.  It makes no claim
       that the pair is invariant for the product of the two targets, that the
       chains meet (exactly, maximally, or at all), that any estimator built
       from the pair is unbiased, or that coupling improves sampling efficiency
       in any sense.  None of those properties is implemented or tested.

   The contract
   ------------

   The one property this module does assert, and test, is:

       Coupling changes only the *joint law* of the two marginals' random
       inputs.  It never changes either marginal's transition function, its
       Metropolis decision rule, or its cached log-density and gradient.

   Concretely, given the same ``(state, standard_normal, uniform)`` triple, each
   marginal here performs exactly the transition an uncoupled HMC marginal built
   from the same target, metric, step size and integration count would perform,
   and returns exactly the same state and info.  The two marginals share a
   uniform but each compares it against **its own** acceptance probability, so a
   shared uniform is emphatically *not* a shared decision.  Each marginal carries
   its own ``logdensity``/``logdensity_grad`` cache.

   Mathematical semantics
   ----------------------

   Write :math:`M` for a marginal's mass matrix and :math:`A` for the momentum
   square root BlackJAX uses, so that a momentum drawn as :math:`p = A z` with
   :math:`z \sim N(0, I)` has covariance :math:`A A^\top = M`.  Both couplings
   below draw a single :math:`z` and hand each marginal a transformed copy:

   ``synchronous``
       Both marginals receive the same :math:`z`.

   ``reflection``
       The first marginal receives :math:`z`; the second receives

       .. math::

           z' = z - 2 e (e^\top z)

       for a unit vector :math:`e`.  That map is an orthogonal reflection, so in
       exact arithmetic it preserves :math:`N(0, I)` and the second marginal's
       momentum law is unchanged **for any** unit :math:`e` that does not depend
       on :math:`z`.  In floating point the normalisation of :math:`e` carries a
       rounding error of a few units in the last place, so the preservation is
       exact in the mathematics and accurate to that tolerance in the code.  That is the whole of what reflection buys here: marginal
       correctness.  It is not a statement about contraction, meeting, or
       coupling quality, and none is made.

       ``e`` comes from ``direction_fn``, which is fixed when the kernel is
       built and is applied to the *incoming* pair of states.  It is given the two
       states and the first metric, and no innovations.  Note that this is a
       statement about its arguments, not about ordering: the kernel draws ``z``
       and the uniform first and only then runs the transition that evaluates
       ``direction_fn``.  Supplying a ``direction_fn`` that is a pure function of
       its arguments is the caller's part of the contract.  The default direction is the
       difference of the two positions whitened by the **first** marginal's
       metric, :math:`e \propto A^{-1}(x_1 - x_2)`.  That particular choice is a
       heuristic: it is the direction along which, to leading order, the two
       chains' displacements respond oppositely, since the velocity
       :math:`\partial K / \partial p` equals :math:`A^{-\top} z` and hence
       :math:`\langle x_1 - x_2, \partial K/\partial p\rangle = \langle
       A^{-1}(x_1 - x_2), z\rangle`.  When the two marginals use *different*
       metrics the default still uses the first marginal's metric only; it is
       then a first-metric-based direction with no claimed property for the
       pair.  A zero direction is the identity map, i.e. reflection degenerates
       to synchronous coupling for that transition.

   Relation to ``blackjax.hmc``
   ----------------------------

   Each marginal applies the same *mathematical* Metropolis rule as ordinary
   HMC: accept with probability :math:`\min(1, e^{\Delta})`.  It realises that
   rule by comparing a supplied uniform against the acceptance probability,
   whereas :func:`~blackjax.mcmc.proposal.static_binomial_sampling` realises it
   with :func:`jax.random.bernoulli`.  The two agree as distributions and do
   **not** agree draw-for-draw: feeding the same key to this kernel and to
   ``blackjax.hmc`` will not reproduce the same accept/reject sequence, and no
   such parity is claimed or tested.

   Usage
   -----

   There is deliberately no top-level ``blackjax.coupled_hmc``; reach this
   module at ``blackjax.mcmc.coupled_hmc``.  Every per-marginal parameter is
   passed as an explicit ``(first, second)`` pair -- including when both
   marginals share a value, which is written ``(value, value)``.  Nothing is
   broadcast or inferred, because positions, metrics and step sizes may
   themselves legitimately be tuples.

   .. code::

       import blackjax.mcmc.coupled_hmc as coupled_hmc

       algorithm = coupled_hmc.as_top_level_api(
           (logdensity_fn, logdensity_fn),
           step_size=(0.1, 0.1),
           inverse_mass_matrix=(inverse_mass_matrix, inverse_mass_matrix),
           num_integration_steps=(10, 10),
           coupling="reflection",
       )
       state = algorithm.init((position_one, position_two))
       state, info = algorithm.step(rng_key, state)



Classes
-------

.. autoapisummary::

   blackjax.mcmc.coupled_hmc.CoupledHMCState
   blackjax.mcmc.coupled_hmc.CoupledHMCInfo


Functions
---------

.. autoapisummary::

   blackjax.mcmc.coupled_hmc.validate_marginal_inputs
   blackjax.mcmc.coupled_hmc.init
   blackjax.mcmc.coupled_hmc.build_kernel
   blackjax.mcmc.coupled_hmc.as_top_level_api


Module Contents
---------------

.. py:class:: CoupledHMCState



   State of a pair of coupled HMC chains.

   The two marginals are held as two complete, independent
   :class:`~blackjax.mcmc.hmc.HMCState` values, so each carries its own
   position and its own cached ``logdensity``/``logdensity_grad``.  Keeping
   them separate makes crossing the two caches an obvious error rather than
   a silent one -- it does not make it impossible, so the test suite checks
   it explicitly.

   first
       State of the first marginal chain.
   second
       State of the second marginal chain.



   .. py:attribute:: first
      :type:  blackjax.mcmc.hmc.HMCState


   .. py:attribute:: second
      :type:  blackjax.mcmc.hmc.HMCState


.. py:class:: CoupledHMCInfo



   Additional information on a coupled HMC transition.

   Both marginals' complete :class:`~blackjax.mcmc.hmc.HMCInfo` objects are
   preserved untouched, so every per-chain diagnostic (momentum, acceptance
   rate, acceptance decision, divergence flag, energy, proposal, integration
   count) stays separately visible.  The remaining three fields describe the
   shared randomness itself.

   first
       Transition information for the first marginal.
   second
       Transition information for the second marginal.
   common_normal
       The flat standard normal vector handed to the first marginal.  The
       second marginal receives this vector under the coupling map, i.e.
       unchanged for ``"synchronous"`` and reflected for ``"reflection"``.
   reflection_unit
       The unit vector used by the reflection, in the same flat coordinates
       as ``common_normal``.  Exactly zero under synchronous coupling, and
       exactly zero for a reflection whose direction was zero (both denote
       the identity map).
   uniform
       The single uniform variate shared by both Metropolis tests.  Each
       marginal compares it against its own acceptance probability.



   .. py:attribute:: first
      :type:  blackjax.mcmc.hmc.HMCInfo


   .. py:attribute:: second
      :type:  blackjax.mcmc.hmc.HMCInfo


   .. py:attribute:: common_normal
      :type:  blackjax.types.Array


   .. py:attribute:: reflection_unit
      :type:  blackjax.types.Array


   .. py:attribute:: uniform
      :type:  blackjax.types.Array


.. py:function:: validate_marginal_inputs(inverse_mass_matrix, step_size, num_integration_steps)

   Eagerly validate one marginal's concrete metric and integration settings.

   These are host-side numerical checks on *concrete* values: they use NumPy
   and cannot run on tracers, so they are performed once when an algorithm is
   constructed and are deliberately absent from the traced kernel.

   .. important::

       Passing this check says the values supplied *now* are admissible.  It
       says nothing about values that appear later inside a jitted or
       vmapped computation, which are never seen by this function.  A caller
       who builds a kernel with :func:`build_kernel` and feeds it traced
       metrics is responsible for their admissibility.

   :raises TypeError, ValueError: If the metric is not a supported kind, is not finite, is not positive
       definite, or the integration settings are not positive and finite.


.. py:function:: init(position: blackjax.base.Position, logdensity_fn: Sequence[Callable]) -> CoupledHMCState

   Initialise a coupled pair.

   :param position: An explicit ``(first_position, second_position)`` pair.  The two
                    positions must share a pytree structure, leaf shapes and one floating
                    dtype.
   :param logdensity_fn: An explicit ``(first_logdensity_fn, second_logdensity_fn)`` pair.  The
                         two may target different distributions.


.. py:function:: build_kernel(integrator: Callable = integrators.velocity_verlet, divergence_threshold: float = 1000, *, coupling: str = 'synchronous', direction_fn: Callable | None = None)

   Build a coupled HMC kernel.

   ``coupling`` and ``direction_fn`` are fixed here, when the kernel is
   built, and not at call time.  ``direction_fn`` is applied to the pair of
   incoming states and the first metric, and is supplied no innovations.

   It is worth being exact about what that does and does not say, because an
   earlier version of this docstring overstated it.  The kernel draws ``z``
   and the uniform *before* running the transition in which ``direction_fn``
   is evaluated, so the guarantee is **not** one of ordering.  It is that the
   innovations are not among the arguments handed over.  A ``direction_fn``
   that closes over innovations, or draws its own randomness, breaks the
   reflection's marginal correctness and nothing here can detect it. Keeping
   it a pure function of the arguments it is given is the caller's part of the
   contract -- the same purity convention every BlackJAX kernel assumes.

   :param integrator: Symplectic integrator used by both marginals.
   :param divergence_threshold: Energy difference above which a marginal transition is flagged
                                divergent.  Applied to each marginal separately.
   :param coupling: ``"synchronous"``, which gives both marginals the same standard
                    normal, or ``"reflection"``, which gives the second marginal a
                    reflected copy.
   :param direction_fn: Only for ``"reflection"``.  A callable
                        ``(first_state, second_state, first_metric) -> Array`` returning a
                        flat direction; defaults to :func:`whitened_difference`.
                        Passing one under synchronous coupling is an error, since it would
                        have no effect.

   :returns: * A kernel ``(rng_key, state, logdensity_fn, step_size,
             * *inverse_mass_matrix, num_integration_steps) -> (CoupledHMCState,*
             * CoupledHMCInfo)`` in which every per-marginal parameter is an explicit
             * ``(first, second)`` pair.


.. py:function:: as_top_level_api(logdensity_fn: Sequence[Callable], step_size: Sequence[float], inverse_mass_matrix: Sequence[blackjax.mcmc.metrics.MetricTypes], num_integration_steps: Sequence[int], *, coupling: str = 'synchronous', direction_fn: Callable | None = None, integrator: Callable = integrators.velocity_verlet, divergence_threshold: float = 1000) -> blackjax.base.SamplingAlgorithm

   Build a ``SamplingAlgorithm`` for a coupled HMC pair.

   There is deliberately no top-level ``blackjax.coupled_hmc``; reach this
   through ``blackjax.mcmc.coupled_hmc.as_top_level_api``.

   Every per-marginal argument is an explicit ``(first, second)`` pair, so a
   shared value is written twice.  The two marginals may target different
   distributions and use different fixed Gaussian Euclidean metrics, step
   sizes and integration counts, but their positions must share a pytree
   structure, leaf shapes and one floating dtype.

   :param logdensity_fn: Pair of log-density functions.
   :param step_size: Pair of integration step sizes.
   :param inverse_mass_matrix: Pair of inverse mass matrices.  Each is a diagonal array, a dense
                               array, or a :class:`~blackjax.mcmc.metrics.LowRankInverseMassMatrix`;
                               callable (Riemannian) metrics and pre-built
                               :class:`~blackjax.mcmc.metrics.Metric` objects are not supported.
   :param num_integration_steps: Pair of integration step counts.
   :param coupling: ``"synchronous"`` or ``"reflection"``; see :func:`build_kernel`.
   :param direction_fn: Reflection direction policy; see :func:`build_kernel`.
   :param integrator: Symplectic integrator used by both marginals.
   :param divergence_threshold: Per-marginal divergence threshold.

   .. rubric:: Notes

   Each marginal's concrete metric and integration settings are checked here
   by :func:`validate_marginal_inputs`.  Those checks read concrete values on
   the host, so they cover the arguments given here and say nothing about
   traced values appearing later under ``jit`` or ``vmap``.  Callers who need
   to supply traced metrics use :func:`build_kernel` directly, which performs
   no eager numerical validation.

   :rtype: A ``SamplingAlgorithm`` whose ``init`` takes a pair of positions.


