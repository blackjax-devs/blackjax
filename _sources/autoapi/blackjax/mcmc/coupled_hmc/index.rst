blackjax.mcmc.coupled_hmc
=========================

.. py:module:: blackjax.mcmc.coupled_hmc

.. autoapi-nested-parse::

   Two HMC marginals coupled only through their random inputs.

   The pair consists of ordinary HMC transitions with one shared normal
   innovation and one shared uniform variate.  Each marginal keeps its own
   transition, acceptance probability, decision, and cached state.  This module
   does not claim product-target invariance, meeting, unbiased estimators, or an
   efficiency improvement.

   The contract
   ------------

   Given the same ``(state, standard_normal, uniform)`` triple and per-marginal
   parameters, each marginal performs the same transition as ordinary HMC.  A
   shared uniform is compared against each marginal's own acceptance probability;
   it is not a shared decision.

   Mathematical semantics
   ----------------------

   Writing :math:`p = A z` for the momentum and :math:`A A^\top = M` for its
   covariance, both couplings draw one :math:`z \sim N(0,I)` and hand each
   marginal a transformed copy:

   ``synchronous``
       Both marginals receive the same :math:`z`.

   ``reflection``
       The first marginal receives :math:`z`; the second receives

       .. math::

           z' = z - 2 e (e^\top z)

       For unit ``e`` independent of ``z``, this orthogonal reflection preserves
       the normal law exactly in mathematics (floating-point normalization is
       accurate to rounding).

       ``e`` comes from ``direction_fn``, fixed at build time and applied to the
       incoming states and first metric only; it receives no innovations and must
       be pure in those arguments.  The default direction is the
       difference of the two positions whitened by the **first** marginal's
       metric, :math:`e \propto A^{-1}(x_1 - x_2)`.  That particular choice is a
       heuristic.  With different metrics it remains only a first-metric-based
       direction with no claimed property for the pair, and a zero direction is
       the identity.

   Relation to ``blackjax.hmc``
   ----------------------------

   Each marginal accepts with probability :math:`\min(1,e^{\Delta})`, comparing
   the supplied uniform directly.  This is distributionally equivalent to
   ordinary HMC's Bernoulli implementation, but not draw-for-draw equivalent.

   Usage
   -----

   There is no top-level ``blackjax.coupled_hmc``; use
   ``blackjax.mcmc.coupled_hmc``.  Every per-marginal argument is an explicit
   ``(first, second)`` pair, including shared values as ``(value, value)``;
   nothing is broadcast or inferred.

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



   Pair of HMC states, each with its own caches.

   Each marginal retains its own position and cached
   ``logdensity``/``logdensity_grad``.

   first
       State of the first marginal chain.
   second
       State of the second marginal chain.



   .. py:attribute:: first
      :type:  blackjax.mcmc.hmc.HMCState


   .. py:attribute:: second
      :type:  blackjax.mcmc.hmc.HMCState


.. py:class:: CoupledHMCInfo



   Per-marginal transition information and the shared random inputs.

   ``first`` and ``second`` retain complete, separate HMC diagnostics.

   first
       Transition information for the first marginal.
   second
       Transition information for the second marginal.
   common_normal
       Standard normal handed to the first marginal; the second receives its
       synchronous or reflected image.
   reflection_unit
       Reflection direction in the same coordinates, or exactly zero for
       synchronous/identity coupling.
   uniform
       Uniform shared by both tests; each marginal compares it with its own
       acceptance probability.



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

   Eagerly validate one marginal's concrete metric and settings.

   Checks use NumPy on concrete host values and are absent from the traced
   kernel.  Passing them validates only the values supplied now; callers using
   traced values through :func:`build_kernel` retain responsibility for them.

   :raises TypeError, ValueError: If the metric is not a supported kind, is not finite, is not positive
       definite, or the integration settings are not positive and finite.


.. py:function:: init(position: blackjax.base.Position, logdensity_fn: Sequence[Callable]) -> CoupledHMCState

   Initialize a coupled pair from explicit position and log-density pairs.

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

   The kernel draws ``z`` and the uniform before the transition evaluates
   ``direction_fn``; the contract concerns its arguments, not ordering.  A
   callable that closes over innovations or draws randomness breaks reflection
   marginal correctness and cannot be detected here, so callers must keep it
   pure in the supplied arguments.

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


