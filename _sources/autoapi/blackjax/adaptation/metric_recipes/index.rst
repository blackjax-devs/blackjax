blackjax.adaptation.metric_recipes
==================================

.. py:module:: blackjax.adaptation.metric_recipes

.. autoapi-nested-parse::

   Metric recipes and the embeddable MetricCore protocol for staged_adaptation.

   Available recipes
   -----------------
   Pass any of the following string names as the ``metric=`` argument to
   :func:`~blackjax.adaptation.staged_adaptation.staged_adaptation`:

   - ``"welford_diag"`` — Stan-default diagonal Welford estimator; reproduces
     :func:`~blackjax.adaptation.window_adaptation.window_adaptation` exactly.
   - ``"welford_dense"`` — Dense Welford covariance, same Stan schedule.
   - ``"fisher_diag"`` — Fisher-divergence-minimising diagonal estimator
     (situational; requires position *and* gradient samples; see registry
     provenance note for operational guidance).
   - ``"fisher_low_rank"`` — Fisher-divergence-minimising LOW-RANK estimator;
     requires position *and* gradient samples; needs a ``buffer_size``
     argument in ``build_core()``.  Algorithm 1 of
     :cite:p:`seyboldt2026preconditioning` / nutpie's mass-matrix estimator.
   - ``"sample_cov_low_rank"`` — Sample-covariance low-rank estimator (MEADS
     / Scheme-B form); draws only, no gradients, no regularisation.  Needs a
     ``buffer_size`` argument in ``build_core()``.

   Usage::

       # String sugar (registry lookup):
       wu = staged_adaptation(nuts, logdensity_fn, metric="welford_diag")

       # Low-rank: pre-build the core with buffer_size, then pass to engine:
       from blackjax.adaptation.metric_recipes import REGISTRY, _build_fisher_low_rank_core
       core = _build_fisher_low_rank_core(buffer_size=256, max_rank=10, gamma=1e-5, cutoff=2.0)
       wu = staged_adaptation(nuts, logdensity_fn, metric=core, schedule_fn=my_schedule_fn)

       # Via the recipe (also needs buffer_size):
       recipe = REGISTRY["fisher_low_rank"]
       core = recipe.build_core(buffer_size=256)
       wu = staged_adaptation(nuts, logdensity_fn, metric=core)

   Design
   ------
   A :class:`MetricRecipe` declares an (estimator, buffer, representation,
   support_gate) tuple with construction-time validation of the coupling
   contract (``needs ⊆ provides`` and ``emits == representation``): incompatible
   combos fail at Python level with a clear message, never inside traced code.

   A :class:`MetricCore` is the embeddable mass-matrix adaptation component —
   the separable piece that the staged_adaptation engine hosts.  Step-size
   dual averaging and the stage schedule are HOST-layer concerns; this core
   handles only the inverse-mass-matrix estimation.

   Layer doctrine:

   - The METRIC CORE (:class:`MetricCore`) handles mass-matrix tuning only.
   - Step-size adaptation and the stage schedule live in the HOST
     (:mod:`~blackjax.adaptation.staged_adaptation`).
   - Step-size/metric decoupling: for HMC/NUTS, the dual-averaging step-size proxy
     is the scalar Metropolis acceptance rate — NOT an eigenvalue quantity of the
     adapted metric.  This matters because in MCLMC-LRD, where step_size ∝ 1/√λ_max,
     feeding the full low-rank metric to the proxy caused a previously observed
     step-size collapse (effective-sample rate dropped 20.6×→1.27×).  For HMC/NUTS
     the full low-rank metric feeds the MCMC kernel (correct coupling), and dual
     averaging reads only acceptance_rate — the step-size/metric decoupling
     principle is naturally satisfied; no diagonal-reference split is needed.  This
     analysis applies to ALL recipes in this module.

   .. rubric:: Notes

   The :class:`MetricRecipe` schema (field types) is provisional:
   estimator/buffer/support_gate are currently string tags.  Future slices will
   replace these with proper constructor objects once the schema is stable across
   all recipe families.  Import directly from ``blackjax.adaptation.metric_recipes``
   — :class:`MetricRecipe` is not exported at the ``blackjax`` top level.



Attributes
----------

.. autoapisummary::

   blackjax.adaptation.metric_recipes.REGISTRY


Classes
-------

.. autoapisummary::

   blackjax.adaptation.metric_recipes.MetricCore
   blackjax.adaptation.metric_recipes.LowRankMetricCoreState
   blackjax.adaptation.metric_recipes.MetricRecipe


Functions
---------

.. autoapisummary::

   blackjax.adaptation.metric_recipes.seed_low_rank_sigma_from_grad
   blackjax.adaptation.metric_recipes.lookup_recipe


Module Contents
---------------

.. py:class:: MetricCore



   Embeddable mass-matrix adaptation core: init/update/final protocol.

   A NamedTuple-of-callables (house style) bundling the three operations that
   together constitute mass-matrix adaptation.  The engine hosts this core;
   step-size adaptation and the stage schedule remain in the host layer.

   This core is hostable by warmups that declare no intrinsic metric adaptation
   scheme (i.e. the metric core can be swapped without changing the host's step-
   size or schedule logic).  It is NOT wired into MEADS, whose fold-based metric
   is co-designed with its damping and step rules and cannot be factored out.

   :param init: ``(n_dims: int) -> MetricCoreState``.  Creates the initial mass-matrix
                adaptation state.  Closes over ``initial_inverse_mass_matrix`` and
                ``imm_shrinkage_to_previous`` when constructed via
                :meth:`MetricRecipe.build_core`.
   :type init: Callable
   :param update: ``(state, position: ArrayLikeTree, grad: ArrayLikeTree | None) -> MetricCoreState``.
                  Accumulates one sample.  For welford-path recipes ``grad`` is accepted
                  (interface uniformity) but ignored.
   :type update: Callable
   :param final: ``(state) -> MetricCoreState``.  Called at each slow-window boundary:
                 computes the new inverse mass matrix, writes it to
                 ``state.inverse_mass_matrix``, resets the accumulator.  The host
                 reads ``new_state.inverse_mass_matrix`` for the next window.
   :type final: Callable

   .. rubric:: Notes

   ``MetricCoreState`` is one of
   :class:`~blackjax.adaptation.mass_matrix.MassMatrixAdaptationState` or
   :class:`~blackjax.adaptation.mass_matrix.FisherMassMatrixAdaptationState` —
   the existing in-tree types; this core is a thin protocol wrapper, not a
   re-implementation.


   .. py:attribute:: init
      :type:  Callable


   .. py:attribute:: update
      :type:  Callable


   .. py:attribute:: final
      :type:  Callable


.. py:class:: LowRankMetricCoreState



   Scan-carry state for the low-rank mass-matrix MetricCore.

   Holds the current low-rank inverse mass matrix factors, the draw/gradient
   circular buffer, and the buffer bookkeeping counters.  The engine reads
   ``inverse_mass_matrix`` at each window boundary; the core's ``final()``
   updates it.

   :param inverse_mass_matrix: Current low-rank IMM as a
                               :class:`~blackjax.mcmc.metrics.LowRankInverseMassMatrix` NamedTuple
                               ``(sigma, U, lam)`` with shapes ``(d,)``, ``(d, max_rank)``,
                               ``(max_rank,)``.  This field is read by the engine at window boundaries
                               (via ``StagedAdaptationState.inverse_mass_matrix``) and by the MCMC
                               kernel's ``default_metric`` dispatch.
   :param mu_star: Optimal translation ``x̄ + σ² ⊙ ᾱ``, shape ``(d,)``.  Not part of the
                   engine's host protocol; the shim reads this from the last adaptation
                   state to re-initialize the chain after warmup.  Always zero for the
                   ``"sample_cov_low_rank"`` recipe (no optimal translation in that
                   estimator).
   :param draws_buffer: Circular buffer of chain positions, shape ``(buffer_size, d)``.
                        The first ``buffer_idx`` rows are valid; the remainder are zero-padded.
                        Dropped (replaced with ``None``) by the default OOM-guard
                        ``adaptation_info_fn`` in the shim to avoid O(num_steps × buffer_size × d)
                        allocations inside ``jax.lax.scan``.
   :param grads_buffer: Circular buffer of log-density gradients, shape ``(buffer_size, d)``.
                        Same layout and OOM-guard treatment as ``draws_buffer``.  Always zeros
                        for the ``"sample_cov_low_rank"`` recipe (not used by that estimator).
   :param buffer_idx: Number of draws written to the buffer so far (monotonically increasing,
                      NOT wrapped).  Modular indexing in ``update()`` handles wrap-around so
                      the most recent ``buffer_size`` draws are always in the buffer.
                      Reset to 0 by ``final()`` under the default ``"reset"`` buffer policy.
   :param background_split: Number of the buffer's leading rows considered "background" (for the
                            accumulating buffer policy only).  Always 0 under the default
                            ``"reset"`` policy.
   :param recompute_counter: Steps since the last metric recompute (for accumulating periodic
                             recompute only).  Always 0 under the default ``"reset"`` policy.


   .. py:attribute:: inverse_mass_matrix
      :type:  blackjax.mcmc.metrics.LowRankInverseMassMatrix


   .. py:attribute:: mu_star
      :type:  blackjax.types.Array


   .. py:attribute:: draws_buffer
      :type:  blackjax.types.Array


   .. py:attribute:: grads_buffer
      :type:  blackjax.types.Array


   .. py:attribute:: buffer_idx
      :type:  int


   .. py:attribute:: background_split
      :type:  int


   .. py:attribute:: recompute_counter
      :type:  int


.. py:function:: seed_low_rank_sigma_from_grad(state: LowRankMetricCoreState, grad: blackjax.types.ArrayLikeTree) -> LowRankMetricCoreState

   Seed the diagonal scale ``sigma`` from the initial log-density gradient.

   Implements nutpie's ``gradient_based_init`` logic: instead of starting from
   the identity (``sigma=1`` for every coordinate), set
   ``sigma_i = 1/sqrt(clip(|grad_i|, 1e-20, 1e20))`` so that the initial
   diagonal inverse mass matrix equals ``M^{-1}_i = sigma_i^2 = 1/|grad_i|``,
   matching ``M = diag(|grad|)`` (a diagonal Hessian approximation at the
   starting point; cf. L-BFGS and paper §3.1).

   Coordinates where ``|grad_i| < 1e-10`` fall back to ``sigma_i = 1.0``
   (identity) rather than the astronomically large ``sigma_i = 1e10`` that the
   raw formula would give.  This defends the real edge case of initialising
   at (or very near) a stationary point of the target — e.g. ``x=0`` on any
   centered/standardised density — where the gradient is exactly zero and an
   extreme initial scale causes near-certain divergence on the very first
   trajectory (root-caused via the Fisher 2×2 calibration study).

   This function is a **named seeding entry point** so that any host (the
   window-adaptation shim, ChEES, etc.) can call the same code path and the
   seeding logic is independently testable.

   :param state: Initial :class:`LowRankMetricCoreState` from ``core.init(n_dims)``
                 (before any gradient information).
   :param grad: Log-density gradient at the initial position.  Must be the same pytree
                structure as the chain's position.

   :returns: State with ``sigma`` replaced by the gradient-seeded values and
             ``inverse_mass_matrix`` updated accordingly (``U``/``lam`` unchanged,
             ``mu_star`` unchanged).
   :rtype: LowRankMetricCoreState


.. py:class:: MetricRecipe

   Configuration bundle for a mass-matrix adaptation recipe.

   Declares an (estimator, buffer, representation, support_gate) tuple with
   construction-time validation of the coupling contract (``needs ⊆ provides``
   and ``emits == representation``): incompatible combos fail at Python level
   with a clear message, never inside traced code.

   .. note::

       **Schema is provisional.**  The field types for ``estimator``,
       ``buffer``, and ``support_gate`` are string tags; future slices will
       replace these with proper constructor objects once the schema is stable
       across all recipe families.  This class is not exported at the
       ``blackjax`` top level — import directly from
       ``blackjax.adaptation.metric_recipes``.

   :param representation: The inverse-mass-matrix representation this recipe produces.
                          Slice-1 values: ``"diag"`` (1D array) or ``"dense"`` (2D array).
   :param estimator: String tag for the estimator function family.
                     Slice-1 values: ``"welford"``, ``"fisher_diag"``.
   :param buffer: String tag for the buffer/data-feeding policy.
                  Slice-1 value: ``"reset_window"``.
   :param support_gate: String tag for the support gate, or ``None`` (slice-1 default — no
                        gating beyond the estimator's intrinsic validation).
   :param needs: ``frozenset[str]`` declaring what data the estimator requires from the
                 buffer.  Validated at construction: ``needs ⊆ provides``.
                 Slice-1 values: ``frozenset({"positions"})`` or
                 ``frozenset({"positions", "gradients"})``.
   :param provides: ``frozenset[str]`` declaring what data the buffer provides.
   :param emits: Representation tag this estimator emits.  Validated at construction:
                 ``emits == representation``.
   :param provenance: Human-readable guidance string, stamped with benchmark evidence where
                      available.


   .. py:attribute:: representation
      :type:  str


   .. py:attribute:: estimator
      :type:  str


   .. py:attribute:: buffer
      :type:  str


   .. py:attribute:: support_gate
      :type:  str | None


   .. py:attribute:: needs
      :type:  frozenset


   .. py:attribute:: provides
      :type:  frozenset


   .. py:attribute:: emits
      :type:  str


   .. py:attribute:: provenance
      :type:  str


   .. py:attribute:: max_rank
      :type:  int | None
      :value: None



   .. py:attribute:: gamma
      :type:  float | None
      :value: None



   .. py:attribute:: cutoff
      :type:  float | None
      :value: None



   .. py:method:: build_core(*, imm_shrinkage_to_previous: float = 0.0, initial_inverse_mass_matrix: blackjax.types.Array | None = None, buffer_size: int | None = None) -> MetricCore

      Build an embeddable :class:`MetricCore` from this recipe.

      :param imm_shrinkage_to_previous: Pseudo-count controlling shrinkage of the per-window IMM toward the
                                        previous window's IMM.  Default ``0.0`` (Stan vanilla, no
                                        persistence).  Forwarded to
                                        :func:`~blackjax.adaptation.mass_matrix.mass_matrix_adaptation`.
                                        Not supported for ``"fisher_diag"`` or the low-rank recipes
                                        (``ValueError`` there per ``mass_matrix_adaptation``'s validation).
      :param initial_inverse_mass_matrix: Optional seed array for the initial inverse mass matrix.  ``None``
                                          (default) uses the standard identity initialisation (``ones(d)``
                                          for diagonal, ``identity(d)`` for dense).  Ignored for the
                                          low-rank estimators (``"fisher_low_rank"``, ``"sample_cov_low_rank"``).
      :param buffer_size: Required for low-rank recipes (``"fisher_low_rank"`` and
                          ``"sample_cov_low_rank"``).  Size of the circular draw/gradient
                          buffer (number of rows).  Use the schedule-derived heuristic in
                          :func:`~blackjax.adaptation.low_rank_adaptation.window_adaptation_low_rank`
                          or compute it yourself:
                          ``min(2 * max(num_steps // 5, 128), max(num_steps, 1))`` for the
                          reset policy; :func:`~blackjax.adaptation.low_rank_adaptation
                          ._accumulating_buffer_capacity` for the accumulating policy.
                          Ignored for diag/dense recipes.

      :returns: Embeddable init/update/final bundle ready for the engine.
      :rtype: MetricCore

      :raises ValueError: If the estimator tag is not supported, or if ``buffer_size`` is
          ``None`` for a low-rank estimator.



.. py:data:: REGISTRY
   :type:  dict[str, MetricRecipe]

.. py:function:: lookup_recipe(name: str) -> MetricRecipe

   Look up a named recipe from the :data:`REGISTRY`.

   :param name: Registry key.  Current names:

                - ``"welford_diag"`` (default; reproduces ``window_adaptation`` exactly)
                - ``"welford_dense"``
                - ``"fisher_diag"``
                - ``"fisher_low_rank"`` (Algorithm 1, seyboldt2026; needs ``buffer_size``)
                - ``"sample_cov_low_rank"`` (MEADS/Scheme-B; needs ``buffer_size``)

   :returns: The registered recipe for ``name``.
   :rtype: MetricRecipe

   :raises ValueError: If ``name`` is not in the registry, with a sorted list of known names.
       Pass a :class:`MetricRecipe` or :class:`MetricCore` directly for
       custom or experimental recipes that are not registry-stamped.


