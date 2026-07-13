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

   Usage::

       # String sugar (registry lookup):
       wu = staged_adaptation(nuts, logdensity_fn, metric="welford_diag")

       # Recipe constructor (testability / custom configuration):
       from blackjax.adaptation.metric_recipes import REGISTRY
       core = REGISTRY["welford_diag"].build_core(imm_shrinkage_to_previous=5.0)
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
   - Step-size proxy independence: for diag/dense representations (this slice),
     the dual-averaging acceptance-rate proxy is not an eigenvalue proxy of the
     adapted metric, so the step-size and metric adaptation are decoupled.  A
     ``diag_reference`` accessor will be added in the low-rank slice where this
     independence no longer holds.

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
   blackjax.adaptation.metric_recipes.MetricRecipe


Functions
---------

.. autoapisummary::

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


   .. py:method:: build_core(*, imm_shrinkage_to_previous: float = 0.0, initial_inverse_mass_matrix: blackjax.types.Array | None = None) -> MetricCore

      Build an embeddable :class:`MetricCore` from this recipe.

      :param imm_shrinkage_to_previous: Pseudo-count controlling shrinkage of the per-window IMM toward the
                                        previous window's IMM.  Default ``0.0`` (Stan vanilla, no
                                        persistence).  Forwarded to
                                        :func:`~blackjax.adaptation.mass_matrix.mass_matrix_adaptation`.
                                        Not supported for ``"fisher_diag"`` (``ValueError`` there per
                                        ``mass_matrix_adaptation``'s validation).
      :param initial_inverse_mass_matrix: Optional seed array for the initial inverse mass matrix.  ``None``
                                          (default) uses the standard identity initialisation (``ones(d)``
                                          for diagonal, ``identity(d)`` for dense).

      :returns: Embeddable init/update/final bundle ready for the engine.
      :rtype: MetricCore

      :raises ValueError: If the estimator tag is not supported by this slice.



.. py:data:: REGISTRY
   :type:  dict[str, MetricRecipe]

.. py:function:: lookup_recipe(name: str) -> MetricRecipe

   Look up a named recipe from the :data:`REGISTRY`.

   :param name: Registry key.  Current slice-1 names:

                - ``"welford_diag"`` (default; reproduces ``window_adaptation`` exactly)
                - ``"welford_dense"``
                - ``"fisher_diag"``

   :returns: The registered recipe for ``name``.
   :rtype: MetricRecipe

   :raises ValueError: If ``name`` is not in the registry, with a sorted list of known names.
       Pass a :class:`MetricRecipe` or :class:`MetricCore` directly for
       custom or experimental recipes that are not registry-stamped.


