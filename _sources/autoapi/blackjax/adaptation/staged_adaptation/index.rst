blackjax.adaptation.staged_adaptation
=====================================

.. py:module:: blackjax.adaptation.staged_adaptation

.. autoapi-nested-parse::

   Staged warmup adaptation engine for the HMC family.

   This module provides the :func:`staged_adaptation` engine and the
   :func:`build_schedule` function (previously in ``window_adaptation.py``;
   re-exported from there for backward compatibility).

   Architecture (layer doctrine)
   ------------------------------
   - :class:`StagedAdaptationState` — the scan-carry for the warmup.
   - :func:`_make_engine` — builds the HOST: stage schedule dispatching +
     step-size dual averaging + metric core hooks.  Only the MetricCore protocol
     crosses the host/core boundary.
   - :func:`staged_adaptation` — public entry point; accepts a recipe name, a
     :class:`~blackjax.adaptation.metric_recipes.MetricRecipe`, or a pre-built
     :class:`~blackjax.adaptation.metric_recipes.MetricCore`.

   The metric core (:class:`~blackjax.adaptation.metric_recipes.MetricCore`) is
   the separable, embeddable component — its init/update/final protocol runs on
   the engine's clock (Stan window schedule for slice 1).  Step-size dual
   averaging lives in the HOST layer (this module), not in the core.

   ``WindowAdaptationState`` in :mod:`~blackjax.adaptation.window_adaptation`
   is defined as ``WindowAdaptationState = StagedAdaptationState``.  Both names
   refer to the same class object; ``isinstance`` checks using either name continue
   to work.

   .. rubric:: Notes

   ``build_schedule`` is defined here (the canonical location) and re-exported
   from ``window_adaptation`` for backward compatibility.  Import it from either
   module; the object is identical.



Classes
-------

.. autoapisummary::

   blackjax.adaptation.staged_adaptation.StagedAdaptationState


Functions
---------

.. autoapisummary::

   blackjax.adaptation.staged_adaptation.build_schedule
   blackjax.adaptation.staged_adaptation.staged_adaptation


Module Contents
---------------

.. py:class:: StagedAdaptationState



   Scan-carry state for the staged adaptation engine.

   Field names intentionally mirror the previous
   ``WindowAdaptationState`` fields so that any downstream code accessing
   adaptation info by field name (``.ss_state``, ``.imm_state``, …)
   continues to work without modification.

   ``WindowAdaptationState`` in :mod:`blackjax.adaptation.window_adaptation`
   is an alias of this type (``WindowAdaptationState = StagedAdaptationState``);
   both names refer to the same NamedTuple class object, so ``isinstance``
   checks using either name are equivalent.

   :param ss_state: Current state of the dual-averaging step-size adaptation.
   :param imm_state: Current mass-matrix adaptation core state.  One of
                     :class:`~blackjax.adaptation.mass_matrix.MassMatrixAdaptationState`
                     or :class:`~blackjax.adaptation.mass_matrix.FisherMassMatrixAdaptationState`.
                     Typed ``Any`` here to avoid a hard dependency on the concrete types;
                     the MetricCore protocol guarantees the right type at construction time.
   :param step_size: Current (exponential-space) step-size estimate; read by the MCMC kernel
                     at every scan step.
   :param inverse_mass_matrix: Current inverse mass matrix; updated at each slow-window boundary and
                               read by the MCMC kernel at every scan step.


   .. py:attribute:: ss_state
      :type:  blackjax.adaptation.step_size.DualAveragingAdaptationState


   .. py:attribute:: imm_state
      :type:  Any


   .. py:attribute:: step_size
      :type:  float


   .. py:attribute:: inverse_mass_matrix
      :type:  blackjax.types.Array


.. py:function:: build_schedule(num_steps: int, initial_buffer_size: int = 75, final_buffer_size: int = 50, first_window_size: int = 25) -> list[tuple[int, bool]]

   Return the schedule for Stan's warmup.

   The schedule below is intended to be as close as possible to Stan's :cite:p:`stan_hmc_param`.
   The warmup period is split into three stages:

   1. An initial fast interval to reach the typical set. Only the step size is
   adapted in this window.
   2. "Slow" parameters that require global information (typically covariance)
   are estimated in a series of expanding intervals with no memory; the step
   size is re-initialized at the end of each window. Each window is twice the
   size of the preceding window.
   3. A final fast interval during which the step size is adapted using the
   computed mass matrix.

   Schematically:

   ```
   +---------+---+------+------------+------------------------+------+
   |  fast   | s | slow |   slow     |        slow            | fast |
   +---------+---+------+------------+------------------------+------+
   ```

   The distinction slow/fast comes from the speed at which the algorithms
   converge to a stable value; in the common case, estimation of covariance
   requires more steps than dual averaging to give an accurate value. See :cite:p:`stan_hmc_param`
   for a more detailed explanation.

   Fast intervals are given the label 0 and slow intervals the label 1.

   :param num_steps: The number of warmup steps to perform.
   :type num_steps: int
   :param initial_buffer_size: The width of the initial fast adaptation interval.
   :type initial_buffer_size: int
   :param first_window_size: The width of the first slow adaptation interval.
   :type first_window_size: int
   :param final_buffer_size: The width of the final fast adaptation interval.
   :type final_buffer_size: int

   :rtype: A list of tuples (window_label, is_middle_window_end).


.. py:function:: staged_adaptation(algorithm, logdensity_fn: Callable, metric: str | blackjax.adaptation.metric_recipes.MetricRecipe | blackjax.adaptation.metric_recipes.MetricCore = 'welford_diag', *, max_grad_budget: int | None = None, imm_shrinkage_to_previous: float = 0.0, initial_inverse_mass_matrix: blackjax.types.Array | None = None, initial_step_size: float = 1.0, target_acceptance_rate: float = 0.8, adaptation_info_fn: Callable = return_all_adapt_info, integrator=mcmc.integrators.velocity_verlet, schedule_fn: Callable | None = None, initial_metric_state: Any = None, **extra_parameters) -> blackjax.base.AdaptationAlgorithm

   Adapt the step size and inverse mass matrix for HMC-family algorithms.

   The :func:`staged_adaptation` engine implements the same Stan warmup
   schedule as :func:`~blackjax.adaptation.window_adaptation.window_adaptation`
   but exposes a composable :class:`~blackjax.adaptation.metric_recipes.MetricCore`
   interface for the mass-matrix adaptation component.  The step-size dual-averaging
   and the stage schedule live in the HOST (this function); the mass-matrix
   estimation is fully delegated to the ``metric`` argument.

   :param algorithm: An algorithm from the HMC family (e.g. :data:`blackjax.nuts`,
                     :data:`blackjax.hmc`).  The algorithm's ``build_kernel`` method is
                     inspected to decide whether to pass an integrator.
   :param logdensity_fn: The log density probability density function to sample.
   :param metric: The mass-matrix adaptation specification.  Accepts:

                  - ``"auto"`` — the meta-adaptation controller
                    (:mod:`~blackjax.adaptation.meta_adaptation`). Automatically selects
                    the diagonal vs low-rank path and the growing-window schedule.
                    Requires ``max_grad_budget`` to be set.  The emitted metric is always
                    a :class:`~blackjax.mcmc.metrics.LowRankInverseMassMatrix` (with
                    U=0, lam=1 when the controller stays diagonal — bit-equivalent to
                    the diagonal metric).

                    .. warning::
                       ``metric="auto"`` is **experimental (v1)**.  The low-rank
                       escalation is not robustly calibrated at high dimension: when
                       the residual spectrum's dominant structure sits near the detection
                       boundary, whether the controller escalates can depend on the
                       random seed.  Use for exploration and algorithm development, not
                       for production efficiency claims.  A multi-chain escalation
                       trigger (planned for v2) is expected to make the decision robust.
                  - **str** — a registry name (``"welford_diag"`` (default),
                    ``"welford_dense"``, ``"fisher_diag"``); looked up via
                    :func:`~blackjax.adaptation.metric_recipes.lookup_recipe` and built
                    with ``imm_shrinkage_to_previous`` and ``initial_inverse_mass_matrix``.
                  - :class:`~blackjax.adaptation.metric_recipes.MetricRecipe` — built
                    with ``imm_shrinkage_to_previous`` and ``initial_inverse_mass_matrix``.
                  - :class:`~blackjax.adaptation.metric_recipes.MetricCore` — used
                    directly as-is; ``imm_shrinkage_to_previous`` and
                    ``initial_inverse_mass_matrix`` are ignored (closed over in the core).
   :param max_grad_budget: Maximum total gradient budget (leapfrog evaluations).  Required when
                           ``metric="auto"``; ignored otherwise.  The meta-adaptation controller
                           converts this to a warmup step count via a conservative divisor (see
                           :mod:`~blackjax.adaptation.meta_adaptation`).  Passed as-is; use
                           :func:`~blackjax.adaptation.meta_adaptation.extract_meta_verdict` after
                           ``warmup.run()`` to get the structured routing verdict and true gradient
                           counts.
   :param imm_shrinkage_to_previous: Pseudo-count controlling shrinkage of the per-window IMM toward the
                                     previous window's IMM (Bayesian persistence).  Default ``0.0``
                                     reproduces Stan's per-window-reset behavior exactly.  Ignored when
                                     ``metric`` is a :class:`~blackjax.adaptation.metric_recipes.MetricCore`.
   :param initial_inverse_mass_matrix: Optional seed array for the initial inverse mass matrix.  Ignored when
                                       ``metric`` is a :class:`~blackjax.adaptation.metric_recipes.MetricCore`.
   :param initial_step_size: Step size used to seed the dual-averaging adaptation.
   :param target_acceptance_rate: Target Metropolis acceptance rate for step-size adaptation.  Default
                                  ``0.80`` (Stan default).
   :param adaptation_info_fn: Function to select the adaptation info returned at each step.  See
                              :func:`~blackjax.adaptation.base.return_all_adapt_info` and
                              :func:`~blackjax.adaptation.base.get_filter_adapt_info_fn`.  By default
                              all information is saved — this can result in excessive memory usage
                              if the information is unused.
   :param integrator: The symplectic integrator passed to ``algorithm.build_kernel``; only
                      used if ``build_kernel`` accepts arguments.  Defaults to
                      :func:`~blackjax.mcmc.integrators.velocity_verlet`.
   :param schedule_fn: Callable ``(num_steps: int) -> Array`` that returns a
                       ``(num_steps, 2)`` array of ``(stage, is_window_end)`` pairs, or
                       ``None`` (default) to use the path-appropriate default.
                       When ``None`` and ``metric="auto"``, the default is
                       :func:`~blackjax.adaptation.low_rank_adaptation.build_growing_window_schedule`
                       (nutpie's proportional-to-tune, 1.5×-growing-window schedule).
                       When ``None`` and any other ``metric``, the default is
                       :func:`build_schedule` (Stan's fixed-absolute, 2×-doubling schedule).
                       An explicit callable is always honored regardless of ``metric``.
   :param initial_metric_state: Optional pre-built mass-matrix adaptation core state.  When not
                                ``None``, overrides the ``metric_core.init(n_dims)`` call at warmup
                                start — the provided state is used as-is.  The object must be a
                                valid state for the chosen ``metric`` core (its
                                ``inverse_mass_matrix`` field is unpacked into
                                :class:`StagedAdaptationState` immediately).  Intended for callers
                                that seed the initial state from external data (e.g., gradient-based
                                diagonal-scale initialisation); ``None`` (the default) reproduces
                                the standard identity/zero initialisation.
   :param \*\*extra_parameters: Additional parameters forwarded to the MCMC kernel at every step, e.g.
                                ``num_integration_steps`` for HMC.

   :returns: An :class:`~blackjax.base.AdaptationAlgorithm` wrapping a ``run``
             function with signature ``(rng_key, position, num_steps=1000)`` that
             returns ``(AdaptationResults, info)``.
   :rtype: AdaptationAlgorithm

   .. rubric:: Notes

   Wrap ``warmup.run(...)`` in :func:`blackjax.progress_bar` to display a
   progress bar, e.g. ``with blackjax.progress_bar(): warmup.run(...)``.

   .. seealso::

      :py:obj:`blackjax.adaptation.window_adaptation.window_adaptation`
          Thin compatibility shim over this engine; preserves the old parameter interface exactly.

      :py:obj:`blackjax.adaptation.metric_recipes.REGISTRY`
          Registry of named :class:`~blackjax.adaptation.metric_recipes.MetricRecipe` objects for the ``metric`` string argument.


