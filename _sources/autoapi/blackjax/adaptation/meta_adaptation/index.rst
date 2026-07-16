blackjax.adaptation.meta_adaptation
===================================

.. py:module:: blackjax.adaptation.meta_adaptation

.. autoapi-nested-parse::

   Meta-adaptation controller for the HMC-family warmup.

   At each window boundary the controller computes two signals: (1) held-out
   score-linearity R² — the curvature gate (funnel R²≈0.007 vs ≥0.54 for all
   metric-fixable classes); (2) S_gap(k) = λ₁/λ_{k+1} of the Welford-whitened
   residual — the magnitude predictor (Spearman 1.0 with measured rank-k payoff).
   Escalate diagonal → rank-k iff R² ≥ _R_MIN AND S_gap ≥ _S_MIN AND stable
   over two consecutive windows AND budget deadline clear. Growing-window schedule
   (nutpie-style) is the default; AIRM-velocity early exit is advisory in v1
   (the scan runs its full length; ``converged_at_step`` records where stopping
   would have helped — the actual early-stop host is the named v1.1 upgrade).

   .. warning::

      ``metric="auto"`` is **experimental (v1)**.  The low-rank escalation is not
      robustly calibrated at high dimension: when the residual spectrum's dominant
      structure sits near the detection boundary, whether the controller escalates
      can depend on the random seed used for sampling.  Use ``metric="auto"`` for
      exploration and algorithm development, not for production efficiency claims.
      A multi-chain escalation trigger (planned for v2) is expected to make the
      escalation decision robust across seeds.

   **Dtype note**: the composed estimator ``_compute_low_rank_metric`` produces
   numerically indefinite metrics under float32 (~98% of runs). Enable x64 via
   ``jax.config.update("jax_enable_x64", True)`` for production use and for the
   production use and for numerical-precision-sensitive acceptance runs; all optpath harnesses ran with x64 enabled.

   See :mod:`blackjax.adaptation.metric_recipes` for the MetricCore protocol and
   :mod:`blackjax.adaptation.staged_adaptation` for the host engine.



Classes
-------

.. autoapisummary::

   blackjax.adaptation.meta_adaptation.MetaAdaptationCoreState
   blackjax.adaptation.meta_adaptation.MetaAdaptationVerdict


Functions
---------

.. autoapisummary::

   blackjax.adaptation.meta_adaptation.build_meta_adaptation_core
   blackjax.adaptation.meta_adaptation.extract_meta_verdict


Module Contents
---------------

.. py:class:: MetaAdaptationCoreState



   Scan-carry state for the meta-adaptation MetricCore.

   Buffer fields mirror ``LowRankMetricCoreState`` so the state is
   interchangeable in the staged_adaptation engine.  The ``inverse_mass_matrix``
   is always a :class:`~blackjax.mcmc.metrics.LowRankInverseMassMatrix`; before
   escalation, U=0 and lam=1 (bit-equivalent to the diagonal metric).


   .. py:attribute:: inverse_mass_matrix
      :type:  blackjax.mcmc.metrics.LowRankInverseMassMatrix


   .. py:attribute:: mu_star
      :type:  blackjax.types.Array


   .. py:attribute:: draws_buffer
      :type:  blackjax.types.Array


   .. py:attribute:: grads_buffer
      :type:  blackjax.types.Array


   .. py:attribute:: buffer_idx
      :type:  blackjax.types.Array


   .. py:attribute:: background_split
      :type:  blackjax.types.Array


   .. py:attribute:: recompute_counter
      :type:  blackjax.types.Array


   .. py:attribute:: has_escalated
      :type:  blackjax.types.Array


   .. py:attribute:: escalation_rank
      :type:  blackjax.types.Array


   .. py:attribute:: s_gap_prev
      :type:  blackjax.types.Array


   .. py:attribute:: s_gap_curr
      :type:  blackjax.types.Array


   .. py:attribute:: r2_latest
      :type:  blackjax.types.Array


   .. py:attribute:: r2_mode
      :type:  blackjax.types.Array


   .. py:attribute:: budget_used
      :type:  blackjax.types.Array


   .. py:attribute:: converged_at_step
      :type:  blackjax.types.Array


   .. py:attribute:: prev_lam
      :type:  blackjax.types.Array


   .. py:attribute:: airm_vel_prev
      :type:  blackjax.types.Array


   .. py:attribute:: airm_vel_curr
      :type:  blackjax.types.Array


   .. py:attribute:: is_slow_mixing
      :type:  blackjax.types.Array


.. py:class:: MetaAdaptationVerdict



   Verdict emitted by :func:`extract_meta_verdict` after the warmup scan.

   All budget numbers are in warmup steps (step-clock proxy) unless the
   info stream is provided for true gradient counts.

   ``budget_returned_steps`` is ADVISORY in v1: the scan runs its full length;
   a stopping host (lax.while) is the named v1.1 upgrade.


   .. py:attribute:: route
      :type:  str


   .. py:attribute:: metric
      :type:  blackjax.mcmc.metrics.LowRankInverseMassMatrix


   .. py:attribute:: effective_rank
      :type:  int


   .. py:attribute:: confidence
      :type:  str


   .. py:attribute:: exit_reason
      :type:  str


   .. py:attribute:: budget_used_steps
      :type:  int


   .. py:attribute:: budget_returned_steps
      :type:  int


   .. py:attribute:: budget_used_grads
      :type:  int


   .. py:attribute:: r2_final
      :type:  float


   .. py:attribute:: s_gap_final
      :type:  float


   .. py:attribute:: transient_mixing_class
      :type:  str


   .. py:attribute:: buffer_policy
      :type:  str


   .. py:attribute:: flags
      :type:  dict


.. py:function:: build_meta_adaptation_core(max_grad_budget: int, *, max_rank: int | None = None, gamma: float = 1e-05, cutoff: float = 2.0) -> blackjax.adaptation.metric_recipes.MetricCore

   Build the meta-adaptation :class:`~blackjax.adaptation.metric_recipes.MetricCore`.

   :param max_grad_budget: Maximum total gradient budget (leapfrog evaluations).  Converted to
                           warmup steps via ``_ASSUMED_AVG_LEAPFROGS_PER_STEP`` at Python time.
   :param max_rank: Maximum low-rank rank; ``None`` uses :data:`_MAX_RANK_CAP`.
   :param gamma: Fisher-estimator parameters; defaults match ``fisher_low_rank`` recipe.
   :param cutoff: Fisher-estimator parameters; defaults match ``fisher_low_rank`` recipe.

   :returns: Embeddable init/update/final bundle.
   :rtype: MetricCore


.. py:function:: extract_meta_verdict(final_state: MetaAdaptationCoreState, max_grad_budget: int, num_warmup_steps: int, adaptation_info: Any = None) -> MetaAdaptationVerdict

   Build a :class:`MetaAdaptationVerdict` from the final core state.

   Call after ``warmup.run()`` completes.  Pass ``adaptation_info`` (the
   second return of ``warmup.run()``) for true gradient counts.

   ``budget_returned_steps`` is ADVISORY: the scan runs its full length in v1;
   this field shows where a stopping host would have saved steps.


