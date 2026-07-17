blackjax.adaptation.meta.builders
=================================

.. py:module:: blackjax.adaptation.meta.builders

.. autoapi-nested-parse::

   Core builders for the meta-adaptation controller.

   This file is the primary entry point for understanding the controller decision
   logic.  Read :mod:`~blackjax.adaptation.meta._calibration` for the calibration
   surface (all thresholds and swappable gate functions).

   Functions
   ---------
   :func:`build_meta_adaptation_core` — single-chain MetricCore builder.
   :func:`build_multi_chain_meta_core` — multi-chain MetricCore builder.



Functions
---------

.. autoapisummary::

   blackjax.adaptation.meta.builders.build_meta_adaptation_core
   blackjax.adaptation.meta.builders.build_multi_chain_meta_core


Module Contents
---------------

.. py:function:: build_meta_adaptation_core(max_grad_budget: int, *, max_rank: int | None = None, gamma: float = 1e-05, cutoff: float = 2.0) -> blackjax.adaptation.metric_recipes.MetricCore

   Build the meta-adaptation :class:`~blackjax.adaptation.metric_recipes.MetricCore`.

   :param max_grad_budget: Maximum total gradient budget (leapfrog evaluations).  Converted to
                           warmup steps via ``_ASSUMED_AVG_LEAPFROGS_PER_STEP`` at Python time.
   :param max_rank: Maximum low-rank rank; ``None`` uses :data:`~blackjax.adaptation.meta._calibration._MAX_RANK_CAP`.
   :param gamma: Fisher-estimator parameters; defaults match ``fisher_low_rank`` recipe.
   :param cutoff: Fisher-estimator parameters; defaults match ``fisher_low_rank`` recipe.

   :returns: Embeddable init/update/final bundle.
   :rtype: MetricCore


.. py:function:: build_multi_chain_meta_core(max_grad_budget: int, n_chains: int = _MULTI_CHAIN_DEFAULT_N_CHAINS, *, max_rank: int | None = None, gamma: float = 1e-05, cutoff: float = 2.0) -> blackjax.adaptation.metric_recipes.MetricCore

   Build the multi-chain meta-adaptation :class:`~blackjax.adaptation.metric_recipes.MetricCore`.

   Runs M independent chains sharing one adapted metric; the escalation
   decision uses pooled M-chain information instead of a single-chain
   stability check.  The pooled between-chain signal makes the escalation
   decision robust to seed variation for near-edge posterior structures.

   The multi-chain gate (replaces the single-chain S_gap-stability check).
   Five conditions must all hold to escalate:

   1. **Magnitude.** Top eigenvalue of the between-chain T matrix exceeds the
      detection edge ``(1 + √(d/(M−1)))²`` (M−1 dof, grand-mean constraint).
   2. **Collinearity.** Fraction of total between-chain scatter in the top
      singular direction f₁ ≥ :data:`~blackjax.adaptation.meta._calibration._MC_COLLINEARITY_TOL`.
      Genuine slow directions produce near-rank-1 concentration (f₁→1);
      isotropic spurious scatter gives f₁ ≈ 1/(M−1).
   3. **Leave-one-out.** Detection must survive dropping any single chain,
      preventing a single outlier chain from driving the verdict.  Leave-two-out
      (dropping any pair) is subsumed by the collinearity + unimodality conjunction
      for the aligned-pair threat model and is deferred to v2.1.
   4. **Support floor.** At least one spike is admitted (k ≥ 1).
   5. **Unimodality guard.** Gap-statistic on the projected chain-means must
      not flag mode-split; mode-separated chains are deferred to the ensemble
      (Paper-3 scope) and reported via ``deferred_to_ensemble=True`` in the
      verdict.

   Plus R² curvature gate and budget deadline (same as single-chain).

   Budget re-allocation: ``max_grad_budget`` is the TOTAL gradient budget,
   shared across all M chains.  Providing ``n_chains=M`` overdispersed
   starting positions to ``run()`` causes each chain to run for
   ``total // M`` leapfrog evaluations — the total cost equals the
   single-chain budget, not M× it.

   For ``n_chains=1`` use :func:`build_meta_adaptation_core` directly to
   obtain exact single-chain (v1) behaviour; the ``staged_adaptation`` engine
   routes to it automatically when ``n_chains=1``.

   :param max_grad_budget: Maximum total gradient budget (leapfrog evaluations) across all M chains.
   :param n_chains: Number of independent chains.  Must be ≥ 2.  Defaults to
                    :data:`~blackjax.adaptation.meta._calibration._MULTI_CHAIN_DEFAULT_N_CHAINS` (8).
   :param max_rank: Same as :func:`build_meta_adaptation_core`.
   :param gamma: Same as :func:`build_meta_adaptation_core`.
   :param cutoff: Same as :func:`build_meta_adaptation_core`.

   :returns: Embeddable init/update/final bundle.  ``update`` expects ``position``
             of shape ``(n_chains, d)`` and ``grad`` of shape ``(n_chains, d)``.
   :rtype: MetricCore


