blackjax.adaptation.meta.verdict
================================

.. py:module:: blackjax.adaptation.meta.verdict

.. autoapi-nested-parse::

   Post-run verdict extractors for the meta-adaptation controller.

   Functions
   ---------
   :func:`extract_meta_verdict` — single-chain verdict from
       :class:`~blackjax.adaptation.meta._state.MetaAdaptationCoreState`.
   :func:`extract_multi_chain_verdict` — multi-chain verdict from
       :class:`~blackjax.adaptation.meta._state.MultiChainMetaAdaptationCoreState`.



Functions
---------

.. autoapisummary::

   blackjax.adaptation.meta.verdict.extract_meta_verdict
   blackjax.adaptation.meta.verdict.extract_multi_chain_verdict


Module Contents
---------------

.. py:function:: extract_meta_verdict(final_state: blackjax.adaptation.meta._state.MetaAdaptationCoreState, max_grad_budget: int, num_warmup_steps: int, adaptation_info: Any = None) -> blackjax.adaptation.meta._state.MetaAdaptationVerdict

   Build a :class:`~blackjax.adaptation.meta._state.MetaAdaptationVerdict` from the final core state.

   Call after ``warmup.run()`` completes.  Pass ``adaptation_info`` (the
   second return of ``warmup.run()``) for true gradient counts.

   ``budget_returned_steps`` is ADVISORY: the scan runs its full length in v1;
   this field shows where a stopping host would have saved steps.


.. py:function:: extract_multi_chain_verdict(final_state: blackjax.adaptation.meta._state.MultiChainMetaAdaptationCoreState, max_grad_budget: int, num_warmup_steps: int, adaptation_info: Any = None, *, pooled_draws_by_window: Any = None) -> blackjax.adaptation.meta._state.MetaAdaptationVerdict

   Build a :class:`~blackjax.adaptation.meta._state.MetaAdaptationVerdict` from a multi-chain final core state.

   Drop-in counterpart of :func:`extract_meta_verdict` for the
   :class:`~blackjax.adaptation.meta._state.MultiChainMetaAdaptationCoreState` produced by
   :func:`~blackjax.adaptation.meta.builders.build_multi_chain_meta_core`.

   :param final_state: Final :class:`~blackjax.adaptation.meta._state.MultiChainMetaAdaptationCoreState` after ``warmup.run()``.
   :param max_grad_budget: Same semantics as :func:`extract_meta_verdict`.
   :param num_warmup_steps: Same semantics as :func:`extract_meta_verdict`.
   :param adaptation_info: Same semantics as :func:`extract_meta_verdict`.
   :param pooled_draws_by_window: Optional per-window pooled draw array exposed for nested-R̂ diagnostics
                                  (shape ``(n_chains, n_per_window, d)`` per window).  Not used internally
                                  — passed through as ``flags["pooled_draws_by_window"]`` for the
                                  evaluation layer.

   :returns: Verdict with multi-chain–specific flags: ``n_chains``,
             ``chain_collinearity``, ``unimodality_gate``, ``deferred_to_ensemble``,
             and ``mode_coverage="multi_chain_certified"`` when all gates passed.
   :rtype: MetaAdaptationVerdict


