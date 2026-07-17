# Copyright 2020- The Blackjax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Meta-adaptation controller for the HMC-family warmup.

At each window boundary the controller computes two signals: (1) held-out
score-linearity R² — the curvature gate (funnel R²≈0.007 vs ≥0.54 for all
metric-fixable classes); (2) S_gap(k) = λ₁/λ_{k+1} of the Welford-whitened
residual — the magnitude predictor (Spearman 1.0 with measured rank-k payoff).
Escalate diagonal → rank-k iff R² ≥ _R_MIN AND S_gap ≥ _S_MIN AND stable
over two consecutive windows AND budget deadline clear. Growing-window schedule
(nutpie-style) is the default; AIRM-velocity early exit is advisory
(``converged_at_step`` records where stopping would have helped).

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
``jax.config.update("jax_enable_x64", True)`` for production use.

See :mod:`blackjax.adaptation.metric_recipes` for the MetricCore protocol and
:mod:`blackjax.adaptation.staged_adaptation` for the host engine.

Submodule layout
----------------
:mod:`~blackjax.adaptation.meta._state`
    State NamedTuples: MetaAdaptationCoreState, MultiChainMetaAdaptationCoreState,
    MetaAdaptationVerdict.
:mod:`~blackjax.adaptation.meta._calibration`
    All module constants and swappable calibration functions.
:mod:`~blackjax.adaptation.meta._signals`
    Single-chain signal computation (spectrum, S_gap, R², mixing).
:mod:`~blackjax.adaptation.meta._detection`
    Multi-chain detection statistics (within-chain, between-chain, LOO, gap-stat).
:mod:`~blackjax.adaptation.meta._router`
    Router functions (geometric-mean scale, projected-gain R², PC-centered pool).
:mod:`~blackjax.adaptation.meta._schedule`
    Multi-chain growing-window schedule.
:mod:`~blackjax.adaptation.meta.builders`
    Core builder functions: build_meta_adaptation_core, build_multi_chain_meta_core.
:mod:`~blackjax.adaptation.meta.verdict`
    Post-run verdict extractors: extract_meta_verdict, extract_multi_chain_verdict.
"""
from blackjax.adaptation.meta._calibration import (
    _mc_detection_edge,
    _mc_unimodality_threshold,
)
from blackjax.adaptation.meta._detection import (
    _between_chain_detection,
    _compute_within_chain_stats,
)
from blackjax.adaptation.meta._state import (
    MetaAdaptationCoreState,
    MetaAdaptationVerdict,
    MultiChainMetaAdaptationCoreState,
)
from blackjax.adaptation.meta.builders import (
    build_meta_adaptation_core,
    build_multi_chain_meta_core,
)
from blackjax.adaptation.meta.verdict import (
    extract_meta_verdict,
    extract_multi_chain_verdict,
)

__all__ = [
    "MetaAdaptationCoreState",
    "MetaAdaptationVerdict",
    "MultiChainMetaAdaptationCoreState",
    "build_meta_adaptation_core",
    "build_multi_chain_meta_core",
    "extract_meta_verdict",
    "extract_multi_chain_verdict",
    "_between_chain_detection",
    "_compute_within_chain_stats",
    "_mc_detection_edge",
    "_mc_unimodality_threshold",
]
