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
"""Back-compatibility shim — use :mod:`blackjax.adaptation.meta` instead.

.. deprecated::
    Importing from ``blackjax.adaptation.meta_adaptation`` is deprecated.
    The module has been reorganised into the
    :mod:`blackjax.adaptation.meta` sub-package.  All public names are
    still importable; update your imports to use the new location:

    - ``from blackjax.adaptation.meta import build_meta_adaptation_core``
    - ``from blackjax.adaptation.meta import build_multi_chain_meta_core``
    - ``from blackjax.adaptation.meta import extract_meta_verdict``
    - ``from blackjax.adaptation.meta import extract_multi_chain_verdict``
    - ``from blackjax.adaptation.meta._calibration import _ASSUMED_AVG_LEAPFROGS_PER_STEP``
    - … (see :mod:`blackjax.adaptation.meta` for the full public surface)
"""
from __future__ import annotations

import warnings as _warnings

_warnings.warn(
    "blackjax.adaptation.meta_adaptation is deprecated; "
    "import from blackjax.adaptation.meta instead.",
    DeprecationWarning,
    stacklevel=1,
)

# Re-export the full public surface so that existing code continues to work.
from blackjax.adaptation.meta._calibration import (  # noqa: E402, F401
    _AIRM_VELOCITY_TOL,
    _ASSUMED_AVG_LEAPFROGS_PER_STEP,
    _DETECTION_BRANCH_BETWEEN_MEANS,
    _DETECTION_BRANCH_BOTH,
    _DETECTION_BRANCH_NONE,
    _DETECTION_BRANCH_POOLED_WITHIN,
    _GAIN_READABILITY_FLOOR,
    _GAIN_THRESHOLD,
    _LAM_NONTRIVIAL_TOL,
    _MAX_RANK_CAP,
    _MC_COLLINEARITY_TOL,
    _MC_MIN_CHAINS,
    _MC_UNIMODALITY_CONFIRM_WINDOWS,
    _MC_UNIMODALITY_Q99_TABLE,
    _MIN_TRAIN_D_RATIO,
    _MIN_TRAIN_K_RATIO,
    _MULTI_CHAIN_DEFAULT_N_CHAINS,
    _R2_DEFERRED,
    _R2_FULL_AFFINE,
    _R2_PROJECTED,
    _R_MIN,
    _S_GAP_STABILITY_TOL,
    _S_MIN,
    _STEP_SIZE_READAPT_BUFFER,
    _TRANSIENT_MIXING_THRESHOLD,
    _W_BRANCH_NULL_EDGE_TW_FACTOR,
    _W_BRANCH_PSI_FLOOR,
    _W_BRANCH_R1_TOL,
    _mc_detection_edge,
    _mc_unimodality_threshold,
    _w_branch_null_edge,
    _w_branch_psi_threshold,
)
from blackjax.adaptation.meta._detection import (  # noqa: E402, F401
    _between_chain_detection,
    _compute_chain_consistency_psi,
    _compute_contraction_stat,
    _compute_lag1_autocorr_top_dir,
    _compute_mode_consistency_flag,
    _compute_pooled_within_spectrum,
    _compute_within_chain_stats,
    _loo_detection_passes,
    _unimodality_gap_stat,
)
from blackjax.adaptation.meta._router import (  # noqa: E402, F401
    _build_pc_centered_time_major_pool,
    _compute_projected_gain_r2_mc,
    _geometric_mean_deploy_scale,
)
from blackjax.adaptation.meta._schedule import (  # noqa: E402, F401
    _build_mc_window_schedule,
)
from blackjax.adaptation.meta._signals import (  # noqa: E402, F401
    _choose_rank,
    _compute_r2_score_linearity,
    _compute_s_gap,
    _compute_transient_mixing_signal,
    _compute_whitened_spectrum,
)
from blackjax.adaptation.meta._state import (  # noqa: E402, F401
    MetaAdaptationCoreState,
    MetaAdaptationVerdict,
    MultiChainMetaAdaptationCoreState,
)
from blackjax.adaptation.meta.builders import (  # noqa: E402, F401
    build_meta_adaptation_core,
    build_multi_chain_meta_core,
)
from blackjax.adaptation.meta.verdict import (  # noqa: E402, F401
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
