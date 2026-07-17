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
"""Window schedule for the multi-chain meta-adaptation controller."""
from __future__ import annotations

from blackjax.adaptation.meta._calibration import _MIN_TRAIN_K_RATIO
from blackjax.types import Array


def _build_mc_window_schedule(num_steps: int, M: int, actual_rank: int) -> Array:
    """Pooled-aware growing-window schedule for the M-chain meta-adaptation path.

    The single-chain schedule's 30%-early phase and 80-step starting window are
    sized for a single chain.  With M chains the detection-relevant count is the
    POOLED ``M * n``, so windows should be sized on pooled samples: the first
    main window is ``n1 = ceil(min_n_proj / M)`` per chain, ensuring
    ``n_pool = n1 * M >= min_n_proj = 8 * (actual_rank + 1)`` (projected-tier
    floor).  This restores early-escalation capability that the single-chain
    schedule loses at M >= 4.

    Example (M=8, actual_rank=25, num_steps=312):
        ``n1 = ceil(208 / 8) = 26``.
        Windows end at steps 1, 27, 66, 124, 265.
        Steps 27, 66, 124 have n_pool ≥ 208 AND budget_remaining ≥ 50 —
        all three are escalation-eligible.

    The ``early_window=0.0`` argument to :func:`build_growing_window_schedule`
    gives a harmless 1-step early window (due to the ``max(..., 1)`` guard in
    that function) followed by the main pooled-aware phase.

    Parameters
    ----------
    num_steps
        Per-chain warmup step count (= total_budget // (LEAPS * M)).
    M
        Number of chains (static Python int).
    actual_rank
        Estimated rank capacity (static Python int; = min(d//2, _MAX_RANK_CAP)).

    Returns
    -------
    Array
        ``(num_steps, 2)`` schedule array in the same ``(stage, is_window_end)``
        format as :func:`~blackjax.adaptation.low_rank_adaptation.build_growing_window_schedule`.
    """
    from blackjax.adaptation.low_rank_adaptation import build_growing_window_schedule

    min_n_proj = 2 * _MIN_TRAIN_K_RATIO * (actual_rank + 1)  # 8*(actual_rank+1)
    n1 = max(-(-min_n_proj // M), 1)  # ceil division, ensure ≥ 1
    return build_growing_window_schedule(
        num_steps,
        early_window=0.0,  # suppress early phase; harmless 1-step leftover is fine
        window_size=n1,
        window_growth=1.5,
    )
