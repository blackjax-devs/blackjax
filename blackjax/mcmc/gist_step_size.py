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
"""Back-compatibility shim — use :mod:`blackjax.mcmc.composed.step_size` instead.

.. deprecated::
    Importing from ``blackjax.mcmc.gist_step_size`` is deprecated. The module
    has been reorganised into the :mod:`blackjax.mcmc.composed` sub-package.
    All public names are still importable; update your imports to use the new
    location:

    - ``from blackjax.mcmc.composed import step_size``
    - ``from blackjax.mcmc.composed.step_size import GISTStepSizeInfo``
    - ``from blackjax.mcmc.composed.step_size import StepSizeTuningParameter``
"""
from __future__ import annotations

import warnings as _warnings

_warnings.warn(
    "blackjax.mcmc.gist_step_size is deprecated; "
    "import from blackjax.mcmc.composed.step_size instead.",
    DeprecationWarning,
    stacklevel=1,
)

# Re-export the full public surface so that existing code continues to work.
from blackjax.mcmc.composed.step_size import (  # noqa: E402, F401
    GISTStepSizeInfo,
    StepSizeTuningParameter,
    _apply_fn,
    _StepSizeExtra,
    _tuning_parameter_fn,
    as_top_level_api,
    build_kernel,
    init,
    step_size_selector,
)

__all__ = [
    "GISTStepSizeInfo",
    "StepSizeTuningParameter",
    "init",
    "step_size_selector",
    "build_kernel",
    "as_top_level_api",
    "_StepSizeExtra",
    "_tuning_parameter_fn",
    "_apply_fn",
]
