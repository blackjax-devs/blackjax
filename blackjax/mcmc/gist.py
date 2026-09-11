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
"""Back-compatibility shim — use :mod:`blackjax.mcmc.composed` instead.

.. deprecated::
    Importing from ``blackjax.mcmc.gist`` is deprecated. The module has been
    reorganised into the :mod:`blackjax.mcmc.composed` sub-package (the
    general GIST kernel spine now lives in
    :mod:`blackjax.mcmc.composed._seam`). All public names are still
    importable; update your imports to use the new location:

    - ``from blackjax.mcmc.composed import GISTState``
    - ``from blackjax.mcmc.composed import GISTInfo``
    - ``from blackjax.mcmc.composed import init``
    - ``from blackjax.mcmc.composed import build_kernel``
    - ``from blackjax.mcmc.composed._seam import as_top_level_api``
"""
from __future__ import annotations

import warnings as _warnings

_warnings.warn(
    "blackjax.mcmc.gist is deprecated; " "import from blackjax.mcmc.composed instead.",
    DeprecationWarning,
    stacklevel=1,
)

# Re-export the full public surface so that existing code continues to work.
from blackjax.mcmc.composed._seam import (  # noqa: E402, F401
    GISTInfo,
    GISTState,
    _step,
    as_top_level_api,
    build_kernel,
    init,
)

__all__ = [
    "GISTState",
    "GISTInfo",
    "init",
    "build_kernel",
    "as_top_level_api",
    "_step",
]
