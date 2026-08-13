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
"""The GIST (Gibbs self-tuning) sampler family: general seam plus instances.

Single-folder package (mirrors :mod:`blackjax.adaptation.meta`) bundling the
general GIST kernel spine with its concrete instances, so the whole family
reviews as one unit. Every instance shares the same seam: augment
``(theta, rho)`` with a Gibbs-refreshed tuning parameter ``alpha``, apply a
measure-preserving involution indexed by ``alpha``, one Metropolis test (see
:mod:`~blackjax.mcmc.composed._seam` for the general kernel and the
measure-preservation argument).

This package currently ships two independently-tunable instances -- a
self-tuning step size (:mod:`~blackjax.mcmc.composed.step_size`, autoStep
-style) and a self-tuning trajectory length
(:mod:`~blackjax.mcmc.composed.trajectory_length`, no-U-turn, not NUTS's
recursive doubling). A third, *composed* instance that selects the step size
first (``h``, via a reversibility-checked ladder) and then the trajectory
length at that selected ``h`` (``L | h``, with the rollout built at ``h``) is
planned as a follow-up addition to this same package -- this module is the
substrate that instance builds on, not the instance itself.

Cite, don't claim
------------------
The *composition principle* behind chaining a local step-size selector into a
local trajectory-length selector is already published: adaptNUTS composes a
GIST step-size selector with NUTS-style path construction [2], and the GIST
survey itself frames the general seam this package implements [1]. Any future
composed instance in this package is an *instance* of that published
machinery -- with its own falsification suite and estimand-aware defaults --
not a claim to the composition principle.

Submodule layout
----------------
:mod:`~blackjax.mcmc.composed._seam`
    The general kernel spine: ``GISTState``/``GISTInfo``, ``init``,
    ``build_kernel``, ``as_top_level_api``. Not user-facing on its own.
:mod:`~blackjax.mcmc.composed.step_size`
    GIST instance (a): self-tuning step size (autoStep-style), exposed as
    ``blackjax.gist_step_size``.
:mod:`~blackjax.mcmc.composed.trajectory_length`
    GIST instance (b): self-tuning trajectory length (no-U-turn), exposed as
    ``blackjax.gist_trajectory_length``.

References
----------
.. [1] Bou-Rabee, Carpenter, Marsden, "GIST: Gibbs self-tuning for locally
   adaptive Hamiltonian Monte Carlo", arXiv:2404.15253, Statistical Surveys
   2026, Vol. 20, pp. 135-179.
.. [2] Bou-Rabee, Carpenter, Kleppe, Marsden, "Incorporating Local Step-Size
   Adaptivity into the No-U-Turn Sampler using Gibbs Self Tuning",
   arXiv:2408.08259, Journal of Chemical Physics.
"""
from blackjax.mcmc.composed import step_size, trajectory_length
from blackjax.mcmc.composed._seam import GISTInfo, GISTState, build_kernel, init

__all__ = [
    "GISTState",
    "GISTInfo",
    "init",
    "build_kernel",
    "step_size",
    "trajectory_length",
]
