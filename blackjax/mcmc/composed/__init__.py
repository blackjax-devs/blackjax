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
general kernel spine (:mod:`~blackjax.mcmc.composed._seam`) with its shipped
instances -- :mod:`~blackjax.mcmc.composed.step_size` (self-tuning step size,
autoStep-style, ``blackjax.gist_step_size``),
:mod:`~blackjax.mcmc.composed.trajectory_length` (self-tuning trajectory
length, no-U-turn, ``blackjax.gist_trajectory_length``), and
:mod:`~blackjax.mcmc.composed._kernel` (composed step-size x trajectory-length
instance, ``h`` then ``L``, ``blackjax.gist_composed``).

References: [1] Bou-Rabee, Carpenter, Marsden, "GIST: Gibbs self-tuning for
locally adaptive Hamiltonian Monte Carlo", arXiv:2404.15253; [2] Bou-Rabee,
Carpenter, Kleppe, Marsden, "Incorporating Local Step-Size Adaptivity into
the No-U-Turn Sampler using Gibbs Self Tuning", arXiv:2408.08259 -- the
published sources of the composition machinery.
"""
from blackjax.mcmc.composed import _kernel, step_size, trajectory_length
from blackjax.mcmc.composed._seam import GISTInfo, GISTState, build_kernel, init

__all__ = [
    "GISTState",
    "GISTInfo",
    "init",
    "build_kernel",
    "step_size",
    "trajectory_length",
    "_kernel",
]
