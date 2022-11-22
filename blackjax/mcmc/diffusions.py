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
"""Solvers for Langevin diffusions."""
from typing import NamedTuple

import jax
import jax.numpy as jnp

from blackjax.types import PyTree
from blackjax.util import generate_gaussian_noise

__all__ = ["overdamped_langevin"]


class DiffusionState(NamedTuple):
    position: PyTree
    logprob: float
    logprob_grad: PyTree


def overdamped_langevin(logprob_grad_fn):
    """Euler solver for overdamped Langevin diffusion."""

    def one_step(rng_key, state: DiffusionState, step_size: float, batch: tuple = ()):
        position, _, logprob_grad = state
        noise = generate_gaussian_noise(rng_key, position)
        position = jax.tree_util.tree_map(
            lambda p, g, n: p + step_size * g + jnp.sqrt(2 * step_size) * n,
            position,
            logprob_grad,
            noise,
        )

        logprob, logprob_grad = logprob_grad_fn(position, *batch)
        return DiffusionState(position, logprob, logprob_grad)

    return one_step
