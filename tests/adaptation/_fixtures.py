# Copyright 2024- The Blackjax Authors.
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
"""Shared test fixtures for adaptation-layer tests."""

import jax
import jax.numpy as jnp

__all__ = ["DIM", "TARGET_MEAN", "TARGET_STD", "logdensity_fn"]

DIM = 3
TARGET_MEAN = jnp.zeros(DIM)
TARGET_STD = jnp.array([0.1, 1.0, 10.0])  # deliberately anisotropic


def logdensity_fn(x):
    return jax.scipy.stats.norm.logpdf(x, loc=TARGET_MEAN, scale=TARGET_STD).sum()
