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
"""Shared test utilities for BlackJAX tests."""
import datetime

import chex
import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree


class BlackJAXTest(chex.TestCase):
    """Base test case with date-based PRNG key management.

    Provides a fresh ``self.key`` for each test method, seeded from today's
    date so results are deterministic within a day and rotate automatically.

    Use ``self.next_key()`` to obtain a fresh subkey for each independent
    random operation — this advances ``self.key`` so no two calls share the
    same underlying key.

    Example
    -------

    .. code::

        class MyTest(BlackJAXTest):
            def test_something(self):
                state = init(self.next_key())
                new_state, info = kernel(self.next_key(), state, ...)
    """

    def setUp(self):
        super().setUp()
        seed = int(datetime.date.today().strftime("%Y%m%d"))
        self.key = jax.random.key(seed)

    def next_key(self):
        """Advance the key state and return a fresh subkey.

        Each call produces an independent subkey and mutates ``self.key`` so
        that successive calls yield different keys.
        """
        self.key, subkey = jax.random.split(self.key)
        return subkey


def std_normal_logdensity(x, scale=1.0):
    """Log density (unnormalised) of a zero-mean Gaussian.

    Default ``scale=1.0`` gives the standard isotropic normal — preserves the
    original signature for callers passing no ``scale``.  Supply a scalar to
    scale uniformly across all dimensions, or an array of shape matching
    ``ravel_pytree(x)[0].shape`` for an anisotropic per-element scale (i.e.
    the per-dim standard deviation of the target).

    The normalisation constants (``-d/2 * log(2*pi)`` and ``-sum log scale``)
    are dropped since they don't affect HMC sampling.

    Works with both array and PyTree positions.
    """
    if jnp.ndim(scale) == 0:
        leaves = jax.tree.leaves(x)
        return -0.5 / (scale**2) * sum(jnp.sum(leaf**2) for leaf in leaves)
    flat_x, _ = ravel_pytree(x)
    return -0.5 * jnp.sum((flat_x / scale) ** 2)
