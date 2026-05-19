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


def std_normal_logdensity(x):
    """Log density of a standard isotropic Gaussian: -0.5 * sum(x**2).

    Works with both array and PyTree positions (sums over all leaves).
    """
    leaves = jax.tree.leaves(x)
    return -0.5 * sum(jnp.sum(leaf**2) for leaf in leaves)


# Anisotropic 3-D Gaussian fixture for tests where the IMM/preconditioner choice
# is supposed to be detectable (deliberately heterogeneous per-dim variance).
ANISOTROPIC_3D_DIM = 3
ANISOTROPIC_3D_MEAN = jnp.zeros(ANISOTROPIC_3D_DIM)
ANISOTROPIC_3D_STD = jnp.array([0.1, 1.0, 10.0])


def anisotropic_3d_gaussian_logdensity(x):
    """Log density of a 3-D anisotropic Gaussian with std [0.1, 1.0, 10.0].

    The wide spread of per-dim variances (1e-2 to 1e2 on the covariance scale)
    makes the choice of inverse-mass-matrix preconditioner detectable in
    short-warmup tests. See `tests/adaptation/test_window_adaptation_imm_seed.py`.
    """
    return jax.scipy.stats.norm.logpdf(
        x, loc=ANISOTROPIC_3D_MEAN, scale=ANISOTROPIC_3D_STD
    ).sum()
