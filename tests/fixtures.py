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


def neal_funnel_logdensity(x):
    """Neal's funnel (unnormalised): the canonical stress test for
    per-location step-size adaptation.

    ``x[0]`` is the "neck" variable ``y ~ N(0, 3**2)``; ``x[1:]`` are
    ``N(0, exp(y))`` i.i.d. given ``y``. A single global step size cannot
    work well here: the conditional scale of ``x[1:]`` ranges over
    ``exp(y / 2)`` as ``y`` sweeps its prior, from tiny near the neck
    (``y`` very negative) to huge in the mouth (``y`` very positive).

    Works with any array position of length >= 2 (one neck coordinate plus
    at least one funnel coordinate).
    """
    y = x[0]
    v = x[1:]
    logdensity_y = -0.5 * (y / 3.0) ** 2
    logdensity_v = -0.5 * jnp.exp(-y) * jnp.sum(v**2) - 0.5 * v.shape[0] * y
    return logdensity_y + logdensity_v


def smooth_skewed_logdensity(x):
    """A smooth, gradient-friendly asymmetric target: ``y = log(u)`` for
    ``u ~ Exponential(1)``, i.e. ``-y ~ Gumbel(0, 1)``.

    Unnormalised log density ``y - exp(y)``, defined and differentiable on
    all of ``R`` (gradient ``1 - exp(y)``, never zero/degenerate). A
    symmetric Gaussian target cannot detect a reversed-direction/sign bug in
    a leapfrog+flip or a forward-vs-reverse rollout, but an asymmetric one
    can (mirrors ``test_slice.py::test_multivariate_skewed_exponential``'s
    rationale) -- **without** the raw ``jnp.where(x > 0, ..., -inf)``
    representation of ``Exponential`` directly, whose *zero* gradient
    outside the support lets a leapfrog trajectory that crosses the
    boundary drift forever under a purely linear potential (no restoring
    force once ``logdensity_grad`` clamps to 0) -- pathologically
    degenerate for a gradient-based (HMC-family) sampler, even though it is
    perfectly fine for a density-evaluation-only sampler like slice
    sampling.

    Closed-form moments (per-coordinate, i.i.d.): mean ``-gamma`` (negative
    Euler-Mascheroni, approx -0.5772), variance ``pi**2 / 6`` (approx
    1.6449), skewness approx -1.1395 (left-skewed).
    """
    return jnp.sum(x - jnp.exp(x))
