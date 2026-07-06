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
"""Tests for blackjax.eca.ensemble_execute_fn on a multi-device mesh.

This module sets XLA_FLAGS to force multiple host (CPU) devices *before*
importing jax, since the device count is fixed at JAX's first use. All
existing ``test_laps*`` tests in ``tests/mcmc/test_sampling.py`` run on a
single-device mesh (``jax.devices()[:1]``), which is exactly why the
shard_map closure bug fixed here (blackjax-devs/blackjax#932) went
undetected: on one device, shard_map's per-device partitioning is a trivial
no-op, so a closed-over sharded array never actually gets checked against
shard_map's "no closed-over NamedSharding" rule.
"""

import os

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=2"

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from absl.testing import absltest  # noqa: E402
from jax.sharding import Mesh  # noqa: E402

from blackjax.eca import ensemble_execute_fn  # noqa: E402


class ECAMultiDeviceTest(absltest.TestCase):
    def setUp(self):
        super().setUp()
        self.assertGreaterEqual(
            jax.device_count(),
            2,
            "This test requires >=2 devices; check that XLA_FLAGS took effect "
            "before jax was first used in this process.",
        )
        self.mesh = Mesh(jax.devices()[:2], "chains")

    def test_args_as_sharded_array_does_not_raise(self):
        """Regression test for blackjax-devs/blackjax#932.

        Passing a sharded JAX array as `args` (as happens in
        `laps_burn_in.py` when `signs`, derived from a prior
        `ensemble_execute_fn` call's output, is threaded into the next call)
        used to raise NotImplementedError on >1 devices because `args` was
        captured in the closure of the inner `shard_map`-wrapped function
        instead of being passed through as a tracked argument.
        """
        num_chains = 4

        def first_step(key, x, args):
            del args
            return x + 1.0, None

        rng_key = jax.random.PRNGKey(0)
        key1, key2 = jax.random.split(rng_key)

        initial_state, equipartition = ensemble_execute_fn(
            first_step,
            key1,
            num_chains,
            self.mesh,
            summary_statistics_fn=lambda y: jnp.mean(y),
        )

        # `signs` mirrors laps_burn_in.py: derived from the previous call's
        # sharded output, and itself a sharded JAX array with NamedSharding.
        signs = jnp.where(equipartition < 1.0, -1.0, 1.0)
        self.assertIsInstance(signs, jax.Array)

        def second_step(key, x, args):
            return x * args, None

        final_state, _ = ensemble_execute_fn(
            second_step,
            key2,
            num_chains,
            self.mesh,
            x=initial_state,
            args=signs,
        )

        np.testing.assert_allclose(final_state, initial_state * signs)

    def test_args_none_still_works(self):
        """The common `args=None` path must be unaffected by the fix."""
        num_chains = 4

        def step(key, x, args):
            del args
            return x + 1.0, None

        rng_key = jax.random.PRNGKey(1)
        result, _ = ensemble_execute_fn(step, rng_key, num_chains, self.mesh)
        np.testing.assert_allclose(result, jnp.ones(num_chains))


if __name__ == "__main__":
    absltest.main()
