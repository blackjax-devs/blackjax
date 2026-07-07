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
"""Multi-device tests for blackjax.eca.ensemble_execute_fn.

Requires XLA_FLAGS="--xla_force_host_platform_device_count=2" to be set in
the environment before JAX is imported (handled automatically by the
test_multidevice CI job).  Tests are skipped when fewer than 2 devices are
available so the file is safe to run locally without the flag.
"""

import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest
from jax.sharding import Mesh, NamedSharding

from blackjax.eca import ensemble_execute_fn


class ECAMultiDeviceTest(absltest.TestCase):
    def setUp(self):
        super().setUp()
        if jax.device_count() < 2:
            self.skipTest(f"Requires >=2 devices, got {jax.device_count()}.")
        self.mesh = Mesh(jax.devices()[:2], "chains")

    def test_args_as_sharded_array_does_not_raise(self):
        """Regression test for blackjax-devs/blackjax#932.

        Mirrors the exact call pattern in laps_burn_in.py: a first
        ensemble_execute_fn call produces a sharded equipartition array;
        a sign-flip derives `signs`; that sharded array is then passed as
        `args` to a second call.  On main before the fix this raised
        NotImplementedError on >1 devices because args was captured in the
        shard_map closure instead of being passed as a tracked argument.
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

        # Derive signs from equipartition output, mirroring laps_burn_in.py.
        signs = jnp.where(equipartition < 1.0, jnp.array(-1.0), jnp.array(1.0))

        # signs must be a NamedSharding array — that is the trigger for #932.
        self.assertIsInstance(signs, jax.Array)
        self.assertIsInstance(signs.sharding, NamedSharding)

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
        """The common args=None path must be unaffected by the fix."""
        num_chains = 4

        def step(key, x, args):
            del args
            return x + 1.0, None

        rng_key = jax.random.PRNGKey(1)
        result, _ = ensemble_execute_fn(step, rng_key, num_chains, self.mesh)
        np.testing.assert_allclose(result, jnp.ones(num_chains))

    def test_laps_end_to_end_two_devices(self):
        """Regression: laps() must complete on a 2-device mesh without the #932 crash.

        The bug path: laps_burn_in passes a sharded `signs` array as `args` to a
        second ensemble_execute_fn call.  Before the fix that raised NotImplementedError.
        """
        from blackjax.adaptation.laps import laps as run_laps

        def logdensity(x):
            return -0.5 * jnp.sum(x**2)

        _, _, _, final_state = run_laps(
            logdensity_fn=logdensity,
            sample_init=lambda key: jax.random.normal(key, shape=(2,)),
            ndims=2,
            num_steps1=20,
            num_steps2=20,
            num_chains=4,  # 4 chains across 2 devices → 2 per device
            mesh=self.mesh,
            rng_key=jax.random.key(0),
            early_stop=False,
            diagonal_preconditioning=True,
            integrator_coefficients=None,
            steps_per_sample=5,
            r_end=0.5,
            diagnostics=False,
            superchain_size=1,
        )

        self.assertEqual(final_state.position.shape, (4, 2))


if __name__ == "__main__":
    absltest.main()
