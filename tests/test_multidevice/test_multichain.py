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
"""Multi-device tests for the shard_map multiple-chains pattern.

See also: docs/examples/howto_sample_multiple_chains.md
"""

import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

import blackjax


class ShardMapMultiChainTest(absltest.TestCase):
    def setUp(self):
        super().setUp()
        if jax.device_count() < 2:
            self.skipTest(f"Requires >=2 devices, got {jax.device_count()}.")
        self.mesh = Mesh(jax.devices()[:2], "chains")

    def test_shard_map_nuts_two_chains(self):
        """Smoke test for the docs' shard_map multiple-chains pattern.

        Each shard_map device runs an independent NUTS chain via lax.scan.
        Checks that output shape is (num_chains, num_steps, ndims).
        """
        ndims = 2
        num_chains = 2   # one chain per device
        num_steps = 20

        def logdensity(x):
            return -0.5 * jnp.sum(x**2)

        nuts = blackjax.nuts(
            logdensity,
            step_size=0.1,
            inverse_mass_matrix=jnp.ones(ndims),
        )

        rng_key = jax.random.key(42)
        init_key, sample_key = jax.random.split(rng_key)

        sharding = NamedSharding(self.mesh, P("chains"))

        # Build sharded initial states
        initial_positions = jax.random.normal(init_key, shape=(num_chains, ndims))
        initial_states = jax.vmap(nuts.init)(initial_positions)
        initial_states = jax.device_put(initial_states, sharding)

        # One key per chain, sharded
        sample_keys = jax.device_put(
            jax.random.split(sample_key, num_chains), sharding
        )

        def run_one_chain(key, state):
            # shard_map gives us the per-device slice: shape (1, ...).
            # Squeeze the leading device-axis off before handing to the kernel.
            key = key[0]
            state = jax.tree.map(lambda x: x[0], state)

            def one_step(carry, step_key):
                new_state, _ = nuts.step(step_key, carry)
                return new_state, new_state.position

            step_keys = jax.random.split(key, num_steps)
            _, positions = jax.lax.scan(one_step, state, step_keys)
            # Add the device-axis back so out_specs=P("chains") can concatenate.
            return positions[None]

        result = jax.jit(
            jax.shard_map(
                run_one_chain,
                mesh=self.mesh,
                in_specs=(P("chains"), P("chains")),
                out_specs=P("chains"),
                check_vma=False,
            )
        )(sample_keys, initial_states)

        self.assertEqual(result.shape, (num_chains, num_steps, ndims))
        # Chains should diverge — if both produce identical positions the
        # key-sharding is broken.
        self.assertFalse(
            np.allclose(
                np.array(result[0]), np.array(result[1])
            ),
            "Both chains produced identical positions — key sharding may be broken.",
        )


if __name__ == "__main__":
    absltest.main()
