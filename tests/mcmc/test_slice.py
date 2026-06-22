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
"""Unit tests for the Slice sampling kernel."""

import chex
import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest

from blackjax.mcmc.slice import (
    SliceInfo,
    SliceState,
    build_kernel,
    init,
)
from tests.fixtures import BlackJAXTest, std_normal_logdensity


class SliceInitTest(BlackJAXTest):
    """Tests for slice.init."""

    def test_init_stores_position_and_logdensity(self):
        """init stores the initial position and the log-density at that position."""
        position = jnp.array([1.0, 2.0, 3.0])
        state = init(position, std_normal_logdensity)
        np.testing.assert_allclose(state.position, position)
        expected_ld = std_normal_logdensity(position)
        np.testing.assert_allclose(float(state.logdensity[0]), float(expected_ld))

    def test_init_widths_are_positive(self):
        """Initial step widths are all positive."""
        position = jnp.ones(4)
        state = init(position, std_normal_logdensity)
        flat_widths, _ = jax.flatten_util.ravel_pytree(state.widths)
        assert jnp.all(flat_widths > 0)

    def test_init_n_is_zero(self):
        """Iteration counter starts at zero."""
        state = init(jnp.zeros(2), std_normal_logdensity)
        np.testing.assert_equal(float(state.n[0]), 0.0)

    def test_init_pytree_position(self):
        """init works with PyTree (dict) positions."""

        def logdensity_fn(pos):
            return -0.5 * (jnp.sum(pos["a"] ** 2) + jnp.sum(pos["b"] ** 2))

        position = {"a": jnp.ones(2), "b": jnp.zeros(3)}
        state = init(position, logdensity_fn)
        self.assertIsInstance(state, SliceState)
        assert jnp.isfinite(state.logdensity[0])
        chex.assert_trees_all_equal_shapes(state.widths, position)



class SliceKernelTest(BlackJAXTest):
    """Tests for the slice sampling kernel."""

    def test_returns_state_and_info(self):
        """Kernel returns a (SliceState, SliceInfo) pair."""
        position = jnp.zeros(3)
        state = init(position, std_normal_logdensity)
        kernel = build_kernel(n_doublings=5)
        new_state, info = kernel(self.next_key(), state, std_normal_logdensity)
        self.assertIsInstance(new_state, SliceState)
        self.assertIsInstance(info, SliceInfo)

    def test_position_shape_preserved(self):
        """Output position has the same shape as input position."""
        ndim = 4
        position = jnp.zeros(ndim)
        state = init(position, std_normal_logdensity)
        kernel = build_kernel(n_doublings=5)
        new_state, _ = kernel(self.next_key(), state, std_normal_logdensity)
        self.assertEqual(new_state.position.shape, (ndim,))

    def test_logdensity_consistent(self):
        """Stored logdensity matches the density evaluated at the new position."""
        position = jnp.zeros(3)
        state = init(position, std_normal_logdensity)
        kernel = build_kernel(n_doublings=5)
        new_state, _ = kernel(self.next_key(), state, std_normal_logdensity)
        expected = std_normal_logdensity(new_state.position)
        np.testing.assert_allclose(
            float(new_state.logdensity[0]), float(expected), atol=1e-5
        )

    def test_n_increments(self):
        """Iteration counter increments by 1 per step."""
        state = init(jnp.zeros(2), std_normal_logdensity)
        kernel = build_kernel(n_doublings=5)
        new_state, _ = kernel(self.next_key(), state, std_normal_logdensity)
        np.testing.assert_equal(float(new_state.n[0]), 1.0)

    def test_info_widths_match_state(self):
        """SliceInfo.widths equals the widths stored in the new state."""
        position = jnp.zeros(3)
        state = init(position, std_normal_logdensity)
        kernel = build_kernel(n_doublings=5)
        new_state, info = kernel(self.next_key(), state, std_normal_logdensity)
        flat_state_widths, _ = jax.flatten_util.ravel_pytree(new_state.widths)
        flat_info_widths, _ = jax.flatten_util.ravel_pytree(info.widths)
        np.testing.assert_allclose(flat_state_widths, flat_info_widths)

    def test_widths_adapt(self):
        """Step widths change from their initial values after several steps."""
        position = jnp.zeros(3)
        state = init(position, std_normal_logdensity)
        kernel = build_kernel(n_doublings=5)
        initial_widths, _ = jax.flatten_util.ravel_pytree(state.widths)

        for _ in range(10):
            state, _ = kernel(self.next_key(), state, std_normal_logdensity)

        final_widths, _ = jax.flatten_util.ravel_pytree(state.widths)
        assert not jnp.allclose(initial_widths, final_widths)

    def test_pytree_position(self):
        """Kernel works with PyTree (dict) positions."""

        def logdensity_fn(pos):
            return -0.5 * (jnp.sum(pos["x"] ** 2) + jnp.sum(pos["y"] ** 2))

        position = {"x": jnp.zeros(2), "y": jnp.zeros(2)}
        state = init(position, logdensity_fn)
        kernel = build_kernel(n_doublings=5)
        new_state, info = kernel(self.next_key(), state, logdensity_fn)
        chex.assert_trees_all_equal_shapes(new_state.position, position)
        chex.assert_trees_all_equal_shapes(info.widths, position)

    def test_jit_compatible(self):
        """Kernel is JIT-compilable."""
        position = jnp.zeros(3)
        state = init(position, std_normal_logdensity)
        kernel = jax.jit(build_kernel(n_doublings=5), static_argnums=(2,))
        new_state, _ = kernel(self.next_key(), state, std_normal_logdensity)
        self.assertEqual(new_state.position.shape, (3,))

    def test_different_keys_give_different_samples(self):
        """Two independent runs from the same state produce different positions."""
        state = init(jnp.zeros(3), std_normal_logdensity)
        kernel = build_kernel(n_doublings=5)
        new_state_1, _ = kernel(self.next_key(), state, std_normal_logdensity)
        new_state_2, _ = kernel(self.next_key(), state, std_normal_logdensity)
        assert not jnp.allclose(new_state_1.position, new_state_2.position)

    def test_logdensity_finite(self):
        """Log-density in the new state is always finite."""
        state = init(jnp.ones(3), std_normal_logdensity)
        kernel = build_kernel(n_doublings=5)
        for _ in range(5):
            state, _ = kernel(self.next_key(), state, std_normal_logdensity)
            assert jnp.isfinite(state.logdensity[0])



class SliceTopLevelAPITest(BlackJAXTest):
    """Tests for the top-level blackjax.slice_sampling API."""

    def test_init_and_step(self):
        """Top-level API: init + step runs and returns SliceState."""
        import blackjax

        algo = blackjax.slice_sampling(std_normal_logdensity, n_doublings=5)
        state = algo.init(jnp.zeros(3))
        new_state, info = algo.step(self.next_key(), state)
        self.assertIsInstance(new_state, SliceState)
        self.assertIsInstance(info, SliceInfo)

    def test_top_level_jit(self):
        """Top-level step is JIT-compilable."""
        import blackjax

        algo = blackjax.slice_sampling(std_normal_logdensity, n_doublings=5)
        state = algo.init(jnp.zeros(3))
        new_state, _ = jax.jit(algo.step)(self.next_key(), state)
        self.assertEqual(new_state.position.shape, (3,))

    def test_default_n_doublings(self):
        """as_top_level_api default n_doublings=10 works without explicit arg."""
        import blackjax

        algo = blackjax.slice_sampling(std_normal_logdensity)
        state = algo.init(jnp.zeros(2))
        new_state, _ = algo.step(self.next_key(), state)
        self.assertEqual(new_state.position.shape, (2,))

    def test_build_kernel_accessible(self):
        """build_kernel is accessible via blackjax.slice_sampling.build_kernel."""
        import blackjax

        kernel = blackjax.slice_sampling.build_kernel(n_doublings=3)
        state = blackjax.slice_sampling.init(jnp.zeros(2), std_normal_logdensity)
        new_state, info = kernel(self.next_key(), state, std_normal_logdensity)
        self.assertIsInstance(new_state, SliceState)


if __name__ == "__main__":
    absltest.main()
