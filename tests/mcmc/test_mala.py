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
"""Unit tests for MALA (and the underlying overdamped Langevin diffusion)."""

import chex
import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest

import blackjax.mcmc.diffusions as diffusions
import blackjax.mcmc.mala as mala
from tests.fixtures import BlackJAXTest, std_normal_logdensity

# ---------------------------------------------------------------------------
# mcmc/diffusions.overdamped_langevin
# ---------------------------------------------------------------------------


class OverdampedLangevinMCMCTest(BlackJAXTest):
    """Tests for blackjax.mcmc.diffusions.overdamped_langevin."""

    def setUp(self):
        super().setUp()
        self.step_size = 1e-3

    def _make_state(self, position):
        grad_fn = jax.value_and_grad(std_normal_logdensity)
        logdensity, logdensity_grad = grad_fn(position)
        return diffusions.DiffusionState(position, logdensity, logdensity_grad)

    def test_returns_diffusion_state(self):
        """One step returns a DiffusionState with correct fields."""
        position = jnp.zeros(3)
        state = self._make_state(position)
        step = diffusions.overdamped_langevin(jax.value_and_grad(std_normal_logdensity))
        new_state = step(self.next_key(), state, self.step_size)
        self.assertIsInstance(new_state, diffusions.DiffusionState)
        self.assertEqual(new_state.position.shape, (3,))
        assert jnp.isfinite(new_state.logdensity)

    def test_position_changes(self):
        """Position should change after one step."""
        position = jnp.zeros(3)
        state = self._make_state(position)
        step = diffusions.overdamped_langevin(jax.value_and_grad(std_normal_logdensity))
        new_state = step(self.next_key(), state, self.step_size)
        assert not jnp.allclose(new_state.position, position)

    def test_gradient_updated(self):
        """DiffusionState gradient is consistent with the new position."""
        position = jnp.zeros(3)
        state = self._make_state(position)
        step = diffusions.overdamped_langevin(jax.value_and_grad(std_normal_logdensity))
        new_state = step(self.next_key(), state, self.step_size)
        expected_grad = jax.grad(std_normal_logdensity)(new_state.position)
        np.testing.assert_allclose(new_state.logdensity_grad, expected_grad, atol=1e-5)

    def test_pytree_position(self):
        """Works with PyTree positions."""

        def logdensity_fn(pos):
            return -0.5 * (jnp.sum(pos["x"] ** 2) + jnp.sum(pos["y"] ** 2))

        position = {"x": jnp.zeros(2), "y": jnp.zeros(3)}
        logdensity, logdensity_grad = jax.value_and_grad(logdensity_fn)(position)
        state = diffusions.DiffusionState(position, logdensity, logdensity_grad)

        step = diffusions.overdamped_langevin(jax.value_and_grad(logdensity_fn))
        new_state = step(self.next_key(), state, self.step_size)
        chex.assert_trees_all_equal_shapes(new_state.position, position)

    def test_jit_compatible(self):
        """Step is JIT-compilable."""
        position = jnp.zeros(2)
        state = self._make_state(position)
        step = jax.jit(
            diffusions.overdamped_langevin(jax.value_and_grad(std_normal_logdensity))
        )
        new_state = step(self.next_key(), state, self.step_size)
        self.assertEqual(new_state.position.shape, (2,))


# ---------------------------------------------------------------------------
# mala.init
# ---------------------------------------------------------------------------


class MALAInitTest(BlackJAXTest):
    """Tests for mala.init."""

    def test_init_computes_logdensity_and_grad(self):
        """init stores logdensity and logdensity_grad at the initial position."""
        position = jnp.array([1.0, 2.0])
        state = mala.init(position, std_normal_logdensity)
        expected_logdensity = std_normal_logdensity(position)
        expected_grad = jax.grad(std_normal_logdensity)(position)
        np.testing.assert_allclose(float(state.logdensity), float(expected_logdensity))
        np.testing.assert_allclose(state.logdensity_grad, expected_grad)

    def test_init_pytree_position(self):
        """init works with PyTree positions."""

        def logdensity_fn(pos):
            return -0.5 * jnp.sum(pos["a"] ** 2)

        position = {"a": jnp.ones(3)}
        state = mala.init(position, logdensity_fn)
        chex.assert_trees_all_equal_shapes(state.position, position)
        assert jnp.isfinite(state.logdensity)


# ---------------------------------------------------------------------------
# mala.build_kernel / kernel
# ---------------------------------------------------------------------------


class MALAKernelTest(BlackJAXTest):
    """Tests for the MALA kernel."""

    def setUp(self):
        super().setUp()
        self.step_size = 1e-2

    def test_returns_state_and_info(self):
        """Kernel returns (MALAState, MALAInfo)."""
        position = jnp.zeros(4)
        state = mala.init(position, std_normal_logdensity)
        kernel = mala.build_kernel()
        new_state, info = kernel(
            self.next_key(), state, std_normal_logdensity, self.step_size
        )
        self.assertIsInstance(new_state, mala.MALAState)
        self.assertIsInstance(info, mala.MALAInfo)

    def test_output_position_shape(self):
        """Output position has same shape as input."""
        position = jnp.zeros(5)
        state = mala.init(position, std_normal_logdensity)
        kernel = mala.build_kernel()
        new_state, _ = kernel(
            self.next_key(), state, std_normal_logdensity, self.step_size
        )
        self.assertEqual(new_state.position.shape, (5,))

    def test_acceptance_rate_in_range(self):
        """Acceptance rate is in [0, 1]."""
        position = jnp.zeros(3)
        state = mala.init(position, std_normal_logdensity)
        kernel = mala.build_kernel()
        _, info = kernel(self.next_key(), state, std_normal_logdensity, self.step_size)
        assert 0.0 <= float(info.acceptance_rate) <= 1.0

    def test_logdensity_updated(self):
        """Stored logdensity is consistent with accepted position."""
        position = jnp.zeros(2)
        state = mala.init(position, std_normal_logdensity)
        kernel = mala.build_kernel()
        new_state, _ = kernel(
            self.next_key(), state, std_normal_logdensity, self.step_size
        )
        expected = std_normal_logdensity(new_state.position)
        np.testing.assert_allclose(
            float(new_state.logdensity), float(expected), atol=1e-5
        )

    def test_large_step_size_low_acceptance(self):
        """With a very large step size, most proposals should be rejected."""
        position = jnp.zeros(2)
        state = mala.init(position, std_normal_logdensity)
        kernel = mala.build_kernel()

        def one_step(state, key):
            new_state, info = kernel(key, state, std_normal_logdensity, step_size=10.0)
            return new_state, info.is_accepted

        keys = jax.random.split(self.next_key(), 200)
        _, accepted = jax.lax.scan(one_step, state, keys)
        acceptance_rate = jnp.mean(accepted.astype(jnp.float32))
        # With step_size=10.0 on a standard normal, acceptance should be low
        assert float(acceptance_rate) < 0.5

    def test_small_step_size_high_acceptance(self):
        """With a tiny step size, almost all proposals should be accepted."""
        position = jnp.zeros(2)
        state = mala.init(position, std_normal_logdensity)
        kernel = mala.build_kernel()

        def one_step(state, key):
            new_state, info = kernel(key, state, std_normal_logdensity, step_size=1e-5)
            return new_state, info.is_accepted

        keys = jax.random.split(self.next_key(), 200)
        _, accepted = jax.lax.scan(one_step, state, keys)
        acceptance_rate = jnp.mean(accepted.astype(jnp.float32))
        assert float(acceptance_rate) > 0.9

    def test_pytree_position(self):
        """Kernel works with PyTree positions."""

        def logdensity_fn(pos):
            return -0.5 * (jnp.sum(pos["a"] ** 2) + jnp.sum(pos["b"] ** 2))

        position = {"a": jnp.zeros(2), "b": jnp.zeros(3)}
        state = mala.init(position, logdensity_fn)
        kernel = mala.build_kernel()
        new_state, info = kernel(self.next_key(), state, logdensity_fn, self.step_size)
        chex.assert_trees_all_equal_shapes(new_state.position, position)
        assert 0.0 <= float(info.acceptance_rate) <= 1.0

    def test_jit_compatible(self):
        """Kernel is JIT-compilable."""
        position = jnp.zeros(3)
        state = mala.init(position, std_normal_logdensity)
        kernel = jax.jit(mala.build_kernel(), static_argnums=(2,))
        new_state, info = kernel(
            self.next_key(), state, std_normal_logdensity, self.step_size
        )
        self.assertEqual(new_state.position.shape, (3,))


# ---------------------------------------------------------------------------
# mala.as_top_level_api
# ---------------------------------------------------------------------------


class MALATopLevelAPITest(BlackJAXTest):
    """Tests for the MALA top-level API."""

    def test_init_and_step(self):
        """Top-level API init + step runs and returns MALAState."""
        position = jnp.zeros(4)
        algo = mala.as_top_level_api(std_normal_logdensity, step_size=1e-2)
        state = algo.init(position)
        new_state, info = algo.step(self.next_key(), state)
        self.assertIsInstance(new_state, mala.MALAState)
        self.assertEqual(new_state.position.shape, (4,))

    def test_top_level_jit(self):
        """Top-level step is JIT-compilable."""
        position = jnp.zeros(3)
        algo = mala.as_top_level_api(std_normal_logdensity, step_size=1e-2)
        state = algo.init(position)
        new_state, info = jax.jit(algo.step)(self.next_key(), state)
        self.assertEqual(new_state.position.shape, (3,))


if __name__ == "__main__":
    absltest.main()
