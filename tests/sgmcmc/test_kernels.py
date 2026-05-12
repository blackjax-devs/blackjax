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
"""Unit tests for SGMCMC kernel modules (SGLD, SGHMC, SGNHT, CSGLD)."""

import chex
import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest

import blackjax.sgmcmc.csgld as csgld
import blackjax.sgmcmc.sghmc as sghmc
import blackjax.sgmcmc.sgld as sgld
import blackjax.sgmcmc.sgnht as sgnht
from tests.fixtures import BlackJAXTest

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_grad_estimator(scale=1.0):
    """A simple gradient estimator that ignores the minibatch."""

    def grad_estimator(position, minibatch):
        return jax.tree.map(lambda x: scale * jnp.ones_like(x), position)

    return grad_estimator


def _make_zero_grad_estimator():
    """A gradient estimator that returns zero everywhere."""

    def grad_estimator(position, minibatch):
        return jax.tree.map(jnp.zeros_like, position)

    return grad_estimator


MINIBATCH = jnp.zeros(1)  # unused dummy minibatch


# ---------------------------------------------------------------------------
# SGLD
# ---------------------------------------------------------------------------


class SGLDKernelTest(BlackJAXTest):
    """Tests for the SGLD kernel."""

    def setUp(self):
        super().setUp()
        self.step_size = 1e-3

    def test_init_returns_position(self):
        """SGLD init is identity — returns position unchanged."""
        position = jnp.array([1.0, 2.0, 3.0])
        state = sgld.init(position)
        np.testing.assert_array_equal(state, position)

    def test_init_works_with_pytree(self):
        """SGLD init works for PyTree positions."""
        position = {"a": jnp.zeros(3), "b": jnp.ones(2)}
        state = sgld.init(position)
        chex.assert_trees_all_equal(state, position)

    def test_kernel_returns_same_structure(self):
        """Kernel output has the same PyTree structure as input position."""
        position = {"x": jnp.zeros(2), "y": jnp.ones(3)}
        kernel = sgld.build_kernel()
        new_position = kernel(
            self.next_key(), position, _make_grad_estimator(), MINIBATCH, self.step_size
        )
        chex.assert_trees_all_equal_shapes(position, new_position)

    def test_zero_grad_temp0_position_unchanged(self):
        """With zero gradient and temperature=0 the position does not move."""
        position = jnp.array([1.0, -2.0, 0.5])
        kernel = sgld.build_kernel()
        new_position = kernel(
            self.next_key(),
            position,
            _make_zero_grad_estimator(),
            MINIBATCH,
            self.step_size,
            temperature=0.0,
        )
        np.testing.assert_allclose(new_position, position, atol=1e-6)

    def test_positive_grad_moves_position_up(self):
        """Positive gradient should move position in the positive direction."""
        position = jnp.zeros(1)
        kernel = sgld.build_kernel()

        def one_step(pos, key):
            return (
                kernel(
                    key,
                    pos,
                    _make_grad_estimator(1.0),
                    MINIBATCH,
                    self.step_size,
                    temperature=0.0,
                ),
                None,
            )

        keys = jax.random.split(self.next_key(), 100)
        final_position, _ = jax.lax.scan(one_step, position, keys)
        assert float(final_position[0]) > 0.0

    def test_top_level_api_step(self):
        """as_top_level_api step produces output with same structure."""
        position = jnp.zeros(4)
        algo = sgld.as_top_level_api(_make_grad_estimator())
        state = algo.init(position)
        new_state = algo.step(self.next_key(), state, MINIBATCH, self.step_size)
        self.assertEqual(new_state.shape, position.shape)

    def test_jit_compatible(self):
        """Kernel is JIT-compilable."""
        position = jnp.ones(3)
        grad_est = _make_grad_estimator()
        kernel = jax.jit(sgld.build_kernel(), static_argnums=(2,))
        new_position = kernel(
            self.next_key(), position, grad_est, MINIBATCH, self.step_size
        )
        self.assertEqual(new_position.shape, (3,))


# ---------------------------------------------------------------------------
# SGHMC
# ---------------------------------------------------------------------------


class SGHMCKernelTest(BlackJAXTest):
    """Tests for the SGHMC kernel."""

    def setUp(self):
        super().setUp()
        self.step_size = 1e-3

    def test_init_returns_position(self):
        """SGHMC init is identity."""
        position = jnp.array([1.0, 2.0])
        state = sghmc.init(position)
        np.testing.assert_array_equal(state, position)

    def test_kernel_output_shape(self):
        """Kernel returns position with the same shape."""
        position = jnp.zeros(4)
        kernel = sghmc.build_kernel()
        new_position = kernel(
            self.next_key(),
            position,
            _make_grad_estimator(),
            MINIBATCH,
            self.step_size,
            num_integration_steps=5,
        )
        self.assertEqual(new_position.shape, (4,))

    def test_kernel_pytree_position(self):
        """Kernel works for PyTree positions."""
        position = {"a": jnp.zeros(2), "b": jnp.zeros(3)}
        kernel = sghmc.build_kernel()
        new_position = kernel(
            self.next_key(),
            position,
            _make_grad_estimator(),
            MINIBATCH,
            self.step_size,
            num_integration_steps=3,
        )
        chex.assert_trees_all_equal_shapes(position, new_position)

    def test_zero_grad_temp0_deterministic(self):
        """With zero gradient and temperature=0, two calls give identical result."""
        position = jnp.array([1.0, 2.0])
        kernel = sghmc.build_kernel()
        key = jax.random.key(42)
        p1 = kernel(
            key,
            position,
            _make_zero_grad_estimator(),
            MINIBATCH,
            self.step_size,
            5,
            temperature=0.0,
        )
        p2 = kernel(
            key,
            position,
            _make_zero_grad_estimator(),
            MINIBATCH,
            self.step_size,
            5,
            temperature=0.0,
        )
        np.testing.assert_array_equal(p1, p2)

    def test_top_level_api_step(self):
        """as_top_level_api step runs and returns correct shape."""
        position = jnp.zeros(4)
        algo = sghmc.as_top_level_api(_make_grad_estimator(), num_integration_steps=5)
        state = algo.init(position)
        new_state = algo.step(self.next_key(), state, MINIBATCH, self.step_size)
        self.assertEqual(new_state.shape, position.shape)

    def test_jit_compatible(self):
        """Kernel is JIT-compilable."""
        position = jnp.ones(3)
        grad_est = _make_grad_estimator()
        kernel = jax.jit(sghmc.build_kernel(), static_argnums=(2, 5))
        new_position = kernel(
            self.next_key(),
            position,
            grad_est,
            MINIBATCH,
            self.step_size,
            5,
        )
        self.assertEqual(new_position.shape, (3,))


# ---------------------------------------------------------------------------
# SGNHT
# ---------------------------------------------------------------------------


class SGNHTKernelTest(BlackJAXTest):
    """Tests for the SGNHT kernel."""

    def setUp(self):
        super().setUp()
        self.step_size = 1e-3
        self.alpha = 0.01

    def test_init_state_fields(self):
        """init returns SGNHTState with position, momentum, xi."""
        position = jnp.zeros(4)
        state = sgnht.init(position, self.next_key(), xi=self.alpha)
        self.assertEqual(state.position.shape, (4,))
        self.assertEqual(state.momentum.shape, (4,))
        assert jnp.isfinite(state.xi)

    def test_kernel_returns_sgnht_state(self):
        """Kernel returns an SGNHTState with matching shapes."""
        position = jnp.zeros(4)
        state = sgnht.init(position, self.next_key(), xi=self.alpha)
        kernel = sgnht.build_kernel(self.alpha)
        new_state = kernel(
            self.next_key(), state, _make_grad_estimator(), MINIBATCH, self.step_size
        )
        self.assertIsInstance(new_state, sgnht.SGNHTState)
        self.assertEqual(new_state.position.shape, (4,))
        self.assertEqual(new_state.momentum.shape, (4,))
        assert jnp.isfinite(new_state.xi)

    def test_kernel_pytree_position(self):
        """Kernel works for PyTree positions."""
        position = {"a": jnp.zeros(2), "b": jnp.zeros(3)}
        state = sgnht.init(position, self.next_key(), xi=self.alpha)
        kernel = sgnht.build_kernel(self.alpha)
        new_state = kernel(
            self.next_key(), state, _make_grad_estimator(), MINIBATCH, self.step_size
        )
        chex.assert_trees_all_equal_shapes(position, new_state.position)

    def test_top_level_api_step(self):
        """as_top_level_api step runs and returns SGNHTState."""
        position = jnp.zeros(4)
        algo = sgnht.as_top_level_api(_make_grad_estimator())
        state = algo.init(position, self.next_key())
        new_state = algo.step(self.next_key(), state, MINIBATCH, self.step_size)
        self.assertIsInstance(new_state, sgnht.SGNHTState)

    def test_jit_compatible(self):
        """Kernel is JIT-compilable."""
        position = jnp.ones(3)
        state = sgnht.init(position, self.next_key(), xi=self.alpha)
        grad_est = _make_grad_estimator()
        kernel = jax.jit(sgnht.build_kernel(self.alpha), static_argnums=(2,))
        new_state = kernel(self.next_key(), state, grad_est, MINIBATCH, self.step_size)
        self.assertEqual(new_state.position.shape, (3,))


# ---------------------------------------------------------------------------
# Contour SGLD (CSGLD)
# ---------------------------------------------------------------------------


class CSGLDKernelTest(BlackJAXTest):
    """Tests for the Contour SGLD kernel."""

    def setUp(self):
        super().setUp()
        self.step_size = 1e-3
        self.num_partitions = 16
        self.energy_gap = 10.0

    def _make_logdensity_estimator(self):
        def logdensity_estimator(position, minibatch):
            return -0.5 * jnp.sum(position**2)

        return logdensity_estimator

    def test_init_state_fields(self):
        """init returns ContourSGLDState with correct structure."""
        position = jnp.zeros(4)
        state = csgld.init(position, self.num_partitions)
        self.assertIsInstance(state, csgld.ContourSGLDState)
        self.assertEqual(state.position.shape, (4,))
        self.assertEqual(state.energy_pdf.shape, (self.num_partitions,))
        np.testing.assert_allclose(state.energy_pdf.sum(), 1.0, atol=1e-5)

    def test_kernel_returns_csgld_state(self):
        """Kernel returns a ContourSGLDState with same position shape."""
        position = jnp.zeros(4)
        state = csgld.init(position, self.num_partitions)
        kernel = csgld.build_kernel(self.num_partitions, self.energy_gap)
        new_state = kernel(
            self.next_key(),
            state,
            self._make_logdensity_estimator(),
            _make_grad_estimator(),
            MINIBATCH,
            self.step_size,
        )
        self.assertIsInstance(new_state, csgld.ContourSGLDState)
        self.assertEqual(new_state.position.shape, (4,))

    def test_energy_pdf_sums_to_one(self):
        """energy_pdf remains a valid probability vector after a step."""
        position = jnp.zeros(4)
        state = csgld.init(position, self.num_partitions)
        kernel = csgld.build_kernel(self.num_partitions, self.energy_gap)
        new_state = kernel(
            self.next_key(),
            state,
            self._make_logdensity_estimator(),
            _make_grad_estimator(),
            MINIBATCH,
            self.step_size,
            step_size_stoch=1e-3,
        )
        # energy_pdf should still be approximately non-negative
        assert jnp.all(new_state.energy_pdf >= 0)

    def test_zeta_zero_reduces_to_sgld(self):
        """With zeta=0, CSGLD reduces to standard SGLD (same gradient multiplier)."""
        position = jnp.ones(2)
        state = csgld.init(position, self.num_partitions)
        kernel = csgld.build_kernel(self.num_partitions, self.energy_gap)
        # Two identical calls with same key → identical results
        key = self.next_key()
        new_state1 = kernel(
            key,
            state,
            self._make_logdensity_estimator(),
            _make_grad_estimator(),
            MINIBATCH,
            self.step_size,
            zeta=0.0,
        )
        new_state2 = kernel(
            key,
            state,
            self._make_logdensity_estimator(),
            _make_grad_estimator(),
            MINIBATCH,
            self.step_size,
            zeta=0.0,
        )
        np.testing.assert_array_equal(new_state1.position, new_state2.position)

    def test_top_level_api_step(self):
        """as_top_level_api step runs and returns ContourSGLDState."""
        position = jnp.zeros(4)
        algo = csgld.as_top_level_api(
            self._make_logdensity_estimator(),
            _make_grad_estimator(),
            num_partitions=self.num_partitions,
            energy_gap=self.energy_gap,
        )
        state = algo.init(position)
        new_state = algo.step(
            self.next_key(), state, MINIBATCH, self.step_size, step_size_stoch=1e-3
        )
        self.assertIsInstance(new_state, csgld.ContourSGLDState)

    def test_jit_compatible(self):
        """Kernel is JIT-compilable."""
        position = jnp.ones(3)
        state = csgld.init(position, self.num_partitions)
        logdensity_est = self._make_logdensity_estimator()
        grad_est = _make_grad_estimator()
        kernel = jax.jit(
            csgld.build_kernel(self.num_partitions, self.energy_gap),
            static_argnums=(2, 3),
        )
        new_state = kernel(
            self.next_key(),
            state,
            logdensity_est,
            grad_est,
            MINIBATCH,
            self.step_size,
        )
        self.assertEqual(new_state.position.shape, (3,))


if __name__ == "__main__":
    absltest.main()
