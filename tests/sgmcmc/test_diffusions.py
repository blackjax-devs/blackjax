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
"""Unit tests for sgmcmc diffusion solvers."""

import chex
import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest, parameterized

import blackjax.sgmcmc.diffusions as diffusions
from tests.fixtures import BlackJAXTest


class OverdampedLangevinTest(BlackJAXTest):
    """Tests for the overdamped Langevin diffusion solver."""

    def setUp(self):
        super().setUp()
        self.step_size = 1e-3
        self.step = diffusions.overdamped_langevin()

    def test_returns_same_pytree_structure(self):
        """Output position has same structure as input."""
        position = {"x": jnp.array([1.0, 2.0]), "y": jnp.array(3.0)}
        grad = jax.tree.map(jnp.ones_like, position)
        new_position = self.step(self.next_key(), position, grad, self.step_size)
        chex.assert_trees_all_equal_shapes(position, new_position)

    def test_zero_gradient_moves_only_via_noise(self):
        """With zero gradient the update is purely noise-driven."""
        position = jnp.zeros(4)
        grad = jnp.zeros(4)
        # With temperature=0 the noise term vanishes → position unchanged
        new_position = self.step(
            self.next_key(), position, grad, self.step_size, temperature=0.0
        )
        np.testing.assert_allclose(new_position, position, atol=1e-6)

    def test_gradient_moves_position_in_correct_direction(self):
        """Positive gradient should increase position on average."""
        position = jnp.zeros(1)
        grad = jnp.array([1.0])

        # Run many steps with temperature=0 to isolate the gradient term

        def one_step(pos, key):
            return self.step(key, pos, grad, self.step_size, temperature=0.0), None

        keys = jax.random.split(self.next_key(), 100)
        final_position, _ = jax.lax.scan(one_step, position, keys)
        assert float(final_position[0]) > 0.0

    @parameterized.parameters(
        {"temperature": 0.5},
        {"temperature": 1.0},
        {"temperature": 2.0},
    )
    def test_jit_compatible(self, temperature):
        """Step function is JIT-compilable."""
        position = jnp.ones(3)
        grad = jnp.ones(3)
        jit_step = jax.jit(self.step)
        new_position = jit_step(
            self.next_key(), position, grad, self.step_size, temperature
        )
        self.assertEqual(new_position.shape, (3,))


class SGHMCDiffusionTest(BlackJAXTest):
    """Tests for the SGHMC diffusion solver."""

    def setUp(self):
        super().setUp()
        self.step_size = 1e-3
        self.alpha = 0.01
        self.beta = 0.0
        self.step = diffusions.sghmc(self.alpha, self.beta)

    def test_returns_position_and_momentum(self):
        """Step returns a (position, momentum) pair with correct shapes."""
        position = jnp.zeros(4)
        momentum = jnp.ones(4)
        grad = jnp.ones(4)
        new_position, new_momentum = self.step(
            self.next_key(), position, momentum, grad, self.step_size
        )
        self.assertEqual(new_position.shape, (4,))
        self.assertEqual(new_momentum.shape, (4,))

    def test_position_update_uses_momentum(self):
        """Position update is position + step_size * momentum (before noise)."""
        position = jnp.zeros(2)
        momentum = jnp.ones(2)
        grad = jnp.zeros(2)
        # With temperature=0 and zero gradient, noise term vanishes.
        new_position, _ = self.step(
            self.next_key(), position, momentum, grad, self.step_size, temperature=0.0
        )
        expected = position + self.step_size * momentum
        np.testing.assert_allclose(new_position, expected, atol=1e-6)

    def test_momentum_friction_reduces_magnitude(self):
        """Friction term (alpha) reduces momentum over time."""
        position = jnp.zeros(2)
        momentum = jnp.ones(2) * 10.0
        grad = jnp.zeros(2)

        def one_step(state, key):
            pos, mom = state
            new_pos, new_mom = self.step(
                key, pos, mom, grad, self.step_size, temperature=0.0
            )
            return (new_pos, new_mom), None

        keys = jax.random.split(self.next_key(), 200)
        (_, final_momentum), _ = jax.lax.scan(one_step, (position, momentum), keys)
        # Momentum should have decayed toward zero
        assert float(jnp.linalg.norm(final_momentum)) < float(jnp.linalg.norm(momentum))

    def test_jit_compatible(self):
        """Step function is JIT-compilable."""
        position = jnp.ones(3)
        momentum = jnp.ones(3)
        grad = jnp.ones(3)
        jit_step = jax.jit(self.step)
        new_position, new_momentum = jit_step(
            self.next_key(), position, momentum, grad, self.step_size
        )
        self.assertEqual(new_position.shape, (3,))
        self.assertEqual(new_momentum.shape, (3,))

    def test_pytree_position(self):
        """Step works with PyTree positions."""
        position = {"a": jnp.zeros(2), "b": jnp.zeros(3)}
        momentum = {"a": jnp.ones(2), "b": jnp.ones(3)}
        grad = jax.tree.map(jnp.zeros_like, position)
        new_position, new_momentum = self.step(
            self.next_key(), position, momentum, grad, self.step_size
        )
        chex.assert_trees_all_equal_shapes(position, new_position)
        chex.assert_trees_all_equal_shapes(momentum, new_momentum)


class SGNHTDiffusionTest(BlackJAXTest):
    """Tests for the SGNHT diffusion solver."""

    def setUp(self):
        super().setUp()
        self.step_size = 1e-3
        self.alpha = 0.01
        self.beta = 0.0
        self.step = diffusions.sgnht(self.alpha, self.beta)

    def test_returns_position_momentum_xi(self):
        """Step returns (position, momentum, xi) with correct shapes."""
        position = jnp.zeros(4)
        momentum = jnp.ones(4)
        xi = 0.01
        grad = jnp.ones(4)
        new_position, new_momentum, new_xi = self.step(
            self.next_key(), position, momentum, xi, grad, self.step_size
        )
        self.assertEqual(new_position.shape, (4,))
        self.assertEqual(new_momentum.shape, (4,))
        assert jnp.isfinite(new_xi)

    def test_xi_increases_when_kinetic_energy_exceeds_temperature(self):
        """Thermostat xi should increase when momentum norm^2/d > temperature."""
        d = 4
        position = jnp.zeros(d)
        # Large momentum → kinetic energy >> temperature → xi increases
        momentum = jnp.ones(d) * 10.0
        xi = 0.01
        grad = jnp.zeros(d)
        _, _, new_xi = self.step(
            self.next_key(),
            position,
            momentum,
            xi,
            grad,
            self.step_size,
            temperature=1.0,
        )
        assert float(new_xi) > float(xi)

    def test_xi_decreases_when_kinetic_energy_below_temperature(self):
        """Thermostat xi should decrease when momentum norm^2/d < temperature."""
        d = 4
        position = jnp.zeros(d)
        # Near-zero momentum → kinetic energy << temperature → xi decreases
        momentum = jnp.zeros(d)
        xi = 1.0
        grad = jnp.zeros(d)
        _, _, new_xi = self.step(
            self.next_key(),
            position,
            momentum,
            xi,
            grad,
            self.step_size,
            temperature=1.0,
        )
        assert float(new_xi) < float(xi)

    def test_jit_compatible(self):
        """Step function is JIT-compilable."""
        position = jnp.ones(3)
        momentum = jnp.ones(3)
        xi = 0.01
        grad = jnp.ones(3)
        jit_step = jax.jit(self.step)
        new_position, new_momentum, new_xi = jit_step(
            self.next_key(), position, momentum, xi, grad, self.step_size
        )
        self.assertEqual(new_position.shape, (3,))


if __name__ == "__main__":
    absltest.main()
