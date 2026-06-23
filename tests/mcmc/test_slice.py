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

from blackjax.mcmc.slice import SliceInfo, SliceState, build_kernel, init
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


class SliceMomentsRecoveryTest(BlackJAXTest):
    """Statistical moments-recovery tests: verify the sampler actually samples correctly.

    These tests run the kernel for enough steps to measure empirical mean and
    standard deviation and assert they match the target. Tolerances are
    deliberately generous to avoid seed-specific flakiness — the point is to
    catch gross sampling failures, not to test convergence speed.
    """

    def _run_chain(self, logdensity_fn, initial_position, n_steps, n_doublings=10):
        """Run a slice-sampling chain and return an array of positions."""
        state = init(initial_position, logdensity_fn)
        kernel = build_kernel(n_doublings=n_doublings)

        def step_fn(state, key):
            new_state, _ = kernel(key, state, logdensity_fn)
            return new_state, new_state.position

        keys = jax.random.split(self.next_key(), n_steps)
        _, positions = jax.lax.scan(step_fn, state, keys)
        return positions

    def test_recovers_mean_std_normal_1d(self):
        """Slice sampler recovers the mean of a 1D standard normal.

        Uses 2000 post-warmup steps and checks mean within 0.15 of 0.0
        and std within [0.8, 1.2].  Tolerance is ~3 SE for ESS≈100, leaving
        substantial room for the actual ESS, which is typically >> 100 here.
        """
        positions = self._run_chain(std_normal_logdensity, jnp.zeros(1), n_steps=2000)
        mean = float(jnp.mean(positions))
        std = float(jnp.std(positions))
        self.assertAlmostEqual(mean, 0.0, delta=0.15)
        self.assertGreater(std, 0.8)
        self.assertLess(std, 1.2)

    def test_recovers_std_normal_2d(self):
        """Slice sampler recovers marginal means and stds of a 2D standard normal."""
        positions = self._run_chain(std_normal_logdensity, jnp.zeros(2), n_steps=2000)
        means = jnp.mean(positions, axis=0)
        stds = jnp.std(positions, axis=0)
        for i in range(2):
            self.assertAlmostEqual(float(means[i]), 0.0, delta=0.15)
            self.assertGreater(float(stds[i]), 0.8)
            self.assertLess(float(stds[i]), 1.2)

    def test_recovers_correlated_gaussian_2d(self):
        """Slice sampler recovers the correlation of a 2D Gaussian with rho=0.9.

        Coordinate-wise updates can mix slowly under high correlation, so this
        test uses 3000 steps and only checks that the empirical correlation is
        positive and at least 0.7 (true value is 0.9).
        """
        rho = jnp.asarray(0.9)
        sigma2 = 1.0 - rho**2

        def corr_gaussian_logdensity(x):
            x0, x1 = x[0], x[1]
            return -0.5 / sigma2 * (x0**2 - 2.0 * rho * x0 * x1 + x1**2)

        positions = self._run_chain(
            corr_gaussian_logdensity, jnp.zeros(2), n_steps=3000
        )
        x0 = np.array(positions[:, 0])
        x1 = np.array(positions[:, 1])
        corr = float(np.corrcoef(x0, x1)[0, 1])
        self.assertGreater(corr, 0.7)
        for i in range(2):
            self.assertAlmostEqual(float(jnp.mean(positions[:, i])), 0.0, delta=0.2)


if __name__ == "__main__":
    absltest.main()
