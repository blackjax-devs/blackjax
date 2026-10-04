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
"""Tests for the single-path Pathfinder algorithm."""

import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest
from jax.extend import core

import blackjax
from blackjax.vi.pathfinder import PathfinderState, approximate, sample
from tests.fixtures import BlackJAXTest, std_normal_logdensity


class PathfinderApproximateTest(BlackJAXTest):
    """Tests for the `approximate` function."""

    def test_batched_elbo_matches_vectorized_path(self):
        """Chunking both maps preserves the optimization path and PRNG keys."""
        position = {"x": jnp.array([1.5, -0.7])}

        def logdensity(x):
            return -jnp.sum(x["x"] ** 2 / 2 + x["x"] ** 4 / 20)

        key = self.next_key()
        kwargs = dict(num_samples=7, maxiter=4, maxcor=3)
        expected = approximate(key, logdensity, position, **kwargs)
        actual = jax.jit(
            lambda k, x: approximate(k, logdensity, x, batch_size=2, **kwargs)
        )(key, position)
        for reference, result in zip(
            jax.tree.leaves(expected), jax.tree.leaves(actual)
        ):
            np.testing.assert_allclose(result, reference, rtol=1e-5, atol=1e-6)

    def test_both_elbo_axes_are_chunked_through_top_level_api(self):
        """Both maps add scans, and the top-level init forwards batch_size."""

        def count_scans(value):
            if isinstance(value, core.Jaxpr):
                return sum(
                    (eq.primitive.name == "scan") + count_scans(eq.params)
                    for eq in value.eqns
                )
            if isinstance(value, core.ClosedJaxpr):
                return count_scans(value.jaxpr)
            if isinstance(value, dict):
                return sum(count_scans(x) for x in value.values())
            if isinstance(value, (tuple, list)):
                return sum(count_scans(x) for x in value)
            return 0

        key = self.next_key()
        position = jnp.array([1.5, -0.7])
        algo = blackjax.pathfinder(std_normal_logdensity)
        kwargs = dict(num_samples=6, maxiter=3, maxcor=3)
        default = jax.make_jaxpr(lambda k, x: algo.init(k, x, **kwargs))(key, position)
        chunked = jax.make_jaxpr(lambda k, x: algo.init(k, x, batch_size=2, **kwargs))(
            key, position
        )
        # The optimizer has its own scans in both graphs. Each chunked ELBO
        # axis must add a scan beyond those, so dropping either map fails.
        self.assertEqual(count_scans(chunked), count_scans(default) + 2)

    def test_returns_pathfinder_state_and_info(self):
        """approximate returns (PathfinderState, PathfinderInfo)."""
        ndim = 2
        initial_position = jnp.zeros(ndim)

        state, info = approximate(
            self.next_key(), std_normal_logdensity, initial_position
        )
        self.assertIsInstance(state, PathfinderState)
        # Info should contain the full path
        self.assertIsInstance(info.path, PathfinderState)

    def test_state_has_finite_elbo(self):
        """Best ELBO from approximate is finite."""
        initial_position = jnp.zeros(3)

        state, _ = approximate(self.next_key(), std_normal_logdensity, initial_position)
        assert jnp.isfinite(state.elbo), f"ELBO is not finite: {state.elbo}"

    def test_state_position_shape(self):
        """state.position has the same shape as initial_position."""
        ndim = 4
        initial_position = jnp.zeros(ndim)

        state, _ = approximate(self.next_key(), std_normal_logdensity, initial_position)
        self.assertEqual(state.position.shape, (ndim,))

    def test_state_position_near_mode(self):
        """For a Gaussian, the best-ELBO position should be near the mode."""
        ndim = 2
        true_mean = jnp.array([2.0, -1.0])
        initial_position = jnp.zeros(ndim)

        def logdensity_fn(x):
            return -0.5 * jnp.sum((x - true_mean) ** 2)

        state, _ = approximate(self.next_key(), logdensity_fn, initial_position)
        # Position at best ELBO should be close to the true mean
        np.testing.assert_allclose(state.position, true_mean, atol=0.5)

    def test_path_elbo_shape(self):
        """Path ELBO has shape (maxiter+1,)."""
        ndim = 2
        initial_position = jnp.zeros(ndim)
        maxiter = 10

        _, info = approximate(
            self.next_key(), std_normal_logdensity, initial_position, maxiter=maxiter
        )
        self.assertEqual(info.path.elbo.shape, (maxiter + 1,))

    def test_pytree_position(self):
        """approximate works with PyTree initial positions."""
        initial_position = {"w": jnp.zeros(2), "b": jnp.zeros(1)}
        state, _ = approximate(self.next_key(), std_normal_logdensity, initial_position)
        self.assertIsInstance(state, PathfinderState)
        assert jnp.isfinite(state.elbo)


class PathfinderSampleTest(BlackJAXTest):
    """Tests for the `sample` function."""

    def _get_state(self, ndim=2):
        """Helper: run approximate and return the state."""
        initial_position = jnp.zeros(ndim)
        state, _ = approximate(
            jax.random.fold_in(self.next_key(), 0),
            std_normal_logdensity,
            initial_position,
        )
        return state

    def test_sample_returns_correct_shape(self):
        """sample returns array of shape (num_samples, ndim)."""
        ndim = 3
        state = self._get_state(ndim)
        samples, logq = sample(self.next_key(), state, num_samples=50)
        self.assertEqual(samples.shape, (50, ndim))
        self.assertEqual(logq.shape, (50,))

    def test_sample_single_returns_1d(self):
        """sample with num_samples=() returns a single sample (no leading dim)."""
        ndim = 2
        state = self._get_state(ndim)
        s, logq = sample(self.next_key(), state, num_samples=())
        self.assertEqual(s.shape, (ndim,))
        self.assertEqual(logq.shape, ())

    def test_logq_is_finite(self):
        """Log-density of drawn samples is finite."""
        state = self._get_state(ndim=2)
        _, logq = sample(self.next_key(), state, num_samples=20)
        assert jnp.all(jnp.isfinite(logq))

    def test_samples_are_finite(self):
        """Drawn samples are finite."""
        state = self._get_state(ndim=3)
        samples, _ = sample(self.next_key(), state, num_samples=30)
        assert jnp.all(jnp.isfinite(samples))

    def test_sample_mean_near_mode(self):
        """Sample mean should be close to the true posterior mean."""
        ndim = 2
        true_mean = jnp.array([1.0, -2.0])
        initial_position = jnp.zeros(ndim)

        def logdensity_fn(x):
            return -0.5 * jnp.sum((x - true_mean) ** 2)

        state, _ = approximate(
            jax.random.fold_in(self.next_key(), 1), logdensity_fn, initial_position
        )
        samples, _ = sample(self.next_key(), state, num_samples=500)
        np.testing.assert_allclose(samples.mean(0), true_mean, atol=0.5)


class PathfinderTopLevelAPITest(BlackJAXTest):
    """Tests for the top-level API (via blackjax.pathfinder)."""

    def test_init_and_sample(self):
        """Top-level API: init then sample gives correct-shaped output."""
        ndim = 2

        algo = blackjax.pathfinder(std_normal_logdensity)
        key_init, key_sample = jax.random.split(self.next_key())
        initial_position = jnp.zeros(ndim)

        state, _ = algo.init(key_init, initial_position, num_samples=50)
        samples, logq = algo.sample(key_sample, state, 100)
        self.assertEqual(samples.shape, (100, ndim))
        self.assertEqual(logq.shape, (100,))

    def test_step_is_noop(self):
        """Pathfinder step is a no-op (returns the same state)."""
        algo = blackjax.pathfinder(std_normal_logdensity)
        key_init, key_step = jax.random.split(self.next_key())
        state, _ = algo.init(key_init, jnp.zeros(2), num_samples=20)
        new_state, info = algo.step(key_step, state)
        np.testing.assert_array_equal(state.elbo, new_state.elbo)


if __name__ == "__main__":
    absltest.main()
