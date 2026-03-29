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
import jax
import jax.numpy as jnp
import jax.scipy.stats as stats
import numpy as np
import optax
from absl.testing import absltest

import blackjax
from blackjax.vi.fullrank_vi import (
    FRVIState,
    generate_fullrank_logdensity,
    init,
    sample,
    step,
)
from tests.util import BlackJAXTest, std_normal_logdensity


class FRVIUnitTest(BlackJAXTest):
    """Unit tests for the full-rank VI building blocks."""

    def setUp(self):
        super().setUp()
        self.optimizer = optax.adam(1e-2)

    def test_init_zeros_mean(self):
        """init sets mu to zero for all leaves."""
        position = jnp.ones(4)
        state = init(position, self.optimizer)
        np.testing.assert_array_equal(state.mu, jnp.zeros(4))

    def test_init_chol_params_shape(self):
        """init chol_params has shape d*(d+1)//2 for d-dimensional position."""
        dim = 4
        position = jnp.ones(dim)
        state = init(position, self.optimizer)
        self.assertEqual(state.chol_params.shape, (dim * (dim + 1) // 2,))

    def test_init_pytree_position(self):
        """init works with PyTree positions."""
        position = {"a": jnp.zeros(3), "b": jnp.ones(2)}
        state = init(position, self.optimizer)
        # mu should match the PyTree structure of position
        jax.tree.map(
            lambda m, p: self.assertEqual(m.shape, p.shape), state.mu, position
        )
        # chol_params for a 5-d problem: 5*6//2 = 15
        self.assertEqual(state.chol_params.shape, (15,))

    def test_step_returns_state_and_info(self):
        """step returns (FRVIState, FRVIInfo) with finite ELBO."""
        position = jnp.zeros(2)
        state = init(position, self.optimizer)

        new_state, info = step(
            self.next_key(), state, std_normal_logdensity, self.optimizer
        )
        self.assertIsInstance(new_state, FRVIState)
        assert jnp.isfinite(info.elbo)

    def test_step_output_shapes_unchanged(self):
        """step output mu/chol_params have same shape as before."""
        position = {"x": jnp.zeros(3), "y": jnp.zeros(2)}
        state = init(position, self.optimizer)

        new_state, _ = step(
            self.next_key(), state, std_normal_logdensity, self.optimizer
        )
        jax.tree.map(
            lambda a, b: self.assertEqual(a.shape, b.shape), state.mu, new_state.mu
        )
        self.assertEqual(state.chol_params.shape, new_state.chol_params.shape)

    def test_elbo_decreases_over_steps(self):
        """ELBO (KL divergence) should decrease after several optimization steps."""
        position = jnp.zeros(2)
        state = init(position, self.optimizer)

        def logdensity_fn(x):
            return -0.5 * jnp.sum((x - 3.0) ** 2)

        initial_elbo = None
        for i in range(50):
            subkey = jax.random.fold_in(self.next_key(), i)
            state, info = jax.jit(step, static_argnums=(2, 3))(
                subkey, state, logdensity_fn, self.optimizer
            )
            if initial_elbo is None:
                initial_elbo = float(info.elbo)

        assert float(info.elbo) < initial_elbo

    def test_sample_shape(self):
        """sample returns (num_samples, dim) shaped output."""
        position = jnp.zeros(3)
        state = init(position, self.optimizer)
        samples = sample(self.next_key(), state, num_samples=20)
        self.assertEqual(samples.shape, (20, 3))

    def test_sample_pytree_shape(self):
        """sample works with PyTree positions."""
        position = {"a": jnp.zeros(2), "b": jnp.zeros(4)}
        state = init(position, self.optimizer)
        samples = sample(self.next_key(), state, num_samples=10)
        self.assertEqual(samples["a"].shape, (10, 2))
        self.assertEqual(samples["b"].shape, (10, 4))

    def test_generate_fullrank_logdensity(self):
        """generate_fullrank_logdensity returns a finite scalar."""
        position = jnp.zeros(3)
        state = init(position, self.optimizer)
        logdensity = generate_fullrank_logdensity(state.mu, state.chol_params)
        val = logdensity(jnp.ones(3))
        self.assertEqual(val.shape, ())
        assert jnp.isfinite(val)

    def test_jit_compatible(self):
        """step is JIT-compilable."""
        position = jnp.zeros(2)
        state = init(position, self.optimizer)

        new_state, info = jax.jit(step, static_argnums=(2, 3))(
            self.next_key(), state, std_normal_logdensity, self.optimizer
        )
        assert jnp.isfinite(info.elbo)


class FRVITest(BlackJAXTest):
    """Integration tests for the full-rank VI top-level API."""

    def test_recover_diagonal_posterior(self):
        """Full-rank VI should recover an independent Gaussian posterior."""
        ground_truth = [
            # loc, scale
            (2, 4),
            (3, 5),
        ]

        def logdensity_fn(x):
            logpdf = stats.norm.logpdf(x["x_1"], *ground_truth[0]) + stats.norm.logpdf(
                x["x_2"], *ground_truth[1]
            )
            return jnp.sum(logpdf)

        initial_position = {"x_1": 0.0, "x_2": 0.0}

        num_steps = 15_000
        num_samples = 100

        optimizer = optax.adam(1e-2)
        frvi = blackjax.fullrank_vi(logdensity_fn, optimizer, num_samples)
        state = frvi.init(initial_position)

        rng_key = self.next_key()
        for i in range(num_steps):
            subkey = jax.random.fold_in(rng_key, i)
            state, _ = jax.jit(frvi.step)(subkey, state)

        loc_1, loc_2 = state.mu["x_1"], state.mu["x_2"]
        # Diagonal entries: chol_params[0] = log(scale_1), chol_params[1] = log(scale_2)
        scale_1 = jnp.exp(state.chol_params[0])
        scale_2 = jnp.exp(state.chol_params[1])
        self.assertAlmostEqual(loc_1, ground_truth[0][0], delta=0.1)
        self.assertAlmostEqual(scale_1, ground_truth[0][1], delta=0.1)
        self.assertAlmostEqual(loc_2, ground_truth[1][0], delta=0.1)
        self.assertAlmostEqual(scale_2, ground_truth[1][1], delta=0.1)

    def test_recover_correlated_posterior(self):
        """Full-rank VI should capture off-diagonal covariance structure.

        Target: 2D Gaussian with mean [1, 2] and correlation 0.8.
        This test exercises the off-diagonal Cholesky entries — the key
        feature that distinguishes full-rank from mean-field VI.
        """
        # Covariance: [[1, 0.8], [0.8, 1]]  (std=1, correlation=0.8)
        target_mean = jnp.array([1.0, 2.0])
        target_cov = jnp.array([[1.0, 0.8], [0.8, 1.0]])
        target_prec = jnp.linalg.inv(target_cov)

        def logdensity_fn(x):
            diff = x - target_mean
            return -0.5 * diff @ target_prec @ diff

        initial_position = jnp.zeros(2)

        num_steps = 10_000
        num_samples = 50

        optimizer = optax.adam(1e-2)
        frvi = blackjax.fullrank_vi(logdensity_fn, optimizer, num_samples)
        state = frvi.init(initial_position)

        rng_key = self.next_key()
        for i in range(num_steps):
            subkey = jax.random.fold_in(rng_key, i)
            state, _ = jax.jit(frvi.step)(subkey, state)

        # Recover mean
        np.testing.assert_allclose(state.mu, target_mean, atol=0.1)

        # Recover Cholesky factor of covariance.
        # Target Cholesky L s.t. L L^T = [[1, 0.8], [0.8, 1]]:
        #   L[0,0] = 1.0, L[1,0] = 0.8, L[1,1] = sqrt(1 - 0.64) = 0.6
        # chol_params: [log(L[0,0]), log(L[1,1]), L[1,0]] = [0, log(0.6), 0.8]
        l00 = jnp.exp(state.chol_params[0])  # should be ~1.0
        l11 = jnp.exp(state.chol_params[1])  # should be ~0.6
        l10 = state.chol_params[2]  # should be ~0.8

        self.assertAlmostEqual(float(l00), 1.0, delta=0.1)
        self.assertAlmostEqual(float(l11), 0.6, delta=0.1)
        self.assertAlmostEqual(float(l10), 0.8, delta=0.1)


if __name__ == "__main__":
    absltest.main()
