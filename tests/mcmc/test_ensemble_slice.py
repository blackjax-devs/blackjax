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
"""Tests for the ensemble slice sampler."""

from functools import partial

import chex
import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest, parameterized

import blackjax
from blackjax.mcmc.ensemble import EnsembleState
from blackjax.mcmc.ensemble_slice import (
    differential_direction,
    gaussian_direction,
    kde_direction,
)
from blackjax.mcmc.slice import SliceInfo
from blackjax.util import run_inference_algorithm
from tests.fixtures import (
    BlackJAXTest,
    assert_chain_mean,
    assert_iid_mean,
    correlated_gaussian,
    mixture_logdensity,
    mixture_moments,
    rosenbrock_logdensity,
    sample_mixture,
    sample_rosenbrock,
    std_normal_logdensity,
    whitened_moments,
)

DIRECTIONS = [
    dict(testcase_name="differential", direction=differential_direction),
    dict(testcase_name="gaussian", direction=gaussian_direction),
    dict(testcase_name="kde", direction=kde_direction()),
]


def run_ensemble(algorithm, initial_position, rng_key, num_steps):
    _, (positions, info) = run_inference_algorithm(
        rng_key,
        algorithm,
        num_steps,
        initial_state=algorithm.init(initial_position),
        transform=lambda state, info: (state.position, info),
    )
    return positions, info


def final_position(algorithm, initial_position, rng_key, num_steps):
    final_state, _ = run_inference_algorithm(
        rng_key,
        algorithm,
        num_steps,
        initial_state=algorithm.init(initial_position),
        transform=lambda state, info: None,
    )
    return final_state.position


class EnsembleSliceTest(BlackJAXTest):
    def test_jit_and_no_recompile(self):
        chex.clear_trace_counter()
        algorithm = blackjax.ensemble_slice(std_normal_logdensity)
        state = algorithm.init(jax.random.normal(self.next_key(), (8, 2)))
        step = jax.jit(chex.assert_max_traces(algorithm.step, n=1))
        state, _ = step(self.next_key(), state)
        state, _ = step(self.next_key(), state)

    @parameterized.named_parameters(DIRECTIONS)
    def test_step(self, direction):
        algorithm = blackjax.ensemble_slice(std_normal_logdensity, direction)
        state = algorithm.init(np.random.default_rng(0).normal(size=(7, 2)))
        new_state, info = algorithm.step(self.next_key(), state)
        self.assertIsInstance(new_state, EnsembleState)
        self.assertIsInstance(info, SliceInfo)
        chex.assert_trees_all_equal_shapes(state, new_state)
        chex.assert_shape(list(info), (7,))
        self.assertTrue(np.all(info.is_accepted))
        np.testing.assert_allclose(
            new_state.logdensity,
            jax.vmap(std_normal_logdensity)(new_state.position),
            rtol=1e-6,
        )

    def test_rank_deficient_covariance(self):
        keys = jax.random.split(self.next_key(), 100)
        complementary_positions = jax.random.normal(self.next_key(), (100, 2, 2))
        direction = jax.vmap(gaussian_direction)
        self.assertTrue(np.all(np.isfinite(direction(keys, complementary_positions))))

    @parameterized.named_parameters(DIRECTIONS)
    def test_position_dtype_is_preserved(self, direction):
        with jax.enable_x64():
            algorithm = blackjax.ensemble_slice(std_normal_logdensity, direction)
            position = jax.random.normal(self.next_key(), (8, 2), jnp.float32)
            new_state, _ = algorithm.step(self.next_key(), algorithm.init(position))
        self.assertEqual(new_state.position.dtype, jnp.float32)

    @parameterized.named_parameters(DIRECTIONS)
    def test_pytree_position(self, direction):
        def logdensity_fn(x):
            return -0.5 * (x["a"] ** 2 + jnp.sum((x["b"] - 2.0) ** 2))

        position = {
            "a": jax.random.normal(self.next_key(), (16,)),
            "b": 2.0 + jax.random.normal(self.next_key(), (16, 3)),
        }
        algorithm = blackjax.ensemble_slice(logdensity_fn, direction)
        positions, _ = run_ensemble(algorithm, position, self.next_key(), 20_000)
        assert_chain_mean(positions["a"][2_000:], 0.0, 0.05)
        assert_chain_mean(positions["b"][2_000:], 2.0, 0.05)


class EnsembleSliceAffineInvarianceTest(BlackJAXTest):
    def test_trajectory_commutes_with_affine_map(self):
        with jax.enable_x64():
            dim, num_walkers, num_steps = 4, 12, 50
            rng = np.random.default_rng(1)
            matrix = rng.normal(size=(dim, dim)) * np.logspace(-1, 1, dim)
            shift = rng.normal(size=dim)
            inverse = np.linalg.inv(matrix)

            def transformed_logdensity(y):
                return std_normal_logdensity(inverse @ (y - shift))

            initial_position = jax.random.normal(
                self.next_key(), (num_walkers, dim), jnp.float64
            )
            key = self.next_key()
            positions, info = run_ensemble(
                blackjax.ensemble_slice(std_normal_logdensity),
                initial_position,
                key,
                num_steps,
            )
            transformed_positions, transformed_info = run_ensemble(
                blackjax.ensemble_slice(transformed_logdensity),
                initial_position @ matrix.T + shift,
                key,
                num_steps,
            )

        np.testing.assert_array_equal(info.num_shrink, transformed_info.num_shrink)
        np.testing.assert_allclose(
            transformed_positions,
            np.asarray(positions) @ matrix.T + shift,
            rtol=1e-8,
            atol=1e-8,
        )


class EnsembleSliceInvarianceTest(BlackJAXTest):
    """Independent ensembles started at exact draws from the target."""

    num_replicates = 2_000
    num_walkers = 16
    num_steps = 30

    def final_positions(self, algorithm, initial_position):
        keys = jax.random.split(self.next_key(), self.num_replicates)
        run = jax.vmap(partial(final_position, algorithm, num_steps=self.num_steps))
        return np.asarray(run(initial_position, keys))

    def test_halves_are_updated_in_turn(self):
        """Updating both halves at once contracts four walkers by about 1%."""
        num_replicates, num_walkers, dim = 200_000, 4, 2
        algorithm = blackjax.ensemble_slice(std_normal_logdensity)
        keys = jax.random.split(self.next_key(), num_replicates)
        initial_position = jax.random.normal(
            self.next_key(), (num_replicates, num_walkers, dim)
        )
        run = jax.vmap(partial(final_position, algorithm, num_steps=10))
        final = run(initial_position, keys)
        assert_iid_mean((np.asarray(final) ** 2).mean(axis=(1, 2)), 1.0)

    @parameterized.named_parameters(DIRECTIONS)
    def test_correlated_gaussian(self, direction):
        with jax.enable_x64():
            mean, cholesky, logdensity_fn, whiten = correlated_gaussian(5)
            noise = jax.random.normal(
                self.next_key(), (self.num_replicates, self.num_walkers, 5)
            )
            initial_position = mean + np.asarray(noise) @ cholesky.T
            algorithm = blackjax.ensemble_slice(logdensity_fn, direction)
            final = self.final_positions(algorithm, initial_position)
        first, first_expected, second, second_expected = whitened_moments(whiten(final))
        assert_iid_mean(first.mean(axis=1), first_expected)
        assert_iid_mean(second.mean(axis=1), second_expected)

    @parameterized.named_parameters(DIRECTIONS)
    def test_rosenbrock(self, direction):
        initial_position = sample_rosenbrock(
            self.next_key(), (self.num_replicates, self.num_walkers)
        )
        algorithm = blackjax.ensemble_slice(rosenbrock_logdensity, direction)
        final = self.final_positions(algorithm, initial_position)
        assert_iid_mean(final.mean(axis=1), np.array([1.0, 11.0]))
        assert_iid_mean(((final[..., 0] - 1.0) ** 2).mean(axis=1), 10.0)
        assert_iid_mean(((final[..., 1] - 11.0) ** 2).mean(axis=1), 240.1)

    @parameterized.named_parameters(DIRECTIONS)
    def test_mixture(self, direction):
        mean, variance, probability_negative = mixture_moments()
        initial_position = sample_mixture(
            self.next_key(), (self.num_replicates, self.num_walkers)
        )
        algorithm = blackjax.ensemble_slice(mixture_logdensity, direction)
        final = self.final_positions(algorithm, initial_position)
        assert_iid_mean(final.mean(axis=1), mean)
        assert_iid_mean(((final - mean) ** 2).mean(axis=1), variance)
        assert_iid_mean((final[..., 0] < 0).mean(axis=1), probability_negative)


class EnsembleSliceConvergenceTest(BlackJAXTest):
    num_walkers = 32

    @parameterized.named_parameters(DIRECTIONS)
    def test_correlated_gaussian(self, direction):
        with jax.enable_x64():
            _, _, logdensity_fn, whiten = correlated_gaussian(5)
            initial_position = jax.random.normal(
                self.next_key(), (self.num_walkers, 5), jnp.float64
            )
            algorithm = blackjax.ensemble_slice(logdensity_fn, direction)
            positions, _ = run_ensemble(
                algorithm, initial_position, self.next_key(), 30_000
            )
        first, first_expected, second, second_expected = whitened_moments(
            whiten(positions[5_000:])
        )
        assert_chain_mean(first, first_expected, 0.05)
        assert_chain_mean(second, second_expected, 0.1)

    @parameterized.named_parameters(DIRECTIONS)
    def test_mixture(self, direction):
        mean, variance, probability_negative = mixture_moments()
        initial_position = jax.random.normal(self.next_key(), (self.num_walkers, 2))
        algorithm = blackjax.ensemble_slice(mixture_logdensity, direction)
        positions, _ = run_ensemble(
            algorithm, initial_position, self.next_key(), 20_000
        )
        positions = np.asarray(positions[2_000:])
        assert_chain_mean(positions, mean, 0.05 * np.sqrt(variance))
        assert_chain_mean((positions - mean) ** 2, variance, 0.1 * variance)
        assert_chain_mean(positions[..., 0] < 0, probability_negative, 0.02)


if __name__ == "__main__":
    absltest.main()
