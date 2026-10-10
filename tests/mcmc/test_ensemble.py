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
"""Tests for the affine-invariant ensemble sampler and its moves."""

from functools import partial

import chex
import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest, parameterized

import blackjax
from blackjax.base import SamplingAlgorithm
from blackjax.mcmc.ensemble import (
    EnsembleInfo,
    EnsembleState,
    differential_evolution_move,
    kde_move,
    snooker_move,
    stretch_move,
    walk_move,
)
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

MOVES = [
    dict(testcase_name="stretch", move=stretch_move()),
    dict(testcase_name="walk", move=walk_move()),
    dict(testcase_name="walk_with_helpers", move=walk_move(6)),
    dict(testcase_name="differential_evolution", move=differential_evolution_move()),
    dict(testcase_name="snooker", move=snooker_move()),
    dict(testcase_name="kde", move=kde_move()),
]


def run_ensemble(algorithm, initial_position, rng_key, num_steps):
    _, (positions, is_accepted) = run_inference_algorithm(
        rng_key,
        algorithm,
        num_steps,
        initial_state=algorithm.init(initial_position),
        transform=lambda state, info: (state.position, info.is_accepted),
    )
    return positions, is_accepted


def final_position(algorithm, initial_position, rng_key, num_steps):
    final_state, _ = run_inference_algorithm(
        rng_key,
        algorithm,
        num_steps,
        initial_state=algorithm.init(initial_position),
        transform=lambda state, info: None,
    )
    return final_state.position


class EnsembleTest(BlackJAXTest):
    def test_init(self):
        position = jax.random.normal(self.next_key(), (8, 3))
        state = blackjax.ensemble.init(position, std_normal_logdensity)
        self.assertIsInstance(state, EnsembleState)
        np.testing.assert_allclose(
            state.logdensity, jax.vmap(std_normal_logdensity)(position), rtol=1e-6
        )

    def test_jit_and_no_recompile(self):
        chex.clear_trace_counter()
        algorithm = blackjax.ensemble(std_normal_logdensity)
        state = algorithm.init(jax.random.normal(self.next_key(), (8, 2)))
        step = jax.jit(chex.assert_max_traces(algorithm.step, n=1))
        state, _ = step(self.next_key(), state)
        state, _ = step(self.next_key(), state)

    @parameterized.named_parameters(MOVES)
    def test_step(self, move):
        algorithm = blackjax.ensemble(std_normal_logdensity, move)
        state = algorithm.init(np.random.default_rng(0).normal(size=(13, 2)))
        new_state, info = algorithm.step(self.next_key(), state)
        self.assertIsInstance(info, EnsembleInfo)
        chex.assert_trees_all_equal_shapes(state, new_state)
        chex.assert_shape([info.acceptance_rate, info.is_accepted], (13,))
        np.testing.assert_allclose(
            new_state.logdensity,
            jax.vmap(std_normal_logdensity)(new_state.position),
            rtol=1e-6,
        )

    @parameterized.named_parameters(MOVES)
    def test_position_dtype_is_preserved(self, move):
        with jax.enable_x64():
            algorithm = blackjax.ensemble(std_normal_logdensity, move)
            position = jax.random.normal(self.next_key(), (16, 2), jnp.float32)
            new_state, _ = algorithm.step(self.next_key(), algorithm.init(position))
        self.assertEqual(new_state.position.dtype, jnp.float32)

    @parameterized.named_parameters(MOVES)
    def test_pytree_position(self, move):
        def logdensity_fn(x):
            return -0.5 * (x["a"] ** 2 + jnp.sum((x["b"] - 2.0) ** 2))

        position = {
            "a": jax.random.normal(self.next_key(), (16,)),
            "b": 2.0 + jax.random.normal(self.next_key(), (16, 3)),
        }
        algorithm = blackjax.ensemble(logdensity_fn, move)
        state = algorithm.init(position)
        new_state, _ = algorithm.step(self.next_key(), state)
        chex.assert_trees_all_equal_shapes(state, new_state)

        positions, _ = run_ensemble(algorithm, position, self.next_key(), 50_000)
        assert_chain_mean(positions["a"][2_000:], 0.0, 0.05)
        assert_chain_mean(positions["b"][2_000:], 2.0, 0.05)

    @parameterized.named_parameters(MOVES)
    def test_walkers_keep_their_positions_when_rejected(self, move):
        algorithm = blackjax.ensemble(std_normal_logdensity, move)
        state = algorithm.init(jax.random.normal(self.next_key(), (32, 4)))
        new_state, info = algorithm.step(self.next_key(), state)
        rejected = ~np.asarray(info.is_accepted)
        self.assertTrue(rejected.any() and not rejected.all())
        np.testing.assert_array_equal(
            new_state.position[rejected], state.position[rejected]
        )
        moved = np.any(new_state.position != state.position, axis=1)
        np.testing.assert_array_equal(moved, ~rejected)

    def test_rank_deficient_covariance(self):
        keys = jax.random.split(self.next_key(), 100)
        complementary_positions = jax.random.normal(self.next_key(), (8, 3))
        move = jax.vmap(walk_move(2), in_axes=(0, None, None))
        new_positions, _ = move(keys, jnp.zeros(3), complementary_positions)
        self.assertTrue(np.all(np.isfinite(new_positions)))

    def test_mixture_of_moves(self):
        moves = [stretch_move(), differential_evolution_move()]
        steps = [blackjax.ensemble(std_normal_logdensity, m).step for m in moves]
        weights = jnp.array([0.7, 0.3])

        def step(rng_key, state):
            key_choice, key_step = jax.random.split(rng_key)
            index = jax.random.choice(key_choice, len(steps), p=weights)
            return jax.lax.switch(index, steps, key_step, state)

        algorithm = SamplingAlgorithm(
            blackjax.ensemble(std_normal_logdensity).init, step
        )
        initial_state = algorithm.init(jax.random.normal(self.next_key(), (16, 2)))
        _, positions = run_inference_algorithm(
            self.next_key(),
            algorithm,
            20_000,
            initial_state=initial_state,
            transform=lambda state, info: state.position,
        )
        assert_chain_mean(positions[2_000:], 0.0, 0.05)
        assert_chain_mean(positions[2_000:] ** 2, 1.0, 0.05)


class EnsembleAffineInvarianceTest(BlackJAXTest):
    @parameterized.named_parameters(
        dict(testcase_name="stretch", move=stretch_move()),
        dict(
            testcase_name="differential_evolution", move=differential_evolution_move()
        ),
    )
    def test_trajectory_commutes_with_affine_map(self, move):
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
            positions, accepted = run_ensemble(
                blackjax.ensemble(std_normal_logdensity, move),
                initial_position,
                key,
                num_steps,
            )
            transformed_positions, transformed_accepted = run_ensemble(
                blackjax.ensemble(transformed_logdensity, move),
                initial_position @ matrix.T + shift,
                key,
                num_steps,
            )

        np.testing.assert_array_equal(accepted, transformed_accepted)
        np.testing.assert_allclose(
            transformed_positions,
            np.asarray(positions) @ matrix.T + shift,
            rtol=1e-8,
            atol=1e-8,
        )


class EnsembleInvarianceTest(BlackJAXTest):
    """Independent ensembles started at exact draws from the target."""

    num_replicates = 2_000
    num_walkers = 16
    num_steps = 30

    def final_positions(self, algorithm, initial_position):
        keys = jax.random.split(self.next_key(), self.num_replicates)
        run = jax.vmap(partial(final_position, algorithm, num_steps=self.num_steps))
        return np.asarray(run(initial_position, keys))

    def test_halves_are_updated_in_turn(self):
        """Updating both halves at once contracts two walkers by about 3%."""
        num_replicates, num_walkers, dim = 200_000, 2, 2
        algorithm = blackjax.ensemble(std_normal_logdensity)
        keys = jax.random.split(self.next_key(), num_replicates)
        initial_position = jax.random.normal(
            self.next_key(), (num_replicates, num_walkers, dim)
        )
        run = jax.vmap(partial(final_position, algorithm, num_steps=10))
        final = run(initial_position, keys)
        assert_iid_mean((np.asarray(final) ** 2).mean(axis=(1, 2)), 1.0)

    @parameterized.named_parameters(MOVES)
    def test_correlated_gaussian(self, move):
        with jax.enable_x64():
            mean, cholesky, logdensity_fn, whiten = correlated_gaussian(5)
            noise = jax.random.normal(
                self.next_key(), (self.num_replicates, self.num_walkers, 5)
            )
            initial_position = mean + np.asarray(noise) @ cholesky.T
            algorithm = blackjax.ensemble(logdensity_fn, move)
            final = self.final_positions(algorithm, initial_position)
        first, first_expected, second, second_expected = whitened_moments(whiten(final))
        assert_iid_mean(first.mean(axis=1), first_expected)
        assert_iid_mean(second.mean(axis=1), second_expected)

    @parameterized.named_parameters(MOVES)
    def test_rosenbrock(self, move):
        initial_position = sample_rosenbrock(
            self.next_key(), (self.num_replicates, self.num_walkers)
        )
        algorithm = blackjax.ensemble(rosenbrock_logdensity, move)
        final = self.final_positions(algorithm, initial_position)
        assert_iid_mean(final.mean(axis=1), np.array([1.0, 11.0]))
        assert_iid_mean(((final[..., 0] - 1.0) ** 2).mean(axis=1), 10.0)
        assert_iid_mean(((final[..., 1] - 11.0) ** 2).mean(axis=1), 240.1)

    @parameterized.named_parameters(MOVES)
    def test_mixture(self, move):
        mean, variance, probability_negative = mixture_moments()
        initial_position = sample_mixture(
            self.next_key(), (self.num_replicates, self.num_walkers)
        )
        algorithm = blackjax.ensemble(mixture_logdensity, move)
        final = self.final_positions(algorithm, initial_position)
        assert_iid_mean(final.mean(axis=1), mean)
        assert_iid_mean(((final - mean) ** 2).mean(axis=1), variance)
        assert_iid_mean((final[..., 0] < 0).mean(axis=1), probability_negative)


class EnsembleConvergenceTest(BlackJAXTest):
    num_walkers = 32

    @parameterized.named_parameters(MOVES)
    def test_correlated_gaussian(self, move):
        with jax.enable_x64():
            _, _, logdensity_fn, whiten = correlated_gaussian(5)
            initial_position = jax.random.normal(
                self.next_key(), (self.num_walkers, 5), jnp.float64
            )
            algorithm = blackjax.ensemble(logdensity_fn, move)
            positions, _ = run_ensemble(
                algorithm, initial_position, self.next_key(), 60_000
            )
        first, first_expected, second, second_expected = whitened_moments(
            whiten(positions[10_000:])
        )
        assert_chain_mean(first, first_expected, 0.05)
        assert_chain_mean(second, second_expected, 0.1)

    def test_stretch_acceptance_rate(self):
        _, _, logdensity_fn, _ = correlated_gaussian(5)
        initial_position = jax.random.normal(self.next_key(), (self.num_walkers, 5))
        algorithm = blackjax.ensemble(logdensity_fn)
        _, is_accepted = run_ensemble(
            algorithm, initial_position, self.next_key(), 20_000
        )
        self.assertBetween(float(is_accepted[10_000:].mean()), 0.45, 0.65)

    @parameterized.named_parameters(MOVES)
    def test_mixture(self, move):
        mean, variance, probability_negative = mixture_moments()
        initial_position = jax.random.normal(self.next_key(), (self.num_walkers, 2))
        algorithm = blackjax.ensemble(mixture_logdensity, move)
        positions, _ = run_ensemble(
            algorithm, initial_position, self.next_key(), 30_000
        )
        positions = np.asarray(positions[5_000:])
        assert_chain_mean(positions, mean, 0.05 * np.sqrt(variance))
        assert_chain_mean((positions - mean) ** 2, variance, 0.1 * variance)
        assert_chain_mean(positions[..., 0] < 0, probability_negative, 0.02)


if __name__ == "__main__":
    absltest.main()
