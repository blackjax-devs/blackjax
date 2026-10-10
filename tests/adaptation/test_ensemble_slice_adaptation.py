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
"""Tests for the bracket-width adaptation of the ensemble slice sampler."""

import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest

import blackjax
from blackjax.adaptation.ensemble_slice_adaptation import base
from blackjax.mcmc.slice import SliceInfo
from blackjax.util import run_inference_algorithm
from tests.fixtures import (
    BlackJAXTest,
    assert_chain_mean,
    correlated_gaussian,
    whitened_moments,
)


def slice_info(num_expansions, num_shrink, is_accepted):
    num_walkers = len(num_expansions)
    return SliceInfo(
        is_accepted=jnp.asarray(is_accepted),
        num_expansions=jnp.asarray(num_expansions),
        num_shrink=jnp.asarray(num_shrink),
        bracket_left=jnp.zeros(num_walkers),
        bracket_right=jnp.zeros(num_walkers),
    )


class EnsembleSliceAdaptationTest(BlackJAXTest):
    def test_width_update(self):
        init, update = base(tolerance=0.05, patience=1)
        state = init(2.0)

        state = update(state, slice_info([1, 2], [3, 2], [True, True]))
        np.testing.assert_allclose(state.width, 2.0 * 2.0 * 3 / (3 + 3))
        self.assertEqual(int(state.num_within_tolerance), 1)

        state = update(state, slice_info([0, 0], [4, 1], [True, False]))
        np.testing.assert_allclose(state.width, 2.0 * 2.0 * 1 / (1 + 4))
        self.assertEqual(int(state.num_within_tolerance), 1)

        state = update(state, slice_info([2, 2], [3, 3], [True, True]))
        frozen_width = 2.0 * 2.0 * 1 / (1 + 4) * 2.0 * 4 / (4 + 4)
        np.testing.assert_allclose(state.width, frozen_width)
        self.assertEqual(int(state.num_within_tolerance), 2)

        state = update(state, slice_info([9, 9], [1, 1], [True, True]))
        np.testing.assert_allclose(state.width, frozen_width)

    def test_adaptation(self):
        mean, cholesky, logdensity_fn, whiten = correlated_gaussian(5)
        noise = jax.random.normal(self.next_key(), (32, 5))
        initial_position = jnp.asarray(
            mean + np.asarray(noise) @ cholesky.T, noise.dtype
        )
        warmup = blackjax.ensemble_slice_adaptation(logdensity_fn)
        (state, parameters), info = warmup.run(self.next_key(), initial_position, 500)

        self.assertGreater(int(info.adaptation_state.num_within_tolerance[-1]), 5)
        num_expansions = info.info.num_expansions[-100:].sum()
        num_contractions = (info.info.num_shrink - info.info.is_accepted)[-100:].sum()
        expansion_fraction = num_expansions / (num_expansions + num_contractions)
        self.assertBetween(float(expansion_fraction), 0.35, 0.65)

        algorithm = blackjax.ensemble_slice(logdensity_fn, **parameters)
        _, positions = run_inference_algorithm(
            self.next_key(),
            algorithm,
            20_000,
            initial_state=state,
            transform=lambda state, info: state.position,
        )
        first, first_expected, second, second_expected = whitened_moments(
            whiten(positions)
        )
        assert_chain_mean(first, first_expected, 0.05)
        assert_chain_mean(second, second_expected, 0.1)


if __name__ == "__main__":
    absltest.main()
