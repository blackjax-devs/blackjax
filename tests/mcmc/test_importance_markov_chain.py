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
"""Tests for the Importance Markov Chain.

The tests go from the building blocks to the full chain:

``OptimalReplicationTest``
    The replication rule, whose properties are known in closed form.

    - ``test_mean_and_variance``: :math:`\\mathbb{E}[N] = \\rho` and
      :math:`\\mathrm{Var}(N) = \\langle\\rho\\rangle(1 - \\langle\\rho\\rangle)`.
    - ``test_integer_rho_is_deterministic``: an integer :math:`\\rho` always
      gives :math:`\\rho` replicas.
    - ``test_invalid_weights_give_no_replica``: NaN and :math:`-\\infty`
      log-weights give 0 replicas.
    - ``test_large_weights_are_clipped``: counts stop at ``MAX_NUM_REPLICAS``.

``LogKappaTest``
    - ``test_expected_number_of_samples``: the chosen :math:`\\kappa` gives
      the requested expected total, to float precision.

``IndependentSamplingTest``
    The vectorised interface for i.i.d. proposals.

    - ``test_moments_and_number_of_samples``: with :math:`\\kappa` from a
      pilot batch, the weighted moments match the target and each proposal
      gives one sample on average.
    - ``test_rejection_sampling_regime``: with
      :math:`\\kappa < 1 / \\sup(\\pi / q)`, every count is 0 or 1 and the
      kept proposals are draws from the target.
    - ``test_pytree_position``: pytree positions keep their structure.

``InitTest``
    - ``test_initial_state``: the copy counter starts at 0 and the state holds
      the instrumental state and its position.

``CorrectnessTest``
    The distribution of the extended chain.

    - ``test_finite_state_space``: the reference check. On four states the
      target is known exactly, so the frequencies of the chain and the number
      of instrumental steps per sample (:math:`1/\\kappa`) are compared
      without approximation error.
    - ``test_rejection_sampling_regime``: with independent instrumental draws
      and a small :math:`\\kappa`, every step is a new draw from the target.
    - ``test_dropping_exhausted_transitions_is_exact``: with a budget of one
      instrumental step, most transitions are exhausted; dropping them still
      recovers the target exactly.

``KernelTest``
    The mechanics of a transition and the library's API contracts.

    - ``test_copies_are_returned_before_new_draws``: remaining copies are
      returned without any instrumental step.
    - ``test_max_num_steps``: without any replica, a transition stops after
      ``max_num_steps`` steps, keeps its sample and reports it.
    - ``test_jit_and_no_recompile``, ``test_pytree_position`` and
      ``test_vmap_over_chains``: the contracts required of every sampler.

Two more tests live elsewhere: ``imc`` in ``tests/test_api_protocols.py``
checks the ``SamplingAlgorithm`` protocol, and ``test_imc`` in
``tests/mcmc/test_sampling.py`` recovers the moments of a univariate normal.

The random keys change every day (see ``BlackJAXTest``), so statistical
tolerances are set to about 5 standard errors or more. They were checked to
pass for 20 different seeds.
"""

from typing import NamedTuple

import chex
import jax
import jax.numpy as jnp
import jax.scipy.stats as stats
from absl.testing import absltest

import blackjax
from blackjax.base import SamplingAlgorithm
from blackjax.mcmc import importance_markov_chain as imc
from blackjax.util import run_inference_algorithm
from tests.fixtures import BlackJAXTest


def target_logdensity(x):
    return stats.norm.logpdf(x, loc=1.0, scale=0.5)


def proposal_logdensity(x):
    return stats.norm.logpdf(x, loc=0.0, scale=2.0)


def proposal_distribution(rng_key):
    return 2.0 * jax.random.normal(rng_key)


class IndependentState(NamedTuple):
    position: jax.Array


def independent_algorithm(distribution):
    """An instrumental algorithm whose every step is a fresh i.i.d. draw."""

    def init(position, rng_key=None):
        return IndependentState(jnp.asarray(position, dtype=jnp.float32))

    def step(rng_key, state):
        return IndependentState(distribution(rng_key)), None

    return SamplingAlgorithm(init, step)


class DiscreteState(NamedTuple):
    position: jax.Array


def discrete_metropolis(probabilities):
    """Metropolis algorithm on {0, ..., K-1} with uniform proposals."""
    num_states = probabilities.shape[0]

    def init(position, rng_key=None):
        return DiscreteState(jnp.asarray(position, dtype=jnp.int32))

    def step(rng_key, state):
        key_proposal, key_accept = jax.random.split(rng_key)
        proposal = jax.random.randint(key_proposal, (), 0, num_states)
        log_ratio = jnp.log(probabilities[proposal]) - jnp.log(
            probabilities[state.position]
        )
        is_accepted = jnp.log(jax.random.uniform(key_accept)) < log_ratio
        return DiscreteState(jnp.where(is_accepted, proposal, state.position)), None

    return SamplingAlgorithm(init, step)


class OptimalReplicationTest(BlackJAXTest):
    """Tests for the minimal-variance replication rule."""

    def test_mean_and_variance(self):
        """E[N] = rho and Var(N) = frac(rho) (1 - frac(rho))."""
        rho = jnp.array([0.0, 0.3, 1.0, 2.7, 12.5])
        keys = jax.random.split(self.next_key(), 100_000)
        draws = jax.vmap(lambda key: imc.optimal_replication(key, jnp.log(rho)))(keys)

        fractional_part = rho - jnp.floor(rho)
        self.assertEqual(draws.dtype, jnp.int32)
        chex.assert_trees_all_close(draws.mean(axis=0), rho, atol=0.01)
        chex.assert_trees_all_close(
            draws.var(axis=0), fractional_part * (1 - fractional_part), atol=0.01
        )

    def test_integer_rho_is_deterministic(self):
        """An integer rho always gives exactly rho replicas."""
        keys = jax.random.split(self.next_key(), 100)
        draws = jax.vmap(lambda key: imc.optimal_replication(key, jnp.log(3.0)))(keys)
        chex.assert_trees_all_equal(draws, jnp.full(100, 3, dtype=jnp.int32))

    def test_invalid_weights_give_no_replica(self):
        """NaN and -inf log-weights give 0 replicas."""
        log_rho = jnp.array([jnp.nan, -jnp.inf])
        draws = imc.optimal_replication(self.next_key(), log_rho)
        chex.assert_trees_all_equal(draws, jnp.zeros(2, dtype=jnp.int32))

    def test_large_weights_are_clipped(self):
        """Counts stop at MAX_NUM_REPLICAS instead of overflowing int32."""
        draw = imc.optimal_replication(self.next_key(), jnp.array(1_000.0))
        self.assertEqual(int(draw), int(imc.MAX_NUM_REPLICAS))


class LogKappaTest(BlackJAXTest):
    """Tests for the choice of kappa from log-weights."""

    def test_expected_number_of_samples(self):
        """The chosen kappa gives the requested expected total."""
        log_weights = jax.random.normal(self.next_key(), (50,))
        log_kappa = imc.log_kappa_for_num_samples(log_weights, 200)
        expected_total = jnp.sum(jnp.exp(log_kappa + log_weights))
        chex.assert_trees_all_close(expected_total, 200.0, rtol=1e-5)


class IndependentSamplingTest(BlackJAXTest):
    """Tests for the vectorised interface with i.i.d. proposals."""

    def test_moments_and_number_of_samples(self):
        """With kappa from a pilot batch, the moments match the target."""
        pilot_keys = jax.random.split(self.next_key(), 100_000)
        pilot_positions = jax.vmap(proposal_distribution)(pilot_keys)
        pilot_log_weights = jax.vmap(target_logdensity)(pilot_positions) - jax.vmap(
            proposal_logdensity
        )(pilot_positions)
        log_kappa = imc.log_kappa_for_num_samples(pilot_log_weights, 100_000)

        positions, num_replicas = blackjax.imc.independent_sampling(
            self.next_key(),
            target_logdensity,
            proposal_distribution,
            proposal_logdensity,
            num_proposals=100_000,
            log_kappa=log_kappa,
        )
        total = jnp.sum(num_replicas)
        mean = jnp.sum(num_replicas * positions) / total
        variance = jnp.sum(num_replicas * (positions - mean) ** 2) / total

        chex.assert_trees_all_close(total / 100_000, 1.0, atol=0.03)
        chex.assert_trees_all_close(mean, 1.0, atol=0.02)
        chex.assert_trees_all_close(variance, 0.25, atol=0.02)

    def test_rejection_sampling_regime(self):
        """With kappa * pi / q <= 1, every proposal has 0 or 1 replica."""
        positions, num_replicas = blackjax.imc.independent_sampling(
            self.next_key(),
            target_logdensity,
            proposal_distribution,
            proposal_logdensity,
            num_proposals=100_000,
            log_kappa=jnp.log(1 / 5),
        )
        accepted = positions[num_replicas == 1]

        self.assertTrue(bool(jnp.all(num_replicas <= 1)))
        chex.assert_trees_all_close(accepted.size / 100_000, 1 / 5, atol=0.01)
        chex.assert_trees_all_close(accepted.mean(), 1.0, atol=0.02)
        chex.assert_trees_all_close(accepted.var(), 0.25, atol=0.02)

    def test_pytree_position(self):
        """Pytree positions keep their structure, with a leading proposal axis."""

        def logdensity(x):
            return -0.5 * (x["a"] ** 2 + jnp.sum(x["b"] ** 2))

        def distribution(rng_key):
            key_a, key_b = jax.random.split(rng_key)
            return {
                "a": 2.0 * jax.random.normal(key_a),
                "b": 2.0 * jax.random.normal(key_b, (3,)),
            }

        def distribution_logdensity(x):
            return -0.125 * (x["a"] ** 2 + jnp.sum(x["b"] ** 2))

        positions, num_replicas = blackjax.imc.independent_sampling(
            self.next_key(),
            logdensity,
            distribution,
            distribution_logdensity,
            num_proposals=10,
            log_kappa=0.0,
        )
        self.assertEqual(positions["a"].shape, (10,))
        self.assertEqual(positions["b"].shape, (10, 3))
        self.assertEqual(num_replicas.shape, (10,))


class InitTest(BlackJAXTest):
    """Tests for the initial state of the chain."""

    def test_initial_state(self):
        """The counter starts at 0 and the state wraps the instrumental state."""
        instrumental_algorithm = independent_algorithm(proposal_distribution)
        state = imc.init(0.5, instrumental_algorithm.init)

        self.assertEqual(int(state.num_remaining), 0)
        self.assertEqual(state.num_remaining.dtype, jnp.int32)
        chex.assert_trees_all_equal(
            state.instrumental_state, instrumental_algorithm.init(0.5)
        )
        chex.assert_trees_all_equal(state.position, state.instrumental_state.position)


class CorrectnessTest(BlackJAXTest):
    """Tests that the extended chain targets the right distribution."""

    def test_finite_state_space(self):
        """The chain recovers the target exactly on a finite state space."""
        target = jnp.array([0.1, 0.2, 0.3, 0.4])
        instrumental = jnp.array([0.4, 0.3, 0.2, 0.1])
        kappa = 1.5

        algorithm = blackjax.imc(
            lambda x: jnp.log(target[x]),
            lambda x: jnp.log(instrumental[x]),
            discrete_metropolis(instrumental),
            log_kappa=jnp.log(kappa),
        )
        num_steps = 200_000
        _, (positions, num_instrumental_steps) = run_inference_algorithm(
            self.next_key(),
            algorithm,
            num_steps,
            initial_position=0,
            transform=lambda state, info: (
                state.position,
                info.num_instrumental_steps,
            ),
        )

        frequencies = jnp.bincount(positions, length=4) / num_steps
        steps_per_sample = jnp.sum(num_instrumental_steps) / num_steps
        chex.assert_trees_all_close(frequencies, target, atol=0.01)
        chex.assert_trees_all_close(steps_per_sample, 1 / kappa, atol=0.02)

    def test_rejection_sampling_regime(self):
        """With kappa * pi / q <= 1, each step is a new draw from the target."""
        algorithm = blackjax.imc(
            target_logdensity,
            proposal_logdensity,
            independent_algorithm(proposal_distribution),
            log_kappa=jnp.log(1 / 5),
        )
        _, (positions, num_remaining) = run_inference_algorithm(
            self.next_key(),
            algorithm,
            20_000,
            initial_position=0.0,
            transform=lambda state, info: (state.position, state.num_remaining),
        )

        chex.assert_trees_all_equal(num_remaining, jnp.zeros_like(num_remaining))
        chex.assert_trees_all_close(positions.mean(), 1.0, atol=0.02)
        chex.assert_trees_all_close(positions.var(), 0.25, atol=0.02)

    def test_dropping_exhausted_transitions_is_exact(self):
        """Dropping exhausted transitions recovers the chain without a budget."""
        target = jnp.array([0.1, 0.2, 0.3, 0.4])
        instrumental = jnp.array([0.4, 0.3, 0.2, 0.1])

        algorithm = blackjax.imc(
            lambda x: jnp.log(target[x]),
            lambda x: jnp.log(instrumental[x]),
            discrete_metropolis(instrumental),
            log_kappa=jnp.log(0.3),
            max_num_steps=1,
        )
        _, (positions, is_exhausted) = run_inference_algorithm(
            self.next_key(),
            algorithm,
            400_000,
            initial_position=0,
            transform=lambda state, info: (state.position, info.is_exhausted),
        )
        kept_positions = positions[~is_exhausted]

        frequencies = jnp.bincount(kept_positions, length=4) / kept_positions.size
        self.assertGreater(float(is_exhausted.mean()), 0.5)
        chex.assert_trees_all_close(frequencies, target, atol=0.01)


class KernelTest(BlackJAXTest):
    """Tests for the mechanics of a transition and the API contracts."""

    def test_copies_are_returned_before_new_draws(self):
        """Remaining copies are returned without any instrumental step."""
        algorithm = blackjax.imc(
            target_logdensity,
            proposal_logdensity,
            independent_algorithm(proposal_distribution),
            log_kappa=0.0,
        )
        state = imc.IMCState(
            jnp.array(0.3),
            jnp.array(2, dtype=jnp.int32),
            IndependentState(jnp.array(0.3)),
        )
        initial_position = state.position
        for expected_remaining in [1, 0]:
            state, info = algorithm.step(self.next_key(), state)
            chex.assert_trees_all_equal(state.position, initial_position)
            self.assertEqual(int(state.num_remaining), expected_remaining)
            self.assertEqual(int(info.num_instrumental_steps), 0)

    def test_max_num_steps(self):
        """Without any replica, a transition stops after max_num_steps steps."""
        algorithm = blackjax.imc(
            target_logdensity,
            proposal_logdensity,
            independent_algorithm(proposal_distribution),
            log_kappa=-jnp.inf,
            max_num_steps=25,
        )
        state = algorithm.init(0.5)
        new_state, info = algorithm.step(self.next_key(), state)

        self.assertTrue(info.is_exhausted)
        self.assertEqual(int(info.num_instrumental_steps), 25)
        self.assertEqual(float(new_state.position), 0.5)
        self.assertEqual(int(new_state.num_remaining), 0)
        self.assertNotEqual(float(new_state.instrumental_state.position), 0.5)

    def test_jit_and_no_recompile(self):
        """The step function is traced once and reused across calls."""
        algorithm = blackjax.imc(
            target_logdensity,
            proposal_logdensity,
            blackjax.irmh(proposal_logdensity, proposal_distribution),
            log_kappa=0.0,
        )

        @chex.assert_max_traces(n=1)
        def step(rng_key, state):
            return algorithm.step(rng_key, state)

        jitted_step = jax.jit(step)
        state = algorithm.init(jnp.array(0.0))
        state, _ = jitted_step(self.next_key(), state)
        state, _ = jitted_step(self.next_key(), state)

    def test_pytree_position(self):
        """A step preserves the structure, shapes and dtypes of the state."""

        def logdensity(x):
            return -0.5 * (x["a"] ** 2 + jnp.sum(x["b"] ** 2))

        def instrumental_logdensity(x):
            return 0.5 * logdensity(x)

        random_walk = blackjax.additive_step_random_walk.normal_random_walk(
            instrumental_logdensity, sigma=jnp.ones(4)
        )
        algorithm = blackjax.imc(
            logdensity, instrumental_logdensity, random_walk, log_kappa=0.0
        )
        state = algorithm.init({"a": jnp.array(0.0), "b": jnp.zeros(3)})
        new_state, _ = algorithm.step(self.next_key(), state)
        chex.assert_trees_all_equal_shapes_and_dtypes(state, new_state)

    def test_vmap_over_chains(self):
        """Several chains can be run in parallel with jax.vmap."""
        algorithm = blackjax.imc(
            target_logdensity,
            proposal_logdensity,
            blackjax.irmh(proposal_logdensity, proposal_distribution),
            log_kappa=0.0,
        )
        num_chains = 8
        initial_positions = jnp.linspace(-1.0, 1.0, num_chains)
        states = jax.vmap(algorithm.init)(initial_positions)
        keys = jax.random.split(self.next_key(), num_chains)
        new_states, infos = jax.vmap(algorithm.step)(keys, states)

        self.assertEqual(new_states.position.shape, (num_chains,))
        self.assertEqual(infos.num_instrumental_steps.shape, (num_chains,))


if __name__ == "__main__":
    absltest.main()
