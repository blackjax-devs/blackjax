"""Tests for mhmc and dmhmc kernels.

Tests exercise the canonical short-name aliases (``blackjax.mhmc``,
``blackjax.dmhmc``) and verify backward-compatible aliases
(``blackjax.multinomial_hmc``, ``blackjax.dynamic_hmc``) are the same objects.
"""

import jax
import jax.numpy as jnp
from absl.testing import absltest

import blackjax
from blackjax.mcmc.dynamic_hmc import DynamicHMCState
from blackjax.mcmc.hmc import HMCInfo, HMCState, multinomial_hmc_proposal
from tests.fixtures import BlackJAXTest, std_normal_logdensity


class MHMCTest(BlackJAXTest):
    """Unit tests for blackjax.mhmc (multinomial HMC, fixed steps)."""

    def test_sampling_algorithm_interface(self):
        """The high-level API returns a SamplingAlgorithm with init/step."""
        sampler = blackjax.mhmc(
            std_normal_logdensity,
            step_size=0.1,
            inverse_mass_matrix=jnp.array([1.0]),
            num_integration_steps=10,
        )
        state = sampler.init(jnp.array(0.5))
        self.assertIsInstance(state, HMCState)

        new_state, info = jax.jit(sampler.step)(self.next_key(), state)
        self.assertIsInstance(new_state, HMCState)
        self.assertIsInstance(info, HMCInfo)

    def test_correct_sampling(self):
        """On a standard normal, the sampler should produce reasonable samples."""
        sampler = blackjax.mhmc(
            std_normal_logdensity,
            step_size=0.5,
            inverse_mass_matrix=jnp.array([1.0]),
            num_integration_steps=20,
        )
        state = sampler.init(jnp.array(0.0))
        step = jax.jit(sampler.step)

        states = []
        for i in range(500):
            key = jax.random.fold_in(self.next_key(), i)
            state, info = step(key, state)
            states.append(state.position)

        samples = jnp.stack(states)
        self.assertAlmostEqual(float(jnp.mean(samples)), 0.0, delta=0.3)
        self.assertAlmostEqual(float(jnp.std(samples)), 1.0, delta=0.3)

    def test_divergence_detection(self):
        """With a huge step size the sampler should flag divergences."""
        sampler = blackjax.mhmc(
            std_normal_logdensity,
            step_size=1000.0,
            inverse_mass_matrix=jnp.array([1.0]),
            num_integration_steps=100,
            divergence_threshold=100,
        )
        state = sampler.init(jnp.array(0.0))
        _, info = jax.jit(sampler.step)(self.next_key(), state)
        self.assertTrue(info.is_divergent)

    def test_acceptance_rate(self):
        """With a well-tuned step size the acceptance rate should be high."""
        sampler = blackjax.mhmc(
            std_normal_logdensity,
            step_size=0.1,
            inverse_mass_matrix=jnp.array([1.0]),
            num_integration_steps=10,
        )
        state = sampler.init(jnp.array(0.0))
        _, info = jax.jit(sampler.step)(self.next_key(), state)
        self.assertGreater(float(info.acceptance_rate), 0.5)

    def test_pytree_position(self):
        """The sampler should handle dict-structured positions."""
        sampler = blackjax.mhmc(
            std_normal_logdensity,
            step_size=0.1,
            inverse_mass_matrix=jnp.array([1.0, 1.0]),
            num_integration_steps=10,
        )
        state = sampler.init({"a": jnp.array(0.0), "b": jnp.array(1.0)})
        new_state, info = jax.jit(sampler.step)(self.next_key(), state)
        self.assertIn("a", new_state.position)
        self.assertIn("b", new_state.position)

    def test_build_kernel_with_build_proposal(self):
        """build_kernel with build_proposal=multinomial_hmc_proposal works."""
        kernel = blackjax.hmc.build_kernel(
            build_proposal=multinomial_hmc_proposal,
        )
        state = blackjax.hmc.init(jnp.array(0.0), std_normal_logdensity)

        new_state, info = jax.jit(kernel, static_argnums=(2,))(
            self.next_key(),
            state,
            std_normal_logdensity,
            0.1,
            jnp.array([1.0]),
            10,
        )
        self.assertIsInstance(new_state, HMCState)
        self.assertIsInstance(info, HMCInfo)
        self.assertTrue(info.is_accepted)

    def test_mhmc_matches_explicit_build_proposal(self):
        """blackjax.mhmc produces the same results as
        blackjax.hmc with build_proposal=multinomial_hmc_proposal.
        """
        kwargs = dict(
            step_size=0.1,
            inverse_mass_matrix=jnp.array([1.0]),
            num_integration_steps=10,
        )

        sampler_alias = blackjax.mhmc(std_normal_logdensity, **kwargs)
        sampler_direct = blackjax.hmc(
            std_normal_logdensity,
            build_proposal=multinomial_hmc_proposal,
            **kwargs,
        )

        state = blackjax.hmc.init(jnp.array(0.0), std_normal_logdensity)
        key = self.next_key()

        new_state_alias, info_alias = jax.jit(sampler_alias.step)(key, state)
        new_state_direct, info_direct = jax.jit(sampler_direct.step)(key, state)

        self.assertEqual(
            float(new_state_alias.logdensity),
            float(new_state_direct.logdensity),
        )
        self.assertEqual(
            float(info_alias.acceptance_rate),
            float(info_direct.acceptance_rate),
        )

    def test_backward_compat_alias(self):
        """blackjax.multinomial_hmc is the same object as blackjax.mhmc."""
        self.assertIs(blackjax.multinomial_hmc, blackjax.mhmc)


class DMHMCTest(BlackJAXTest):
    """Smoke tests for blackjax.dmhmc (dynamic steps + multinomial proposal)."""

    def test_alias_returns_dynamic_hmc_state(self):
        sampler = blackjax.dmhmc(
            std_normal_logdensity,
            step_size=0.1,
            inverse_mass_matrix=jnp.array([1.0]),
        )
        state = sampler.init(jnp.array(0.0), self.next_key())
        self.assertIsInstance(state, DynamicHMCState)
        new_state, info = jax.jit(sampler.step)(self.next_key(), state)
        self.assertIsInstance(new_state, DynamicHMCState)
        self.assertIsInstance(info, HMCInfo)

    def test_is_accepted_always_true(self):
        """Multinomial proposal has no M-H rejection step."""
        sampler = blackjax.dmhmc(
            std_normal_logdensity,
            step_size=0.1,
            inverse_mass_matrix=jnp.array([1.0]),
        )
        state = sampler.init(jnp.array(0.0), self.next_key())
        _, info = jax.jit(sampler.step)(self.next_key(), state)
        self.assertTrue(bool(info.is_accepted))

    def test_dmhmc_matches_explicit_build_proposal(self):
        """blackjax.dmhmc produces the same result as
        blackjax.dhmc(build_proposal=multinomial_hmc_proposal)."""
        kwargs = dict(step_size=0.1, inverse_mass_matrix=jnp.array([1.0]))
        sampler_alias = blackjax.dmhmc(std_normal_logdensity, **kwargs)
        sampler_explicit = blackjax.dhmc(
            std_normal_logdensity,
            build_proposal=multinomial_hmc_proposal,
            **kwargs,
        )
        init_key = self.next_key()
        state = sampler_alias.init(jnp.array(0.0), init_key)
        key = self.next_key()

        new_alias, info_alias = jax.jit(sampler_alias.step)(key, state)
        new_explicit, info_explicit = jax.jit(sampler_explicit.step)(key, state)

        self.assertEqual(float(new_alias.logdensity), float(new_explicit.logdensity))
        self.assertEqual(
            float(info_alias.acceptance_rate), float(info_explicit.acceptance_rate)
        )

    def test_backward_compat_alias(self):
        """blackjax.dynamic_hmc is the same object as blackjax.dhmc."""
        self.assertIs(blackjax.dynamic_hmc, blackjax.dhmc)


if __name__ == "__main__":
    absltest.main()
