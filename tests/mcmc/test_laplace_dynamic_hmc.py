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
"""Tests for laplace_dhmc and laplace_dmhmc."""
import chex
import jax
import jax.numpy as jnp
import jax.scipy.stats as stats
from absl.testing import absltest

import blackjax
from blackjax.mcmc.hmc import HMCInfo, multinomial_hmc_proposal
from blackjax.mcmc.laplace_dynamic_hmc import LaplaceDynamicHMCState
from blackjax.mcmc.laplace_marginal import laplace_marginal_factory
from tests.fixtures import BlackJAXTest


def make_gaussian_model(y):
    def log_joint(theta, log_sigma):
        sigma = jnp.exp(log_sigma)
        log_prior_theta = stats.norm.logpdf(theta, 0.0, sigma).sum()
        log_lik = stats.norm.logpdf(y, theta, 1.0).sum()
        return log_prior_theta + log_lik

    return log_joint


class TestLaplaceDynamicHMCState(BlackJAXTest):
    """State structure and init."""

    def setUp(self):
        super().setUp()
        self.n = 4
        self.y = jax.random.normal(self.next_key(), (self.n,))
        self.theta_init = jnp.zeros(self.n)
        self.log_joint = make_gaussian_model(self.y)
        self.laplace = laplace_marginal_factory(
            self.log_joint, self.theta_init, maxiter=200
        )

    def test_init_returns_laplace_dynamic_hmc_state(self):
        from blackjax.mcmc.laplace_dynamic_hmc import init

        phi = jnp.array(0.0)
        state = init(phi, self.laplace, self.next_key())
        self.assertIsInstance(state, LaplaceDynamicHMCState)

    def test_init_all_fields_finite(self):
        from blackjax.mcmc.laplace_dynamic_hmc import init

        phi = jnp.array(0.0)
        state = init(phi, self.laplace, self.next_key())
        self.assertTrue(jnp.isfinite(state.logdensity))
        self.assertTrue(jnp.isfinite(state.logdensity_grad))
        self.assertTrue(jnp.all(jnp.isfinite(state.theta_star)))
        self.assertEqual(state.theta_star.shape, (self.n,))
        self.assertIsNotNone(
            state.random_generator_arg
        )  # shape depends on PRNG key style

    def test_init_theta_star_consistent(self):
        from blackjax.mcmc.laplace_dynamic_hmc import init

        phi = jnp.array(0.5)
        state = init(phi, self.laplace, self.next_key())
        expected = self.laplace.solve_theta(phi)
        import numpy as np

        np.testing.assert_allclose(state.theta_star, expected, atol=1e-4)


class TestLaplaceDHMCKernel(BlackJAXTest):
    """Kernel and top-level API tests for laplace_dhmc."""

    def setUp(self):
        super().setUp()
        self.n = 4
        self.y = jax.random.normal(self.next_key(), (self.n,))
        self.theta_init = jnp.zeros(self.n)
        self.log_joint = make_gaussian_model(self.y)
        self.kwargs = dict(
            step_size=0.1,
            inverse_mass_matrix=jnp.ones(1),
            maxiter=200,
        )
        self.sampler = blackjax.laplace_dhmc(
            self.log_joint, self.theta_init, **self.kwargs
        )
        self.state = self.sampler.init(jnp.array(0.0), self.next_key())

    def test_step_returns_laplace_dynamic_hmc_state(self):
        new_state, _ = self.sampler.step(self.next_key(), self.state)
        self.assertIsInstance(new_state, LaplaceDynamicHMCState)

    def test_step_returns_hmc_info(self):
        _, info = self.sampler.step(self.next_key(), self.state)
        self.assertIsInstance(info, HMCInfo)

    def test_step_all_fields_finite(self):
        new_state, _ = self.sampler.step(self.next_key(), self.state)
        self.assertTrue(jnp.isfinite(new_state.logdensity))
        self.assertTrue(jnp.isfinite(new_state.logdensity_grad))
        self.assertTrue(jnp.all(jnp.isfinite(new_state.theta_star)))

    def test_random_generator_arg_advances(self):
        """random_generator_arg must change between steps."""
        new_state, _ = self.sampler.step(self.next_key(), self.state)
        self.assertFalse(
            jnp.array_equal(
                new_state.random_generator_arg, self.state.random_generator_arg
            )
        )

    def test_jit_compatible(self):
        step = jax.jit(self.sampler.step)
        new_state, info = step(self.next_key(), self.state)
        self.assertTrue(jnp.isfinite(new_state.logdensity))
        self.assertTrue(jnp.isfinite(info.acceptance_rate))

    @chex.assert_max_traces(n=2)
    def test_no_excess_retracing(self):
        step = jax.jit(self.sampler.step)
        for _ in range(3):
            self.state, _ = step(self.next_key(), self.state)

    def test_blackjax_top_level_alias(self):
        """blackjax.laplace_dhmc exposes .init and .step."""
        sampler = blackjax.laplace_dhmc(self.log_joint, self.theta_init, **self.kwargs)
        state = sampler.init(jnp.array(0.0), self.next_key())
        new_state, info = jax.jit(sampler.step)(self.next_key(), state)
        self.assertIsInstance(new_state, LaplaceDynamicHMCState)
        self.assertTrue(jnp.isfinite(info.acceptance_rate))


class TestLaplaceDMHMC(BlackJAXTest):
    """Smoke tests for blackjax.laplace_dmhmc (multinomial variant)."""

    def setUp(self):
        super().setUp()
        self.n = 4
        self.y = jax.random.normal(self.next_key(), (self.n,))
        self.theta_init = jnp.zeros(self.n)
        self.log_joint = make_gaussian_model(self.y)
        self.kwargs = dict(
            step_size=0.1,
            inverse_mass_matrix=jnp.ones(1),
            maxiter=200,
        )

    def test_alias_returns_laplace_dynamic_hmc_state(self):
        sampler = blackjax.laplace_dmhmc(self.log_joint, self.theta_init, **self.kwargs)
        state = sampler.init(jnp.array(0.0), self.next_key())
        self.assertIsInstance(state, LaplaceDynamicHMCState)
        new_state, _ = jax.jit(sampler.step)(self.next_key(), state)
        self.assertIsInstance(new_state, LaplaceDynamicHMCState)

    def test_is_accepted_always_true(self):
        """Multinomial proposal has no M-H rejection step."""
        sampler = blackjax.laplace_dmhmc(self.log_joint, self.theta_init, **self.kwargs)
        state = sampler.init(jnp.array(0.0), self.next_key())
        _, info = jax.jit(sampler.step)(self.next_key(), state)
        self.assertTrue(bool(info.is_accepted))

    def test_alias_matches_explicit_build_proposal(self):
        """laplace_dmhmc produces the same result as
        laplace_dhmc(build_proposal=multinomial_hmc_proposal)."""
        sampler_alias = blackjax.laplace_dmhmc(
            self.log_joint, self.theta_init, **self.kwargs
        )
        sampler_explicit = blackjax.laplace_dhmc(
            self.log_joint,
            self.theta_init,
            build_proposal=multinomial_hmc_proposal,
            **self.kwargs,
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


if __name__ == "__main__":
    absltest.main()
