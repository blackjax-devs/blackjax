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
"""Tests for the Laplace-HMC sampler (laplace_hmc)."""
import chex
import jax
import jax.numpy as jnp
import jax.scipy.stats as stats
import numpy as np
from absl.testing import absltest

import blackjax
from blackjax.mcmc.laplace_hmc import LaplaceHMCState, as_top_level_api, init
from blackjax.mcmc.laplace_marginal import laplace_marginal_factory
from blackjax.util import run_inference_algorithm
from tests.fixtures import BlackJAXTest

# ---------------------------------------------------------------------------
# Shared model: Gaussian-Gaussian (Laplace is exact)
#
#   phi   ~ N(0, 10²)                      [hyperparameter, scalar]
#   theta | phi ~ N(0, exp(phi)² * I_n)    [latent, n-vector]
#   y     | theta ~ N(theta, I_n)          [observations, n-vector]
# ---------------------------------------------------------------------------


def make_gaussian_model(y):
    def log_joint(theta, log_sigma):
        sigma = jnp.exp(log_sigma)
        log_prior_theta = stats.norm.logpdf(theta, 0.0, sigma).sum()
        log_lik = stats.norm.logpdf(y, theta, 1.0).sum()
        return log_prior_theta + log_lik

    def exact_log_marginal(log_sigma):
        sigma = jnp.exp(log_sigma)
        var_marg = sigma**2 + 1.0
        return stats.norm.logpdf(y, 0.0, jnp.sqrt(var_marg)).sum()

    return log_joint, exact_log_marginal


class TestLaplaceHMCState(BlackJAXTest):
    """Tests for LaplaceHMCState structure and init."""

    def setUp(self):
        super().setUp()
        self.n = 4
        self.y = jax.random.normal(self.next_key(), (self.n,))
        self.theta_init = jnp.zeros(self.n)
        log_joint, _ = make_gaussian_model(self.y)
        self.laplace = laplace_marginal_factory(log_joint, self.theta_init, maxiter=200)

    def test_init_returns_laplace_hmc_state(self):
        phi = jnp.array(0.0)
        state = init(phi, self.laplace)
        self.assertIsInstance(state, LaplaceHMCState)

    def test_init_state_fields(self):
        phi = jnp.array(0.0)
        state = init(phi, self.laplace)
        self.assertEqual(state.position.shape, ())
        self.assertTrue(jnp.isfinite(state.logdensity))
        self.assertTrue(jnp.isfinite(state.logdensity_grad))
        self.assertEqual(state.theta_star.shape, (self.n,))
        self.assertTrue(jnp.all(jnp.isfinite(state.theta_star)))

    def test_init_theta_star_matches_solve_theta(self):
        phi = jnp.array(0.5)
        state = init(phi, self.laplace)
        expected = self.laplace.solve_theta(phi)
        np.testing.assert_allclose(state.theta_star, expected, atol=1e-4)

    def test_init_logdensity_consistent(self):
        """logdensity in state must equal laplace(phi)[0]."""
        phi = jnp.array(0.0)
        state = init(phi, self.laplace)
        lp_ref, _ = self.laplace(phi)
        np.testing.assert_allclose(state.logdensity, lp_ref, atol=1e-5)


class TestLaplaceHMCKernel(BlackJAXTest):
    """Tests for build_kernel and as_top_level_api."""

    def setUp(self):
        super().setUp()
        self.n = 4
        self.y = jax.random.normal(self.next_key(), (self.n,))
        self.theta_init = jnp.zeros(self.n)
        self.log_joint, _ = make_gaussian_model(self.y)

        self.step_size = 0.1
        self.inv_mass = jnp.ones(1)
        self.num_steps = 3

        self.sampler = as_top_level_api(
            self.log_joint,
            self.theta_init,
            self.step_size,
            self.inv_mass,
            self.num_steps,
            maxiter=200,
        )
        phi_init = jnp.array(0.0)
        self.state = self.sampler.init(phi_init)

    def test_step_returns_laplace_hmc_state(self):
        new_state, _ = self.sampler.step(self.next_key(), self.state)
        self.assertIsInstance(new_state, LaplaceHMCState)

    def test_step_state_is_finite(self):
        new_state, _ = self.sampler.step(self.next_key(), self.state)
        self.assertTrue(jnp.isfinite(new_state.position))
        self.assertTrue(jnp.isfinite(new_state.logdensity))
        self.assertTrue(jnp.isfinite(new_state.logdensity_grad))
        self.assertTrue(jnp.all(jnp.isfinite(new_state.theta_star)))

    def test_theta_star_shape_preserved(self):
        new_state, _ = self.sampler.step(self.next_key(), self.state)
        self.assertEqual(new_state.theta_star.shape, (self.n,))

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

    def test_sample_theta_from_mcmc_states(self):
        """After running laplace_hmc, sample theta ~ N(theta_star, H^{-1}) for each phi."""
        laplace = laplace_marginal_factory(self.log_joint, self.theta_init, maxiter=200)

        n_samples = 30
        keys = jax.random.split(self.next_key(), n_samples)
        rng_keys = jax.random.split(self.next_key(), n_samples)

        def one_step(state, key):
            new_state, _ = self.sampler.step(key, state)
            return new_state, new_state

        _, states = jax.lax.scan(one_step, self.state, keys)

        theta_samples = jax.vmap(laplace.sample_theta)(
            rng_keys, states.position, states.theta_star
        )
        self.assertEqual(theta_samples.shape, (n_samples, self.n))
        self.assertTrue(jnp.all(jnp.isfinite(theta_samples)))

    def test_blackjax_top_level_api(self):
        """blackjax.laplace_hmc exposes .init and .step."""
        sampler = blackjax.laplace_hmc(
            self.log_joint,
            self.theta_init,
            self.step_size,
            self.inv_mass,
            self.num_steps,
            maxiter=200,
        )
        state = sampler.init(jnp.array(0.0))
        new_state, info = jax.jit(sampler.step)(self.next_key(), state)
        self.assertIsInstance(new_state, LaplaceHMCState)
        self.assertTrue(jnp.isfinite(info.acceptance_rate))


class TestLaplaceHMCSampling(BlackJAXTest):
    """End-to-end sampling test: posterior mean of phi should recover true value."""

    def test_posterior_mean_gaussian_model(self):
        # phi ~ N(0, 2²), theta | phi ~ N(0, exp(phi)² I), y | theta ~ N(theta, I)
        # True log_sigma = 0.0 (sigma = 1.0).  Posterior of phi is proper and
        # concentrated near 0 for large n.
        # Fixed y with sum(y²)/n ≈ 2, matching the marginal variance sigma²+1=2
        # at the true log_sigma=0.  This pins the posterior near 0 regardless of
        # the PRNG sequence used for sampling.
        n = 10
        y = jnp.array([1.2, 1.5, -1.3, -1.4, 1.6, -1.5, 0.9, -1.8, 1.4, -1.2])

        def log_joint(theta, log_sigma):
            log_prior_phi = stats.norm.logpdf(log_sigma, 0.0, 2.0)
            sigma = jnp.exp(log_sigma)
            log_prior_theta = stats.norm.logpdf(theta, 0.0, sigma).sum()
            log_lik = stats.norm.logpdf(y, theta, 1.0).sum()
            return log_prior_phi + log_prior_theta + log_lik

        sampler = as_top_level_api(
            log_joint,
            jnp.zeros(n),
            step_size=0.1,
            inverse_mass_matrix=jnp.ones(1),
            num_integration_steps=5,
            maxiter=200,
        )

        phi_init = jnp.array(0.0)
        initial_state = sampler.init(phi_init)

        warmup_state, _ = run_inference_algorithm(
            self.next_key(), sampler, 500, initial_state=initial_state
        )
        _, samples = run_inference_algorithm(
            self.next_key(),
            sampler,
            1000,
            initial_state=warmup_state,
            transform=lambda state, info: state.position,
        )

        # Posterior should be near 0 (true log_sigma).  N(0, 2) prior + n=10
        # observations at sigma=1 gives a well-defined posterior.
        posterior_mean = float(jnp.mean(samples))
        self.assertLess(abs(posterior_mean), 1.5)  # loose but meaningful


class TestLaplaceHMCFunnel(BlackJAXTest):
    """Neal's funnel: laplace_hmc vs NCP-NUTS baseline.

    Model (centered parameterisation, used by laplace_hmc):
        phi      ~ N(0, 3²)
        theta_i  ~ N(0, exp(phi)²)   i = 1 … n
        y_i      ~ N(theta_i, 1)

    The non-centered parameterisation (NCP) removes the funnel geometry and
    lets standard NUTS explore freely.  It serves as a reference posterior for
    phi because its geometry is benign.

    NCP (used by NUTS baseline):
        phi      ~ N(0, 3²)
        z_i      ~ N(0, 1)           (non-centred latent)
        theta_i   = z_i * exp(phi)
        y_i      ~ N(theta_i, 1)
    """

    def setUp(self):
        super().setUp()
        self.n = 5
        rng = self.next_key()
        phi_true = 0.5
        theta_true = jax.random.normal(rng, (self.n,)) * jnp.exp(phi_true)
        key_obs = self.next_key()
        self.y = theta_true + jax.random.normal(key_obs, (self.n,))

    def _run_laplace_hmc(self, n_warmup=500, n_samples=5000):
        y = self.y
        n = self.n

        def log_joint(theta, phi):
            return (
                stats.norm.logpdf(phi, 0.0, 3.0)
                + stats.norm.logpdf(theta, 0.0, jnp.exp(phi)).sum()
                + stats.norm.logpdf(y, theta, 1.0).sum()
            )

        laplace = laplace_marginal_factory(log_joint, jnp.zeros(n), maxiter=200)
        sampler = as_top_level_api(
            log_joint,
            jnp.zeros(n),
            step_size=0.3,
            inverse_mass_matrix=jnp.ones(1),
            num_integration_steps=5,
            maxiter=200,
        )
        initial_state = sampler.init(jnp.array(0.0))

        warmup_state, _ = run_inference_algorithm(
            self.next_key(), sampler, n_warmup, initial_state=initial_state
        )
        _, (phi_samples, theta_star_samples) = run_inference_algorithm(
            self.next_key(),
            sampler,
            n_samples,
            initial_state=warmup_state,
            transform=lambda state, info: (state.position, state.theta_star),
        )

        # Draw theta ~ N(theta_star, H^{-1}) for each phi sample.
        rng_keys = jax.random.split(self.next_key(), n_samples)
        theta_samples = jax.vmap(laplace.sample_theta)(
            rng_keys, phi_samples, theta_star_samples
        )
        return phi_samples, theta_samples  # (n_samples,), (n_samples, n)

    def _run_ncp_nuts(self, n_warmup=500, n_samples=5000):
        """Window-adapted NUTS on the NCP — the reference posterior for phi and theta."""
        y = self.y
        n = self.n

        def log_joint_ncp(flat):
            z = flat[:n]
            phi = flat[n]
            theta = z * jnp.exp(phi)
            return (
                stats.norm.logpdf(phi, 0.0, 3.0)
                + stats.norm.logpdf(z, 0.0, 1.0).sum()
                + stats.norm.logpdf(y, theta, 1.0).sum()
            )

        warmup = blackjax.window_adaptation(blackjax.nuts, log_joint_ncp)
        (warmup_state, params), _ = warmup.run(
            self.next_key(), jnp.zeros(n + 1), num_steps=n_warmup
        )
        nuts_algo = blackjax.nuts(log_joint_ncp, **params)

        _, (phi_samples, theta_samples) = run_inference_algorithm(
            self.next_key(),
            nuts_algo,
            n_samples,
            initial_state=warmup_state,
            transform=lambda state, info: (
                state.position[n],
                state.position[:n] * jnp.exp(state.position[n]),
            ),
        )
        return phi_samples, theta_samples  # (n_samples,), (n_samples, n)

    def test_posterior_matches_ncp_nuts(self):
        """laplace_hmc phi and theta posteriors must agree with NCP-NUTS reference.

        phi mean/std and theta mean/std are all checked from a single pair of
        runs to avoid paying the sampling cost multiple times.

        For theta, we compare the mean and std pooled across components — the
        model is exchangeable in theta, so the pooled statistics are the most
        powerful single-number summary.
        """
        phi_laplace, theta_laplace = self._run_laplace_hmc()
        phi_ncp, theta_ncp = self._run_ncp_nuts()

        # --- phi ---
        mean_phi_laplace, mean_phi_ncp = float(jnp.mean(phi_laplace)), float(
            jnp.mean(phi_ncp)
        )
        std_phi_laplace, std_phi_ncp = float(jnp.std(phi_laplace)), float(
            jnp.std(phi_ncp)
        )
        np.testing.assert_allclose(
            mean_phi_laplace,
            mean_phi_ncp,
            atol=0.1,
            err_msg="phi mean: laplace_hmc {:.3f} vs NCP-NUTS {:.3f}".format(
                mean_phi_laplace, mean_phi_ncp
            ),
        )
        # Allow up to 40% relative deviation — Laplace underestimates variance,
        # especially for small n.
        np.testing.assert_allclose(
            std_phi_laplace,
            std_phi_ncp,
            rtol=0.4,
            err_msg="phi std: laplace_hmc {:.3f} vs NCP-NUTS {:.3f}".format(
                std_phi_laplace, std_phi_ncp
            ),
        )

        # --- theta (pooled across components) ---
        mean_theta_laplace, mean_theta_ncp = float(jnp.mean(theta_laplace)), float(
            jnp.mean(theta_ncp)
        )
        std_theta_laplace, std_theta_ncp = float(jnp.std(theta_laplace)), float(
            jnp.std(theta_ncp)
        )
        np.testing.assert_allclose(
            mean_theta_laplace,
            mean_theta_ncp,
            atol=0.2,
            err_msg="theta mean: laplace_hmc {:.3f} vs NCP-NUTS {:.3f}".format(
                mean_theta_laplace, mean_theta_ncp
            ),
        )
        np.testing.assert_allclose(
            std_theta_laplace,
            std_theta_ncp,
            rtol=0.3,
            err_msg="theta std: laplace_hmc {:.3f} vs NCP-NUTS {:.3f}".format(
                std_theta_laplace, std_theta_ncp
            ),
        )


if __name__ == "__main__":
    absltest.main()
