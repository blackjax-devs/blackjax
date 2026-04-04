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
"""Tests for the Laplace-approximated marginal log-density."""
import jax
import jax.numpy as jnp
import jax.scipy.stats as stats
import numpy as np
from absl.testing import absltest

import chex
from tests.fixtures import BlackJAXTest
from blackjax.mcmc.laplace_marginal import LaplaceMarginal, laplace_marginal_factory


# ---------------------------------------------------------------------------
# Shared model: Gaussian-Gaussian (Laplace is exact)
#
#   phi   ~ N(0, 10²)                         [hyperparameter, scalar]
#   theta | phi ~ N(0, exp(phi)² * I_n)        [latent, n-vector]
#   y     | theta ~ N(theta, I_n)              [observations, n-vector]
#
# Exact log-marginal (up to constant in phi):
#   log p(y | phi) = sum_i N(y_i; 0, exp(2*phi) + 1)
#   grad_phi log p(y | phi) = exp(2*phi) * (||y||² / (exp(2*phi)+1)² - n/(exp(2*phi)+1))
# ---------------------------------------------------------------------------

def make_gaussian_model(y):
    """Return log_joint for the Gaussian-Gaussian model."""

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


class TestLaplaceMarginalFactory(BlackJAXTest):
    """Tests for laplace_marginal_factory and LaplaceMarginal."""

    def setUp(self):
        super().setUp()
        self.n = 4
        self.y = jax.random.normal(self.next_key(), (self.n,))
        self.theta_init = jnp.zeros(self.n)
        self.log_joint, self.exact_log_marginal = make_gaussian_model(self.y)
        self.laplace = laplace_marginal_factory(
            self.log_joint, self.theta_init, maxiter=200
        )

    # --- factory output -------------------------------------------------------

    def test_returns_laplace_marginal_instance(self):
        self.assertIsInstance(self.laplace, LaplaceMarginal)

    def test_has_required_callables(self):
        self.assertTrue(callable(self.laplace.solve_theta))
        self.assertTrue(callable(self.laplace.get_theta_star))
        self.assertTrue(callable(self.laplace.log_marginal))

    # --- solve_theta: mode finding in isolation -------------------------------

    def test_solve_theta_finds_correct_mode(self):
        # For log_joint(theta, phi) with Gaussian prior N(0, sigma^2) and
        # Gaussian likelihood N(theta, 1), the mode is:
        #   theta* = sigma^2 / (sigma^2 + 1) * y
        log_sigma = jnp.array(0.0)
        sigma = jnp.exp(log_sigma)
        expected_mode = (sigma**2 / (sigma**2 + 1.0)) * self.y

        theta_star = self.laplace.solve_theta(log_sigma)
        np.testing.assert_allclose(theta_star, expected_mode, atol=1e-4)

    def test_solve_theta_warm_starts(self):
        # Fewer L-BFGS iterations should be needed from a warm start.
        log_sigma = jnp.array(0.5)
        cold_result = laplace_marginal_factory(
            self.log_joint, self.theta_init, maxiter=3
        ).solve_theta(log_sigma)
        warm_result = laplace_marginal_factory(
            self.log_joint, self.theta_init, maxiter=3
        ).solve_theta(log_sigma, theta_prev=self.laplace.solve_theta(log_sigma))

        # Warm start from the true mode should already satisfy the optimality
        # condition; cold start from zeros needs iterations to get there.
        log_sigma_vec = jnp.array([0.5])
        grad_at_cold = jax.grad(self.log_joint, argnums=0)(cold_result, log_sigma_vec[0])
        grad_at_warm = jax.grad(self.log_joint, argnums=0)(warm_result, log_sigma_vec[0])
        self.assertLess(
            float(jnp.linalg.norm(grad_at_warm)),
            float(jnp.linalg.norm(grad_at_cold)) + 1e-4,
        )

    # --- log_marginal value ---------------------------------------------------

    def test_value_matches_exact_gaussian(self):
        # Laplace is exact for Gaussian-Gaussian models.
        for log_sigma_val in [-1.0, 0.0, 0.5, 1.0]:
            phi = jnp.array(log_sigma_val)
            (approx, _), _ = jax.value_and_grad(
                self.laplace, has_aux=True
            )(phi)
            exact = self.exact_log_marginal(phi)
            np.testing.assert_allclose(approx, exact, rtol=1e-4, atol=1e-4)

    def test_returns_theta_star_as_aux(self):
        phi = jnp.array(0.0)
        (lp, theta_star), _ = jax.value_and_grad(
            self.laplace, has_aux=True
        )(phi)
        self.assertEqual(theta_star.shape, (self.n,))
        # theta_star should match solve_theta output
        expected_mode = self.laplace.solve_theta(phi)
        np.testing.assert_allclose(theta_star, expected_mode, atol=1e-4)

    # --- gradient correctness -------------------------------------------------

    def test_gradient_matches_exact_gaussian(self):
        # Exact gradient of the Gaussian-Gaussian log-marginal.
        for log_sigma_val in [-0.5, 0.0, 0.5]:
            phi = jnp.array(log_sigma_val)
            (_, _), approx_grad = jax.value_and_grad(
                self.laplace, has_aux=True
            )(phi)
            exact_grad = jax.grad(self.exact_log_marginal)(phi)
            np.testing.assert_allclose(approx_grad, exact_grad, rtol=1e-3, atol=1e-3)

    def test_gradient_matches_finite_differences(self):
        # 2D phi: (log_sigma, mu_shift) with non-trivial joint.
        def log_joint_2d(theta, phi):
            sigma = jnp.exp(phi[0])
            mu = phi[1]
            log_prior = stats.norm.logpdf(theta, mu, sigma).sum()
            log_lik = stats.norm.logpdf(self.y, theta, 1.0).sum()
            return log_prior + log_lik

        laplace_2d = laplace_marginal_factory(
            log_joint_2d, self.theta_init, maxiter=200
        )
        phi0 = jnp.array([0.0, 0.5])
        (_, _), approx_grad = jax.value_and_grad(
            laplace_2d, has_aux=True
        )(phi0)

        eps = 1e-2  # cbrt(machine_eps) is optimal for float32 central FD
        fd_grad = jnp.array([
            (laplace_2d(phi0.at[0].set(phi0[0] + eps))[0] -
             laplace_2d(phi0.at[0].set(phi0[0] - eps))[0]) / (2 * eps),
            (laplace_2d(phi0.at[1].set(phi0[1] + eps))[0] -
             laplace_2d(phi0.at[1].set(phi0[1] - eps))[0]) / (2 * eps),
        ])
        np.testing.assert_allclose(approx_grad, fd_grad, rtol=0.02, atol=0.02)

    def test_known_model_mode_and_gradient(self):
        # Exact closed-form test from the model:
        #   phi   ~ N(0, 10²)
        #   theta | phi ~ N(phi, 1)
        #   y=1.0 | theta ~ N(theta, 1)
        # => theta*(phi) = (phi + 1) / 2
        # => d/dphi log p̂ = -0.5*(phi - 1) - phi/100
        def log_joint(theta, phi):
            log_prior_phi = -0.5 * jnp.sum((phi / 10.0) ** 2)
            log_prior_theta = -0.5 * jnp.sum((theta - phi) ** 2)
            log_lik = -0.5 * jnp.sum((1.0 - theta) ** 2)
            return log_prior_phi + log_prior_theta + log_lik

        laplace = laplace_marginal_factory(
            log_joint, jnp.array([0.0]), maxiter=200
        )
        phi = jnp.array([2.0])
        (lp, theta_star), grad = jax.value_and_grad(
            laplace, has_aux=True
        )(phi)

        np.testing.assert_allclose(theta_star, jnp.array([1.5]), atol=1e-3)
        expected_grad = -0.5 * (phi - 1.0) - phi / 100.0
        np.testing.assert_allclose(grad, expected_grad, atol=1e-3)

    # --- JAX compatibility ----------------------------------------------------

    def test_jit_value_and_grad(self):
        phi = jnp.array(0.0)
        (lp, theta_star), grad = jax.jit(
            jax.value_and_grad(self.laplace, has_aux=True)
        )(phi)
        self.assertTrue(jnp.isfinite(lp))
        self.assertTrue(jnp.isfinite(grad))
        self.assertTrue(jnp.all(jnp.isfinite(theta_star)))

    @chex.assert_max_traces(n=1)
    def test_no_excess_retracing(self):
        jit_fn = jax.jit(self.laplace)
        for log_sigma_val in [0.0, 0.5, 1.0]:
            jit_fn(jnp.array(log_sigma_val))

    def test_pytree_theta(self):
        n_half = self.n // 2

        def log_joint_tree(theta_tree, log_sigma):
            theta = jnp.concatenate([theta_tree["a"], theta_tree["b"]])
            return self.log_joint(theta, log_sigma)

        theta_init_tree = {"a": jnp.zeros(n_half), "b": jnp.zeros(n_half)}
        laplace_tree = laplace_marginal_factory(
            log_joint_tree, theta_init_tree, maxiter=200
        )
        phi = jnp.array(0.0)
        (lp, theta_star), grad = jax.value_and_grad(
            laplace_tree, has_aux=True
        )(phi)
        self.assertTrue(jnp.isfinite(lp))
        self.assertTrue(jnp.isfinite(grad))
        self.assertIn("a", theta_star)
        self.assertIn("b", theta_star)


if __name__ == "__main__":
    absltest.main()
