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
import chex
import jax
import jax.numpy as jnp
import jax.scipy.stats as stats
import numpy as np
import scipy.integrate
from absl.testing import absltest

from blackjax.mcmc.laplace_marginal import LaplaceMarginal, laplace_marginal_factory
from tests.fixtures import BlackJAXTest

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

    # --- log_marginal value ---------------------------------------------------

    def test_value_matches_exact_gaussian(self):
        # Laplace is exact for Gaussian-Gaussian models.
        for log_sigma_val in [-1.0, 0.0, 0.5, 1.0]:
            phi = jnp.array(log_sigma_val)
            (approx, _), _ = jax.value_and_grad(self.laplace, has_aux=True)(phi)
            exact = self.exact_log_marginal(phi)
            np.testing.assert_allclose(approx, exact, rtol=1e-4, atol=1e-4)

    def test_returns_theta_star_as_aux(self):
        phi = jnp.array(0.0)
        (lp, theta_star), _ = jax.value_and_grad(self.laplace, has_aux=True)(phi)
        self.assertEqual(theta_star.shape, (self.n,))
        # theta_star should match solve_theta output
        expected_mode = self.laplace.solve_theta(phi)
        np.testing.assert_allclose(theta_star, expected_mode, atol=1e-4)

    # --- gradient correctness -------------------------------------------------

    def test_gradient_matches_exact_gaussian(self):
        # Exact gradient of the Gaussian-Gaussian log-marginal.
        for log_sigma_val in [-0.5, 0.0, 0.5]:
            phi = jnp.array(log_sigma_val)
            (_, _), approx_grad = jax.value_and_grad(self.laplace, has_aux=True)(phi)
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
        (_, _), approx_grad = jax.value_and_grad(laplace_2d, has_aux=True)(phi0)

        eps = 1e-2  # cbrt(machine_eps) is optimal for float32 central FD
        fd_grad = jnp.array(
            [
                (
                    laplace_2d(phi0.at[0].set(phi0[0] + eps))[0]
                    - laplace_2d(phi0.at[0].set(phi0[0] - eps))[0]
                )
                / (2 * eps),
                (
                    laplace_2d(phi0.at[1].set(phi0[1] + eps))[0]
                    - laplace_2d(phi0.at[1].set(phi0[1] - eps))[0]
                )
                / (2 * eps),
            ]
        )
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

        laplace = laplace_marginal_factory(log_joint, jnp.array([0.0]), maxiter=200)
        phi = jnp.array([2.0])
        (lp, theta_star), grad = jax.value_and_grad(laplace, has_aux=True)(phi)

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
        (lp, theta_star), grad = jax.value_and_grad(laplace_tree, has_aux=True)(phi)
        self.assertTrue(jnp.isfinite(lp))
        self.assertTrue(jnp.isfinite(grad))
        self.assertIn("a", theta_star)
        self.assertIn("b", theta_star)

    # --- sample_theta: draw from Laplace approximate posterior ---------------

    def test_sample_theta_shape(self):
        phi = jnp.array(0.0)
        theta_star = self.laplace.solve_theta(phi)
        sample = self.laplace.sample_theta(self.next_key(), phi, theta_star)
        self.assertEqual(sample.shape, (self.n,))

    def test_sample_theta_mean_and_variance_match_exact_posterior(self):
        """Empirical mean ≈ theta_star and variance ≈ sigma^2/(sigma^2+1).

        For the Gaussian-Gaussian model the Laplace posterior is exact:
            theta | phi, y ~ N(mu_post, Sigma_post)
            mu_post    = sigma^2 / (sigma^2 + 1) * y
            Sigma_post = sigma^2 / (sigma^2 + 1) * I
        """
        phi = jnp.array(0.0)  # log_sigma = 0  =>  sigma = 1
        sigma = jnp.exp(phi)
        exact_mean = (sigma**2 / (sigma**2 + 1.0)) * self.y
        exact_var = sigma**2 / (sigma**2 + 1.0)

        theta_star = self.laplace.solve_theta(phi)
        keys = jax.random.split(self.next_key(), 5000)
        samples = jax.vmap(lambda k: self.laplace.sample_theta(k, phi, theta_star))(
            keys
        )

        np.testing.assert_allclose(jnp.mean(samples, axis=0), exact_mean, atol=0.05)
        np.testing.assert_allclose(jnp.var(samples, axis=0), exact_var, rtol=0.05)


class TestLaplaceAdjointAnalytical(BlackJAXTest):
    """Validates JAX AD gradient against the adjoint formula from Algorithm 2 of
    Margossian et al. (2020), using a Poisson-LogNormal model where the derivation
    can be carried out in closed form.

    Model:  K(phi) = exp(phi) * I
        phi                              hyperparameter (scalar)
        theta | phi ~ N(0, exp(phi)*I)   latent, n-vector
        y_i | theta ~ Poisson(exp(theta_i))

    Because K is a scaled identity and the Poisson observations are conditionally
    independent, every matrix in the adjoint is diagonal.

    Analytical gradient (derived from the total derivative of log p̂):

        d/dphi log p̂ = partial_phi - 1/2 * d(log det H)/dphi

    where:
        k            = exp(phi)                             K = k * I
        a            = K^{-1} theta* = theta* / k
        W            = diag(exp(theta*))                    Poisson W_ii = exp(theta_i*)
        H            = K^{-1} + W = diag(1/k + W_i)         neg-Hessian of log joint
        d3_i         = -exp(theta_i*)                       Poisson third derivative
        partial_phi  = -n/2 + ||theta*||^2 / (2k)           direct phi-deriv of log prior
        dtheta*/dphi = H^{-1} a   (IFT: positive sign)      implicit function theorem
        dH/dphi      = -1/k + W * dtheta*/dphi              chain rule through H
        d_logdet     = tr(H^{-1} dH/dphi)                   = sum(H_inv * dH_diag)

    The s2 term from Algorithm 2 enters through d_logdet:
        s2 = -1/2 * diag(H^{-1}) * d3 = 1/2 * H_inv * W
        -1/2 * sum(W * H_inv^2 * a) = sum(s2 * H_inv * a)

    This confirms that JAX's custom_root + slogdet approach correctly captures the
    third-order information that Algorithm 2's manual adjoint was designed for.
    """

    def setUp(self):
        super().setUp()
        self.n = 6
        rng = self.next_key()
        # Upper bound capped at 5 to keep Poisson rates moderate: large y pushes
        # exp(theta*) high, ill-conditioning the Hessian in float32 and causing
        # gradient errors above atol=1e-2.
        self.y = jax.random.randint(rng, (self.n,), 1, 5).astype(jnp.float32)
        self.theta_init = jnp.zeros(self.n)

        def log_joint(theta, phi):
            # K(phi) = exp(phi) * I  =>  theta ~ N(0, exp(phi)*I)
            log_prior = stats.norm.logpdf(theta, 0.0, jnp.exp(0.5 * phi)).sum()
            log_lik = jax.scipy.stats.poisson.logpmf(self.y, jnp.exp(theta)).sum()
            return log_prior + log_lik

        self.log_joint = log_joint
        self.laplace = laplace_marginal_factory(log_joint, self.theta_init, maxiter=500)

    def _analytical_gradient(self, phi, theta_star):
        """Closed-form adjoint gradient for the Poisson-LogNormal LGM.

        Implements the total derivative d/dphi log p̂ accounting for the implicit
        dependence theta*(phi) via the IFT.
        """
        k = jnp.exp(phi)  # K = k * I
        n = theta_star.shape[0]

        a = theta_star / k  # K^{-1} theta*
        W = jnp.exp(theta_star)  # Poisson: W_ii = exp(theta_i*)
        H_inv = 1.0 / (1.0 / k + W)  # (K^{-1} + W)^{-1} diagonal

        # IFT: differentiating the optimality condition wrt phi gives
        #   -(1/k + W_i) * dtheta_i*/dphi + theta_i*/k = 0
        # => dtheta_i*/dphi = a_i / H_ii = H_inv_i * a_i  (positive)
        dtheta_dphi = H_inv * a

        # Total derivative of H = K^{-1} + W wrt phi:
        #   dK^{-1}/dphi = -1/k * I
        #   dW/dphi = diag(W) * dtheta*/dphi  [chain rule; d3_i = -W_i enters here]
        dH_diag = -1.0 / k + W * dtheta_dphi  # = -1/k + W * H_inv * a

        # Direct phi-derivative of log p(theta*|phi) at fixed theta*:
        #   log p = -n/2 * log(2pi*k) - ||theta*||^2/(2k)
        partial_phi = -n / 2.0 + jnp.sum(theta_star**2) / (2.0 * k)

        # d/dphi log det H = tr(H^{-1} dH/dphi)
        d_logdet = jnp.sum(H_inv * dH_diag)

        return partial_phi - 0.5 * d_logdet

    def test_jax_grad_matches_analytical_adjoint(self):
        """JAX AD gradient (custom_root + slogdet VJP) matches closed-form adjoint.

        Tests multiple phi values to confirm the match is not coincidental.
        """
        for phi_val in [-1.0, 0.0, 0.5, 1.0]:
            phi = jnp.array(phi_val)
            (_, theta_star), jax_grad = jax.value_and_grad(self.laplace, has_aux=True)(
                phi
            )
            analytical = self._analytical_gradient(phi, theta_star)
            # Tolerance is set by float32 L-BFGS convergence, not by the formula.
            np.testing.assert_allclose(
                float(jax_grad),
                float(analytical),
                rtol=1e-2,
                atol=1e-2,
                err_msg=f"phi={phi_val}",
            )


class TestLaplacePoissonQuadrature(BlackJAXTest):
    """Gold-standard value test: log-marginal matches 1D numerical quadrature.

    Mirrors Stan's ``aki_ex_test.cpp``, which validates the Laplace marginal
    against ``integrate_1d`` for a scalar-theta Poisson-LogNormal model.
    Scalar theta makes the exact marginal tractable by quadrature, giving an
    model-agnostic ground truth that requires no closed-form derivation.

    Model:
        phi                ~ (hyperparameter under test)
        theta | phi        ~ N(0, exp(phi/2))   [scalar latent, variance = exp(phi)]
        y_i   | theta      ~ Poisson(exp(theta))
    """

    def setUp(self):
        super().setUp()
        self.y = jnp.array([3.0, 1.0, 4.0, 1.0, 5.0])

        def log_joint(theta, phi):
            # theta scalar; variance of prior = exp(phi), std = exp(phi/2)
            log_prior = stats.norm.logpdf(theta, 0.0, jnp.exp(0.5 * phi))
            log_lik = jax.scipy.stats.poisson.logpmf(self.y, jnp.exp(theta)).sum()
            return log_prior + log_lik

        self.log_joint = log_joint
        self.laplace = laplace_marginal_factory(log_joint, jnp.array(0.0), maxiter=500)

    def _exact_log_marginal(self, phi_val):
        """1D quadrature over theta to get the exact log p(y | phi)."""

        def integrand(theta):
            log_prior = float(
                stats.norm.logpdf(jnp.array(theta), 0.0, np.exp(0.5 * phi_val))
            )
            log_lik = float(jax.scipy.stats.poisson.logpmf(self.y, np.exp(theta)).sum())
            return np.exp(log_prior + log_lik)

        result, _ = scipy.integrate.quad(integrand, -15.0, 15.0, limit=200)
        return np.log(result)

    def test_log_marginal_value_matches_quadrature(self):
        """Laplace log-marginal matches 1D quadrature within atol=0.1 nats.

        atol=0.1 matches the ~5% relative tolerance used in Stan's aki_ex_test.
        The Laplace approximation is accurate here because the posterior
        p(theta | phi, y) is unimodal and well-concentrated for these y values.
        """
        for phi_val in [-1.0, 0.0, 0.5, 1.0]:
            phi = jnp.array(phi_val)
            approx, _ = self.laplace(phi)
            exact = self._exact_log_marginal(phi_val)
            np.testing.assert_allclose(
                float(approx),
                exact,
                atol=0.1,
                err_msg="phi={}: Laplace={:.4f}, quadrature={:.4f}".format(
                    phi_val, float(approx), exact
                ),
            )


if __name__ == "__main__":
    absltest.main()
