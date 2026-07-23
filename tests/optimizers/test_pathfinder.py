"""Test the pathfinder algorithm."""
import functools

import chex
import jax
import jax.numpy as jnp
import jax.scipy.stats as stats
import numpy as np
from absl.testing import absltest, parameterized

import blackjax
from blackjax.optimizers.lbfgs import bfgs_sample, lbfgs_inverse_hessian_factors


class PathfinderTest(chex.TestCase):
    def setUp(self):
        super().setUp()
        self.key = jax.random.key(1)

    @chex.all_variants(without_device=False, with_pmap=False)
    @parameterized.parameters(
        [(1,), (2,)],
    )
    def test_recover_posterior(self, ndim):
        """Test if pathfinder is able to estimate well enough the posterior of a
        normal-normal conjugate model"""

        def logp_posterior_conjugate_normal_model(
            x, observed, prior_mu, prior_prec, true_prec
        ):
            n = observed.shape[0]
            posterior_cov = jnp.linalg.inv(prior_prec + n * true_prec)
            posterior_mu = (
                posterior_cov
                @ (
                    prior_prec @ prior_mu[:, None]
                    + n * true_prec @ observed.mean(0)[:, None]
                )
            )[:, 0]
            return stats.multivariate_normal.logpdf(x, posterior_mu, posterior_cov)

        def logp_unnormalized_posterior(x, observed, prior_mu, prior_prec, true_cov):
            logp = 0.0
            logp += stats.multivariate_normal.logpdf(x, prior_mu, prior_prec)
            logp += stats.multivariate_normal.logpdf(observed, x, true_cov).sum()
            return logp

        rng_key_chol, rng_key_observed, rng_key_pathfinder = jax.random.split(
            self.key, 3
        )

        L = jnp.tril(jax.random.normal(rng_key_chol, (ndim, ndim)))
        true_mu = jnp.arange(ndim)
        true_cov = L @ L.T
        true_prec = jnp.linalg.pinv(true_cov)

        prior_mu = jnp.zeros(ndim)
        prior_prec = jnp.eye(ndim)

        observed = jax.random.multivariate_normal(
            rng_key_observed, true_mu, true_cov, shape=(10_000,)
        )

        logp_model = functools.partial(
            logp_unnormalized_posterior,
            observed=observed,
            prior_mu=prior_mu,
            prior_prec=prior_prec,
            true_cov=true_cov,
        )

        x0 = jnp.ones(ndim)
        pathfinder = blackjax.pathfinder(logp_model)
        out, _ = self.variant(pathfinder.init)(rng_key_pathfinder, x0)

        sim_p, log_p = bfgs_sample(
            rng_key_pathfinder,
            10_000,
            out.position,
            out.grad_position,
            out.alpha,
            out.beta,
            out.gamma,
        )

        log_q = logp_posterior_conjugate_normal_model(
            sim_p, observed, prior_mu, prior_prec, true_prec
        )

        kl = (log_p - log_q).mean()
        # TODO(junpenglao): Make this test more robust.
        self.assertAlmostEqual(kl, 0.0, delta=2.5)

    def test_bfgs_sample_logdet_finite_large_n(self):
        """Regression pin for the -inf logdet overflow (issue #1007).

        The old formula ``jnp.log(jnp.prod(alpha))`` overflows to inf for large N
        (``2.0 ** 3000`` exceeds any IEEE 754 float range).  The PR replaces it with
        ``jnp.sum(jnp.log(alpha))`` which is numerically stable.

        Setting beta=0 / gamma=0 isolates the logdet term so only the
        sum(log(alpha)) path is exercised.  Asserts the returned logdensity is finite.
        """
        N = 3000
        alpha = 2.0 * jnp.ones(N)
        # beta=0 / gamma=0: only the logdet = sum(log(alpha)) term is non-trivial.
        beta = jnp.zeros((N, 2))
        gamma = jnp.zeros((2, 2))
        position = jnp.zeros(N)
        grad_position = jnp.zeros(N)

        _phi, logq = bfgs_sample(
            jax.random.key(0), 1, position, grad_position, alpha, beta, gamma
        )
        self.assertTrue(jnp.all(jnp.isfinite(logq)))

    def test_bfgs_sample_mu_reassociation(self):
        """Regression pin for the mu re-association (issue #1007).

        The PR rewrites the O(N²) form:
            position + diag(alpha) @ grad + beta @ gamma @ beta.T @ grad
        as the memory-efficient factored form:
            position + alpha * grad + beta @ (gamma @ (beta.T @ grad))

        Exercises the bfgs_sample code path by reproducing the internal noise
        tensor with the same key, subtracting it from the samples to recover the
        deterministic mu, then comparing to the dense reference.
        Z = S satisfies the curvature condition s^T z = ||s||^2 > 0, guaranteeing
        valid L-BFGS inverse-Hessian factors and a positive-definite Cholesky target.
        """
        with jax.enable_x64():
            N, J = 50, 5

            k0, k1, k2, k3, k4, k5 = jax.random.split(jax.random.key(42), 6)
            alpha = jnp.abs(jax.random.normal(k0, (N,))) + 0.5
            S = jax.random.normal(k1, (N, J))
            Z = S  # curvature condition: s^T z = ||s||^2 > 0
            beta, gamma = lbfgs_inverse_hessian_factors(S, Z, alpha)
            position = jax.random.normal(k3, (N,))
            grad_position = jax.random.normal(k4, (N,))

            num_samples = 4
            phi, _ = bfgs_sample(
                k5, num_samples, position, grad_position, alpha, beta, gamma
            )

            # Reproduce the noise term with the same key to recover mu from phi.
            u = jax.random.normal(k5, (num_samples,) + (N, 1))
            Q, R = jnp.linalg.qr(beta / jnp.sqrt(alpha)[:, None], mode="reduced")
            Id = jnp.identity(R.shape[0])
            L = jnp.linalg.cholesky(Id + R @ gamma @ R.T)
            noise = jnp.sqrt(alpha)[:, None] * (Q @ (L - Id) @ (Q.T @ u) + u)
            mu_from_phi = phi - noise[..., 0]  # (num_samples, N)

            # Dense reference: the pre-PR formula using the O(N²) diag(alpha) intermediate.
            mu_ref = (
                position
                + jnp.diag(alpha) @ grad_position
                + beta @ gamma @ beta.T @ grad_position
            )
            np.testing.assert_allclose(
                mu_from_phi,
                jnp.broadcast_to(mu_ref, (num_samples, N)),
                rtol=1e-10,
                atol=1e-10,
            )


if __name__ == "__main__":
    absltest.main()
