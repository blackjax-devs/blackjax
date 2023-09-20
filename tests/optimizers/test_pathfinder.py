"""Test the pathfinder algorithm."""
import functools

import chex
import jax
import jax.numpy as jnp
import jax.scipy.stats as stats
from absl.testing import absltest, parameterized

import blackjax
from blackjax.optimizers.lbfgs import bfgs_sample


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
        out, _ = self.variant(pathfinder.approximate)(rng_key_pathfinder, x0)

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


if __name__ == "__main__":
    absltest.main()
