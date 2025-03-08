"""Test the pathfinder algorithm."""

import functools

import chex
import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest, parameterized

import blackjax


class PathfinderTest(chex.TestCase):
    def setUp(self):
        super().setUp()
        self.key = jax.random.key(1)

    @chex.all_variants(with_pmap=False)
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
            return jax.scipy.stats.multivariate_normal.logpdf(
                x, posterior_mu, posterior_cov
            )

        def logp_unnormalized_posterior(x, observed, prior_mu, prior_prec, true_cov):
            logp = 0.0
            logp += jax.scipy.stats.multivariate_normal.logpdf(x, prior_mu, prior_prec)
            logp += jax.scipy.stats.multivariate_normal.logpdf(
                observed, x, true_cov
            ).sum()
            return logp

        rng_key_chol, rng_key_observed, rng_key_path, rng_key_choice = jax.random.split(
            self.key, 4
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
        num_paths = 4

        pathfinder = blackjax.pathfinder(logp_model, num_paths=num_paths)
        initial_positions = self.variant(pathfinder.init)(initial_position=x0)

        path_keys = jax.random.split(rng_key_path, num_paths)

        samples, logq = self.variant(jax.vmap(pathfinder.pathfinder))(
            path_keys, initial_positions
        )

        samples = samples.reshape((-1, ndim))
        logq = logq.ravel()

        logp = logp_posterior_conjugate_normal_model(
            samples, observed, prior_mu, prior_prec, true_prec
        )

        kl = (logp - logq).mean()
        # TODO(junpenglao): Make this test more robust.
        self.assertAlmostEqual(kl, 0.0, delta=2.5)

        result = blackjax.multi_pathfinder(
            rng_key=self.key,
            logdensity_fn=logp_model,
            initial_position=x0,
            # jitter_amount=12.0,
            num_paths=num_paths,
            parallel_method="vectorize",
        )

        self.assertAlmostEqual(result.samples.mean(), 0.0, delta=2.5)

    @chex.all_variants(with_pmap=False)
    @parameterized.parameters(
        [(2,), (6,)],
    )
    def test_recover_posterior_eight_schools(self, maxcor):
        J = 8
        y = jnp.array([28.0, 8.0, -3.0, 7.0, -1.0, 1.0, 18.0, 12.0])
        sigma = jnp.array([15.0, 10.0, 16.0, 11.0, 9.0, 11.0, 10.0, 18.0])

        def eight_schools_log_density(y, sigma, mu, tau, theta):
            logp = 0.0
            # Prior for mu
            logp += jax.scipy.stats.norm.logpdf(mu, loc=0.0, scale=10.0)
            # Prior for tau
            logp += jax.scipy.stats.gamma.logpdf(tau, 5, 1)
            # Prior for theta
            logp += jax.scipy.stats.norm.logpdf(theta, loc=0.0, scale=1.0).sum()
            # Likelihood
            logp += jax.scipy.stats.norm.logpdf(
                y, loc=mu + tau * theta, scale=sigma
            ).sum()
            return logp

        def logdensity_fn(param):
            def inner(param):
                mu, tau, *theta = param
                mu = jnp.atleast_1d(mu)
                tau = jnp.atleast_1d(tau)
                theta = jnp.array(theta)
                return eight_schools_log_density(y, sigma, mu, tau, theta)

            return inner(param).squeeze()

        mu_prior = jnp.array([0.0])
        tau_prior = jnp.array([5.0])
        theta_prior = jnp.array([0.0] * J)
        base_position = jnp.concatenate([mu_prior, tau_prior, theta_prior])

        mp = functools.partial(
            blackjax.multi_pathfinder,
            logdensity_fn=logdensity_fn,
            num_paths=20,
            maxcor=maxcor,
            parallel_method="vectorize",
        )

        result = self.variant(mp)(
            rng_key=self.key,
            base_position=base_position,
            jitter_amount=12.0,
        )

        np.testing.assert_allclose(result.samples[:, 0].mean(), 5.0, atol=1.6)
        np.testing.assert_allclose(result.samples[:, 1].mean(), 4.15, atol=1.5)


if __name__ == "__main__":
    absltest.main()
