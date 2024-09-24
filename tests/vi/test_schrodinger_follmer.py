import functools

import chex
import jax
import jax.numpy as jnp
import jax.scipy.stats as stats
from absl.testing import absltest

from blackjax.vi.schrodinger_follmer import as_top_level_api as schrodinger_follmer


class SchrodingerFollmerTest(chex.TestCase):
    def setUp(self):
        super().setUp()
        self.key = jax.random.key(1)

    @chex.all_variants(with_pmap=True)
    def test_recover_posterior(self):
        """Simple Normal mean test"""

        ndim = 2

        rng_key_chol, rng_key_observed, rng_key_init = jax.random.split(self.key, 3)
        L = jnp.tril(jax.random.normal(rng_key_chol, (ndim, ndim)))
        true_mu = jnp.arange(ndim)
        true_cov = L @ L.T
        true_prec = jnp.linalg.pinv(true_cov)

        def logp_posterior_conjugate_normal_model(
            observed, prior_mu, prior_prec, true_prec
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
            return posterior_mu

        def logp_unnormalized_posterior(x, observed, prior_mu, prior_prec, true_cov):
            logp = 0.0
            logp += stats.multivariate_normal.logpdf(x, prior_mu, prior_prec)
            logp += stats.multivariate_normal.logpdf(observed, x, true_cov).sum()
            return logp

        prior_mu = jnp.zeros(ndim)
        prior_prec = jnp.eye(ndim)

        # Simulate the data
        observed = jax.random.multivariate_normal(
            rng_key_observed, true_mu, true_cov, shape=(25,)
        )

        logp_model = functools.partial(
            logp_unnormalized_posterior,
            observed=observed,
            prior_mu=prior_mu,
            prior_prec=prior_prec,
            true_cov=true_cov,
        )

        initial_position = jnp.zeros((ndim,))
        posterior_mu = logp_posterior_conjugate_normal_model(
            observed, prior_mu, prior_prec, true_prec
        )

        schrodinger_follmer_algo = schrodinger_follmer(logp_model, 50, 25)

        initial_state = schrodinger_follmer_algo.init(initial_position)
        schrodinger_follmer_algo_sample = self.variant(
            lambda k, s: schrodinger_follmer_algo.sample(k, s, 100)
        )
        sampled_states = schrodinger_follmer_algo_sample(rng_key_init, initial_state)
        sampled_position = sampled_states.position
        chex.assert_trees_all_close(
            sampled_position.mean(0), posterior_mu, rtol=1e-2, atol=1e-1
        )

        # make sure basic interface is independently covered
        _ = schrodinger_follmer_algo.step(rng_key_init, initial_state)


if __name__ == "__main__":
    absltest.main()
