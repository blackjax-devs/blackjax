import functools

import chex
import jax
import jax.numpy as jnp
import jax.scipy.stats as stats
from absl.testing import absltest, parameterized
from jax.flatten_util import ravel_pytree

from blackjax.vi.schrodinger_follmer import as_top_level_api as schrodinger_follmer
from tests.fixtures import BlackJAXTest


class SchrodingerFollmerTest(BlackJAXTest):
    @chex.all_variants(with_pmap=True)
    @parameterized.named_parameters(
        ("scalar", ()),
        ("vector", (5,)),
        ("matrix_matching_inner_samples", (4, 3)),
        ("matrix", (2, 3)),
        ("tensor", (2, 2, 3)),
        ("mixed_pytree", None),
    )
    def test_coordinate_shape_invariance(self, shape):
        if shape is None:
            position = {
                "matrix": jnp.zeros((4, 3)),
                "vector": jnp.zeros(3),
                "scalar": jnp.array(0.0),
            }
        else:
            position = jnp.zeros(shape)
        flat_position, _ = ravel_pytree(position)

        def logdensity(x):
            return -0.5 * sum(jnp.sum((a - 0.4) ** 2) for a in jax.tree.leaves(x))

        algorithm = schrodinger_follmer(logdensity, n_steps=4, n_inner_samples=4)
        state = algorithm.init(position)
        flat_state = algorithm.init(flat_position)
        key = jax.random.key(0)
        next_state, info = self.variant(algorithm.step)(key, state)
        flat_next_state, flat_info = algorithm.step(key, flat_state)
        chex.assert_trees_all_close(
            ravel_pytree(info.drift)[0], flat_info.drift, atol=1e-6
        )
        chex.assert_trees_all_close(
            ravel_pytree(next_state.position)[0], flat_next_state.position, atol=1e-6
        )

        samples = self.variant(lambda k, s: algorithm.sample(k, s, 3))(key, state)
        flat_samples = algorithm.sample(key, flat_state, 3)
        flattened_samples = jax.vmap(lambda x: ravel_pytree(x)[0])(samples.position)
        chex.assert_trees_all_close(flattened_samples, flat_samples.position, atol=1e-6)

    @chex.all_variants(with_pmap=True)
    def test_recover_posterior(self):
        """Simple Normal mean test"""

        ndim = 2

        rng_key_observed, rng_key_init = jax.random.split(self.next_key(), 2)
        true_mu = jnp.arange(ndim, dtype=float)
        true_cov = jnp.array([[1.0, 0.5], [0.5, 2.0]])
        true_prec = jnp.linalg.inv(true_cov)

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

        # n_steps=100 SDE steps with n_inner_samples=100 drift Monte Carlo
        # samples — previously (50, 25), which gave a variational bias of order
        # ~0.13 on the second posterior component for an unlucky `BlackJAXTest`
        # date-rotated seed (e.g. 2026-05-20 → seed=20260520). Doubling each
        # parameter cuts the worst-seed max-diff from 0.131 → 0.060, giving a
        # comfortable safety margin under `atol=0.1` across 20 sampled daily
        # seeds for ~0.25s of extra wall time.
        schrodinger_follmer_algo = schrodinger_follmer(logp_model, 100, 100)

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
