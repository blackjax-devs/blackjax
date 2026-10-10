import itertools

import chex
import jax
import jax.numpy as jnp
import jax.scipy.stats as stats
import numpy as np
from absl.testing import absltest, parameterized

import blackjax
from blackjax.mcmc.marginal_latent_gaussian import generate_mean_shifted_logprob
from blackjax.util import run_inference_algorithm


def _expected_mean_and_cov(prior_mean, prior_cov, obs, obs_cov):
    S = obs_cov + prior_cov
    gain = prior_cov @ np.linalg.inv(S)
    mean = prior_mean + gain @ (obs - prior_mean)
    cov = prior_cov - gain @ prior_cov
    return mean, cov


class MarginalLatentGaussianTest(chex.TestCase):
    @parameterized.parameters(
        itertools.product([1234, 5678], [True, False], [True, False])
    )
    def test_gaussian_statistics(self, seed, use_mean, tree_input):
        key = jax.random.key(seed)
        key1, key2, key3, key4, key5 = jax.random.split(key, 5)

        D = 4
        C = jax.random.normal(key1, (D, D**2))
        C = C @ C.T
        prior_mean = jax.random.normal(key2, (D,)) if use_mean else None
        R = jax.random.normal(key3, (D, D**2))
        R = R @ R.T
        obs = jax.random.normal(key4, (D,))

        if tree_input:
            to_pytree = lambda x: {"a": x[: D // 2], "b": x[D // 2 :]}
            init_position = to_pytree(np.zeros(D))
            log_pdf = lambda x: stats.multivariate_normal.logpdf(
                jnp.concatenate([x["a"], x["b"]]), obs, R
            )
            if prior_mean is not None:
                log_pdf = generate_mean_shifted_logprob(
                    log_pdf, to_pytree(prior_mean), C
                )
        else:
            init_position = np.zeros(D)
            log_pdf = lambda x: stats.multivariate_normal.logpdf(x, obs, R)
            if prior_mean is not None:
                log_pdf = generate_mean_shifted_logprob(log_pdf, prior_mean, C)

        sampler = blackjax.mgrad_gaussian(log_pdf, covariance=C, step_size=50.0)
        _, history = run_inference_algorithm(
            key5,
            sampler,
            num_steps=500_000,
            initial_position=init_position,
            transform=lambda state, info: state.position,
        )

        if tree_input:
            positions = jnp.concatenate([history["a"], history["b"]], axis=-1)
        else:
            positions = history

        expected_mean, expected_cov = _expected_mean_and_cov(
            prior_mean if use_mean else np.zeros(D), C, obs, R
        )
        chex.assert_trees_all_close(
            np.mean(positions, 0), expected_mean, atol=1e-1, rtol=1e-2
        )
        chex.assert_trees_all_close(
            np.cov(positions, rowvar=False), expected_cov, atol=1e-1, rtol=1e-1
        )


if __name__ == "__main__":
    absltest.main()
