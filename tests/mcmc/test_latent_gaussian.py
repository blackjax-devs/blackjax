import itertools

import chex
import jax
import jax.scipy.stats as stats
import numpy as np
from absl.testing import absltest, parameterized

from blackjax.mcmc.marginal_latent_gaussian import (
    build_kernel,
    generate_mean_shifted_logprob,
    init,
    svd_from_covariance,
)


class GaussianTest(chex.TestCase):
    @parameterized.parameters(itertools.product([1234, 5678], [True, False]))
    def test_gaussian(self, seed, mean):
        n_samples = 500_000

        key = jax.random.key(seed)
        key1, key2, key3, key4, key5 = jax.random.split(key, 5)

        D = 5
        C = jax.random.normal(key1, (D, D**2))
        C = C @ C.T
        prior_mean = jax.random.normal(key2, (D,)) if mean else None
        R = jax.random.normal(key3, (D, D**2))
        R = R @ R.T

        obs = jax.random.normal(key4, (D,))
        log_pdf = lambda x: stats.multivariate_normal.logpdf(x, obs, R)
        if prior_mean is not None:
            log_pdf = generate_mean_shifted_logprob(log_pdf, prior_mean, C)

        DELTA = 50.0
        cov_svd = svd_from_covariance(C)
        _step = build_kernel(cov_svd)
        step = jax.jit(lambda key, state, delta: _step(key, state, log_pdf, delta))

        init_x = np.zeros((D,))
        init_state = init(init_x, log_pdf, cov_svd.U_t)

        keys = jax.random.split(key5, n_samples)

        def body(carry, key):
            curr_n_accepted, state = carry
            state, info = step(key, state, DELTA)
            carry = curr_n_accepted + info.is_accepted, state
            return carry, state

        (n_accepted, _), states = jax.lax.scan(body, (0, init_state), keys)
        if mean:
            expected_mean, expected_cov = _expected_mean_and_cov(prior_mean, C, obs, R)
        else:
            expected_mean, expected_cov = _expected_mean_and_cov(
                np.zeros((D,)), C, obs, R
            )

        chex.assert_trees_all_close(
            np.mean(states.position, 0), expected_mean, atol=1e-1, rtol=1e-2
        )
        chex.assert_trees_all_close(
            np.cov(states.position, rowvar=False), expected_cov, atol=1e-1, rtol=1e-1
        )


def _expected_mean_and_cov(prior_mean, prior_cov, obs, obs_cov):
    S = obs_cov + prior_cov
    gain = prior_cov @ np.linalg.inv(S)
    mean = prior_mean + gain @ (obs - prior_mean)
    cov = prior_cov - gain @ prior_cov
    return mean, cov


if __name__ == "__main__":
    absltest.main()
