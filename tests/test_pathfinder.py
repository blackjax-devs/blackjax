"""Test the pathfinder algorithm."""
import functools

import chex
import jax
import jax.numpy as jnp
import jax.scipy.stats as stats
import numpy as np
from absl.testing import absltest, parameterized
from jax.flatten_util import ravel_pytree
from jaxopt._src.lbfgs import inv_hessian_product

from blackjax.kernels import pathfinder
from blackjax.optimizers.lbfgs import (
    lbfgs_inverse_hessian_factors,
    lbfgs_inverse_hessian_formula_1,
    lbfgs_inverse_hessian_formula_2,
    lbfgs_sample,
    minimize_lbfgs,
)


class PathfinderTest(chex.TestCase):
    def setUp(self):
        super().setUp()
        self.key = jax.random.PRNGKey(1)

    @parameterized.parameters(
        [(1, 10), (10, 1), (10, 20)],
    )
    def test_inverse_hessian(self, maxiter, maxcor):
        """Test if dot product between approximate inverse hessian and gradient is
        the same between two loop recursion algorthm of LBFGS and formulas of the
        pathfinder paper"""

        def regression_logprob(scale, coefs, preds, x):
            """Linear regression"""
            logpdf = 0
            logpdf += stats.expon.logpdf(scale, 0, 2)
            logpdf += stats.norm.logpdf(coefs, 3 * jnp.ones(x.shape[-1]), 2)
            y = jnp.dot(x, coefs)
            logpdf += stats.norm.logpdf(preds, y, scale)
            return jnp.sum(logpdf)

        def regression_model(key):
            init_key0, init_key1 = jax.random.split(key, 2)
            x_data = jax.random.normal(init_key0, shape=(10_000, 1))
            y_data = 3 * x_data + jax.random.normal(init_key1, shape=x_data.shape)

            logposterior_fn_ = functools.partial(
                regression_logprob, x=x_data, preds=y_data
            )
            logposterior_fn = lambda x: logposterior_fn_(**x)

            return logposterior_fn

        fn = regression_model(self.key)
        b0 = {"scale": 1.0, "coefs": 2.0}
        b0_flatten, unravel_fn = ravel_pytree(b0)
        objective_fn = lambda x: -fn(unravel_fn(x))
        (result, status), history = minimize_lbfgs(
            objective_fn, b0_flatten, maxiter=maxiter, maxcor=maxcor
        )

        i = status.iter_num
        i_offset = history.x.shape[0] - status.iter_num + i - 2

        pk = inv_hessian_product(
            -history.g[i_offset + 1],
            status.s_history,
            status.y_history,
            status.rho_history,
            history.gamma[i_offset],
            status.iter_num % maxcor,
        )

        s = jnp.diff(history.x.T).at[:, -status.iter_num - 1].set(0.0)
        z = jnp.diff(history.g.T).at[:, -status.iter_num - 1].set(0.0)

        S = jax.lax.dynamic_slice(s, (0, i_offset - maxcor + 1), (2, maxcor))
        Z = jax.lax.dynamic_slice(z, (0, i_offset - maxcor + 1), (2, maxcor))

        alpha_scalar = history.gamma[i_offset + 1]
        alpha = alpha_scalar * jnp.ones(S.shape[0])
        beta, gamma = lbfgs_inverse_hessian_factors(S, Z, alpha)
        inv_hess_1 = lbfgs_inverse_hessian_formula_1(alpha, beta, gamma)
        inv_hess_2 = lbfgs_inverse_hessian_formula_2(alpha, beta, gamma)

        np.testing.assert_array_almost_equal(
            pk, -inv_hess_1 @ history.g[i_offset + 1], decimal=3
        )
        np.testing.assert_array_almost_equal(
            pk, -inv_hess_2 @ history.g[i_offset + 1], decimal=3
        )

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
        kernel = pathfinder(rng_key_pathfinder, logp_model)
        out = self.variant(kernel.init)(x0)

        sim_p, log_p = lbfgs_sample(
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
        self.assertAlmostEqual(kl, 0.0, delta=1e-0)


if __name__ == "__main__":
    absltest.main()
