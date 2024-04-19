"""Test optimizers."""
import functools

import chex
import jax
import jax.numpy as jnp
import jax.scipy.stats as stats
import numpy as np
from absl.testing import absltest, parameterized
from jax.flatten_util import ravel_pytree
from jaxopt._src.lbfgs import compute_gamma, inv_hessian_product

from blackjax.optimizers.dual_averaging import dual_averaging
from blackjax.optimizers.lbfgs import (
    lbfgs_inverse_hessian_factors,
    lbfgs_inverse_hessian_formula_1,
    lbfgs_inverse_hessian_formula_2,
    lbfgs_recover_alpha,
    minimize_lbfgs,
)


class OptimizerTest(chex.TestCase):
    def setUp(self):
        super().setUp()
        self.key = jax.random.key(1)

    @chex.all_variants(with_pmap=False)
    def test_dual_averaging(self):
        """We test the dual averaging algorithm by searching for the point that
        minimizes the gradient of a simple function.
        """

        # we need to wrap the gradient in a namedtuple as we optimize for a target
        # acceptance probability in the context of HMC.
        f = lambda x: (x - 1) ** 2
        grad_f = jax.jit(jax.grad(f))

        # Our target gradient is 0. we increase the rate of convergence by
        # increasing the value of gamma (see documentation of the algorithm).
        init, update, final = dual_averaging(gamma=0.3)
        unpdate_fn = self.variant(update)

        da_state = init(3)
        for _ in range(100):
            x = jnp.exp(da_state.log_x)
            g = grad_f(x)
            da_state = unpdate_fn(da_state, g)

        self.assertAlmostEqual(final(da_state), 1.0, delta=1e-1)

    @chex.all_variants(with_pmap=False)
    @parameterized.parameters(
        [(5, 10), (10, 2), (10, 20)],
    )
    def test_minimize_lbfgs(self, maxiter, maxcor):
        """Test if dot product between approximate inverse hessian and gradient is
        the same between two loop recursion algorthm of LBFGS and formulas of the
        pathfinder paper"""

        def regression_logprob(log_scale, coefs, preds, x):
            """Linear regression"""
            scale = jnp.exp(log_scale)
            scale_prior = stats.expon.logpdf(scale, 0, 1) + log_scale
            coefs_prior = stats.norm.logpdf(coefs, 0, 5)
            y = jnp.dot(x, coefs)
            logpdf = stats.norm.logpdf(preds, y, scale)
            return sum(x.sum() for x in [scale_prior, coefs_prior, logpdf])

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
        b0 = {"log_scale": 0.0, "coefs": 2.0}
        b0_flatten, unravel_fn = ravel_pytree(b0)
        objective_fn = lambda x: -fn(unravel_fn(x))
        (_, status), history = self.variant(
            functools.partial(
                minimize_lbfgs, objective_fn, maxiter=maxiter, maxcor=maxcor
            )
        )(b0_flatten)
        history = jax.tree.map(lambda x: x[: status.iter_num + 1], history)

        # Test recover alpha
        S = jnp.diff(history.x, axis=0)
        Z = jnp.diff(history.g, axis=0)
        alpha0 = history.alpha[0]

        def scan_fn(alpha, val):
            alpha_l, mask_l = lbfgs_recover_alpha(alpha, *val)
            return alpha_l, (alpha_l, mask_l)

        _, (alpha, mask) = jax.lax.scan(scan_fn, alpha0, (S, Z))
        np.testing.assert_array_almost_equal(alpha, history.alpha[1:])
        np.testing.assert_array_equal(mask, history.update_mask[1:])

        # Test inverse hessian product
        S_partial = S[-maxcor:].T
        Z_partial = Z[-maxcor:].T
        alpha = history.alpha[-1]

        beta, gamma = lbfgs_inverse_hessian_factors(S_partial, Z_partial, alpha)
        inv_hess_1 = lbfgs_inverse_hessian_formula_1(alpha, beta, gamma)
        inv_hess_2 = lbfgs_inverse_hessian_formula_2(alpha, beta, gamma)

        gamma = compute_gamma(S_partial, Z_partial, -1)
        pk = inv_hessian_product(
            -history.g[-1],
            status.s_history,
            status.y_history,
            status.rho_history,
            gamma,
            status.iter_num % maxcor,
        )

        np.testing.assert_allclose(pk, -inv_hess_1 @ history.g[-1], atol=1e-3)
        np.testing.assert_allclose(pk, -inv_hess_2 @ history.g[-1], atol=1e-3)

    @chex.all_variants(with_pmap=False)
    def test_recover_diag_inv_hess(self):
        "Compare inverse Hessian estimation from LBFGS with known groundtruth."
        nd = 5
        mean = np.linspace(3.0, 50.0, nd)
        cov = np.diag(np.linspace(1.0, 10.0, nd))

        def loss_fn(x):
            return -stats.multivariate_normal.logpdf(x, mean, cov)

        (result, status), history = self.variant(
            functools.partial(minimize_lbfgs, loss_fn, maxiter=50)
        )(np.zeros(nd))
        history = jax.tree.map(lambda x: x[: status.iter_num + 1], history)

        np.testing.assert_allclose(result, mean, rtol=0.01)

        S_partial = jnp.diff(history.x, axis=0)[-10:].T
        Z_partial = jnp.diff(history.g, axis=0)[-10:].T
        alpha = history.alpha[-1]

        beta, gamma = lbfgs_inverse_hessian_factors(S_partial, Z_partial, alpha)
        inv_hess_1 = lbfgs_inverse_hessian_formula_1(alpha, beta, gamma)
        inv_hess_2 = lbfgs_inverse_hessian_formula_2(alpha, beta, gamma)

        np.testing.assert_allclose(np.diag(inv_hess_1), np.diag(cov), rtol=0.01)
        np.testing.assert_allclose(inv_hess_1, inv_hess_2, rtol=0.01)


if __name__ == "__main__":
    absltest.main()
