"""Test optimizers."""
import functools

import chex
import jax
import jax.numpy as jnp
import jax.scipy.stats as stats
import numpy as np
from absl.testing import absltest, parameterized
from jax.flatten_util import ravel_pytree
from jaxopt._src.lbfgs import inv_hessian_product

import blackjax.optimizers.dual_averaging as dual_averaging
from blackjax.optimizers.lbfgs import (
    lbfgs_inverse_hessian_factors,
    lbfgs_inverse_hessian_formula_1,
    lbfgs_inverse_hessian_formula_2,
    minimize_lbfgs,
)


class OptimizerTest(chex.TestCase):
    def setUp(self):
        super().setUp()
        self.key = jax.random.PRNGKey(1)

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
        init, update, final = dual_averaging.dual_averaging(gamma=0.3)
        unpdate_fn = self.variant(update)

        da_state = init(3)
        for _ in range(100):
            x = jnp.exp(da_state.log_x)
            g = grad_f(x)
            da_state = unpdate_fn(da_state, g)

        self.assertAlmostEqual(final(da_state), 1.0, delta=1e-1)

    @parameterized.parameters(
        [(1, 10), (10, 1), (10, 20)],
    )
    def test_lbfgs_inverse_hessian(self, maxiter, maxcor):
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


if __name__ == "__main__":
    absltest.main()
