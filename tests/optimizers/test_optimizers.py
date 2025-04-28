"""Test optimizers."""

import functools

import chex
import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest, parameterized

from blackjax.optimizers.dual_averaging import dual_averaging
from blackjax.optimizers.lbfgs import (
    lbfgs_diff_history_matrix,
    lbfgs_inverse_hessian_factors,
    lbfgs_inverse_hessian_formula_1,
    lbfgs_inverse_hessian_formula_2,
    lbfgs_recover_alpha,
    minimize_lbfgs,
)


def compute_inverse_hessian_1_and_2(history, maxcor):
    not_converged_mask = jnp.logical_not(history.converged.at[1:].get())

    # jax.jit would not work with truncated history, so we keep the full history
    position = history.x
    grad_position = history.g

    alpha, s, z, update_mask = lbfgs_recover_alpha(
        position, grad_position, not_converged_mask
    )

    s = jnp.diff(position, axis=0)
    z = jnp.diff(grad_position, axis=0)
    S = lbfgs_diff_history_matrix(s, update_mask, maxcor)
    Z = lbfgs_diff_history_matrix(z, update_mask, maxcor)

    position = position.at[1:].get()
    grad_position = grad_position.at[1:].get()

    beta, gamma = jax.vmap(lbfgs_inverse_hessian_factors)(S, Z, alpha)

    inv_hess_1 = jax.vmap(lbfgs_inverse_hessian_formula_1)(alpha, beta, gamma)
    inv_hess_2 = jax.vmap(lbfgs_inverse_hessian_formula_2)(alpha, beta, gamma)

    return inv_hess_1, inv_hess_2


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
        [(4, 12), (12, 12), (4, 20), (20, 20)],
    )
    def test_minimize_lbfgs(self, maxcor, n):
        """Test if dot product between approximate inverse hessian and gradient is
        the same between two loop recursion algorthm of LBFGS and formulas of the
        pathfinder paper"""

        def quadratic(
            x: np.ndarray, A: np.ndarray, b: np.ndarray, c: float = 0.0
        ) -> float:
            """
            Quadratic function: f(x) = 0.5 * x^T * A * x - b^T * x + c

            Parameters:
                x: Input vector of shape (n,)
                A: Symmetric positive definite matrix of shape (n, n)
                b: Vector of shape (n,)
                c: Scalar constant

            Returns:
                Function value at x
            """
            return 0.5 * x.dot(A).dot(x) - b.dot(x) + c

        def create_spd_matrix(rng_key, n):
            """Create a symmetric positive definite matrix of shape (n, n)."""
            rand = jax.random.normal(rng_key, (n, n))
            A = jnp.dot(rand, rand.T) + n * jnp.eye(n)
            assert np.all(jnp.linalg.eigh(A)[0] > 0)
            return A

        spd_key, b_key, init_key = jax.random.split(self.key, 3)

        A = create_spd_matrix(spd_key, n)
        b = jax.random.normal(b_key, (n,))

        # initial guess
        x0 = jax.random.normal(init_key, shape=(n,))

        # run the optimizer
        quadratic_fn = functools.partial(quadratic, A=A, b=b)
        (result, (last_lbfgs_state, last_ls_state)), history = self.variant(
            functools.partial(minimize_lbfgs, quadratic_fn, maxcor=maxcor)
        )(x0)

        # check if the result is close to the expected minimum
        gt_minimum = np.linalg.solve(A, b)
        np.testing.assert_allclose(
            result,
            gt_minimum,
            atol=1e-2,
            err_msg=f"Expected {gt_minimum}, got {result}",
        )

        gt_inverse_hessian = jnp.linalg.inv(A)
        inv_hess_1, inv_hess_2 = compute_inverse_hessian_1_and_2(history, maxcor=maxcor)

        np.testing.assert_allclose(gt_inverse_hessian, inv_hess_1[-1], atol=1e-1)

        np.testing.assert_allclose(gt_inverse_hessian, inv_hess_2[-1], atol=1e-1)

    @chex.all_variants(with_pmap=False)
    def test_recover_diag_inv_hess(self):
        "Compare inverse Hessian estimation from LBFGS with known groundtruth."
        nd = 5
        maxcor = 6

        mean = np.linspace(3.0, 50.0, nd)
        cov = np.diag(np.linspace(1.0, 10.0, nd))

        def loss_fn(x):
            return -jax.scipy.stats.multivariate_normal.logpdf(x, mean, cov)

        x0 = jnp.zeros(nd)
        (result, (last_lbfgs_state, last_ls_state)), history = self.variant(
            functools.partial(minimize_lbfgs, loss_fn, maxcor=maxcor)
        )(x0)

        np.testing.assert_allclose(result, mean, rtol=0.05)

        inv_hess_1, inv_hess_2 = compute_inverse_hessian_1_and_2(history, maxcor=maxcor)

        np.testing.assert_allclose(np.diag(inv_hess_1[-1]), np.diag(cov), rtol=0.05)
        np.testing.assert_allclose(inv_hess_1[-1], inv_hess_2[-1], rtol=0.05)


if __name__ == "__main__":
    absltest.main()
