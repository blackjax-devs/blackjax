import contextlib

import chex
import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest, parameterized
from jax import random
from jax.scipy import linalg

from blackjax.mcmc import metrics


def _x64_ctx(dtype):
    """Return jax.enable_x64() for float64 tests, nullcontext otherwise."""
    if dtype == "float64":
        return jax.enable_x64()
    return contextlib.nullcontext()


class CovarianceFormattingTest(chex.TestCase):
    def setUp(self):
        super().setUp()
        self.key = random.key(0)
        self.dtype = "float32"

    @parameterized.named_parameters(
        {"testcase_name": "0d", "shape": (), "is_inv": False},
        {"testcase_name": "0d_inv", "shape": (), "is_inv": True},
        {"testcase_name": "3d", "shape": (1, 2, 3), "is_inv": False},
        {"testcase_name": "3d_inv", "shape": (1, 2, 3), "is_inv": True},
    )
    def test_invalid(self, shape, is_inv):
        """Test formatting raises error for invalid shapes"""
        mass_matrix = jnp.zeros(shape=shape)
        with self.assertRaisesRegex(
            ValueError, "The mass matrix has the wrong number of dimensions"
        ):
            metrics._format_covariance(mass_matrix, is_inv)

    @parameterized.named_parameters(
        {"testcase_name": "inv", "is_inv": True},
        {"testcase_name": "no_inv", "is_inv": False},
    )
    def test_dim_1(self, is_inv):
        """Test formatting for 1D mass matrix"""
        mass_matrix = jnp.asarray([1 / 4], dtype=self.dtype)
        mass_matrix_sqrt, inv_mass_matrix_sqrt, diag = metrics._format_covariance(
            mass_matrix, is_inv
        )
        if is_inv:
            chex.assert_trees_all_close(inv_mass_matrix_sqrt, mass_matrix**0.5)
            chex.assert_trees_all_close(mass_matrix_sqrt, mass_matrix**-0.5)
        else:
            chex.assert_trees_all_close(mass_matrix_sqrt, mass_matrix**0.5)
            chex.assert_trees_all_close(inv_mass_matrix_sqrt, mass_matrix**-0.5)

        chex.assert_trees_all_close(diag(mass_matrix), mass_matrix)

    @parameterized.named_parameters(
        {"testcase_name": "inv", "is_inv": True},
        {"testcase_name": "no_inv", "is_inv": False},
    )
    def test_dim_2(self, is_inv):
        """Test formatting for 2D mass matrix"""
        mass_matrix = jnp.asarray([[2 / 3, 0.5], [0.5, 3 / 4]], dtype=self.dtype)
        mass_matrix_sqrt, inv_mass_matrix_sqrt, diag = metrics._format_covariance(
            mass_matrix, is_inv
        )
        if is_inv:
            chex.assert_trees_all_close(
                mass_matrix_sqrt @ mass_matrix_sqrt.T, linalg.inv(mass_matrix)
            )
            chex.assert_trees_all_close(
                inv_mass_matrix_sqrt @ inv_mass_matrix_sqrt.T, mass_matrix
            )

        else:
            chex.assert_trees_all_close(
                mass_matrix_sqrt @ mass_matrix_sqrt.T, mass_matrix
            )
            chex.assert_trees_all_close(
                inv_mass_matrix_sqrt @ inv_mass_matrix_sqrt.T, linalg.inv(mass_matrix)
            )

    def test_dim2_inv_and_not_inv_agree(self):
        mass_matrix = jnp.asarray([[2 / 3, 0.5], [0.5, 3 / 4]], dtype=self.dtype)
        mass_matrix_sqrt, inv_mass_matrix_sqrt, _ = metrics._format_covariance(
            mass_matrix, False
        )
        mass_matrix_sqrt_inv, inv_mass_matrix_sqrt_inv, _ = metrics._format_covariance(
            linalg.inv(mass_matrix), True
        )

        chex.assert_trees_all_close(
            mass_matrix_sqrt @ mass_matrix_sqrt.T,
            mass_matrix_sqrt_inv @ mass_matrix_sqrt_inv.T,
        )
        chex.assert_trees_all_close(
            inv_mass_matrix_sqrt @ inv_mass_matrix_sqrt.T,
            inv_mass_matrix_sqrt_inv @ inv_mass_matrix_sqrt_inv.T,
        )


class GaussianEuclideanMetricsTest(chex.TestCase):
    def setUp(self):
        super().setUp()
        self.key = random.key(0)
        self.dtype = "float32"

    @parameterized.named_parameters(
        {"testcase_name": "0d", "shape": ()},
        {"testcase_name": "3d", "shape": (1, 2, 3)},
    )
    def test_gaussian_euclidean_ndim_invalid(self, shape):
        """Test Gaussian Euclidean Function returns correct function invalid ndim"""
        x = jnp.ones(shape=shape)
        with self.assertRaisesRegex(
            ValueError, "The mass matrix has the wrong number of dimensions"
        ):
            _ = metrics.gaussian_euclidean(x)

    @chex.all_variants(with_pmap=False)
    def test_gaussian_euclidean_dim_1(self):
        """Test Gaussian Euclidean Function with ndim 1"""
        inverse_mass_matrix = jnp.asarray([1 / 4], dtype=self.dtype)
        momentum, kinetic_energy, _, scale = metrics.gaussian_euclidean(
            inverse_mass_matrix
        )

        arbitrary_position = jnp.asarray([12345], dtype=self.dtype)
        momentum_val = self.variant(momentum)(self.key, arbitrary_position)

        # 2 is square root inverse of 1/4
        expected_momentum_val = 2 * random.normal(self.key)

        kinetic_energy_val = self.variant(kinetic_energy)(momentum_val)
        velocity = inverse_mass_matrix * momentum_val
        expected_kinetic_energy_val = 0.5 * velocity * momentum_val

        assert momentum_val == expected_momentum_val
        assert kinetic_energy_val == expected_kinetic_energy_val

        inv_scaled_momentum = scale(
            arbitrary_position, momentum_val, inv=True, trans=False
        )
        scaled_momentum = scale(
            arbitrary_position, momentum_val, inv=False, trans=False
        )

        expected_scaled_momentum = momentum_val / jnp.sqrt(inverse_mass_matrix)
        expected_inv_scaled_momentum = momentum_val * jnp.sqrt(inverse_mass_matrix)

        chex.assert_trees_all_close(inv_scaled_momentum, expected_inv_scaled_momentum)
        chex.assert_trees_all_close(scaled_momentum, expected_scaled_momentum)

    @chex.all_variants(with_pmap=False)
    def test_gaussian_euclidean_dim_2(self):
        """Test Gaussian Euclidean Function with ndim 2"""
        inverse_mass_matrix = jnp.asarray(
            [[2 / 3, 0.5], [0.5, 3 / 4]], dtype=self.dtype
        )
        momentum, kinetic_energy, _, scale = metrics.gaussian_euclidean(
            inverse_mass_matrix
        )

        arbitrary_position = jnp.asarray([12345, 23456], dtype=self.dtype)
        momentum_val = self.variant(momentum)(self.key, arbitrary_position)

        L_inv = linalg.inv(linalg.cholesky(inverse_mass_matrix, lower=False))

        expected_momentum_val = L_inv @ random.normal(self.key, shape=(2,))

        kinetic_energy_val = self.variant(kinetic_energy)(momentum_val)
        velocity = jnp.dot(inverse_mass_matrix, momentum_val)
        expected_kinetic_energy_val = 0.5 * jnp.matmul(velocity, momentum_val)

        np.testing.assert_allclose(expected_momentum_val, momentum_val)
        np.testing.assert_allclose(kinetic_energy_val, expected_kinetic_energy_val)

        inv_scaled_momentum = scale(
            arbitrary_position, momentum_val, inv=True, trans=False
        )
        scaled_momentum = scale(
            arbitrary_position, momentum_val, inv=False, trans=False
        )

        expected_inv_scaled_momentum = jnp.linalg.inv(L_inv).T @ momentum_val
        expected_scaled_momentum = L_inv @ momentum_val

        chex.assert_trees_all_close(inv_scaled_momentum, expected_inv_scaled_momentum)
        chex.assert_trees_all_close(scaled_momentum, expected_scaled_momentum)


class GaussianRiemannianMetricsTest(chex.TestCase):
    def setUp(self):
        super().setUp()
        self.key = random.key(0)
        self.dtype = "float32"

    @parameterized.named_parameters(
        {"testcase_name": "0d", "shape": ()},
        {"testcase_name": "3d", "shape": (1, 2, 3)},
    )
    def test_gaussian_riemannian_value_errors(self, shape):
        x = jnp.ones(shape=shape)
        metric = metrics.gaussian_riemannian(lambda _: x)
        with self.assertRaisesRegex(
            ValueError, "The mass matrix has the wrong number of dimensions"
        ):
            metric.sample_momentum(self.key, x)

        with self.assertRaisesRegex(
            ValueError, "The mass matrix has the wrong number of dimensions"
        ):
            metric.kinetic_energy(x, position=x)

        with self.assertRaisesRegex(
            ValueError, "must be called with the position specified"
        ):
            metric.kinetic_energy(x)

    @chex.all_variants(with_pmap=False)
    def test_gaussian_riemannian_dim_1(self):
        inverse_mass_matrix = jnp.asarray([1 / 4], dtype=self.dtype)
        mass_matrix = jnp.asarray([4.0], dtype=self.dtype)
        momentum, kinetic_energy, _, scale = metrics.gaussian_riemannian(
            lambda _: mass_matrix
        )

        arbitrary_position = jnp.asarray([12345], dtype=self.dtype)
        momentum_val = self.variant(momentum)(self.key, arbitrary_position)

        # 2 is square root inverse of 1/4
        expected_momentum_val = 2 * random.normal(self.key)

        kinetic_energy_val = self.variant(kinetic_energy)(
            momentum_val, position=arbitrary_position
        )
        velocity = inverse_mass_matrix * momentum_val
        expected_kinetic_energy_val = 0.5 * velocity * momentum_val
        expected_kinetic_energy_val += 0.5 * jnp.sum(jnp.log(2 * jnp.pi * mass_matrix))

        np.testing.assert_allclose(expected_momentum_val, momentum_val)
        np.testing.assert_allclose(kinetic_energy_val, expected_kinetic_energy_val)

        inv_scaled_momentum = scale(
            arbitrary_position, momentum_val, inv=True, trans=False
        )
        scaled_momentum = scale(
            arbitrary_position, momentum_val, inv=False, trans=False
        )
        expected_scaled_momentum = momentum_val / jnp.sqrt(inverse_mass_matrix)
        expected_inv_scaled_momentum = momentum_val * jnp.sqrt(inverse_mass_matrix)

        chex.assert_trees_all_close(inv_scaled_momentum, expected_inv_scaled_momentum)
        chex.assert_trees_all_close(scaled_momentum, expected_scaled_momentum)

    @chex.all_variants(with_pmap=False)
    def test_gaussian_riemannian_dim_2(self):
        inverse_mass_matrix = jnp.asarray(
            [[2 / 3, 0.5], [0.5, 3 / 4]], dtype=self.dtype
        )
        mass_matrix = jnp.linalg.inv(inverse_mass_matrix)
        momentum, kinetic_energy, _, scale = metrics.gaussian_riemannian(
            lambda _: mass_matrix
        )

        arbitrary_position = jnp.asarray([12345, 23456], dtype=self.dtype)
        momentum_val = self.variant(momentum)(self.key, arbitrary_position)

        L_inv = linalg.cholesky(linalg.inv(inverse_mass_matrix), lower=True)
        expected_momentum_val = L_inv @ random.normal(self.key, shape=(2,))

        sqrt_mass_matrix, inv_sqrt_mass_matrix, _ = metrics._format_covariance(
            inverse_mass_matrix, True
        )

        kinetic_energy_val = self.variant(kinetic_energy)(
            momentum_val, position=arbitrary_position
        )
        velocity = jnp.dot(inverse_mass_matrix, momentum_val)
        expected_kinetic_energy_val = 0.5 * jnp.matmul(velocity, momentum_val)
        expected_kinetic_energy_val += 0.5 * jnp.linalg.slogdet(mass_matrix)[1]
        expected_kinetic_energy_val += 0.5 * len(mass_matrix) * jnp.log(2 * jnp.pi)

        np.testing.assert_allclose(expected_momentum_val, momentum_val)
        np.testing.assert_allclose(kinetic_energy_val, expected_kinetic_energy_val)

        inv_scaled_momentum = scale(
            arbitrary_position, momentum_val, inv=True, trans=False
        )
        scaled_momentum = scale(
            arbitrary_position, momentum_val, inv=False, trans=False
        )
        expected_inv_scaled_momentum = jnp.linalg.inv(L_inv).T @ momentum_val
        expected_scaled_momentum = L_inv @ momentum_val

        chex.assert_trees_all_close(inv_scaled_momentum, expected_inv_scaled_momentum)
        chex.assert_trees_all_close(scaled_momentum, expected_scaled_momentum)


class GaussianEuclideanLowRankTest(chex.TestCase):
    """Tests for gaussian_euclidean_low_rank metric."""

    def setUp(self):
        super().setUp()
        self.key = random.key(20240101)
        d, k = 8, 3
        # Non-trivial sigma and U not aligned with basis vectors
        sigma_key, u_key = random.split(self.key)
        self.sigma = jnp.exp(random.normal(sigma_key, (d,)) * 0.3)  # positive
        Q, _ = jnp.linalg.qr(random.normal(u_key, (d, k)))
        self.U = Q
        self.lam = jnp.array([4.0, 0.5, 2.5])
        self.d = d
        self.metric = metrics.gaussian_euclidean_low_rank(self.sigma, self.U, self.lam)
        self.pos = jnp.zeros(d)

        # True M = D^{-1} C D^{-1}, M^{-1} = D A D
        D_inv = jnp.diag(1.0 / self.sigma)
        C = jnp.eye(d) + self.U @ jnp.diag(1.0 / self.lam - 1.0) @ self.U.T
        self.M_true = D_inv @ C @ D_inv
        A = jnp.eye(d) + self.U @ jnp.diag(self.lam - 1.0) @ self.U.T
        self.M_inv_true = jnp.diag(self.sigma) @ A @ jnp.diag(self.sigma)

    def _scale_matrix(self, inv, trans):
        """Return the d×d matrix represented by metric.scale(·, inv=inv, trans=trans)."""
        import jax
        from jax.flatten_util import ravel_pytree

        rows = jax.vmap(
            lambda e: ravel_pytree(
                self.metric.scale(self.pos, e, inv=inv, trans=trans)
            )[0]
        )(jnp.eye(self.d))
        return rows  # rows[i] = scale(e_i), so rows.T = the matrix

    def test_kinetic_energy_formula(self):
        """K(p) = 0.5 * p^T M^{-1} p."""
        key = random.fold_in(self.key, 1)
        p = random.normal(key, (self.d,))
        K_metric = self.metric.kinetic_energy(p)
        K_direct = 0.5 * p @ self.M_inv_true @ p
        np.testing.assert_allclose(float(K_metric), float(K_direct), rtol=1e-5)

    def test_expected_kinetic_energy(self):
        """E[K(p)] = d/2 when p ~ N(0, M)."""
        import jax
        from jax.flatten_util import ravel_pytree

        n = 50_000
        keys = random.split(self.key, n)
        ps = jax.vmap(
            lambda k: ravel_pytree(self.metric.sample_momentum(k, self.pos))[0]
        )(keys)
        Ks = jax.vmap(self.metric.kinetic_energy)(ps)
        np.testing.assert_allclose(float(Ks.mean()), self.d / 2, rtol=0.02)

    def test_momentum_covariance(self):
        """E[pp^T] = M when p ~ sample_momentum."""
        import jax
        from jax.flatten_util import ravel_pytree

        n = 80_000
        keys = random.split(self.key, n)
        ps = jax.vmap(
            lambda k: ravel_pytree(self.metric.sample_momentum(k, self.pos))[0]
        )(keys)
        M_emp = (ps.T @ ps) / n
        np.testing.assert_allclose(M_emp, self.M_true, atol=0.05)

    def test_scale_ff(self):
        """scale(e, inv=False, trans=False) = M^{1/2}: S^T S = M."""
        S = self._scale_matrix(False, False)
        np.testing.assert_allclose(S.T @ S, self.M_true, atol=1e-5)

    def test_scale_ft(self):
        """scale(e, inv=False, trans=True) = (M^{1/2})^T: S S^T = M."""
        S = self._scale_matrix(False, True)
        np.testing.assert_allclose(S @ S.T, self.M_true, atol=1e-5)

    def test_scale_tf(self):
        """scale(e, inv=True, trans=False) = M^{-1/2}: S^T S = M^{-1}."""
        S = self._scale_matrix(True, False)
        np.testing.assert_allclose(S.T @ S, self.M_inv_true, atol=1e-5)

    def test_scale_tt(self):
        """scale(e, inv=True, trans=True) = (M^{-1/2})^T: S S^T = M^{-1}."""
        S = self._scale_matrix(True, True)
        np.testing.assert_allclose(S @ S.T, self.M_inv_true, atol=1e-5)

    def test_diagonal_case_matches_euclidean(self):
        """With lam=ones and U=zeros the metric reduces to diagonal gaussian_euclidean."""
        from jax.flatten_util import ravel_pytree

        U_zero = jnp.zeros((self.d, 2))
        lam_one = jnp.ones(2)
        lr_metric = metrics.gaussian_euclidean_low_rank(self.sigma, U_zero, lam_one)

        # diagonal inverse mass matrix = sigma^2
        diag_metric = metrics.gaussian_euclidean(self.sigma**2)

        key = random.fold_in(self.key, 99)
        p_lr, _ = ravel_pytree(lr_metric.sample_momentum(key, self.pos))
        p_diag, _ = ravel_pytree(diag_metric.sample_momentum(key, self.pos))
        np.testing.assert_allclose(p_lr, p_diag, rtol=1e-5)

        p_test = random.normal(key, (self.d,))
        np.testing.assert_allclose(
            lr_metric.kinetic_energy(p_test),
            diag_metric.kinetic_energy(p_test),
            rtol=1e-5,
        )

    def test_is_turning_detects_uturn(self):
        """is_turning returns True for a perfect U-turn."""
        p = jnp.ones(self.d)
        # p_left = p, p_right = -p → p_sum = 0, velocity · 0 = 0 → turning
        turning = self.metric.check_turning(p, -p, jnp.zeros(self.d))
        self.assertTrue(bool(turning))

    def test_is_turning_false_for_straight(self):
        """is_turning returns False when trajectory is moving straight."""
        p = jnp.ones(self.d)
        # p_left = p_right = p_sum/2 = p (constant momentum → not turning)
        not_turning = self.metric.check_turning(p, p, 2 * p)
        self.assertFalse(bool(not_turning))


# ---------------------------------------------------------------------------
# R1 Golden Tests — representation-layer consolidation, zero behaviour change
# ---------------------------------------------------------------------------
# These tests act as a "golden" gate: they compare the new helper-based
# implementations against frozen reference expressions (the original inline
# code, copied verbatim into the test body).  They must pass at all supported
# dtypes and across a range of (d, k) shapes before merging.
#
# (a) _low_rank_matvec parity vs frozen inline expression
# (b) ESH forward_L / adjoint_L parity vs frozen original closures
# (c) Refactored gaussian_euclidean_low_rank parity vs frozen metric impl
# (d) lbfgs_inverse_hessian_to_low_rank_metric parity vs formula_1 dense
# ---------------------------------------------------------------------------


def _make_orthonormal_U(key, d, k):
    """Return a (d, k) matrix with orthonormal columns, or (d, 0) for k=0."""
    if k == 0:
        return jnp.zeros((d, 0))
    Q, _ = jnp.linalg.qr(random.normal(key, (d, max(k, d))))
    return Q[:, :k]


def _make_positive_vector(key, n, lo=0.2, hi=3.0):
    """Return a random positive vector of length n drawn log-uniformly."""
    return jnp.exp(random.uniform(key, (n,), minval=jnp.log(lo), maxval=jnp.log(hi)))


class LowRankMatvecParityTest(chex.TestCase):
    """(a) _low_rank_matvec matches the frozen inline expression for all (d,k,dtype)."""

    @parameterized.named_parameters(
        # (d, k, dtype)
        {"testcase_name": "d2_k0_f32", "d": 2, "k": 0, "dtype": "float32"},
        {"testcase_name": "d2_k1_f32", "d": 2, "k": 1, "dtype": "float32"},
        {"testcase_name": "d7_k0_f32", "d": 7, "k": 0, "dtype": "float32"},
        {"testcase_name": "d7_k1_f32", "d": 7, "k": 1, "dtype": "float32"},
        {"testcase_name": "d7_k3_f32", "d": 7, "k": 3, "dtype": "float32"},
        {"testcase_name": "d50_k1_f32", "d": 50, "k": 1, "dtype": "float32"},
        {"testcase_name": "d50_k3_f32", "d": 50, "k": 3, "dtype": "float32"},
        {"testcase_name": "d50_k25_f32", "d": 50, "k": 25, "dtype": "float32"},
        {"testcase_name": "d2_k0_f64", "d": 2, "k": 0, "dtype": "float64"},
        {"testcase_name": "d2_k1_f64", "d": 2, "k": 1, "dtype": "float64"},
        {"testcase_name": "d7_k0_f64", "d": 7, "k": 0, "dtype": "float64"},
        {"testcase_name": "d7_k1_f64", "d": 7, "k": 1, "dtype": "float64"},
        {"testcase_name": "d7_k3_f64", "d": 7, "k": 3, "dtype": "float64"},
        {"testcase_name": "d50_k1_f64", "d": 50, "k": 1, "dtype": "float64"},
        {"testcase_name": "d50_k3_f64", "d": 50, "k": 3, "dtype": "float64"},
        {"testcase_name": "d50_k25_f64", "d": 50, "k": 25, "dtype": "float64"},
    )
    def test_parity_with_frozen_expression(self, d, k, dtype):
        """_low_rank_matvec == frozen y + U @ ((s-1)*(U.T @ y)) for all (d, k, dtype)."""
        atol = 1e-6 if dtype == "float32" else 1e-12
        key = random.key(2024_01_01 + d * 100 + k)

        with _x64_ctx(dtype):
            k1, k2, k3 = random.split(key, 3)
            U = _make_orthonormal_U(k1, d, k).astype(dtype)
            s = _make_positive_vector(k2, k).astype(dtype)
            y = random.normal(k3, (d,)).astype(dtype)

            # Frozen reference: original inline expression
            frozen = y + U @ ((s - 1.0) * (U.T @ y))
            # New helper
            result = metrics._low_rank_matvec(y, U, s)

        np.testing.assert_allclose(result, frozen, atol=atol, rtol=0)


class EshOperatorParityTest(chex.TestCase):
    """(b) ESH forward_L / adjoint_L via _low_rank_matvec == frozen original closures."""

    @parameterized.named_parameters(
        {"testcase_name": "d2_k1_f32", "d": 2, "k": 1, "dtype": "float32"},
        {"testcase_name": "d7_k1_f32", "d": 7, "k": 1, "dtype": "float32"},
        {"testcase_name": "d7_k3_f32", "d": 7, "k": 3, "dtype": "float32"},
        {"testcase_name": "d50_k3_f32", "d": 50, "k": 3, "dtype": "float32"},
        {"testcase_name": "d50_k25_f32", "d": 50, "k": 25, "dtype": "float32"},
        {"testcase_name": "d2_k1_f64", "d": 2, "k": 1, "dtype": "float64"},
        {"testcase_name": "d7_k1_f64", "d": 7, "k": 1, "dtype": "float64"},
        {"testcase_name": "d7_k3_f64", "d": 7, "k": 3, "dtype": "float64"},
        {"testcase_name": "d50_k3_f64", "d": 50, "k": 3, "dtype": "float64"},
        {"testcase_name": "d50_k25_f64", "d": 50, "k": 25, "dtype": "float64"},
    )
    def test_forward_and_adjoint_parity(self, d, k, dtype):
        """forward_L and adjoint_L via _low_rank_matvec match frozen original closures."""
        atol = 1e-6 if dtype == "float32" else 1e-12
        key = random.key(2024_01_02 + d * 100 + k)

        with _x64_ctx(dtype):
            k1, k2, k3, k4, k5 = random.split(key, 5)
            sigma = _make_positive_vector(k1, d).astype(dtype)
            U = _make_orthonormal_U(k2, d, k).astype(dtype)
            lam = _make_positive_vector(k3, k).astype(dtype)
            sqrt_lam = jnp.sqrt(lam)
            y = random.normal(k4, (d,)).astype(dtype)
            g = random.normal(k5, (d,)).astype(dtype)

            # Frozen original forward_L (verbatim from integrators.py pre-refactor)
            frozen_forward_L = lambda v: sigma * (
                v + U @ ((sqrt_lam - 1.0) * (U.T @ v))
            )

            # Frozen original adjoint_L
            def frozen_adjoint_L(grad):  # noqa: E306
                g_scaled = sigma * grad
                return g_scaled + U @ ((sqrt_lam - 1.0) * (U.T @ g_scaled))

            # New implementations using _low_rank_matvec
            new_forward_L = lambda v: sigma * metrics._low_rank_matvec(v, U, sqrt_lam)
            new_adjoint_L = lambda grad: metrics._low_rank_matvec(
                sigma * grad, U, sqrt_lam
            )

            np.testing.assert_allclose(
                new_forward_L(y), frozen_forward_L(y), atol=atol, rtol=0
            )
            np.testing.assert_allclose(
                new_adjoint_L(g), frozen_adjoint_L(g), atol=atol, rtol=0
            )

    def test_forward_adjoint_compose_to_inv_mass(self):
        """forward_L ∘ adjoint_L = M^{-1}: verifies the algebraic identity holds."""
        d, k = 8, 3
        key = random.key(2024_01_03)
        k1, k2, k3, k4 = random.split(key, 4)

        sigma = _make_positive_vector(k1, d)
        U = _make_orthonormal_U(k2, d, k)
        lam = _make_positive_vector(k3, k) + 0.1  # keep positive
        sqrt_lam = jnp.sqrt(lam)
        g = random.normal(k4, (d,))

        forward_L = lambda v: sigma * metrics._low_rank_matvec(v, U, sqrt_lam)
        adjoint_L = lambda grad: metrics._low_rank_matvec(sigma * grad, U, sqrt_lam)

        # forward_L(adjoint_L(g)) should equal M^{-1} g
        M_inv_g_via_ops = forward_L(adjoint_L(g))

        # Dense M^{-1} for reference
        M_inv = (
            jnp.diag(sigma)
            @ (jnp.eye(d) + U @ jnp.diag(lam - 1.0) @ U.T)
            @ jnp.diag(sigma)
        )
        M_inv_g_dense = M_inv @ g

        np.testing.assert_allclose(M_inv_g_via_ops, M_inv_g_dense, atol=1e-5)


class GaussianEuclideanLowRankRefactorParityTest(chex.TestCase):
    """(c) Refactored gaussian_euclidean_low_rank parity vs frozen reference implementation.

    We inline a frozen reference metric (using the original pre-refactor formulas)
    and verify that the shipped (refactored) metric produces identical outputs for
    kinetic energy, momentum sampling, is_turning, and all four scale modes.
    """

    def _make_frozen_metric(self, sigma, U, lam):
        """Frozen reference implementation of gaussian_euclidean_low_rank pre-refactor."""
        from jax.flatten_util import ravel_pytree

        from blackjax.util import generate_gaussian_noise

        inv_sigma = 1.0 / sigma
        sqrt_lam = jnp.sqrt(lam)
        inv_sqrt_lam = 1.0 / sqrt_lam

        def frozen_momentum_generator(rng_key, position):
            noise = generate_gaussian_noise(rng_key, position)
            eps, unravel_fn = ravel_pytree(noise)
            # FROZEN: original inline form
            v = eps + U @ ((inv_sqrt_lam - 1.0) * (U.T @ eps))
            return unravel_fn(inv_sigma * v)

        def frozen_kinetic_energy(momentum, position=None):
            del position
            p, _ = ravel_pytree(momentum)
            q = sigma * p
            alpha = U.T @ q
            # FROZEN: original two-term form
            return 0.5 * (jnp.dot(q, q) + jnp.dot(alpha, (lam - 1.0) * alpha))

        def frozen_is_turning(ml, mr, ms, pl=None, pr=None):
            del pl, pr
            m_left, _ = ravel_pytree(ml)
            m_right, _ = ravel_pytree(mr)
            m_sum, _ = ravel_pytree(ms)

            def _inv_mass_times(p):
                q = sigma * p
                # FROZEN: original inline form
                return sigma * (q + U @ ((lam - 1.0) * (U.T @ q)))

            vel_left = _inv_mass_times(m_left)
            vel_right = _inv_mass_times(m_right)
            rho = m_sum - (m_right + m_left) / 2
            return (jnp.dot(vel_left, rho) <= 0) | (jnp.dot(vel_right, rho) <= 0)

        def frozen_scale(position, element, *, inv, trans):
            del position
            e, unravel_fn = ravel_pytree(element)
            if not inv and not trans:
                v = e + U @ ((inv_sqrt_lam - 1.0) * (U.T @ e))
                scaled = inv_sigma * v
            elif not inv and trans:
                q = inv_sigma * e
                scaled = q + U @ ((inv_sqrt_lam - 1.0) * (U.T @ q))
            elif inv and not trans:
                v = e + U @ ((sqrt_lam - 1.0) * (U.T @ e))
                scaled = sigma * v
            else:
                q = sigma * e
                scaled = q + U @ ((sqrt_lam - 1.0) * (U.T @ q))
            return unravel_fn(scaled)

        return (
            frozen_momentum_generator,
            frozen_kinetic_energy,
            frozen_is_turning,
            frozen_scale,
        )

    @parameterized.named_parameters(
        {"testcase_name": "d2_k1_f32", "d": 2, "k": 1, "dtype": "float32"},
        {"testcase_name": "d5_k5_f32", "d": 5, "k": 5, "dtype": "float32"},
        {"testcase_name": "d7_k3_f32", "d": 7, "k": 3, "dtype": "float32"},
        {"testcase_name": "d50_k25_f32", "d": 50, "k": 25, "dtype": "float32"},
        {"testcase_name": "d2_k1_f64", "d": 2, "k": 1, "dtype": "float64"},
        {"testcase_name": "d5_k5_f64", "d": 5, "k": 5, "dtype": "float64"},
        {"testcase_name": "d7_k3_f64", "d": 7, "k": 3, "dtype": "float64"},
        {"testcase_name": "d50_k25_f64", "d": 50, "k": 25, "dtype": "float64"},
    )
    def test_metric_operations_parity(self, d, k, dtype):
        """All metric operations match frozen original across d, k, dtype."""
        from jax.flatten_util import ravel_pytree

        atol = 1e-5 if dtype == "float32" else 1e-11
        key = random.key(2024_01_04 + d * 100 + k)

        with _x64_ctx(dtype):
            k1, k2, k3, k4, k5 = random.split(key, 5)
            sigma = _make_positive_vector(k1, d).astype(dtype)
            U = _make_orthonormal_U(k2, d, k).astype(dtype)
            lam = _make_positive_vector(k3, k).astype(dtype) + 0.1
            p = random.normal(k4, (d,)).astype(dtype)

            # Build the two metric implementations
            refactored = metrics.gaussian_euclidean_low_rank(sigma, U, lam)
            (
                frozen_gen,
                frozen_ke,
                frozen_turning,
                frozen_scale,
            ) = self._make_frozen_metric(sigma, U, lam)

            # Kinetic energy
            np.testing.assert_allclose(
                float(refactored.kinetic_energy(p)),
                float(frozen_ke(p)),
                atol=atol,
                rtol=0,
                err_msg=f"kinetic_energy mismatch at d={d}, k={k}, dtype={dtype}",
            )

            # Momentum generator (deterministic given same key)
            p_refactored = refactored.sample_momentum(k5, jnp.zeros(d, dtype=dtype))
            p_ref_flat, _ = ravel_pytree(p_refactored)
            p_frz_flat, _ = ravel_pytree(frozen_gen(k5, jnp.zeros(d, dtype=dtype)))
            np.testing.assert_allclose(p_ref_flat, p_frz_flat, atol=atol, rtol=0)

            # is_turning: same result on arbitrary momenta
            p_l = random.normal(k4, (d,)).astype(dtype)
            p_r = random.normal(k5, (d,)).astype(dtype)
            p_sum = p_l + p_r
            self.assertEqual(
                bool(refactored.check_turning(p_l, p_r, p_sum)),
                bool(frozen_turning(p_l, p_r, p_sum)),
            )

            # scale: all 4 (inv, trans) modes
            e = random.normal(k4, (d,)).astype(dtype)
            pos = jnp.zeros(d, dtype=dtype)
            for inv in (False, True):
                for trans in (False, True):
                    s_new_flat, _ = ravel_pytree(
                        refactored.scale(pos, e, inv=inv, trans=trans)
                    )
                    s_frz_flat, _ = ravel_pytree(
                        frozen_scale(pos, e, inv=inv, trans=trans)
                    )
                    np.testing.assert_allclose(
                        s_new_flat,
                        s_frz_flat,
                        atol=atol,
                        rtol=0,
                        err_msg=f"scale(inv={inv}, trans={trans}) mismatch at d={d}, k={k}",
                    )


def _materialize_low_rank_metric(lrim):
    """Materialise a LowRankInverseMassMatrix to a dense (d,d) matrix M^{-1}."""
    sigma, U, lam = lrim.sigma, lrim.U, lrim.lam
    d = sigma.shape[0]
    inner = jnp.eye(d) + U @ jnp.diag(lam - 1.0) @ U.T
    return jnp.diag(sigma) @ inner @ jnp.diag(sigma)


class LbfgsAdapterParityTest(chex.TestCase):
    """(c) lbfgs_inverse_hessian_to_low_rank_metric materializes to formula_1 dense output."""

    def _make_lbfgs_factors(self, d, m, key):
        """Produce (alpha, beta, gamma) from a plausible L-BFGS history.

        Generates valid S/Z pairs by drawing S randomly and computing Z from
        a diagonal curvature matrix A, ensuring the curvature condition
        S_i^T Z_i > 0 holds for all columns.
        """
        k1, k2, k3 = random.split(key, 3)
        # Diagonal curvature (ensures PD Hessian)
        A_diag = _make_positive_vector(k1, d)
        # Position differences S: (d, m)
        S = random.normal(k2, (d, m)) * 0.1
        # Gradient differences Z = A S (Hessian-times-S)
        Z = jnp.diag(A_diag) @ S
        # Diagonal alpha: small positive values
        alpha = _make_positive_vector(k3, d)
        # Compute (beta, gamma) from lbfgs_inverse_hessian_factors
        from blackjax.optimizers.lbfgs import (
            lbfgs_inverse_hessian_factors,
            lbfgs_inverse_hessian_formula_1,
        )

        beta, gamma = lbfgs_inverse_hessian_factors(S, Z, alpha)
        return alpha, beta, gamma, lbfgs_inverse_hessian_formula_1

    @parameterized.named_parameters(
        {"testcase_name": "d4_m1", "d": 4, "m": 1},
        {"testcase_name": "d8_m3", "d": 8, "m": 3},
        {"testcase_name": "d20_m5", "d": 20, "m": 5},
        {"testcase_name": "d50_m10", "d": 50, "m": 10},
    )
    def test_adapter_parity_f64(self, d, m):
        """Adapter output materialises to formula_1 dense matrix (f64, atol=1e-9)."""
        with jax.enable_x64():
            key = random.key(2024_01_05 + d * 100 + m)
            alpha, beta, gamma, formula_1 = self._make_lbfgs_factors(d, m, key)
            alpha = alpha.astype("float64")
            beta = beta.astype("float64")
            gamma = gamma.astype("float64")

            lrim = metrics.lbfgs_inverse_hessian_to_low_rank_metric(alpha, beta, gamma)

            # Materialize the low-rank representation to a dense (d, d) matrix
            dense_lrim = _materialize_low_rank_metric(lrim)
            # Reference dense matrix from the original formula
            dense_formula = formula_1(alpha, beta, gamma)

            np.testing.assert_allclose(dense_lrim, dense_formula, atol=1e-9, rtol=0)

    @parameterized.named_parameters(
        {"testcase_name": "d4_m1", "d": 4, "m": 1},
        {"testcase_name": "d8_m3", "d": 8, "m": 3},
        {"testcase_name": "d20_m5", "d": 20, "m": 5},
    )
    def test_adapter_parity_f32(self, d, m):
        """Adapter output materialises to formula_1 dense matrix (f32, atol=1e-4)."""
        key = random.key(2024_01_06 + d * 100 + m)
        alpha, beta, gamma, formula_1 = self._make_lbfgs_factors(d, m, key)
        alpha = alpha.astype("float32")
        beta = beta.astype("float32")
        gamma = gamma.astype("float32")

        lrim = metrics.lbfgs_inverse_hessian_to_low_rank_metric(alpha, beta, gamma)

        dense_lrim = _materialize_low_rank_metric(lrim)
        dense_formula = formula_1(alpha, beta, gamma)

        # f32 accumulates more rounding error in the QR + eigh vs direct formula
        np.testing.assert_allclose(dense_lrim, dense_formula, atol=1e-4, rtol=1e-4)

    def test_empty_history_edge(self):
        """Adapter with m=0 (empty L-BFGS history) returns pure diagonal metric."""
        d = 7
        key = random.key(2024_01_07)
        alpha = _make_positive_vector(key, d)
        # Empty beta/gamma
        beta = jnp.zeros((d, 0))
        gamma = jnp.zeros((0, 0))

        lrim = metrics.lbfgs_inverse_hessian_to_low_rank_metric(alpha, beta, gamma)

        self.assertEqual(lrim.U.shape, (d, 0))
        self.assertEqual(lrim.lam.shape, (0,))
        np.testing.assert_allclose(lrim.sigma, jnp.sqrt(alpha))

        # Materialize should give diag(alpha) since U is empty (no correction)
        dense = _materialize_low_rank_metric(lrim)
        np.testing.assert_allclose(dense, jnp.diag(alpha), atol=1e-6)

    def test_sigma_equals_sqrt_alpha(self):
        """sigma field always equals sqrt(alpha) across shapes."""
        for d, m in [(2, 1), (8, 3), (50, 5)]:
            key = random.key(2024_01_08 + d)
            alpha, beta, gamma, _ = self._make_lbfgs_factors(d, m, key)
            lrim = metrics.lbfgs_inverse_hessian_to_low_rank_metric(alpha, beta, gamma)
            np.testing.assert_allclose(lrim.sigma, jnp.sqrt(alpha), atol=1e-6)

    def test_u_has_orthonormal_columns(self):
        """U columns are orthonormal (Uᵀ U = I)."""
        d, m = 20, 5
        key = random.key(2024_01_09)
        alpha, beta, gamma, _ = self._make_lbfgs_factors(d, m, key)
        lrim = metrics.lbfgs_inverse_hessian_to_low_rank_metric(alpha, beta, gamma)
        r = lrim.U.shape[-1]
        UTU = lrim.U.T @ lrim.U
        np.testing.assert_allclose(UTU, jnp.eye(r), atol=1e-5)


if __name__ == "__main__":
    absltest.main()
