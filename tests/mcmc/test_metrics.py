import chex
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest, parameterized
from jax import random
from jax.scipy import linalg

from blackjax.mcmc import metrics


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
        self.key = random.PRNGKey(0)
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


if __name__ == "__main__":
    absltest.main()
