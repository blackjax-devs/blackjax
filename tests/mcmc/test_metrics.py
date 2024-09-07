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
        {"testcase_name": "0d", "shape": (), "get_inv": False},
        {"testcase_name": "0d_inv", "shape": (), "get_inv": True},
        {"testcase_name": "3d", "shape": (1, 2, 3), "get_inv": False},
        {"testcase_name": "3d_inv", "shape": (1, 2, 3), "get_inv": True},
    )
    def test_invalid(self, shape, get_inv):
        """Test formatting raises error for invalid shapes"""
        mass_matrix = jnp.zeros(shape=shape)
        with self.assertRaisesRegex(
                ValueError, "The mass matrix has the wrong number of dimensions"
        ):
            metrics._format_covariance(mass_matrix, get_inv)

    @parameterized.named_parameters(
        {"testcase_name": "inv", "get_inv": True},
        {"testcase_name": "no_inv", "get_inv": False},
    )
    def test_dim_1(self, get_inv):
        """Test formatting for 1D mass matrix"""
        mass_matrix = jnp.asarray([1 / 4], dtype=self.dtype)
        mass_matrix_sqrt, inv_mass_matrix_sqrt, diag = metrics._format_covariance(
            mass_matrix, get_inv
        )
        chex.assert_trees_all_close(mass_matrix_sqrt, mass_matrix ** 0.5)
        if get_inv:
            chex.assert_trees_all_close(inv_mass_matrix_sqrt, mass_matrix ** -0.5)
        else:
            assert inv_mass_matrix_sqrt is None

        chex.assert_trees_all_close(diag(mass_matrix), mass_matrix)

    @parameterized.named_parameters(
        {"testcase_name": "inv", "get_inv": True},
        {"testcase_name": "no_inv", "get_inv": False},
    )
    def test_dim_2(self, get_inv):
        """Test formatting for 2D mass matrix"""
        mass_matrix = jnp.asarray([[1 / 9, 0.5], [0.5, 1 / 4]], dtype=self.dtype)
        mass_matrix_sqrt, inv_mass_matrix_sqrt, diag = metrics._format_covariance(
            mass_matrix, get_inv
        )
        chex.assert_trees_all_close(
            mass_matrix_sqrt, linalg.cholesky(mass_matrix, lower=True)
        )
        if get_inv:
            chex.assert_trees_all_close(
                inv_mass_matrix_sqrt, linalg.inv(mass_matrix_sqrt)
            )
        else:
            assert inv_mass_matrix_sqrt is None

        chex.assert_trees_all_close(diag(mass_matrix), jnp.diag(mass_matrix))


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

        inv_scaled_momentum, scaled_momentum = scale(
            arbitrary_position, ((momentum_val, True), (momentum_val, False))
        )
        expected_scaled_momentum = momentum_val / jnp.sqrt(inverse_mass_matrix)
        expected_inv_scaled_momentum = momentum_val * jnp.sqrt(inverse_mass_matrix)

        chex.assert_trees_all_close(inv_scaled_momentum, expected_inv_scaled_momentum)
        chex.assert_trees_all_close(scaled_momentum, expected_scaled_momentum)

    @chex.all_variants(with_pmap=False)
    def test_gaussian_euclidean_dim_2(self):
        """Test Gaussian Euclidean Function with ndim 2"""
        inverse_mass_matrix = jnp.asarray(
            [[1 / 9, 0.5], [0.5, 1 / 4]], dtype=self.dtype
        )
        momentum, kinetic_energy, _, scale = metrics.gaussian_euclidean(
            inverse_mass_matrix
        )

        arbitrary_position = jnp.asarray([12345, 23456], dtype=self.dtype)
        momentum_val = self.variant(momentum)(self.key, arbitrary_position)

        L_inv = linalg.cholesky(linalg.inv(inverse_mass_matrix), lower=True)
        expected_momentum_val = L_inv @ random.normal(self.key, shape=(2,))

        kinetic_energy_val = self.variant(kinetic_energy)(momentum_val)
        velocity = jnp.dot(inverse_mass_matrix, momentum_val)
        expected_kinetic_energy_val = 0.5 * jnp.matmul(velocity, momentum_val)

        np.testing.assert_allclose(expected_momentum_val, momentum_val)
        np.testing.assert_allclose(kinetic_energy_val, expected_kinetic_energy_val)

        inv_scaled_momentum, scaled_momentum = scale(
            arbitrary_position, ((momentum_val, True), (momentum_val, False))
        )
        expected_inv_scaled_momentum = momentum_val @ linalg.cholesky(
            inverse_mass_matrix, lower=True
        )
        expected_scaled_momentum = momentum_val @ linalg.inv(
            linalg.cholesky(inverse_mass_matrix, lower=True)
        )

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

        inv_scaled_momentum, scaled_momentum = scale(
            arbitrary_position, ((momentum_val, True), (momentum_val, False))
        )
        expected_scaled_momentum = momentum_val / jnp.sqrt(inverse_mass_matrix)
        expected_inv_scaled_momentum = momentum_val * jnp.sqrt(inverse_mass_matrix)

        chex.assert_trees_all_close(inv_scaled_momentum, expected_inv_scaled_momentum)
        chex.assert_trees_all_close(scaled_momentum, expected_scaled_momentum)

    @chex.all_variants(with_pmap=False)
    def test_gaussian_euclidean_dim_2(self):
        inverse_mass_matrix = jnp.asarray(
            [[1 / 9, 0.5], [0.5, 1 / 4]], dtype=self.dtype
        )
        mass_matrix = jnp.linalg.inv(inverse_mass_matrix)
        momentum, kinetic_energy, _, scale = metrics.gaussian_riemannian(
            lambda _: mass_matrix
        )

        arbitrary_position = jnp.asarray([12345, 23456], dtype=self.dtype)
        momentum_val = self.variant(momentum)(self.key, arbitrary_position)

        L_inv = linalg.cholesky(linalg.inv(inverse_mass_matrix), lower=True)
        expected_momentum_val = L_inv @ random.normal(self.key, shape=(2,))

        kinetic_energy_val = self.variant(kinetic_energy)(
            momentum_val, position=arbitrary_position
        )
        velocity = jnp.dot(inverse_mass_matrix, momentum_val)
        expected_kinetic_energy_val = 0.5 * jnp.matmul(velocity, momentum_val)
        expected_kinetic_energy_val += 0.5 * jnp.linalg.slogdet(mass_matrix)[1]
        expected_kinetic_energy_val += 0.5 * len(mass_matrix) * jnp.log(2 * jnp.pi)

        np.testing.assert_allclose(expected_momentum_val, momentum_val)
        np.testing.assert_allclose(kinetic_energy_val, expected_kinetic_energy_val)

        inv_scaled_momentum, scaled_momentum = scale(
            arbitrary_position, ((momentum_val, True), (momentum_val, False))
        )
        expected_inv_scaled_momentum = momentum_val @ linalg.cholesky(
            inverse_mass_matrix, lower=True
        )
        expected_scaled_momentum = momentum_val @ linalg.inv(
            linalg.cholesky(inverse_mass_matrix, lower=True)
        )

        chex.assert_trees_all_close(inv_scaled_momentum, expected_inv_scaled_momentum)
        chex.assert_trees_all_close(scaled_momentum, expected_scaled_momentum)


if __name__ == "__main__":
    absltest.main()
