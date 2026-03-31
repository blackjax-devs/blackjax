# Copyright 2020- The Blackjax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Unit tests for SGMCMC gradient estimators."""
import chex
import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest

import blackjax.sgmcmc.gradients as gradients

# ---------------------------------------------------------------------------
# Simple model: Gaussian prior + Gaussian likelihood
# ---------------------------------------------------------------------------


def _logprior_fn(position):
    """Standard normal prior log-density."""
    return -0.5 * jnp.sum(position**2)


def _loglikelihood_fn(position, x):
    """Gaussian likelihood: log p(x | theta) = -0.5*(x - theta)^2."""
    return -0.5 * jnp.sum((x - position) ** 2)


class LogdensityEstimatorTest(chex.TestCase):
    """Tests for the logdensity_estimator function."""

    def setUp(self):
        self.data_size = 100
        self.estimator = gradients.logdensity_estimator(
            _logprior_fn, _loglikelihood_fn, self.data_size
        )

    def test_returns_scalar(self):
        """Estimator returns a scalar value."""
        position = jnp.zeros(2)
        minibatch = jnp.ones((10, 2))
        result = self.estimator(position, minibatch)
        self.assertEqual(result.shape, ())

    def test_finite_value(self):
        """Estimator returns a finite value for a simple input."""
        position = jnp.zeros(2)
        minibatch = jnp.zeros((5, 2))
        result = self.estimator(position, minibatch)
        assert jnp.isfinite(result)

    def test_scales_with_data_size(self):
        """Likelihood term is scaled by data_size / batch_size."""
        position = jnp.zeros(1)
        minibatch = jnp.ones((10, 1))

        est_100 = gradients.logdensity_estimator(_logprior_fn, _loglikelihood_fn, 100)
        est_200 = gradients.logdensity_estimator(_logprior_fn, _loglikelihood_fn, 200)

        val_100 = est_100(position, minibatch)
        val_200 = est_200(position, minibatch)

        # Prior is the same; likelihood is scaled differently
        # val_200 - val_100 should equal (200 - 100) * mean_loglik
        mean_loglik = float(
            jnp.mean(
                jax.vmap(_loglikelihood_fn, in_axes=(None, 0))(position, minibatch)
            )
        )
        np.testing.assert_allclose(
            float(val_200 - val_100), 100 * mean_loglik, rtol=1e-5
        )

    def test_jit_compatible(self):
        """Estimator is JIT-compilable."""
        position = jnp.zeros(3)
        minibatch = jnp.ones((8, 3))
        result = jax.jit(self.estimator)(position, minibatch)
        assert jnp.isfinite(result)

    def test_works_with_pytree_position(self):
        """Estimator works with PyTree positions."""

        def logprior_pytree(pos):
            return -0.5 * (jnp.sum(pos["w"] ** 2) + jnp.sum(pos["b"] ** 2))

        def loglik_pytree(pos, x):
            return -0.5 * jnp.sum((x - pos["w"]) ** 2)

        est = gradients.logdensity_estimator(logprior_pytree, loglik_pytree, 100)
        position = {"w": jnp.zeros(2), "b": jnp.zeros(1)}
        minibatch = jnp.ones((5, 2))
        result = est(position, minibatch)
        assert jnp.isfinite(result)


class GradEstimatorTest(chex.TestCase):
    """Tests for the grad_estimator function."""

    def setUp(self):
        self.data_size = 100
        self.grad_est = gradients.grad_estimator(
            _logprior_fn, _loglikelihood_fn, self.data_size
        )

    def test_gradient_same_shape_as_position(self):
        """Gradient has the same shape as the position."""
        position = jnp.zeros(3)
        minibatch = jnp.ones((10, 3))
        grad = self.grad_est(position, minibatch)
        self.assertEqual(grad.shape, position.shape)

    def test_gradient_is_finite(self):
        """Gradient is finite for simple inputs."""
        position = jnp.ones(2)
        minibatch = jnp.zeros((5, 2))
        grad = self.grad_est(position, minibatch)
        assert jnp.all(jnp.isfinite(grad))

    def test_gradient_direction_at_zero(self):
        """At position=0 with data>0, gradient should be positive (pushed toward data)."""
        position = jnp.zeros(1)
        # Data all equal to 2.0; posterior mode is pulled toward 2.0
        minibatch = 2.0 * jnp.ones((10, 1))
        grad = self.grad_est(position, minibatch)
        assert float(grad[0]) > 0.0

    def test_jit_compatible(self):
        """Gradient estimator is JIT-compilable."""
        position = jnp.zeros(2)
        minibatch = jnp.ones((5, 2))
        grad = jax.jit(self.grad_est)(position, minibatch)
        self.assertEqual(grad.shape, (2,))

    def test_pytree_position_gradient(self):
        """Gradient estimator works with PyTree positions."""

        def logprior_pytree(pos):
            return -0.5 * jnp.sum(pos["w"] ** 2)

        def loglik_pytree(pos, x):
            return -0.5 * jnp.sum((x - pos["w"]) ** 2)

        grad_est = gradients.grad_estimator(logprior_pytree, loglik_pytree, 50)
        position = {"w": jnp.zeros(2)}
        minibatch = jnp.ones((5, 2))
        grad = grad_est(position, minibatch)
        chex.assert_trees_all_equal_shapes(position, grad)


class ControlVariatesTest(chex.TestCase):
    """Tests for the control_variates gradient estimator."""

    def setUp(self):
        self.data_size = 100
        self.base_grad_est = gradients.grad_estimator(
            _logprior_fn, _loglikelihood_fn, self.data_size
        )
        self.full_data = jnp.zeros((self.data_size, 2))
        # Use MAP (zero for standard normal prior + zero-mean data) as centering
        self.centering_position = jnp.zeros(2)
        self.cv_estimator = gradients.control_variates(
            self.base_grad_est, self.centering_position, self.full_data
        )

    def test_gradient_same_shape_as_position(self):
        """CV gradient has the same shape as position."""
        position = jnp.ones(2)
        minibatch = jnp.zeros((10, 2))
        grad = self.cv_estimator(position, minibatch)
        self.assertEqual(grad.shape, position.shape)

    def test_gradient_finite(self):
        """CV gradient is finite for simple inputs."""
        position = jnp.array([0.5, -0.5])
        minibatch = jnp.zeros((10, 2))
        grad = self.cv_estimator(position, minibatch)
        assert jnp.all(jnp.isfinite(grad))

    def test_centering_position_gives_full_data_gradient(self):
        """At the centering position, CV gradient equals base gradient on full data."""
        position = self.centering_position
        grad_cv = self.cv_estimator(position, self.full_data)
        grad_base = self.base_grad_est(position, self.full_data)
        # At centering position, base_grad(pos, batch) - base_grad(center, batch)
        # = 0, so result = cv_grad_value = base_grad(center, full_data)
        np.testing.assert_allclose(grad_cv, grad_base, atol=1e-5)

    def test_jit_compatible(self):
        """CV estimator is JIT-compilable."""
        position = jnp.ones(2)
        minibatch = jnp.zeros((10, 2))
        grad = jax.jit(self.cv_estimator)(position, minibatch)
        self.assertEqual(grad.shape, (2,))


if __name__ == "__main__":
    absltest.main()
