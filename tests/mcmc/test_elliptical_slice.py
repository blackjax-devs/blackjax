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
"""Unit tests for the Elliptical Slice Sampling kernel."""

import chex
import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest

from blackjax.mcmc.elliptical_slice import (
    EllipSliceInfo,
    EllipSliceState,
    build_kernel,
    ellipsis,
    init,
)
from tests.fixtures import BlackJAXTest, std_normal_logdensity

# ---------------------------------------------------------------------------
# ellipsis geometry
# ---------------------------------------------------------------------------


class EllipsisTest(BlackJAXTest):
    """Tests for the ellipsis helper function."""

    def test_theta_zero_returns_position(self):
        """At theta=0 the proposal equals the original position."""
        position = jnp.array([1.0, 2.0, 3.0])
        momentum = jnp.array([4.0, 5.0, 6.0])
        mean = jnp.zeros(3)
        p, m = ellipsis(position, momentum, theta=0.0, mean=mean)
        np.testing.assert_allclose(p, position, atol=1e-6)
        np.testing.assert_allclose(m, momentum, atol=1e-6)

    def test_theta_pi_negates_position(self):
        """At theta=pi the proposal is -position (centered around mean=0)."""
        position = jnp.array([1.0, 0.0])
        momentum = jnp.array([0.0, 1.0])
        mean = jnp.zeros(2)
        p, _ = ellipsis(position, momentum, theta=jnp.pi, mean=mean)
        np.testing.assert_allclose(p, -position, atol=1e-5)

    def test_theta_half_pi_swaps(self):
        """At theta=pi/2 the proposal equals the original momentum."""
        position = jnp.array([1.0, 0.0])
        momentum = jnp.array([0.0, 2.0])
        mean = jnp.zeros(2)
        p, m = ellipsis(position, momentum, theta=jnp.pi / 2, mean=mean)
        np.testing.assert_allclose(p, momentum, atol=1e-5)
        np.testing.assert_allclose(m, -position, atol=1e-5)

    def test_preserves_norm(self):
        """The L2 norm of (position - mean, momentum - mean) is preserved."""
        position = jnp.array([3.0, 1.0])
        momentum = jnp.array([-1.0, 2.0])
        mean = jnp.array([0.5, 0.5])
        theta = 1.23
        p, m = ellipsis(position, momentum, theta=theta, mean=mean)
        original_norm = jnp.sqrt(
            jnp.sum((position - mean) ** 2) + jnp.sum((momentum - mean) ** 2)
        )
        new_norm = jnp.sqrt(jnp.sum((p - mean) ** 2) + jnp.sum((m - mean) ** 2))
        np.testing.assert_allclose(float(new_norm), float(original_norm), atol=1e-5)


# ---------------------------------------------------------------------------
# init
# ---------------------------------------------------------------------------


class EllipSliceInitTest(BlackJAXTest):
    """Tests for elliptical_slice.init."""

    def test_init_stores_logdensity(self):
        """init stores the logdensity evaluated at the initial position."""
        position = jnp.array([1.0, 2.0])
        state = init(position, std_normal_logdensity)
        expected = std_normal_logdensity(position)
        np.testing.assert_allclose(float(state.logdensity), float(expected))

    def test_init_pytree_position(self):
        """init works with PyTree positions."""

        def loglik(pos):
            return -0.5 * jnp.sum(pos["a"] ** 2)

        position = {"a": jnp.ones(3)}
        state = init(position, loglik)
        self.assertIsInstance(state, EllipSliceState)
        assert jnp.isfinite(state.logdensity)


# ---------------------------------------------------------------------------
# build_kernel / kernel
# ---------------------------------------------------------------------------


class EllipSliceKernelTest(BlackJAXTest):
    """Tests for the elliptical slice kernel."""

    def test_diagonal_cov_returns_state_and_info(self):
        """Kernel with diagonal covariance returns (EllipSliceState, EllipSliceInfo)."""
        ndim = 3
        position = jnp.zeros(ndim)
        cov = jnp.ones(ndim)  # diagonal
        mean = jnp.zeros(ndim)
        state = init(position, std_normal_logdensity)
        kernel = build_kernel(cov, mean)
        new_state, info = kernel(self.next_key(), state, std_normal_logdensity)
        self.assertIsInstance(new_state, EllipSliceState)
        self.assertIsInstance(info, EllipSliceInfo)
        self.assertEqual(new_state.position.shape, (ndim,))

    def test_full_cov_returns_state(self):
        """Kernel with full covariance matrix works correctly."""
        ndim = 2
        position = jnp.zeros(ndim)
        cov = jnp.eye(ndim)
        mean = jnp.zeros(ndim)
        state = init(position, std_normal_logdensity)
        kernel = build_kernel(cov, mean)
        new_state, _ = kernel(self.next_key(), state, std_normal_logdensity)
        self.assertEqual(new_state.position.shape, (ndim,))

    def test_logdensity_consistent(self):
        """Stored logdensity matches loglikelihood at the new position."""
        ndim = 2
        position = jnp.zeros(ndim)
        cov = jnp.ones(ndim)
        mean = jnp.zeros(ndim)
        state = init(position, std_normal_logdensity)
        kernel = build_kernel(cov, mean)
        new_state, _ = kernel(self.next_key(), state, std_normal_logdensity)
        expected = std_normal_logdensity(new_state.position)
        np.testing.assert_allclose(
            float(new_state.logdensity), float(expected), atol=1e-5
        )

    def test_subiter_positive(self):
        """subiter in info is >= 1."""
        ndim = 2
        cov = jnp.ones(ndim)
        mean = jnp.zeros(ndim)
        state = init(jnp.zeros(ndim), std_normal_logdensity)
        kernel = build_kernel(cov, mean)
        _, info = kernel(self.next_key(), state, std_normal_logdensity)
        assert int(info.subiter) >= 1

    def test_accepts_for_flat_likelihood(self):
        """With a flat likelihood (constant), the chain always accepts at theta=0 is not guaranteed,
        but the position is always inside the support and logdensity is finite."""
        ndim = 3
        cov = jnp.ones(ndim)
        mean = jnp.zeros(ndim)

        def flat_loglik(x):
            return jnp.array(0.0)

        state = init(jnp.ones(ndim), flat_loglik)
        kernel = build_kernel(cov, mean)
        new_state, _ = kernel(self.next_key(), state, flat_loglik)
        assert jnp.isfinite(new_state.logdensity)

    def test_pytree_position(self):
        """Kernel works with PyTree positions."""

        def loglik_fn(pos):
            return -0.5 * (jnp.sum(pos["x"] ** 2) + jnp.sum(pos["y"] ** 2))

        ndim = 2
        position = {"x": jnp.zeros(ndim), "y": jnp.zeros(ndim)}
        # For PyTree we need flat cov: use diagonal of size 2*ndim
        cov = jnp.ones(2 * ndim)
        mean = jnp.zeros(2 * ndim)
        state = init(position, loglik_fn)
        kernel = build_kernel(cov, mean)
        new_state, info = kernel(self.next_key(), state, loglik_fn)
        chex.assert_trees_all_equal_shapes(new_state.position, position)

    def test_jit_compatible(self):
        """Kernel is JIT-compilable."""
        ndim = 3
        position = jnp.zeros(ndim)
        cov = jnp.ones(ndim)
        mean = jnp.zeros(ndim)
        state = init(position, std_normal_logdensity)
        kernel = jax.jit(build_kernel(cov, mean), static_argnums=(2,))
        new_state, _ = kernel(self.next_key(), state, std_normal_logdensity)
        self.assertEqual(new_state.position.shape, (ndim,))

    def test_wrong_cov_dim_raises(self):
        """build_kernel raises ValueError for 3-d cov_matrix."""
        import pytest

        with pytest.raises(ValueError, match="wrong number of dimensions"):
            build_kernel(jnp.ones((2, 2, 2)), jnp.zeros(2))


# ---------------------------------------------------------------------------
# Top-level API (via blackjax.elliptical_slice)
# ---------------------------------------------------------------------------


class EllipSliceTopLevelAPITest(BlackJAXTest):
    """Tests for the top-level API."""

    def test_init_and_step(self):
        """Top-level API: init + step runs and returns EllipSliceState."""
        import blackjax

        ndim = 3
        cov = jnp.eye(ndim)
        mean = jnp.zeros(ndim)
        algo = blackjax.elliptical_slice(std_normal_logdensity, mean=mean, cov=cov)
        state = algo.init(jnp.zeros(ndim))
        new_state, info = algo.step(self.next_key(), state)
        self.assertIsInstance(new_state, EllipSliceState)
        self.assertEqual(new_state.position.shape, (ndim,))

    def test_top_level_jit(self):
        """Top-level step is JIT-compilable."""
        import blackjax

        ndim = 2
        cov = jnp.eye(ndim)
        mean = jnp.zeros(ndim)
        algo = blackjax.elliptical_slice(std_normal_logdensity, mean=mean, cov=cov)
        state = algo.init(jnp.zeros(ndim))
        new_state, _ = jax.jit(algo.step)(self.next_key(), state)
        self.assertEqual(new_state.position.shape, (ndim,))


if __name__ == "__main__":
    absltest.main()
