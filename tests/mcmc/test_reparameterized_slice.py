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
"""Tests for the reparameterized random-direction slice sampler."""

import chex
import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest
from jax.scipy.special import logsumexp

import blackjax
from blackjax.mcmc.reparameterized_slice import (
    ReparameterizedSliceInfo,
    ReparameterizedSliceState,
    SliceRootFindingConfig,
    _bisect_root,
    _choose_bracket,
    build_kernel,
    init,
)
from tests.fixtures import BlackJAXTest


def _gaussian_params(mu, log_var):
    return jnp.concatenate((mu, log_var))


def _unpack_gaussian_params(params):
    half = params.shape[0] // 2
    return params[:half], params[half:]


def diagonal_gaussian_logdensity(x, params):
    mu, log_var = _unpack_gaussian_params(params)
    var = jnp.exp(log_var)
    return -0.5 * jnp.sum((x - mu) ** 2 / var + log_var)


def conditional_gaussian_logdensity(x, params, shift):
    mu, log_var = _unpack_gaussian_params(params)
    centered = x - (mu + shift)
    var = jnp.exp(log_var)
    return -0.5 * jnp.sum(centered**2 / var + log_var)


def split_gaussian_logdensity(x, mu, log_var):
    var = jnp.exp(log_var)
    return -0.5 * jnp.sum((x - mu) ** 2 / var + log_var)


def gaussian_logdensity_1d(x, mu, log_sigma):
    sigma = jnp.exp(log_sigma)
    return -0.5 * ((x - mu) / sigma) ** 2 - log_sigma


def standard_normal_logdensity(x):
    return -0.5 * jnp.sum(x**2)


def trimodal_mixture_logdensity(x, params):
    x = jnp.reshape(x, ())
    mu1, mu2, mu3 = params
    var1, var2, var3 = 2.0, 1.0, 1.5
    weight = jnp.log(1.0 / 3.0)
    components = jnp.array(
        [
            -0.5 * (x - mu1) ** 2 / var1 - 0.5 * jnp.log(2.0 * jnp.pi * var1) + weight,
            -0.5 * (x - mu2) ** 2 / var2 - 0.5 * jnp.log(2.0 * jnp.pi * var2) + weight,
            -0.5 * (x - mu3) ** 2 / var3 - 0.5 * jnp.log(2.0 * jnp.pi * var3) + weight,
        ]
    )
    return logsumexp(components)


def normalized_diagonal_gaussian_logdensity(x, params):
    mu, log_var = _unpack_gaussian_params(params)
    var = jnp.exp(log_var)
    return -0.5 * jnp.sum((x - mu) ** 2 / var + log_var + jnp.log(2.0 * jnp.pi))


def _finite_difference_grad(fun, x, eps=1e-4):
    basis = jnp.eye(x.shape[0], dtype=x.dtype)
    grads = []
    for direction in basis:
        grads.append(
            (fun(x + eps * direction) - fun(x - eps * direction)) / (2.0 * eps)
        )
    return jnp.stack(grads)


def _finite_difference_second(fun, x, eps=1e-3):
    x = float(x)
    return (fun(x + eps) - 2.0 * fun(x) + fun(x - eps)) / (eps * eps)


class RootFinderTest(BlackJAXTest):
    def test_bracketing_and_bisection_find_unit_roots(self):
        config = SliceRootFindingConfig(
            root_tolerance=1e-8,
            near_zero=1e-8,
            bracket_log_start=-3.0,
            bracket_log_space=0.2,
            bracket_max_steps=100,
            bisection_max_steps=100,
        )

        def func(alpha):
            return 1.0 - alpha**2

        left, right, steps = _choose_bracket(func, config)
        self.assertGreater(int(steps), 0)
        self.assertLess(float(left), -1.0)
        self.assertGreater(float(right), 1.0)

        left_root, _ = _bisect_root(
            func,
            left,
            -config.near_zero,
            config.root_tolerance,
            config.bisection_max_steps,
        )
        right_root, _ = _bisect_root(
            func,
            config.near_zero,
            right,
            config.root_tolerance,
            config.bisection_max_steps,
        )
        np.testing.assert_allclose(float(left_root), -1.0, atol=1e-5)
        np.testing.assert_allclose(float(right_root), 1.0, atol=1e-5)


class ReparameterizedSliceKernelTest(BlackJAXTest):
    def setUp(self):
        super().setUp()
        self.ndim = 2
        self.params = _gaussian_params(
            mu=jnp.array([0.5, -0.25]),
            log_var=jnp.array([0.1, -0.2]),
        )
        self.position = jnp.array([1.0, -1.0])

    def test_init_stores_logdensity(self):
        state = init(self.position, diagonal_gaussian_logdensity, self.params)
        expected = diagonal_gaussian_logdensity(self.position, self.params)
        np.testing.assert_allclose(float(state.logdensity), float(expected), atol=1e-6)

    def test_init_and_step_support_standard_no_arg_usage(self):
        algo = blackjax.reparameterized_slice(standard_normal_logdensity)
        state = algo.init(self.position, rng_key=self.next_key())
        new_state, _ = algo.step(self.next_key(), state)
        self.assertEqual(new_state.position.shape, self.position.shape)

    def test_kernel_returns_state_and_info(self):
        state = init(self.position, diagonal_gaussian_logdensity, self.params)
        kernel = build_kernel()
        new_state, info = kernel(
            self.next_key(), state, diagonal_gaussian_logdensity, self.params
        )
        self.assertIsInstance(new_state, ReparameterizedSliceState)
        self.assertIsInstance(info, ReparameterizedSliceInfo)
        np.testing.assert_allclose(
            float(new_state.logdensity),
            float(diagonal_gaussian_logdensity(new_state.position, self.params)),
            atol=1e-6,
        )

    def test_kernel_supports_pytree_positions(self):
        params = _gaussian_params(
            mu=jnp.array([0.0, 0.5]),
            log_var=jnp.zeros(2),
        )
        position = {"x": jnp.array([1.0]), "y": jnp.array([-1.0])}

        def pytree_logdensity(pos, params):
            flat = jnp.concatenate((pos["x"], pos["y"]))
            return diagonal_gaussian_logdensity(flat, params)

        state = init(position, pytree_logdensity, params)
        kernel = build_kernel()
        new_state, _ = kernel(self.next_key(), state, pytree_logdensity, params)
        chex.assert_trees_all_equal_shapes(new_state.position, position)

    def test_top_level_api_and_jit(self):
        algo = blackjax.reparameterized_slice(diagonal_gaussian_logdensity)
        state = algo.init(self.position, self.params)
        new_state, _ = jax.jit(algo.step)(self.next_key(), state, self.params)
        self.assertEqual(new_state.position.shape, (self.ndim,))

    def test_vmap_step_matches_multiple_chains_interface(self):
        algo = blackjax.reparameterized_slice(diagonal_gaussian_logdensity)
        positions = jnp.stack((self.position, -self.position))
        states = jax.vmap(algo.init, in_axes=(0, None))(positions, self.params)
        keys = jax.random.split(self.next_key(), positions.shape[0])
        step = jax.jit(jax.vmap(algo.step, in_axes=(0, 0, None)))
        new_states, infos = step(keys, states, self.params)
        self.assertEqual(new_states.position.shape, positions.shape)
        self.assertEqual(infos.alpha_left.shape, (positions.shape[0],))

    def test_extra_logdensity_args_smoke(self):
        shift = jnp.array([0.2, -0.3])
        algo = blackjax.reparameterized_slice(conditional_gaussian_logdensity)
        state = algo.init(self.position, self.params, shift)
        new_state, _ = algo.step(self.next_key(), state, self.params, shift)
        self.assertEqual(new_state.position.shape, (self.ndim,))

    def test_runtime_keyword_args_are_rejected(self):
        shift = jnp.array([0.2, -0.3])
        algo = blackjax.reparameterized_slice(conditional_gaussian_logdensity)
        with self.assertRaises(TypeError):
            _ = algo.init(self.position, self.params, shift=shift)

        state = algo.init(self.position, self.params, shift)
        with self.assertRaises(TypeError):
            _ = algo.step(self.next_key(), state, self.params, shift=shift)

    def test_standard_normal_sampling_is_reasonable(self):
        params = _gaussian_params(mu=jnp.zeros(1), log_var=jnp.zeros(1))
        algo = blackjax.reparameterized_slice(diagonal_gaussian_logdensity)
        initial_positions = jnp.zeros((256, 1))
        states = jax.vmap(algo.init, in_axes=(0, None))(initial_positions, params)
        keys = jax.random.split(self.next_key(), 256 * 40).reshape(40, 256)

        def one_step(states, keys):
            return jax.vmap(algo.step, in_axes=(0, 0, None))(keys, states, params)

        final_states, _ = jax.lax.scan(one_step, states, keys)
        samples = final_states.position[:, 0]
        self.assertLess(abs(float(jnp.mean(samples))), 0.2)
        self.assertLess(abs(float(jnp.var(samples) - 1.0)), 0.25)

    def test_trimodal_mixture_empirical_cdf_is_reasonable(self):
        """BlackJAX-style port of :cite:p:`princetonlips2021slicereparam` tests.tests.test_sampler_cdf."""

        params = jnp.array([-4.0, 0.0, 4.0])
        algo = blackjax.reparameterized_slice(trimodal_mixture_logdensity)
        num_chains = 64
        num_steps = 1_000
        initial_positions = jax.random.normal(self.next_key(), (num_chains, 1))
        states = jax.vmap(algo.init, in_axes=(0, None))(initial_positions, params)
        keys = jax.random.split(self.next_key(), num_steps * num_chains).reshape(
            num_steps, num_chains
        )

        def one_step(states, step_keys):
            new_states, _ = jax.vmap(algo.step, in_axes=(0, 0, None))(
                step_keys, states, params
            )
            return new_states, new_states.position

        _, positions = jax.lax.scan(one_step, states, keys)
        samples = positions.reshape(-1)

        dx = 0.02
        x_range = jnp.arange(-12.0, 12.0, dx)
        pdf = jnp.exp(
            jax.vmap(lambda x: trimodal_mixture_logdensity(x, params))(x_range)
        )
        numerical_cdf = jnp.cumsum(pdf / jnp.sum(pdf))
        empirical_cdf = jax.vmap(lambda x: jnp.mean(samples < x))(x_range)

        self.assertLess(float(jnp.linalg.norm(numerical_cdf - empirical_cdf)), 0.22)


class ReparameterizedSliceGradientTest(BlackJAXTest):
    def setUp(self):
        super().setUp()
        self.params = _gaussian_params(
            mu=jnp.array([0.3, -0.4]),
            log_var=jnp.array([0.1, -0.2]),
        )
        self.x0 = jnp.array([0.75, -0.5])
        self.algo = blackjax.reparameterized_slice(
            diagonal_gaussian_logdensity,
            root_tolerance=1e-7,
            near_zero=1e-8,
        )

    def _x64_gradient_test_setup(self):
        params = _gaussian_params(
            mu=jnp.array([0.3, -0.4], dtype=jnp.float64),
            log_var=jnp.array([0.1, -0.2], dtype=jnp.float64),
        )
        x0 = jnp.array([0.75, -0.5], dtype=jnp.float64)
        algo = blackjax.reparameterized_slice(
            diagonal_gaussian_logdensity,
            root_tolerance=1e-9,
            near_zero=1e-10,
        )
        return algo, params, x0

    def test_one_step_gradient_matches_finite_difference(self):
        """BlackJAX-style port of :cite:p:`princetonlips2021slicereparam` tests.tests.test_custom_vjp_finite_difference."""

        with jax.enable_x64():
            algo, params, x0 = self._x64_gradient_test_setup()
            key = jax.random.key(0)

            def loss_from_params(params):
                state = algo.init(x0, params)
                new_state, _ = algo.step(key, state, params)
                return jnp.sum(new_state.position**2)

            def loss_from_x0(x0):
                state = algo.init(x0, params)
                new_state, _ = algo.step(key, state, params)
                return jnp.sum(new_state.position**2)

            grad_params_ad = jax.grad(loss_from_params)(params)
            grad_params_fd = _finite_difference_grad(loss_from_params, params)
            grad_x0_ad = jax.grad(loss_from_x0)(x0)
            grad_x0_fd = _finite_difference_grad(loss_from_x0, x0)

            np.testing.assert_allclose(
                grad_params_ad, grad_params_fd, atol=3e-4, rtol=3e-4
            )
            np.testing.assert_allclose(grad_x0_ad, grad_x0_fd, atol=3e-4, rtol=3e-4)

    def test_scan_gradient_matches_finite_difference(self):
        """BlackJAX-style port of :cite:p:`princetonlips2021slicereparam` tests.tests.test_custom_vjp_finite_difference."""

        with jax.enable_x64():
            algo, params, x0 = self._x64_gradient_test_setup()
            keys = jax.random.split(jax.random.key(1), 4)

            def loss_from_params(params):
                state = algo.init(x0, params)

                def body_fn(state, step_key):
                    return algo.step(step_key, state, params)

                final_state, _ = jax.lax.scan(body_fn, state, keys)
                return jnp.sum(final_state.position**2)

            def loss_from_x0(x0):
                state = algo.init(x0, params)

                def body_fn(state, step_key):
                    return algo.step(step_key, state, params)

                final_state, _ = jax.lax.scan(body_fn, state, keys)
                return jnp.sum(final_state.position**2)

            grad_params_ad = jax.grad(loss_from_params)(params)
            grad_params_fd = _finite_difference_grad(loss_from_params, params)
            grad_x0_ad = jax.grad(loss_from_x0)(x0)
            grad_x0_fd = _finite_difference_grad(loss_from_x0, x0)

            np.testing.assert_allclose(
                grad_params_ad, grad_params_fd, atol=5e-4, rtol=5e-4
            )
            np.testing.assert_allclose(grad_x0_ad, grad_x0_fd, atol=5e-4, rtol=5e-4)

    def test_full_trajectory_gradient_matches_finite_difference(self):
        """BlackJAX-style port of :cite:p:`princetonlips2021slicereparam` tests.tests.test_finite_difference."""

        with jax.enable_x64():
            algo, params, _ = self._x64_gradient_test_setup()

            for seed, (num_chains, num_steps) in enumerate(((1, 1), (2, 5)), start=2):
                position_key, step_key = jax.random.split(jax.random.key(seed))
                initial_positions = jax.random.normal(
                    position_key, (num_chains, 2), dtype=jnp.float64
                )
                keys = jax.random.split(step_key, num_steps * num_chains).reshape(
                    num_steps, num_chains
                )

                def loss_from_params(params):
                    states = jax.vmap(algo.init, in_axes=(0, None))(
                        initial_positions, params
                    )

                    def body_fn(states, step_keys):
                        new_states, _ = jax.vmap(algo.step, in_axes=(0, 0, None))(
                            step_keys, states, params
                        )
                        return new_states, new_states.position

                    _, positions = jax.lax.scan(body_fn, states, keys)
                    return jnp.sum(positions**2) / num_chains

                grad_params_ad = jax.grad(loss_from_params)(params)
                grad_params_fd = _finite_difference_grad(loss_from_params, params)
                np.testing.assert_allclose(
                    grad_params_ad, grad_params_fd, atol=6e-4, rtol=6e-4
                )

    def test_diagonal_gaussian_kl_gradient_matches_analytic(self):
        """BlackJAX-style port of :cite:p:`princetonlips2021slicereparam` tests.tests.test_grad_diagonal_gaussian_KL."""

        dim = 3
        num_chains = 4_096
        params = _gaussian_params(
            mu=jnp.array([0.15, -0.1, 0.05]),
            log_var=jnp.array([0.05, -0.1, 0.08]),
        )
        algo = blackjax.reparameterized_slice(
            diagonal_gaussian_logdensity,
            root_tolerance=1e-7,
            near_zero=1e-8,
        )
        init_noise = jax.random.normal(self.next_key(), (num_chains, dim))
        step_keys = jax.random.split(self.next_key(), num_chains)

        def loss_from_params(params):
            mu, log_var = _unpack_gaussian_params(params)
            initial_positions = mu + jnp.exp(0.5 * log_var) * init_noise
            states = jax.vmap(algo.init, in_axes=(0, None))(initial_positions, params)
            new_states, _ = jax.vmap(algo.step, in_axes=(0, 0, None))(
                step_keys, states, params
            )
            xs = new_states.position
            target = jax.vmap(lambda x: -0.5 * jnp.sum(x**2 + jnp.log(2.0 * jnp.pi)))(
                xs
            )
            proposal = jax.vmap(
                normalized_diagonal_gaussian_logdensity, in_axes=(0, None)
            )(xs, params)
            return jnp.mean(proposal - target)

        def true_kl_gradient(params):
            mu, log_var = _unpack_gaussian_params(params)
            grad_mu = mu
            grad_log_var = 0.5 * (jnp.exp(log_var) - 1.0)
            return jnp.concatenate((grad_mu, grad_log_var))

        grad_estimate = jax.grad(loss_from_params)(params)
        grad_true = true_kl_gradient(params)
        np.testing.assert_allclose(grad_estimate, grad_true, atol=5e-2, rtol=2e-1)

    def test_scan_and_vmap_compose_under_jit(self):
        params = self.params
        positions = jnp.stack((self.x0, -self.x0))
        initial_states = jax.vmap(self.algo.init, in_axes=(0, None))(positions, params)
        keys = jax.random.split(self.next_key(), positions.shape[0] * 3).reshape(3, 2)

        def run_chain(states, keys):
            def body_fn(states, step_keys):
                return jax.vmap(self.algo.step, in_axes=(0, 0, None))(
                    step_keys, states, params
                )

            return jax.lax.scan(body_fn, states, keys)

        final_states, infos = jax.jit(run_chain)(initial_states, keys)
        self.assertEqual(final_states.position.shape, positions.shape)
        self.assertEqual(infos.alpha_right.shape, (3, 2))

    def test_grad_with_extra_args_runs(self):
        shift = jnp.array([0.1, -0.2])
        algo = blackjax.reparameterized_slice(
            conditional_gaussian_logdensity,
            root_tolerance=1e-7,
            near_zero=1e-8,
        )
        key = self.next_key()

        def loss(params):
            state = algo.init(self.x0, params, shift)
            new_state, _ = algo.step(key, state, params, shift)
            return jnp.sum(new_state.position**2)

        grad_val = jax.grad(loss)(self.params)
        self.assertEqual(grad_val.shape, self.params.shape)

    def test_grad_with_multiple_positional_args_runs(self):
        mu = self.params[:2]
        log_var = self.params[2:]
        algo = blackjax.reparameterized_slice(
            split_gaussian_logdensity,
            root_tolerance=1e-7,
            near_zero=1e-8,
        )
        key = self.next_key()

        def loss(mu, log_var):
            state = algo.init(self.x0, mu, log_var)
            new_state, _ = algo.step(key, state, mu, log_var)
            return jnp.sum(new_state.position**2)

        grad_mu = jax.grad(loss, argnums=0)(mu, log_var)
        grad_log_var = jax.grad(loss, argnums=1)(mu, log_var)
        self.assertEqual(grad_mu.shape, mu.shape)
        self.assertEqual(grad_log_var.shape, log_var.shape)

    def test_forward_mode_jvp_matches_grad_dot_tangent(self):
        key = self.next_key()
        tangent = jnp.array([0.2, -0.3, 0.4, -0.1])

        def loss(params):
            state = self.algo.init(self.x0, params)
            new_state, _ = self.algo.step(key, state, params)
            return jnp.sum(new_state.position**2) + 0.1 * new_state.logdensity

        value, tangent_value = jax.jvp(loss, (self.params,), (tangent,))
        grad_value = jax.grad(loss)(self.params)
        grad_dot_tangent = jnp.vdot(grad_value, tangent)

        self.assertTrue(jnp.isfinite(value))
        np.testing.assert_allclose(
            tangent_value, grad_dot_tangent, atol=1e-6, rtol=1e-6
        )

    def test_second_order_derivative_matches_finite_difference(self):
        with jax.enable_x64():
            algo = blackjax.reparameterized_slice(
                gaussian_logdensity_1d,
                root_tolerance=1e-10,
                near_zero=1e-12,
            )
            x0 = jnp.array(0.1, dtype=jnp.float64)
            mu0 = jnp.array(0.3, dtype=jnp.float64)
            log_sigma = jnp.array(-0.2, dtype=jnp.float64)
            key = self.next_key()

            def loss(mu):
                state = algo.init(x0, mu, log_sigma)
                new_state, _ = algo.step(key, state, mu, log_sigma)
                return new_state.position + 0.3 * new_state.logdensity

            hessian_value = jax.hessian(loss)(mu0)
            jacrev_grad_value = jax.jacrev(jax.grad(loss))(mu0)
            fd_value = _finite_difference_second(
                lambda mu: float(loss(jnp.array(mu, dtype=mu0.dtype))),
                float(mu0),
            )

            np.testing.assert_allclose(hessian_value, fd_value, atol=2e-4, rtol=2e-4)
            np.testing.assert_allclose(
                jacrev_grad_value, fd_value, atol=2e-4, rtol=2e-4
            )


if __name__ == "__main__":
    absltest.main()
