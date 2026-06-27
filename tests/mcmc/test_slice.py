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
"""Tests for the unified Slice sampling family."""
import chex
import jax
import jax.numpy as jnp
import jax.scipy.stats as st
import numpy as np
from absl.testing import absltest, parameterized

import blackjax
from blackjax.mcmc.slice import (
    SliceInfo,
    SliceState,
    build_coordinate_kernel,
    build_kernel,
    coordinate_slice,
    direction_proposal,
    doubling,
    fixed_order,
    init,
    random_order,
    sample_direction,
    stepping_out,
)


def std_normal(x):
    return st.norm.logpdf(x).sum()


def run_chain(algo, position, key, n):
    state = algo.init(position)

    def body(s, k):
        s, info = algo.step(k, s)
        return s, (s.position, info.is_accepted)

    _, (positions, accepted) = jax.lax.scan(body, state, jax.random.split(key, n))
    return positions, np.asarray(accepted)


INTERVALS = [doubling, stepping_out]  # interval procedures are passed as callables


class SliceInitTest(chex.TestCase):
    def test_init_stores_position_and_logdensity(self):
        position = jnp.array([1.0, 2.0, 3.0])
        state = init(position, std_normal)
        np.testing.assert_allclose(state.position, position)
        np.testing.assert_allclose(float(state.logdensity), float(std_normal(position)))

    def test_init_pytree(self):
        logp = lambda p: std_normal(p["a"]) + std_normal(p["b"])
        position = {"a": jnp.ones(2), "b": jnp.zeros(3)}
        state = init(position, logp)
        self.assertIsInstance(state, SliceState)
        chex.assert_trees_all_equal_shapes(state.position, position)


class IntervalProcedureTest(chex.TestCase):
    """The interval finders must bracket the slice and contain the origin (x0)."""

    @parameterized.parameters(*INTERVALS)
    def test_brackets_contain_origin_and_slice(self, interval):
        in_slice = lambda t: jnp.abs(t) < 3.0  # x0 at t=0 is inside
        left, right, _, _ = interval(
            jax.random.key(0), in_slice, width=1.0, max_expansions=10
        )
        self.assertLessEqual(float(left), 0.0)
        self.assertGreaterEqual(float(right), 0.0)
        self.assertTrue(float(left) < -3.0 or float(right) > 3.0)


class SingleStepTest(chex.TestCase):
    @parameterized.parameters(*INTERVALS)
    def test_multivariate_step_shapes(self, interval):
        kernel = build_kernel(interval=interval)
        state = init(jnp.zeros(3), std_normal)
        new_state, info = kernel(
            jax.random.key(0), state, std_normal, direction_proposal(), 1.0
        )
        self.assertIsInstance(new_state, SliceState)
        self.assertIsInstance(info, SliceInfo)
        self.assertEqual(new_state.position.shape, (3,))
        np.testing.assert_allclose(
            float(new_state.logdensity),
            float(std_normal(new_state.position)),
            atol=1e-6,
        )

    @parameterized.parameters(*INTERVALS)
    def test_coordinate_step_shapes(self, interval):
        kernel = build_coordinate_kernel(interval=interval)
        state = init(jnp.zeros(4), std_normal)
        new_state, info = kernel(jax.random.key(0), state, std_normal)
        self.assertEqual(new_state.position.shape, (4,))
        chex.assert_trees_all_equal_shapes(info.bracket_left, new_state.position)

    def test_jit(self):
        algo = blackjax.slice_sampling(std_normal)
        state = algo.init(jnp.zeros(3))
        ns, _ = jax.jit(algo.step)(jax.random.key(0), state)
        self.assertEqual(ns.position.shape, (3,))


class DirectionProposalTest(chex.TestCase):
    """``scale`` (scalar / vector / dense) shapes a unit-norm random direction."""

    @parameterized.parameters(
        (1.0,),  # scalar (isotropic, the default)
        (2.5,),  # scalar
        (np.array([0.5, 2.0, 1.0]),),  # vector (per-coordinate / diagonal)
        (np.array([[2.0, 0.5, 0.0], [0.0, 1.0, 0.3], [0.0, 0.0, 1.5]]),),  # dense
    )
    def test_direction_is_unit_norm(self, scale):
        d = sample_direction(jax.random.key(0), jnp.zeros(3), jnp.asarray(scale))
        np.testing.assert_allclose(float(jnp.linalg.norm(d)), 1.0, atol=1e-6)

    def test_vector_scale_biases_direction(self):
        # A large scale on one axis tilts the direction toward that axis.
        scale = jnp.array([10.0, 0.1])
        keys = jax.random.split(jax.random.key(1), 400)
        ds = jax.vmap(lambda k: sample_direction(k, jnp.zeros(2), scale))(keys)
        mean_abs = jnp.mean(jnp.abs(ds), axis=0)
        self.assertGreater(float(mean_abs[0]), float(mean_abs[1]))


class MomentRecoveryTest(chex.TestCase):
    """Statistical correctness: every sampler/interval recovers known moments."""

    @parameterized.parameters(*INTERVALS)
    def test_multivariate_isotropic_std_normal(self, interval):
        algo = blackjax.slice_sampling(std_normal, interval=interval)
        pos, acc = run_chain(algo, jnp.zeros(3), jax.random.key(1), 5000)
        s = np.asarray(pos[2500:])
        np.testing.assert_allclose(s.mean(), 0.0, atol=0.1)
        np.testing.assert_allclose(s.std(), 1.0, rtol=0.12)
        self.assertTrue(acc.all())

    @parameterized.parameters(*INTERVALS)
    def test_coordinate_std_normal(self, interval):
        algo = coordinate_slice(std_normal, interval=interval)
        pos, acc = run_chain(algo, jnp.zeros(3), jax.random.key(1), 4000)
        s = np.asarray(pos[2000:])
        np.testing.assert_allclose(s.mean(), 0.0, atol=0.1)
        np.testing.assert_allclose(s.std(), 1.0, rtol=0.12)
        self.assertTrue(acc.all())

    @parameterized.parameters(*INTERVALS)
    def test_multivariate_correlated_gaussian(self, interval):
        Sigma = jnp.array([[2.0, 1.2], [1.2, 1.0]])
        Sinv = jnp.linalg.inv(Sigma)
        logp = lambda x: -0.5 * x @ Sinv @ x
        # Precondition the direction with a dense scale (Cholesky factor of Sigma).
        scale = jnp.linalg.cholesky(Sigma)
        algo = blackjax.slice_sampling(
            logp, proposal_generator=direction_proposal(scale), interval=interval
        )
        pos, _ = run_chain(algo, jnp.zeros(2), jax.random.key(2), 6000)
        emp = np.cov(np.asarray(pos[3000:]), rowvar=False)
        np.testing.assert_allclose(emp, np.asarray(Sigma), atol=0.45)

    def test_nonzero_mean(self):
        logp = lambda x: st.norm.logpdf(x - 3.0).sum()
        algo = blackjax.slice_sampling(logp)
        pos, _ = run_chain(algo, jnp.full(2, 3.0), jax.random.key(3), 4000)
        np.testing.assert_allclose(np.asarray(pos[2000:]).mean(), 3.0, atol=0.12)


class ConstraintTest(chex.TestCase):
    """Constrained slice: a half-normal via constraint_fn, both samplers."""

    def _check_halfnormal(self, algo):
        pos, acc = run_chain(algo, jnp.array([1.0]), jax.random.key(4), 5000)
        s = np.asarray(pos[2500:])
        self.assertTrue((s > 0).all())
        np.testing.assert_allclose(s.mean(), np.sqrt(2 / np.pi), atol=0.08)
        self.assertGreater(acc.mean(), 0.99)

    def test_multivariate_constrained(self):
        # Constraints are added by *overriding the proposal* (the NSS hook), not
        # by built-in machinery: gate is_valid in a custom slice_fn (here x > 0).
        def proposal(rng_key, position, logdensity_fn):
            base = direction_proposal()(rng_key, position, logdensity_fn)

            def slice_fn(t):
                state, is_valid = base(t)
                return state, is_valid & jnp.all(state.position > 0)

            return slice_fn

        self._check_halfnormal(
            blackjax.slice_sampling(std_normal, proposal_generator=proposal)
        )

    def test_coordinate_constrained(self):
        self._check_halfnormal(
            coordinate_slice(std_normal, constraint_fn=lambda x: jnp.all(x > 0))
        )


class PyTreeTest(chex.TestCase):
    def _logp(self, p):
        return st.norm.logpdf(p["a"]).sum() + st.norm.logpdf(p["b"] - 2.0).sum()

    def test_multivariate_pytree(self):
        position = {"a": jnp.zeros(2), "b": jnp.zeros(1)}
        algo = blackjax.slice_sampling(self._logp)
        pos, _ = run_chain(algo, position, jax.random.key(5), 4000)
        np.testing.assert_allclose(np.asarray(pos["a"])[2000:].mean(), 0.0, atol=0.12)
        np.testing.assert_allclose(np.asarray(pos["b"])[2000:].mean(), 2.0, atol=0.12)

    def test_coordinate_pytree(self):
        position = {"a": jnp.zeros(2), "b": jnp.zeros(1)}
        algo = coordinate_slice(self._logp)
        pos, _ = run_chain(algo, position, jax.random.key(5), 4000)
        np.testing.assert_allclose(np.asarray(pos["b"])[2000:].mean(), 2.0, atol=0.12)


class CoordinateOrderTest(chex.TestCase):
    @parameterized.parameters(random_order, fixed_order)
    def test_order_procedures_recover_moments(self, order):
        algo = coordinate_slice(std_normal, coordinate_order=order)
        pos, _ = run_chain(algo, jnp.zeros(3), jax.random.key(6), 4000)
        np.testing.assert_allclose(np.asarray(pos[2000:]).mean(), 0.0, atol=0.1)

    def test_per_dimension_widths(self):
        algo = coordinate_slice(std_normal, initial_widths=jnp.array([0.5, 2.0, 5.0]))
        pos, _ = run_chain(algo, jnp.zeros(3), jax.random.key(6), 4000)
        np.testing.assert_allclose(np.asarray(pos[2000:]).std(), 1.0, rtol=0.15)


class TopLevelAPITest(chex.TestCase):
    def test_top_level_is_multivariate(self):
        # blackjax.slice_sampling is the multivariate slice sampler.
        algo = blackjax.slice_sampling(
            std_normal, proposal_generator=direction_proposal(jnp.eye(2)), width=2.0
        )
        state = algo.init(jnp.zeros(2))
        _, info = algo.step(jax.random.key(8), state)
        self.assertIsInstance(info, SliceInfo)
        # A single multivariate move: the bracket is a scalar t-offset.
        self.assertEqual(np.asarray(info.bracket_left).ndim, 0)

    def test_default_is_isotropic_multivariate(self):
        # No cov -> isotropic multivariate slice; recovers a standard normal.
        algo = blackjax.slice_sampling(std_normal)
        pos, _ = run_chain(algo, jnp.zeros(2), jax.random.key(7), 3000)
        np.testing.assert_allclose(np.asarray(pos[1500:]).mean(), 0.0, atol=0.15)


if __name__ == "__main__":
    absltest.main()
