"""Tests for the GIST self-tuning trajectory-length sampler (no-U-turn,
NOT NUTS's recursive doubling).

CI note: run with ``--benchmark-disable`` when parallelizing under xdist
(e.g. ``-n 2``) -- a bare ``-n 2`` run hits an ``INTERNALERROR`` from
pytest-benchmark x xdist under this project's ``filterwarnings = error``
(the same masking-bug family as the probdiffeq migration saga). Not a code
issue in this module; the documented blackjax test command already
includes ``--benchmark-disable``.
"""
import chex
import jax
import jax.numpy as jnp
import jax.scipy.stats as st
import numpy as np
from absl.testing import absltest, parameterized

import blackjax
from blackjax.mcmc import gist, gist_trajectory_length, integrators, metrics
from tests.fixtures import (
    BlackJAXTest,
    assert_mean_within_ess_gated_tolerance,
    neal_funnel_logdensity,
    smooth_skewed_logdensity,
    std_normal_logdensity,
)


def run_chain(algo, position, key, n):
    state = algo.init(position)

    def body(s, k):
        s, info = algo.step(k, s)
        return s, (s.position, info)

    _, (positions, infos) = jax.lax.scan(body, state, jax.random.split(key, n))
    return positions, infos


class InitTest(chex.TestCase):
    def test_init_stores_position_and_gradients(self):
        position = jnp.array([1.0, 2.0])
        state = gist_trajectory_length.init(position, std_normal_logdensity)
        self.assertIsInstance(state, gist.GISTState)
        np.testing.assert_allclose(state.position, position)
        np.testing.assert_allclose(
            float(state.logdensity), float(std_normal_logdensity(position))
        )


class SingleStepTest(chex.TestCase):
    @parameterized.parameters(0.0, 0.5)
    def test_step_shapes_and_types(self, path_fraction):
        algo = blackjax.gist_trajectory_length(
            std_normal_logdensity,
            inverse_mass_matrix=jnp.ones(3),
            step_size=0.3,
            path_fraction=path_fraction,
        )
        state = algo.init(jnp.zeros(3))
        new_state, info = algo.step(jax.random.key(0), state)
        self.assertIsInstance(new_state, gist.GISTState)
        self.assertIsInstance(info, gist_trajectory_length.GISTTrajectoryLengthInfo)
        self.assertEqual(new_state.position.shape, (3,))
        np.testing.assert_allclose(
            float(new_state.logdensity),
            float(std_normal_logdensity(new_state.position)),
            atol=1e-5,
        )

    def test_jit(self):
        algo = blackjax.gist_trajectory_length(
            std_normal_logdensity, inverse_mass_matrix=jnp.ones(3), step_size=0.2
        )
        state = algo.init(jnp.zeros(3))
        new_state, _ = jax.jit(algo.step)(jax.random.key(0), state)
        self.assertEqual(new_state.position.shape, (3,))


class CompilationTest(chex.TestCase):
    def test_no_excess_retracing(self):
        """The logdensity should compile at most 4 times: init, plus 3
        within one kernel trace -- the forward U-turn rollout, the accepted
        trajectory build, and the reverse U-turn rollout (section 2.2.4)
        each need their own gradient evaluation, unlike hmc/nuts's single
        forward trajectory (n=2 there). Verified empirically: the count
        stabilizes at 4 after the first `step()` call and does not grow on
        further calls with the same shapes.
        """

        @chex.assert_max_traces(n=4)
        def logdensity_fn(x):
            return jnp.sum(st.norm.logpdf(x))

        chex.clear_trace_counter()

        algo = blackjax.gist_trajectory_length(
            logdensity_fn, inverse_mass_matrix=jnp.ones(2), step_size=0.3
        )
        state = algo.init(jnp.zeros(2))
        step = jax.jit(algo.step)

        rng_key = jax.random.key(0)
        for i in range(5):
            sample_key = jax.random.fold_in(rng_key, i)
            state, _ = step(sample_key, state)


class StationarityTest(BlackJAXTest):
    """If the population starts exactly at stationarity, it should stay
    there (modeled on ``tests/mcmc/test_barker.py::test_invariance``)."""

    @parameterized.parameters(0.0, 0.5)
    def test_stationarity_from_exact_draws(self, path_fraction):
        d = 2
        n_samples, m_steps = 1500, 15

        algo = blackjax.gist_trajectory_length(
            std_normal_logdensity,
            inverse_mass_matrix=jnp.ones(d),
            step_size=0.5,
            path_fraction=path_fraction,
            max_num_steps=64,
        )

        init_key, inference_key = jax.random.split(self.next_key())
        init_samples = jax.random.normal(init_key, shape=(n_samples, d))
        inference_keys = jax.random.split(inference_key, n_samples)

        def loop(state, key_):
            state, _ = algo.step(key_, state)
            return state, None

        def get_samples(init_sample, key_):
            state = algo.init(init_sample)
            out, _ = jax.lax.scan(loop, state, jax.random.split(key_, m_steps))
            return out.position

        samples = jax.vmap(get_samples)(init_samples, inference_keys)
        chex.assert_trees_all_close(
            jnp.mean(samples, axis=0), jnp.zeros((d,)), atol=0.15, rtol=0.15
        )
        chex.assert_trees_all_close(jnp.cov(samples.T), jnp.eye(d), atol=0.2, rtol=0.2)


class MomentRecoveryTest(BlackJAXTest):
    def test_isotropic_std_normal(self):
        algo = blackjax.gist_trajectory_length(
            std_normal_logdensity, inverse_mass_matrix=jnp.ones(3), step_size=0.4
        )
        pos, infos = run_chain(algo, jnp.zeros(3), self.next_key(), 3000)
        s = np.asarray(pos[1500:])
        np.testing.assert_allclose(s.mean(), 0.0, atol=0.12)
        np.testing.assert_allclose(s.std(), 1.0, rtol=0.15)
        self.assertGreater(float(jnp.mean(infos.acceptance_rate)), 0.05)

    def test_correlated_gaussian_dense_metric(self):
        # Exercises a non-identity (dense) inverse_mass_matrix -- the direct
        # test of the metric-generalization decision for num_steps_to_uturn.
        Sigma = jnp.array([[2.0, 1.2], [1.2, 1.0]])
        Sinv = jnp.linalg.inv(Sigma)
        logp = lambda x: -0.5 * x @ Sinv @ x
        algo = blackjax.gist_trajectory_length(
            logp, inverse_mass_matrix=Sigma, step_size=0.3
        )
        pos, _ = run_chain(algo, jnp.zeros(2), self.next_key(), 4000)
        emp = np.cov(np.asarray(pos[2000:]), rowvar=False)
        np.testing.assert_allclose(emp, np.asarray(Sigma), atol=0.6)

    def test_smooth_skewed_target(self):
        # Asymmetric target: the single highest-value test for this
        # instance's rollout direction -- a symmetric Gaussian cannot
        # detect a reversed-direction/sign bug in the forward-vs-reverse
        # U-turn rollout or the momentum flip (mirrors
        # test_slice.py::test_multivariate_skewed_exponential's rationale,
        # but with a smooth log-space target -- see
        # tests/fixtures.py::smooth_skewed_logdensity for why the raw
        # Exponential representation is a poor fit for a gradient-based
        # sampler: its zero gradient outside the support can make a
        # boundary-crossing trajectory drift forever, degenerate for the
        # no-U-turn rollout specifically).
        algo = blackjax.gist_trajectory_length(
            smooth_skewed_logdensity, inverse_mass_matrix=jnp.ones(2), step_size=0.3
        )
        pos, infos = run_chain(algo, jnp.zeros(2), self.next_key(), 6000)
        s = np.asarray(pos[3000:])
        # The mean/std bounds are the actual reversed-direction/sign-bug
        # detector here (confirmed empirically against a planted direction
        # bug in review: a broken rollout inflates std well outside this
        # band, well before the skew sign would flip). The skew check below
        # is a secondary correctness check, tightened to a band around the
        # closed-form truth (-1.1395) rather than a bare sign check, so it
        # earns its own assertion.
        np.testing.assert_allclose(s.mean(axis=0), -0.5772, atol=0.25)
        np.testing.assert_allclose(s.std(axis=0), 1.2825, rtol=0.25)
        skew = np.mean(((s - s.mean(0)) / s.std(0)) ** 3, axis=0)
        np.testing.assert_allclose(skew, -1.1395, atol=1.0)
        self.assertGreater(float(jnp.mean(infos.acceptance_rate)), 0.05)

    def test_neal_funnel_neck_marginal(self):
        # The canonical stress test for step-size adaptation (a single
        # global step size cannot work). Check only the well-behaved "neck"
        # marginal y ~ N(0, 3**2) exactly -- the funnel coordinates' marginal
        # variance is a log-normal mixture (heavy-tailed, high MC variance),
        # not a useful numeric target at feasible sample sizes.
        # (Rationale mirrors test_gist_step_size.py::test_neal_funnel_neck_marginal)
        algo = blackjax.gist_trajectory_length(
            neal_funnel_logdensity,
            inverse_mass_matrix=jnp.ones(3),
            step_size=0.15,
            max_num_steps=128,
        )
        pos, infos = run_chain(algo, jnp.zeros(3), self.next_key(), 6000)
        y = np.asarray(pos[3000:, 0])
        # Self-calibrating ESS-gated assertion: replaces fixed atol=0.7,
        # robust across JAX versions and environments (see lesson
        # worklog/lessons/code-patterns/2026-05-11-single-realization-mc-noisy-assertion.md).
        assert_mean_within_ess_gated_tolerance(
            y, expected_mean=0.0, ess_min=80, k_sigma=5.0
        )
        self.assertTrue(np.all(np.isfinite(np.asarray(pos))))


class ClosedFormCrossCheckTest(BlackJAXTest):
    """Section 4.3: cheap, exact-to-float-tolerance derivation cross-checks."""

    def test_psi_zero_reduces_to_paper_simple_form(self):
        # At psi=0, Lo (and Lo') are identically 1, so the general-psi
        # formula must reduce to a_GIST = 1 wedge [e^{-DeltaH} M/N 1{L<=N}]
        # (section 2.2.4). Cross-check against the *actual* apply_fn
        # (not a reimplementation of the U-turn rollout).
        state = gist.init(jnp.zeros(2), std_normal_logdensity)
        metric = metrics.default_metric(jnp.ones(2))
        integrator_state = integrators.IntegratorState(
            jnp.zeros(2), jnp.array([1.0, 0.5]), state.logdensity, state.logdensity_grad
        )
        uturn_fn = gist_trajectory_length.num_steps_to_uturn(
            integrators.velocity_verlet, step_size=0.3, metric=metric, max_num_steps=50
        )
        forward = uturn_fn(integrator_state, std_normal_logdensity)
        L = jnp.minimum(forward, jnp.asarray(2))

        apply_fn = gist_trajectory_length._apply_fn(
            integrators.velocity_verlet,
            step_size=0.3,
            max_num_steps=50,
            path_fraction=0.0,
        )
        _, log_ratio, extra = apply_fn(
            integrator_state, L, forward, std_normal_logdensity, metric
        )
        reverse = extra.num_steps_to_uturn_reverse
        expected = jnp.where(
            L <= reverse,
            jnp.log(forward.astype(jnp.float32)) - jnp.log(reverse.astype(jnp.float32)),
            -jnp.inf,
        )
        np.testing.assert_allclose(float(log_ratio), float(expected), atol=1e-6)

    def test_num_steps_to_uturn_quarter_period_anchor_d1(self):
        # d=1 standard normal: the exact Hamiltonian flow is a rotation with
        # period 2*pi. Starting at theta0=0 (a clean special case), the
        # no-return condition (theta(t)-theta0)*rho(t) < 0 first fires at
        # exactly the quarter period t=pi/2 (GIST paper section 4). Leapfrog
        # approximates the exact flow well for small step_size.
        step_size = 0.01
        metric = metrics.default_metric(jnp.ones(1))
        state = integrators.IntegratorState(
            jnp.array([0.0]), jnp.array([1.0]), jnp.array(0.0), jnp.array([0.0])
        )
        uturn_fn = gist_trajectory_length.num_steps_to_uturn(
            integrators.velocity_verlet, step_size, metric, max_num_steps=1000
        )
        n = int(uturn_fn(state, std_normal_logdensity))
        expected = float(jnp.pi / 2) / step_size
        np.testing.assert_allclose(n, expected, rtol=0.05)

    def test_metric_corrected_velocity_used_not_raw_momentum(self):
        # `[DECISION -- TL ratify]` option (ii): the no-U-turn dot product
        # must use the metric-corrected velocity M^{-1} rho, not raw
        # momentum. A strongly anisotropic diagonal metric makes the two
        # disagree -- confirms the correction is load-bearing, not a no-op.
        inv_mass = jnp.array([100.0, 0.01])
        metric = metrics.default_metric(inv_mass)
        state = integrators.IntegratorState(
            jnp.array([0.0, 0.0]),
            jnp.array([1.0, 1.0]),
            jnp.array(0.0),
            jnp.array([0.0, 0.0]),
        )
        uturn_fn = gist_trajectory_length.num_steps_to_uturn(
            integrators.velocity_verlet,
            step_size=0.05,
            metric=metric,
            max_num_steps=200,
        )
        corrected = int(uturn_fn(state, std_normal_logdensity))

        symplectic_integrator = integrators.velocity_verlet(
            std_normal_logdensity, metric.kinetic_energy
        )
        theta0 = state.position
        current = state
        n_raw = 0
        while n_raw < 200:
            nxt = symplectic_integrator(current, 0.05)
            delta = nxt.position - theta0
            if float(jnp.dot(delta, nxt.momentum)) < 0.0:  # raw momentum, not velocity
                break
            current = nxt
            n_raw += 1

        self.assertNotEqual(corrected, n_raw)


class EdgeCaseTest(BlackJAXTest):
    def test_all_reject_on_absurd_step_size(self):
        algo = blackjax.gist_trajectory_length(
            std_normal_logdensity,
            inverse_mass_matrix=jnp.ones(2),
            step_size=1e6,
            max_num_steps=8,
        )
        pos, infos = run_chain(algo, jnp.zeros(2), self.next_key(), 200)
        self.assertTrue(np.all(np.isfinite(np.asarray(pos))))
        np.testing.assert_allclose(np.asarray(pos), 0.0)  # chain never moved
        self.assertFalse(bool(jnp.any(infos.is_accepted)))

    def test_hard_constraint_boundary_no_crash(self):
        logp = lambda x: jnp.where(x[0] > 0, -0.5 * jnp.sum(x**2), -jnp.inf)
        algo = blackjax.gist_trajectory_length(
            logp, inverse_mass_matrix=jnp.ones(2), step_size=1.0
        )
        pos, _ = run_chain(algo, jnp.array([0.5, 0.5]), self.next_key(), 500)
        self.assertTrue(np.all(np.isfinite(np.asarray(pos))))
        self.assertTrue(np.all(np.asarray(pos[:, 0]) > 0))

    def test_nan_gradient_region_no_crash(self):
        logp = lambda x: -jnp.sum(jnp.sqrt(x))
        algo = blackjax.gist_trajectory_length(
            logp, inverse_mass_matrix=jnp.ones(2), step_size=0.5, max_num_steps=16
        )
        pos, _ = run_chain(algo, jnp.array([1.0, 1.0]), self.next_key(), 500)
        self.assertTrue(np.all(np.isfinite(np.asarray(pos))))

    def test_no_return_rejection_direct(self):
        # Direct unit test: pick L far larger than any plausible reverse
        # U-turn count N, so L must fall outside [Lo', N].
        state = gist.init(jnp.zeros(2), std_normal_logdensity)
        metric = metrics.default_metric(jnp.ones(2))
        integrator_state = integrators.IntegratorState(
            jnp.zeros(2), jnp.array([1.0, 0.5]), state.logdensity, state.logdensity_grad
        )
        apply_fn = gist_trajectory_length._apply_fn(
            integrators.velocity_verlet,
            step_size=0.3,
            max_num_steps=50,
            path_fraction=0.5,
        )
        implausibly_large_L = jnp.asarray(50)
        _, log_ratio, extra = apply_fn(
            integrator_state,
            implausibly_large_L,
            jnp.asarray(3),
            std_normal_logdensity,
            metric,
        )
        self.assertTrue(bool(extra.is_no_return_rejected))
        self.assertEqual(float(log_ratio), float("-inf"))

    def test_max_num_steps_cap_used_as_is(self):
        # A tiny max_num_steps caps num_steps_to_uturn without erroring; the
        # capped U is still used as an exact (not approximate) density, so
        # the chain should keep running validly, not crash or NaN out.
        algo = blackjax.gist_trajectory_length(
            std_normal_logdensity,
            inverse_mass_matrix=jnp.ones(2),
            step_size=0.3,
            max_num_steps=2,
        )
        pos, infos = run_chain(algo, jnp.zeros(2), self.next_key(), 300)
        self.assertTrue(np.all(np.isfinite(np.asarray(pos))))
        self.assertTrue(np.all(np.asarray(infos.num_steps_to_uturn_forward) <= 2))


if __name__ == "__main__":
    absltest.main()
