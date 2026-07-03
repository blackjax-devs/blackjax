"""Tests for the GIST self-tuning step-size sampler (autoStep-style)."""
import chex
import jax
import jax.numpy as jnp
import jax.scipy.stats as st
import numpy as np
from absl.testing import absltest, parameterized

import blackjax
from blackjax.mcmc import gist, gist_step_size, integrators, metrics
from tests.fixtures import (
    BlackJAXTest,
    neal_funnel_logdensity,
    smooth_skewed_logdensity,
    std_normal_logdensity,
)

CRITERIA = ["symmetric", "asymmetric"]


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
        state = gist_step_size.init(position, std_normal_logdensity)
        self.assertIsInstance(state, gist.GISTState)
        np.testing.assert_allclose(state.position, position)
        np.testing.assert_allclose(
            float(state.logdensity), float(std_normal_logdensity(position))
        )


class SingleStepTest(chex.TestCase):
    @parameterized.parameters(*CRITERIA)
    def test_step_shapes_and_types(self, criterion):
        algo = blackjax.gist_step_size(
            std_normal_logdensity,
            inverse_mass_matrix=jnp.ones(3),
            initial_step_size=0.5,
            criterion=criterion,
        )
        state = algo.init(jnp.zeros(3))
        new_state, info = algo.step(jax.random.key(0), state)
        self.assertIsInstance(new_state, gist.GISTState)
        self.assertIsInstance(info, gist_step_size.GISTStepSizeInfo)
        self.assertEqual(new_state.position.shape, (3,))
        np.testing.assert_allclose(
            float(new_state.logdensity),
            float(std_normal_logdensity(new_state.position)),
            atol=1e-5,
        )

    def test_jit(self):
        algo = blackjax.gist_step_size(
            std_normal_logdensity, inverse_mass_matrix=jnp.ones(3), initial_step_size=0.3
        )
        state = algo.init(jnp.zeros(3))
        new_state, _ = jax.jit(algo.step)(jax.random.key(0), state)
        self.assertEqual(new_state.position.shape, (3,))

    def test_invalid_criterion_raises(self):
        with self.assertRaises(ValueError):
            gist_step_size.build_kernel(criterion="not-a-criterion")


class CompilationTest(chex.TestCase):
    def test_no_excess_retracing(self):
        """The logdensity should compile at most 4 times: init, plus 3 within
        one kernel trace -- the forward selector search, the accepted-move
        trajectory, and the reverse-selector re-check (section 2.1.3) each
        need their own gradient evaluation, unlike hmc/nuts's single forward
        trajectory (n=2 there). Verified empirically: the count stabilizes
        at 4 after the first `step()` call and does not grow on further
        calls with the same shapes (no spurious per-call retracing).
        """

        @chex.assert_max_traces(n=4)
        def logdensity_fn(x):
            return jnp.sum(st.norm.logpdf(x))

        chex.clear_trace_counter()

        algo = blackjax.gist_step_size(
            logdensity_fn, inverse_mass_matrix=jnp.ones(2), initial_step_size=0.3
        )
        state = algo.init(jnp.zeros(2))
        step = jax.jit(algo.step)

        rng_key = jax.random.key(0)
        for i in range(5):
            sample_key = jax.random.fold_in(rng_key, i)
            state, _ = step(sample_key, state)


class StationarityTest(BlackJAXTest):
    """If the population starts exactly at stationarity, it should stay there.

    Modeled on ``tests/mcmc/test_barker.py::test_invariance``: a cheaper,
    directly automatable proxy for reversibility than a literal pairwise
    detailed-balance check. This is exactly the test to catch a bug in the
    ``j' = j`` reversibility-check reconciliation (section 2.1.3) -- such a
    bug would show up as population drift even from an exact stationary
    start.
    """

    @parameterized.parameters(*CRITERIA)
    def test_stationarity_from_exact_draws(self, criterion):
        d = 2
        n_samples, m_steps = 2000, 20

        algo = blackjax.gist_step_size(
            std_normal_logdensity,
            inverse_mass_matrix=jnp.ones(d),
            initial_step_size=0.7,
            criterion=criterion,
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
        chex.assert_trees_all_close(
            jnp.cov(samples.T), jnp.eye(d), atol=0.2, rtol=0.2
        )


class MomentRecoveryTest(BlackJAXTest):
    def test_isotropic_std_normal(self):
        algo = blackjax.gist_step_size(
            std_normal_logdensity, inverse_mass_matrix=jnp.ones(3), initial_step_size=0.8
        )
        pos, infos = run_chain(algo, jnp.zeros(3), self.next_key(), 6000)
        s = np.asarray(pos[3000:])
        np.testing.assert_allclose(s.mean(), 0.0, atol=0.12)
        np.testing.assert_allclose(s.std(), 1.0, rtol=0.15)
        self.assertGreater(float(jnp.mean(infos.acceptance_rate)), 0.05)

    def test_correlated_gaussian_dense_metric(self):
        # Exercises a non-identity (dense) inverse_mass_matrix.
        Sigma = jnp.array([[2.0, 1.2], [1.2, 1.0]])
        Sinv = jnp.linalg.inv(Sigma)
        logp = lambda x: -0.5 * x @ Sinv @ x
        algo = blackjax.gist_step_size(logp, inverse_mass_matrix=Sigma, initial_step_size=0.5)
        pos, _ = run_chain(algo, jnp.zeros(2), self.next_key(), 8000)
        emp = np.cov(np.asarray(pos[4000:]), rowvar=False)
        np.testing.assert_allclose(emp, np.asarray(Sigma), atol=0.5)

    def test_smooth_skewed_target(self):
        # Asymmetric target: a symmetric Gaussian cannot detect a
        # reversed-direction/sign bug in the leapfrog+flip, but a skewed
        # target can (mirrors test_slice.py's
        # test_multivariate_skewed_exponential rationale, but with a smooth
        # log-space target -- see
        # tests/fixtures.py::smooth_skewed_logdensity for why the raw
        # Exponential representation is a poor fit for a gradient-based
        # sampler).
        algo = blackjax.gist_step_size(
            smooth_skewed_logdensity, inverse_mass_matrix=jnp.ones(2), initial_step_size=0.4
        )
        pos, infos = run_chain(algo, jnp.zeros(2), self.next_key(), 8000)
        s = np.asarray(pos[4000:])
        np.testing.assert_allclose(s.mean(axis=0), -0.5772, atol=0.2)
        np.testing.assert_allclose(s.std(axis=0), 1.2825, rtol=0.2)
        skew = np.mean(((s - s.mean(0)) / s.std(0)) ** 3, axis=0)
        self.assertTrue((skew < 0.0).all())  # left-skewed, not mirrored

    def test_neal_funnel_neck_marginal(self):
        # The canonical stress test for step-size adaptation (a single
        # global step size cannot work). Check only the well-behaved "neck"
        # marginal y ~ N(0, 3**2) exactly -- the funnel coordinates' marginal
        # variance is a log-normal mixture (heavy-tailed, high MC variance),
        # not a useful numeric target at feasible sample sizes.
        algo = blackjax.gist_step_size(
            neal_funnel_logdensity,
            inverse_mass_matrix=jnp.ones(3),
            initial_step_size=0.3,
            max_search_steps=20,
        )
        pos, infos = run_chain(algo, jnp.zeros(3), self.next_key(), 6000)
        y = np.asarray(pos[3000:, 0])
        np.testing.assert_allclose(y.mean(), 0.0, atol=0.6)
        np.testing.assert_allclose(y.std(), 3.0, rtol=0.35)
        self.assertTrue(np.all(np.isfinite(np.asarray(pos))))
        # Regression guard: the symmetric default must not silently degrade
        # to near-zero acceptance the way the asymmetric criterion can
        # ([AutoStep] Fig. 1).
        self.assertGreater(float(jnp.mean(infos.acceptance_rate)), 0.05)


class SelectorUnitTest(BlackJAXTest):
    """Direct tests of ``step_size_selector``'s mu function, section 2.1.2."""

    def test_final_halving_on_successful_expansion(self):
        # A tiny initial_step_size on a std normal, with wide (a, b), should
        # trigger v=+1 (expand) and terminate normally; the "final halving"
        # means the reported step_index is one less than the index at which
        # the loop actually detected termination.
        selector = gist_step_size.step_size_selector(
            integrators.velocity_verlet,
            num_integration_steps=1,
            initial_step_size=1e-4,
            max_search_steps=30,
        )
        state = gist.init(jnp.zeros(2), std_normal_logdensity)
        metric = metrics.default_metric(jnp.ones(2))
        integrator_state = integrators.IntegratorState(
            state.position,
            jnp.ones(2),
            state.logdensity,
            state.logdensity_grad,
        )
        step_index, search_exhausted = selector(
            integrator_state, jnp.array(0.4), jnp.array(0.6), std_normal_logdensity, metric
        )
        self.assertFalse(bool(search_exhausted))
        self.assertGreater(int(step_index), 0)  # expanded away from the tiny step

    def test_search_exhausted_on_zero_budget(self):
        # An absurdly large initial_step_size forces v=-1 (shrink) for
        # virtually any (a, b) in (0, 1); with max_search_steps=0 the search
        # cannot even take one step, so it must report search_exhausted.
        selector = gist_step_size.step_size_selector(
            integrators.velocity_verlet,
            num_integration_steps=1,
            initial_step_size=1e8,
            max_search_steps=0,
        )
        state = gist.init(jnp.zeros(2), std_normal_logdensity)
        metric = metrics.default_metric(jnp.ones(2))
        integrator_state = integrators.IntegratorState(
            state.position,
            jnp.ones(2),
            state.logdensity,
            state.logdensity_grad,
        )
        _, search_exhausted = selector(
            integrator_state, jnp.array(0.3), jnp.array(0.7), std_normal_logdensity, metric
        )
        self.assertTrue(bool(search_exhausted))


class EdgeCaseTest(BlackJAXTest):
    def test_search_exhausted_forces_rejection(self):
        algo = blackjax.gist_step_size(
            std_normal_logdensity,
            inverse_mass_matrix=jnp.ones(2),
            initial_step_size=1e8,  # forces v=-1 for virtually any (a, b)
            max_search_steps=0,
        )
        state = algo.init(jnp.zeros(2))
        new_state, info = jax.jit(algo.step)(self.next_key(), state)
        self.assertTrue(bool(info.search_exhausted))
        self.assertFalse(bool(info.is_accepted))
        np.testing.assert_allclose(new_state.position, state.position)

    def test_hard_constraint_boundary_no_crash(self):
        # Half-normal: logdensity_fn = -inf outside x > 0. A large step size
        # naturally crosses the boundary; the transition must reject cleanly
        # (Delta_energy = +inf) rather than crash or return NaNs.
        logp = lambda x: jnp.where(x[0] > 0, -0.5 * jnp.sum(x**2), -jnp.inf)
        algo = blackjax.gist_step_size(
            logp, inverse_mass_matrix=jnp.ones(2), initial_step_size=3.0
        )
        pos, infos = run_chain(algo, jnp.array([0.5, 0.5]), self.next_key(), 500)
        self.assertTrue(np.all(np.isfinite(np.asarray(pos))))
        self.assertTrue(np.all(np.asarray(pos[:, 0]) > 0))

    def test_nan_gradient_region_no_crash(self):
        # sqrt has a NaN *gradient* (not just a NaN value) for x < 0: the
        # reciprocal-sqrt derivative formula itself inherits the NaN.
        logp = lambda x: -jnp.sum(jnp.sqrt(x))
        algo = blackjax.gist_step_size(
            logp, inverse_mass_matrix=jnp.ones(2), initial_step_size=2.0, max_search_steps=5
        )
        pos, infos = run_chain(algo, jnp.array([1.0, 1.0]), self.next_key(), 500)
        self.assertTrue(np.all(np.isfinite(np.asarray(pos))))

    def test_reversibility_check_failure_forces_rejection(self):
        # Direct unit test of _apply_fn with a mock selector that
        # deliberately disagrees between the forward and reverse call, so
        # that j' != j regardless of the energy term. Use a tiny step size
        # so the *energy* term alone would almost certainly accept.
        initial_position = jnp.zeros(2)
        state = gist.init(initial_position, std_normal_logdensity)
        metric = metrics.default_metric(jnp.ones(2))
        integrator_state = integrators.IntegratorState(
            state.position, jnp.ones(2), state.logdensity, state.logdensity_grad
        )

        def mock_selector(s, a, b, logdensity_fn, metric, *, build_trajectory=None):
            del build_trajectory
            is_initial = jnp.allclose(s.position, initial_position)
            return jnp.where(is_initial, 0, 5), jnp.asarray(False)

        apply_fn = gist_step_size._apply_fn(
            integrators.velocity_verlet,
            num_integration_steps=1,
            initial_step_size=1e-6,  # tiny: energy term alone would accept
            selector=mock_selector,
        )
        alpha = gist_step_size.StepSizeTuningParameter(
            jnp.array(0.3), jnp.array(0.7), jnp.array(0)
        )
        _, log_tuning_density_ratio, extra_info = apply_fn(
            integrator_state, alpha, jnp.asarray(False), std_normal_logdensity, metric
        )
        self.assertEqual(float(log_tuning_density_ratio), float("-inf"))
        self.assertEqual(int(extra_info.reverse_step_index), 5)


if __name__ == "__main__":
    absltest.main()
