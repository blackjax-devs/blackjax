"""Tests for the GIST self-tuning trajectory-length sampler (no-U-turn,
NOT NUTS's recursive doubling).

CI note: run with ``--benchmark-disable`` when parallelizing under xdist
(e.g. ``-n 2``) -- a bare ``-n 2`` run hits an ``INTERNALERROR`` from
pytest-benchmark x xdist under this project's ``filterwarnings = error``
(the same masking-bug family as the probdiffeq migration saga). Not a code
issue in this module; the documented blackjax test command already
includes ``--benchmark-disable``.
"""
from typing import Callable, NamedTuple

import chex
import jax
import jax.numpy as jnp
import jax.scipy.stats as st
import numpy as np
from absl.testing import absltest, parameterized
from jax.flatten_util import ravel_pytree

import blackjax
from blackjax.mcmc import (
    gist,
    gist_trajectory_length,
    hmc,
    integrators,
    metrics,
    trajectory,
)
from tests.fixtures import (
    BlackJAXTest,
    assert_grand_mean_within_robust_tolerance,
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


def uturn_count_reference(metric, step_size, max_num_steps, pairing):
    """Reference U-turn rollout with an explicit choice of dot-product pairing.

    ``pairing="momentum"`` reproduces the shipped criterion (GIST eq. 33);
    ``pairing="velocity"`` reproduces the pre-fix criterion that paired the
    displacement with ``G rho``. Everything else -- integrator, cap, and in
    particular the **counting convention** -- mirrors
    ``num_steps_to_uturn``: ``n`` is incremented on the step at which the
    condition fires.

    Getting that convention right is the whole point of this helper. The
    hand-rolled rollout it replaces counted one step *fewer* than the
    kernel, so its ``assertNotEqual`` against the kernel was satisfied by
    the off-by-one alone and would have passed under either pairing.
    """
    velocity_fn = jax.grad(metric.kinetic_energy)

    def count_fn(state, logdensity_fn):
        symplectic_integrator = integrators.velocity_verlet(
            logdensity_fn, metric.kinetic_energy
        )
        theta0, _ = ravel_pytree(state.position)

        def cond_fn(carry):
            n, _, no_return = carry
            return jnp.logical_not(no_return) & (n < max_num_steps)

        def body_fn(carry):
            n, current, _ = carry
            nxt = symplectic_integrator(current, step_size)
            delta = ravel_pytree(nxt.position)[0] - theta0
            if pairing == "velocity":
                paired, _ = ravel_pytree(velocity_fn(nxt.momentum, nxt.position))
            else:
                paired, _ = ravel_pytree(nxt.momentum)
            return n + 1, nxt, jnp.dot(delta, paired) < 0.0

        n_final, _, _ = jax.lax.while_loop(
            cond_fn, body_fn, (jnp.asarray(0), state, jnp.asarray(False))
        )
        return n_final

    return count_fn


def _reference_tuning_parameter_fn(
    integrator: Callable, step_size: float, max_num_steps: int, path_fraction: float
) -> Callable:
    """The forward GIBBS seam exactly as ``gist_trajectory_length.py``
    implemented it *before* Issue#1058's rollout-caching fix: only the
    no-U-turn count is kept, every intermediate leapfrog state is discarded.
    """

    def tuning_parameter_fn(rng_key, state, logdensity_fn, metric):
        uturn_fn = gist_trajectory_length.num_steps_to_uturn(
            integrator, step_size, metric, max_num_steps
        )
        forward = uturn_fn(state, logdensity_fn)
        lo, _ = gist_trajectory_length._step_distribution(forward, path_fraction)
        num_steps = jax.random.randint(rng_key, shape=(), minval=lo, maxval=forward + 1)
        return num_steps, forward

    return tuning_parameter_fn


def _reference_apply_fn(
    integrator: Callable, step_size: float, max_num_steps: int, path_fraction: float
) -> Callable:
    """The involution seam exactly as ``gist_trajectory_length.py``
    implemented it *before* Issue#1058: the accepted-move proposal is built
    by re-integrating ``num_steps`` leapfrog steps from scratch via
    ``trajectory.static_integration``, not gathered from a cache.
    """

    def apply_fn(state, alpha, aux, logdensity_fn, metric):
        num_steps = alpha
        forward = aux

        symplectic_integrator = integrator(logdensity_fn, metric.kinetic_energy)
        build_trajectory = trajectory.static_integration(symplectic_integrator)
        proposal_state = build_trajectory(state, step_size, num_steps)
        proposal_state = hmc.flip_momentum(proposal_state)

        uturn_fn = gist_trajectory_length.num_steps_to_uturn(
            integrator, step_size, metric, max_num_steps
        )
        reverse = uturn_fn(proposal_state, logdensity_fn)

        _, width_forward = gist_trajectory_length._step_distribution(
            forward, path_fraction
        )
        lo_reverse, width_reverse = gist_trajectory_length._step_distribution(
            reverse, path_fraction
        )

        is_in_reverse_interval = (num_steps >= lo_reverse) & (num_steps <= reverse)
        log_tuning_density_ratio = jnp.where(
            is_in_reverse_interval,
            jnp.log(width_forward.astype(jnp.float32))
            - jnp.log(width_reverse.astype(jnp.float32)),
            -jnp.inf,
        )
        extra_info = gist_trajectory_length._TrajectoryLengthExtra(
            num_integration_steps=num_steps,
            num_steps_to_uturn_forward=forward,
            num_steps_to_uturn_reverse=reverse,
            is_no_return_rejected=jnp.logical_not(is_in_reverse_interval),
        )
        return proposal_state, log_tuning_density_ratio, extra_info

    return apply_fn


def _reference_build_kernel(
    integrator: Callable,
    divergence_threshold: float,
    path_fraction: float,
    max_num_steps: int,
) -> Callable:
    """Mirrors ``gist_trajectory_length.build_kernel`` exactly, wired to the
    pre-Issue#1058 (uncached) ``tuning_parameter_fn``/``apply_fn`` pair
    above instead of the shipped, cached ones.
    """

    def kernel(rng_key, state, logdensity_fn, step_size, inverse_mass_matrix):
        tuning_parameter_fn = _reference_tuning_parameter_fn(
            integrator, step_size, max_num_steps, path_fraction
        )
        apply_fn = _reference_apply_fn(
            integrator, step_size, max_num_steps, path_fraction
        )
        new_state, info, raw_extra_info = gist._step(
            rng_key,
            state,
            logdensity_fn,
            tuning_parameter_fn,
            apply_fn,
            inverse_mass_matrix,
            divergence_threshold,
        )
        extra_info = raw_extra_info
        trajectory_length_info = gist_trajectory_length.GISTTrajectoryLengthInfo(
            info.momentum,
            info.tuning_parameter,
            info.is_accepted,
            info.is_divergent,
            info.acceptance_rate,
            info.energy,
            info.num_integration_steps,
            extra_info.num_steps_to_uturn_forward,
            extra_info.num_steps_to_uturn_reverse,
            extra_info.is_no_return_rejected,
        )
        return new_state, trajectory_length_info

    return kernel


def _dict_position_logdensity(x):
    """A dict-pytree-position standard normal, for the pytree regression case."""
    return -0.5 * (x["a"] ** 2 + jnp.sum(x["b"] ** 2))


class DrawIdentityRegressionTest(chex.TestCase):
    """Regression guard for Issue#1058: caching the forward rollout must not
    change a single bit of the kernel's output.

    Compares the shipped (cached, buffer-gather) ``build_kernel`` against
    ``_reference_build_kernel`` above, which reimplements this module's
    ``tuning_parameter_fn``/``apply_fn`` pair exactly as it stood before
    Issue#1058 (a fresh ``trajectory.static_integration`` re-integration for
    every accepted move). Both kernels share ``num_steps_to_uturn``
    (untouched by this change) and ``gist._step``'s Gibbs-refresh /
    Metropolis-test machinery (also untouched) -- the *only* code path that
    differs between them is how the ``alpha``-length proposal state itself
    is built (buffer gather vs. re-integration), so any exact-equality
    mismatch here can only be attributed to that.

    Both sides are jitted identically (rather than comparing a jitted cached
    kernel against an un-jitted reference, or vice versa) so a difference
    cannot be a jit-vs-eager operation-fusion artifact unrelated to the
    caching change itself.
    """

    @parameterized.named_parameters(
        (
            "identity_metric",
            jnp.zeros(3),
            jnp.ones(3),
            std_normal_logdensity,
        ),
        (
            "diagonal_metric",
            jnp.zeros(3),
            jnp.array([0.3, 1.0, 2.5]),
            std_normal_logdensity,
        ),
        (
            "dense_metric",
            jnp.zeros(2),
            jnp.array([[2.0, 1.2], [1.2, 1.0]]),
            lambda x: -0.5
            * x
            @ jnp.linalg.inv(jnp.array([[2.0, 1.2], [1.2, 1.0]]))
            @ x,
        ),
        (
            "dict_pytree_position",
            {"a": jnp.array(0.0), "b": jnp.array([1.0, -0.5])},
            jnp.ones(3),
            _dict_position_logdensity,
        ),
    )
    def test_bit_identical_to_uncached_reference(
        self, init_position, inverse_mass_matrix, logdensity_fn
    ):
        step_size = 0.3
        max_num_steps = 64
        path_fraction = 0.5
        divergence_threshold = 1000.0

        cached_kernel = gist_trajectory_length.build_kernel(
            integrators.velocity_verlet,
            divergence_threshold,
            path_fraction,
            max_num_steps,
        )
        reference_kernel = _reference_build_kernel(
            integrators.velocity_verlet,
            divergence_threshold,
            path_fraction,
            max_num_steps,
        )
        # Close over the non-array arguments (logdensity_fn, step_size,
        # inverse_mass_matrix) rather than passing logdensity_fn through
        # jax.jit positionally, so both sides jit with the same, simplest
        # signature.
        cached_step = jax.jit(
            lambda key, state: cached_kernel(
                key, state, logdensity_fn, step_size, inverse_mass_matrix
            )
        )
        reference_step = jax.jit(
            lambda key, state: reference_kernel(
                key, state, logdensity_fn, step_size, inverse_mass_matrix
            )
        )

        cached_state = gist_trajectory_length.init(init_position, logdensity_fn)
        reference_state = gist_trajectory_length.init(init_position, logdensity_fn)
        chex.assert_trees_all_equal(cached_state, reference_state)

        rng_key = jax.random.key(0)
        for i in range(50):
            step_key = jax.random.fold_in(rng_key, i)
            cached_state, cached_info = cached_step(step_key, cached_state)
            reference_state, reference_info = reference_step(step_key, reference_state)
            chex.assert_trees_all_equal(cached_state, reference_state)
            chex.assert_trees_all_equal(cached_info, reference_info)


# Fixed constants for the partial-preconditioning fixture below. Found with a
# scratch sweep over d in {12, 16}, kappa(Sigma) in {1e3, 1e4} and step-size
# fractions {0.15, 0.3}: every combination separated the two pairings on
# >=90% of momentum seeds, and this is the cheapest of them (longest rollout
# 299 steps). Measured at these constants: kappa(Sigma) = 1000.0, residual
# kappa_res = 788.4, kernel == whitened ground truth on 20/20 momentum seeds,
# velocity pairing == ground truth on only 1/20. The counts are bit-identical
# under ``jax.enable_x64()``, so the exact-equality assertions below do not
# rest on float32 coincidence.
_PRECOND_DIM = 12
_PRECOND_SIGMA_SEED = 1000
_PRECOND_KAPPA = 1e3
_PRECOND_STEP_FRACTION = 0.3
_PRECOND_MAX_STEPS = 600
_PRECOND_MOMENTUM_SEEDS = tuple(50_000 + s for s in range(20))


class _PartialPreconditioningFixture(NamedTuple):
    """A correlated Gaussian target paired with a *diagonal* preconditioner.

    ``G = diag(Sigma)`` is what real diagonal window adaptation produces
    against a correlated target: it removes the marginal scales but leaves
    the correlation structure, i.e. a large residual anisotropy
    ``kappa_res = cond(G^{-1/2} Sigma G^{-1/2})``. That residual is exactly
    what separates the two candidate U-turn pairings -- under *exact*
    preconditioning (``G = Sigma``, kappa_res = 1) they agree by
    construction, which is why a whitening-based test cannot discriminate.

    The fixture also carries the equivalent whitened problem
    (``phi = G^{-1/2} theta``, target covariance ``G^{-1/2} Sigma G^{-1/2}``,
    metric ``I``). In an identity metric the two pairings coincide, so the
    identity-metric rollout is an unambiguous ground truth.
    """

    dim: int
    kappa_res: float
    step_size: float
    metric: metrics.Metric
    whitened_metric: metrics.Metric
    logdensity_fn: Callable
    whitened_logdensity_fn: Callable
    sigma_sqrt: np.ndarray
    diag_scale: np.ndarray

    def states(self, seed):
        """``(state, whitened_state)`` for one fixed momentum seed."""
        rng = np.random.default_rng(seed)
        theta0 = jnp.asarray(self.sigma_sqrt @ rng.standard_normal(self.dim))
        rho0 = jnp.asarray(rng.standard_normal(self.dim) / self.diag_scale)
        sqrt_g = jnp.asarray(self.diag_scale)

        logdensity, grad = jax.value_and_grad(self.logdensity_fn)(theta0)
        state = integrators.IntegratorState(theta0, rho0, logdensity, grad)

        phi0 = theta0 / sqrt_g
        rho_phi0 = sqrt_g * rho0
        logdensity_w, grad_w = jax.value_and_grad(self.whitened_logdensity_fn)(phi0)
        whitened_state = integrators.IntegratorState(
            phi0, rho_phi0, logdensity_w, grad_w
        )
        return state, whitened_state


def _partial_preconditioning_fixture():
    d = _PRECOND_DIM
    rng = np.random.default_rng(_PRECOND_SIGMA_SEED)
    Q, _ = np.linalg.qr(rng.standard_normal((d, d)))
    eigvals = np.exp(np.linspace(0.0, np.log(_PRECOND_KAPPA), d))
    Sigma = Q @ np.diag(eigvals) @ Q.T
    Sigma = 0.5 * (Sigma + Sigma.T)

    sqrt_g = np.sqrt(np.diag(Sigma))
    Sigma_w = Sigma / np.outer(sqrt_g, sqrt_g)  # G^{-1/2} Sigma G^{-1/2}
    residual_eigvals = np.linalg.eigvalsh(Sigma_w)

    # Step size as a fraction of the shortest oscillation period present in
    # the preconditioned system, so the rollout resolves every mode.
    step_size = float(_PRECOND_STEP_FRACTION * np.sqrt(residual_eigvals.min()))

    Sigma_inv = jnp.asarray(np.linalg.inv(Sigma))
    Sigma_w_inv = jnp.asarray(np.linalg.inv(Sigma_w))
    sigma_eigvals, sigma_eigvecs = np.linalg.eigh(Sigma)

    return _PartialPreconditioningFixture(
        dim=d,
        kappa_res=float(residual_eigvals.max() / residual_eigvals.min()),
        step_size=step_size,
        metric=metrics.default_metric(jnp.asarray(np.diag(Sigma))),
        whitened_metric=metrics.default_metric(jnp.ones(d)),
        logdensity_fn=lambda x: -0.5 * x @ Sigma_inv @ x,
        whitened_logdensity_fn=lambda x: -0.5 * x @ Sigma_w_inv @ x,
        sigma_sqrt=sigma_eigvecs @ np.diag(np.sqrt(sigma_eigvals)) @ sigma_eigvecs.T,
        diag_scale=sqrt_g,
    )


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
        """The logdensity should compile at most 3 times: init, plus 2
        within one kernel trace -- the forward U-turn rollout and the
        reverse U-turn rollout (section 2.2.4) each need their own gradient
        evaluation, unlike hmc/nuts's single forward trajectory (n=2 there).
        There is no longer a third, separate trace for the accepted-move
        build (Issue#1058): the forward rollout now buffers every state it
        visits, so the proposal for the selected `alpha` is a gather from
        that buffer, not a fresh `static_integration` call with its own
        `symplectic_integrator` closure. Regression guard for the caching
        change -- this count going back up to 4 would mean a re-integration
        path crept back in. Verified empirically: the count stabilizes at 3
        after the first `step()` call and does not grow on further calls
        with the same shapes.
        """

        @chex.assert_max_traces(n=3)
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
        # The canonical stress test for trajectory-length adaptation (a single
        # global step size cannot work well here). Check only the well-behaved
        # "neck" marginal y ~ N(0, 3**2) -- the funnel coordinates' marginal
        # variance is a log-normal mixture (heavy-tailed, high MC variance),
        # not a useful numeric target at feasible sample sizes.
        #
        # Multi-chain robust grand-mean gate (replaces the ESS-gated single-
        # chain assertion that escaped 3 times, issue #970; lineage
        # #957 → #959 → #971 skip): K=24 independent chains via vmap; grand
        # mean over chain-means with MAD-based robust SE and an absolute-
        # tolerance floor.  A single stuck chain cannot inflate the threshold
        # and hide a real bias (MAD breakdown ~50%).  See
        # tests/fixtures.py::assert_grand_mean_within_robust_tolerance for
        # the full design rationale.
        algo = blackjax.gist_trajectory_length(
            neal_funnel_logdensity,
            inverse_mass_matrix=jnp.ones(3),
            step_size=0.15,
            max_num_steps=128,
        )

        K = 24
        keys = jax.random.split(self.next_key(), K)

        def one_chain(key):
            pos, infos = run_chain(algo, jnp.zeros(3), key, 3000)
            return pos[1500:, 0], infos.acceptance_rate

        ys, accs = jax.vmap(one_chain)(keys)  # ys: (K, T)
        assert_grand_mean_within_robust_tolerance(
            np.asarray(ys), expected_mean=0.0, atol_floor=1.0, k_sigma=5.0
        )
        self.assertGreater(float(jnp.mean(accs)), 0.05)
        self.assertTrue(np.all(np.isfinite(np.asarray(ys))))


class AffineEquivarianceRegressionTest(chex.TestCase):
    """Regression guard: the no-U-turn rollout must be affine-equivariant.

    ``num_steps_to_uturn`` pairs the position displacement with the **raw
    momentum** ``rho`` (GIST eq. 33). With ``p ~ N(0, G^-1)`` and
    ``K(p) = ½ pᵀGp``, whitening ``phi = G^{-1/2}theta`` and
    ``p_phi = G^{1/2}p`` gives ``(Δtheta)ᵀp = (Δphi)ᵀp_phi`` -- exactly
    invariant under a change of metric. Pairing with the metric-corrected
    velocity ``G rho`` instead gives ``(Δphi)ᵀ G p_phi``, which re-weights
    the whitened modes and is NOT invariant.

    So the test is: run the kernel's rollout on a preconditioned problem,
    run it again on the equivalent whitened problem under an identity
    metric (where the two pairings provably coincide, making it an
    unambiguous ground truth), and demand the two integer counts are
    **exactly** equal. No tolerance: the correct pairing is exactly
    equivariant, so exactness is the entire signal and a tolerance would
    destroy the test's power.

    Two earlier attempts at this guard both passed on the unfixed code:

    * ``test_uturn_invariant_to_whitening`` whitened *exactly* (``G = Σ``),
      i.e. ``kappa_res = 1`` -- precisely the regime where both pairings
      agree, because after exact whitening every coordinate shares one
      period and re-weighting in-phase sines rescales the sum without
      moving its zero crossing. (Its comment claiming ``kappa_res ≈ 72``
      confused ``cond(Σ)`` with the residual.) Structurally incapable of
      failing.
    * ``test_uturn_with_partial_preconditioning_identity_metric`` had the
      right setup but only asserted non-degeneracy (``0 < n < max_steps``,
      ``5 < mean < 100``) -- properties both pairings satisfy. It never
      compared against a ground truth.
    """

    def setUp(self):
        super().setUp()
        self.fixture = _partial_preconditioning_fixture()

    def _counts(self, pairing=None):
        f = self.fixture
        if pairing is None:
            rollout = gist_trajectory_length.num_steps_to_uturn(
                integrators.velocity_verlet, f.step_size, f.metric, _PRECOND_MAX_STEPS
            )
        else:
            rollout = uturn_count_reference(
                f.metric, f.step_size, _PRECOND_MAX_STEPS, pairing
            )
        ground_truth = gist_trajectory_length.num_steps_to_uturn(
            integrators.velocity_verlet,
            f.step_size,
            f.whitened_metric,
            _PRECOND_MAX_STEPS,
        )
        # jit once and reuse across seeds: only the initial state varies.
        count_fn = jax.jit(lambda s: rollout(s, f.logdensity_fn))
        truth_fn = jax.jit(lambda s: ground_truth(s, f.whitened_logdensity_fn))

        counts, truths = [], []
        for seed in _PRECOND_MOMENTUM_SEEDS:
            state, whitened_state = f.states(seed)
            counts.append(int(count_fn(state)))
            truths.append(int(truth_fn(whitened_state)))
        return counts, truths

    def test_uturn_count_matches_whitened_ground_truth(self):
        # The guards come first, in the same test rather than a separate one:
        # a standalone fixture-guard would pass on the unfixed kernel (it
        # never calls it), and a test that passes on the unfixed kernel is
        # exactly what this file already had two of.
        #
        # Guard 1 -- the fixture must leave a large RESIDUAL anisotropy after
        # diagonal preconditioning. That, not cond(Sigma), is the quantity
        # the velocity-pairing bias scales with: at kappa_res ~ 10 the two
        # pairings are indistinguishable, and the separation only appears
        # from kappa_res in the hundreds upward.
        self.assertGreater(self.fixture.kappa_res, 500.0)

        # Guard 2 -- and it must actually separate them here. The velocity
        # pairing disagrees with the ground truth on 19 of these 20 momentum
        # seeds (measured); assert a comfortable majority so the check
        # documents the discriminating power without being brittle.
        velocity_counts, truths = self._counts(pairing="velocity")
        n_disagree = sum(n != t for n, t in zip(velocity_counts, truths))
        self.assertGreaterEqual(n_disagree, 15)

        # Guard 3 -- a capped rollout would agree vacuously on both sides.
        counts, truths = self._counts()
        self.assertTrue(all(n < _PRECOND_MAX_STEPS for n in counts + truths))

        # The regression assertion. Exact equality on every seed.
        self.assertEqual(counts, truths)


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
        # `_apply_fn`'s `aux` is now the cached `_ForwardRollout` (Issue#1058),
        # not the bare `forward` scalar -- build it with the real forward
        # rollout so `rollout.states` is actually populated up to `forward`.
        uturn_fn = gist_trajectory_length._num_steps_to_uturn_with_rollout(
            integrators.velocity_verlet, step_size=0.3, metric=metric, max_num_steps=50
        )
        rollout = uturn_fn(integrator_state, std_normal_logdensity)
        forward = rollout.num_steps_to_uturn
        L = jnp.minimum(forward, jnp.asarray(2))

        apply_fn = gist_trajectory_length._apply_fn(
            integrators.velocity_verlet,
            step_size=0.3,
            max_num_steps=50,
            path_fraction=0.0,
        )
        _, log_ratio, extra = apply_fn(
            integrator_state, L, rollout, std_normal_logdensity, metric
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

    def test_raw_momentum_pairing_used_not_metric_corrected_velocity(self):
        """The rollout pairs the displacement with ``rho``, not ``G rho``.

        This **reverses** the earlier ``[DECISION -- TL ratify] option (ii)``,
        which selected the metric-corrected velocity ``G rho``. That decision
        was ratified on the strength of a test that never verified it -- the
        test asserted only ``assertNotEqual(corrected, n_raw)`` between the
        kernel and a hand-rolled rollout that counted one step fewer, so the
        off-by-one satisfied the assertion under *either* pairing. It also
        stated a premise that is empirically false at its own construction:
        at ``inverse_mass_matrix = [100.0, 0.01]`` on a standard normal, the
        two pairings give the *same* count (both 3), so d=2 with a ~3-step
        rollout had no discriminating power at all -- the effect is
        distributional and scales with residual anisotropy.

        The reason for the reversal is affine equivariance: only ``Δthetaᵀrho``
        is invariant under a change of metric (GIST eq. 33), and it is
        exactly the criterion Stan and BlackJAX's own HMC/NUTS use. See
        ``AffineEquivarianceRegressionTest`` for the ground-truth comparison;
        this test pins the pairing directly.
        """
        fixture = _partial_preconditioning_fixture()
        state, _ = fixture.states(_PRECOND_MOMENTUM_SEEDS[0])
        kernel_count = int(
            gist_trajectory_length.num_steps_to_uturn(
                integrators.velocity_verlet,
                fixture.step_size,
                fixture.metric,
                _PRECOND_MAX_STEPS,
            )(state, fixture.logdensity_fn)
        )
        by_pairing = {
            pairing: int(
                uturn_count_reference(
                    fixture.metric, fixture.step_size, _PRECOND_MAX_STEPS, pairing
                )(state, fixture.logdensity_fn)
            )
            for pairing in ("momentum", "velocity")
        }

        # Guard first: the two pairings must actually disagree here, or the
        # assertion below would be vacuous (as it was in the original).
        self.assertNotEqual(by_pairing["momentum"], by_pairing["velocity"])
        self.assertEqual(kernel_count, by_pairing["momentum"])


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
        #
        # `aux` is now the cached `_ForwardRollout` (Issue#1058): `apply_fn`
        # gathers `rollout.states[L - 1]` instead of re-integrating, so the
        # synthetic `aux` must have a buffer entry there. Hand-construct one
        # by broadcasting `integrator_state` itself (position exactly at the
        # origin) across every buffer slot -- `test_num_steps_to_uturn_quarter_
        # period_anchor_d1` above establishes that at position 0 the no-return
        # rollout fires after a step-size-independent ~(pi/2)/step_size steps
        # *regardless* of the momentum's magnitude or direction, so the
        # reverse rollout from this hand-picked proposal deterministically
        # returns a small count (~5 at step_size=0.3), guaranteed far below
        # `implausibly_large_L`.
        state = gist.init(jnp.zeros(2), std_normal_logdensity)
        metric = metrics.default_metric(jnp.ones(2))
        integrator_state = integrators.IntegratorState(
            jnp.zeros(2), jnp.array([1.0, 0.5]), state.logdensity, state.logdensity_grad
        )
        max_num_steps = 50
        implausibly_large_L = jnp.asarray(max_num_steps)
        rollout = gist_trajectory_length._ForwardRollout(
            num_steps_to_uturn=implausibly_large_L,
            states=jax.tree.map(
                lambda leaf: jnp.broadcast_to(leaf, (max_num_steps,) + leaf.shape),
                integrator_state,
            ),
        )
        apply_fn = gist_trajectory_length._apply_fn(
            integrators.velocity_verlet,
            step_size=0.3,
            max_num_steps=max_num_steps,
            path_fraction=0.5,
        )
        _, log_ratio, extra = apply_fn(
            integrator_state,
            implausibly_large_L,
            rollout,
            std_normal_logdensity,
            metric,
        )
        # Guard: the reverse count must actually be far below L, or the
        # rejection below would be checking nothing.
        self.assertLess(int(extra.num_steps_to_uturn_reverse), max_num_steps // 2)
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
