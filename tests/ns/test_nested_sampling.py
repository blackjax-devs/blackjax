"""Test the Nested Sampling algorithms"""

import functools

import chex
import jax
import jax.numpy as jnp
import jax.scipy.stats as stats
from absl.testing import absltest, parameterized

from blackjax.mcmc import random_walk
from blackjax.mcmc.slice import fixed_order
from blackjax.ns import adaptive, base, from_mcmc, integrator, nss, utils


def gaussian_logprior(x):
    """Standard normal prior"""
    return stats.norm.logpdf(x).sum()


def gaussian_loglikelihood(x):
    """Gaussian likelihood with offset"""
    return stats.norm.logpdf(x - 1.0).sum()


def make_init_state_fn(logprior_fn, loglikelihood_fn):
    """Helper to create init_state_fn from logprior and loglikelihood functions."""
    return functools.partial(
        base.init_state_strategy,
        logprior_fn=logprior_fn,
        loglikelihood_fn=loglikelihood_fn,
    )


def make_mock_nsinfo(positions, loglikelihood, loglikelihood_birth, logdensity):
    """Helper to create NSInfo with correct structure."""
    particles = base.StateWithLogLikelihood(
        position=positions,
        logdensity=logdensity,
        loglikelihood=loglikelihood,
        loglikelihood_birth=loglikelihood_birth,
    )
    return base.NSInfo(particles=particles, update_info={})


def uniform_logprior_2d(x):
    """Uniform prior on [-5, 5]^2"""
    return jnp.where(jnp.all(jnp.abs(x) <= 5.0), 0.0, -jnp.inf)


def gaussian_mixture_loglikelihood(x):
    """2D Gaussian mixture for multi-modal testing"""
    mixture1 = stats.norm.logpdf(x - jnp.array([2.0, 0.0])).sum()
    mixture2 = stats.norm.logpdf(x - jnp.array([-2.0, 0.0])).sum()
    return jnp.logaddexp(mixture1, mixture2)


def build_rw_nested_sampler(
    logprior_fn, loglikelihood_fn, num_inner_steps, num_delete=1, scale=0.3
):
    """Nested sampling with a generic random-walk inner kernel.

    Wraps blackjax's additive random-walk kernel through
    ``from_mcmc.reject_constrained_step`` (the propose-then-reject path for any
    MCMC kernel): the RW proposal samples the prior and the likelihood contour is
    enforced by the constrained step. The per-axis proposal std is adapted from
    the live-point spread each NS step. Returns ``(init_fn, step_kernel)``.
    """
    init_state_fn = make_init_state_fn(logprior_fn, loglikelihood_fn)
    rw_kernel = random_walk.build_additive_step()

    def mcmc_init_fn(position, logdensity_fn):
        return random_walk.init(position, logdensity_fn)

    def mcmc_step_fn(rng_key, state, logdensity_fn, sigma):
        return rw_kernel(rng_key, state, logdensity_fn, random_walk.normal(sigma))

    constrained_step = from_mcmc.reject_constrained_step(
        init_state_fn, logprior_fn, mcmc_init_fn, mcmc_step_fn
    )

    def live_sigma(rng_key, state, info, params=None):
        return {"sigma": scale * jnp.std(state.particles.position, axis=0)}

    kernel = from_mcmc.build_kernel(
        constrained_step, num_inner_steps, live_sigma, num_delete
    )

    def init_fn(positions):
        return adaptive.init(
            positions,
            init_state_fn=jax.vmap(init_state_fn),
            update_inner_kernel_params_fn=live_sigma,
        )

    return init_fn, kernel


@functools.lru_cache(maxsize=1)
def _finalised_ns_run():
    """A short hit-and-run NS chain, run once and cached, so the finalise/sample
    tests share a single (compiled) run instead of each spinning up their own.
    Returns ``(final_state, dead_list)``."""
    algo = nss.as_top_level_api(
        gaussian_logprior, gaussian_loglikelihood, num_inner_steps=5, num_delete=2
    )
    state = algo.init(jax.random.normal(jax.random.key(0), (20, 2)))
    step = jax.jit(algo.step)
    dead = []
    for i in range(4):
        state, info = step(jax.random.key(i + 1), state)
        dead.append(info)
    return state, dead


class NestedSamplingTest(chex.TestCase):
    def setUp(self):
        super().setUp()
        self.key = jax.random.key(42)

    def test_base_ns_init(self):
        """Test basic NS initialization"""
        key = jax.random.key(123)
        num_live = 50

        # Generate initial particles
        positions = jax.random.normal(key, (num_live,))

        # Initialize NS state using the correct API
        init_state_fn = jax.vmap(
            make_init_state_fn(gaussian_logprior, gaussian_loglikelihood)
        )
        state = base.init(positions, init_state_fn)

        # Check state structure - particles is now a StateWithLogLikelihood
        chex.assert_shape(state.particles.position, (num_live,))
        chex.assert_shape(state.particles.loglikelihood, (num_live,))
        chex.assert_shape(state.particles.logdensity, (num_live,))
        chex.assert_shape(state.particles.loglikelihood_birth, (num_live,))

        # Check that loglikelihood and logprior are properly computed
        expected_loglik = jax.vmap(gaussian_loglikelihood)(positions)
        expected_logprior = jax.vmap(gaussian_logprior)(positions)

        chex.assert_trees_all_close(state.particles.loglikelihood, expected_loglik)
        chex.assert_trees_all_close(state.particles.logdensity, expected_logprior)

    def test_delete_fn(self):
        """Test particle deletion function"""
        key = jax.random.key(456)
        num_live = 20
        num_delete = 3

        positions = jax.random.normal(key, (num_live,))
        init_state_fn = jax.vmap(
            make_init_state_fn(gaussian_logprior, gaussian_loglikelihood)
        )
        state = base.init(positions, init_state_fn)

        dead_idx, target_idx = base.delete_fn(state, num_delete)

        # Check correct number of deletions
        chex.assert_shape(dead_idx, (num_delete,))
        chex.assert_shape(target_idx, (num_delete,))

        # Check that worst particles are selected
        worst_loglik = jnp.sort(state.particles.loglikelihood)[:num_delete]
        selected_loglik = state.particles.loglikelihood[dead_idx]
        chex.assert_trees_all_close(jnp.sort(selected_loglik), worst_loglik)

    @parameterized.parameters([1, 2, 5])
    def test_ns_step_consistency(self, num_delete):
        """Test NS step maintains particle count"""
        key = jax.random.key(789)
        num_live = 50

        positions = jax.random.normal(key, (num_live, 2))
        init_state_fn = jax.vmap(
            make_init_state_fn(uniform_logprior_2d, gaussian_mixture_loglikelihood)
        )
        state = base.init(positions, init_state_fn)

        # Mock inner kernel for testing — num_delete closed over from outer scope
        def mock_inner_kernel(rng_key, state, loglikelihood_0):
            particles = state.particles

            # Select start particles from survivors
            choice_key, sample_key = jax.random.split(rng_key)
            weights = (particles.loglikelihood > loglikelihood_0).astype(jnp.float32)
            weights = jnp.where(weights.sum() > 0.0, weights, jnp.ones_like(weights))
            start_idx = jax.random.choice(
                choice_key,
                len(weights),
                shape=(num_delete,),
                p=weights / weights.sum(),
                replace=True,
            )
            start_state = jax.tree.map(lambda x: x[start_idx], particles)

            # Simple random walk for testing
            def single_step(rng_key, state):
                new_pos = (
                    state.position
                    + jax.random.normal(rng_key, state.position.shape) * 0.1
                )
                new_state = base.init_state_strategy(
                    new_pos,
                    uniform_logprior_2d,
                    gaussian_mixture_loglikelihood,
                    loglikelihood_birth=loglikelihood_0,
                )
                return new_state

            sample_keys = jax.random.split(sample_key, num_delete)
            new_particles = jax.vmap(single_step)(sample_keys, start_state)
            return new_particles, {}

        delete_fn = functools.partial(base.delete_fn, num_delete=num_delete)
        kernel = base.build_kernel(delete_fn, mock_inner_kernel)

        # Test that the kernel can be constructed with mock components
        self.assertTrue(callable(kernel))

        # Test delete function works
        dead_idx, target_idx = base.delete_fn(state, num_delete)
        chex.assert_shape(dead_idx, (num_delete,))
        chex.assert_shape(target_idx, (num_delete,))

        # Actually run the kernel and check post-conditions
        new_state, info = kernel(key, state)

        # Particle count preserved
        chex.assert_shape(
            new_state.particles.position,
            state.particles.position.shape,
        )
        # Dead particles returned in info
        chex.assert_shape(info.particles.loglikelihood, (num_delete,))
        # Dead particles are the worst from original state
        worst_loglik = jnp.sort(state.particles.loglikelihood)[:num_delete]
        chex.assert_trees_all_close(
            jnp.sort(info.particles.loglikelihood), worst_loglik
        )

    def test_utils_functions(self):
        """Test utility functions"""
        key = jax.random.key(101112)

        # Create mock dead info
        n_dead = 20
        dead_loglik = jnp.sort(jax.random.uniform(key, (n_dead,))) * 10 - 5
        dead_loglik_birth = jnp.full_like(dead_loglik, -jnp.inf)

        # Create StateWithLogLikelihood for particles
        particles = base.StateWithLogLikelihood(
            position=jnp.zeros((n_dead, 2)),
            logdensity=jnp.zeros(n_dead),
            loglikelihood=dead_loglik,
            loglikelihood_birth=dead_loglik_birth,
        )

        mock_info = base.NSInfo(particles=particles, update_info={})

        # Test compute_num_live
        num_live = utils.compute_num_live(mock_info)
        chex.assert_shape(num_live, (n_dead,))

        # Test logX simulation
        logX_seq, logdX_seq = utils.logX(key, mock_info, shape=10)
        chex.assert_shape(logX_seq, (n_dead, 10))
        chex.assert_shape(logdX_seq, (n_dead, 10))

        # Check logX is decreasing
        self.assertTrue(jnp.all(logX_seq[1:] <= logX_seq[:-1]))


class AdaptiveNestedSamplingTest(chex.TestCase):
    def setUp(self):
        super().setUp()
        self.key = jax.random.key(42)

    def test_adaptive_init(self):
        """Test adaptive NS initialization"""
        key = jax.random.key(123)
        num_live = 30

        positions = jax.random.normal(key, (num_live,))

        def mock_update_params_fn(rng_key, state, info, current_params):
            return {"test_param": 1.0}

        init_state_fn = jax.vmap(
            make_init_state_fn(gaussian_logprior, gaussian_loglikelihood)
        )
        state = adaptive.init(
            positions,
            init_state_fn,
            update_inner_kernel_params_fn=mock_update_params_fn,
        )

        # Check that inner kernel params were set
        self.assertEqual(state.inner_kernel_params["test_param"], 1.0)


class NestedSliceSamplingTest(chex.TestCase):
    def setUp(self):
        super().setUp()
        self.key = jax.random.key(42)

    def test_nss_direction_functions(self):
        """NSS direction: covariance-shaped, scaled to Mahalanobis norm 2."""
        key = jax.random.key(456)
        cov = jnp.array([[2.0, 0.5, 0.0], [0.5, 1.0, 0.0], [0.0, 0.0, 1.5]])

        direction = nss.sample_direction_from_covariance(key, jnp.zeros(3), cov)

        chex.assert_shape(direction, (3,))
        mahalanobis = jnp.sqrt(direction @ jnp.linalg.inv(cov) @ direction)
        self.assertAlmostEqual(float(mahalanobis), 2.0, places=4)

    def test_covariance_proposal(self):
        """The factored NSS proposal: covariance step, likelihood-gated is_valid."""
        init_state_fn = make_init_state_fn(gaussian_logprior, gaussian_loglikelihood)
        position = jnp.array([1.0, 0.0, 0.0])
        loglik = gaussian_loglikelihood(position)

        # Threshold below the current likelihood: the current point (t=0) passes.
        gen = nss.covariance_proposal(init_state_fn, loglik - 1.0, jnp.eye(3))
        state0, valid0 = gen(self.key, position, None)(0.0)
        chex.assert_trees_all_close(state0.position, position)
        self.assertAlmostEqual(float(state0.loglikelihood), float(loglik), places=5)
        self.assertTrue(bool(valid0))

        # Threshold above it: the same point is gated out (is_valid is False).
        gen_high = nss.covariance_proposal(init_state_fn, loglik + 1.0, jnp.eye(3))
        _, valid_high = gen_high(self.key, position, None)(0.0)
        self.assertFalse(bool(valid_high))

    def test_nss_kernel_construction(self):
        """Test NSS kernel can be constructed"""
        init_state_fn = make_init_state_fn(gaussian_logprior, gaussian_loglikelihood)
        kernel = nss.build_kernel(init_state_fn, num_inner_steps=10)

        # Test that kernel is callable
        self.assertTrue(callable(kernel))

    def test_top_level_api_seams(self):
        """proposal / inner_kernel_params are overridable on the top-level API."""

        def my_params(rng_key, state, info, params=None):
            return {"cov": 2.0 * jnp.eye(2)}

        algo = nss.as_top_level_api(
            gaussian_logprior,
            gaussian_loglikelihood,
            num_inner_steps=4,
            proposal=nss.covariance_proposal,
            inner_kernel_params=my_params,
        )
        state = algo.init(jnp.zeros((20, 2)))
        # init seeds the params with the supplied function, not just the default
        chex.assert_trees_all_close(state.inner_kernel_params["cov"], 2.0 * jnp.eye(2))
        # and a full step runs end-to-end with the custom seams
        new_state, _ = jax.jit(algo.step)(self.key, state)
        chex.assert_shape(new_state.particles.position, (20, 2))

    def test_swig_kernel_construction(self):
        """The SwiG kernel builds and the top-level API exposes the seam triplet."""
        init_state_fn = make_init_state_fn(gaussian_logprior, gaussian_loglikelihood)
        kernel = nss.build_swig_kernel(init_state_fn, num_inner_steps=4)
        self.assertTrue(callable(kernel))

    def test_swig_top_level_api_seams(self):
        """coordinate_order / inner_kernel_params are overridable on the SwiG API,
        and there is no bespoke init -- the shared generic init seeds the widths."""

        def my_widths(rng_key, state, info, params=None):
            return {"widths": jnp.array([0.5, 2.0])}

        algo = nss.swig_as_top_level_api(
            gaussian_logprior,
            gaussian_loglikelihood,
            num_inner_steps=4,
            coordinate_order=fixed_order,
            inner_kernel_params=my_widths,
        )
        state = algo.init(jnp.zeros((20, 2)))
        # init seeds the per-axis widths with the supplied function
        chex.assert_trees_all_close(
            state.inner_kernel_params["widths"], jnp.array([0.5, 2.0])
        )
        # and a full SwiG sweep step runs end-to-end with the custom seams
        new_state, _ = jax.jit(algo.step)(self.key, state)
        chex.assert_shape(new_state.particles.position, (20, 2))


class NestedSamplingStatisticalTest(chex.TestCase):
    """Statistical correctness tests for nested sampling algorithms."""

    def setUp(self):
        super().setUp()
        self.key = jax.random.key(42)

    def tearDown(self):
        # Release compiled XLA kernels between heavy end-to-end tests to avoid
        # cumulative memory pressure under pytest-xdist parallel workers.
        jax.clear_caches()
        super().tearDown()

    def test_1d_gaussian_evidence_estimation(self):
        """Test evidence estimation with analytic validation for unnormalized Gaussian."""

        # Simple case: unnormalized Gaussian likelihood exp(-0.5*x²), uniform prior [-3,3]
        prior_a, prior_b = -3.0, 3.0

        def logprior_fn(x):
            return jnp.where(
                (x >= prior_a) & (x <= prior_b), -jnp.log(prior_b - prior_a), -jnp.inf
            )

        def loglikelihood_fn(x):
            # Unnormalized Gaussian: exp(-0.5 * x²)
            return -0.5 * x**2

        # Analytic evidence: Z = ∫[-3,3] (1/6) * exp(-0.5*x²) dx
        # = (1/6) * √(2π) * [Φ(3) - Φ(-3)]
        from scipy.stats import norm

        prior_width = prior_b - prior_a
        integral_part = jnp.sqrt(2 * jnp.pi) * (norm.cdf(3.0) - norm.cdf(-3.0))
        analytical_evidence = integral_part / prior_width
        analytical_log_evidence = jnp.log(analytical_evidence)

        # Generate mock nested sampling data
        num_steps = 60
        key = jax.random.key(42)

        # Create positions spanning the prior range
        positions = jnp.linspace(prior_a + 0.05, prior_b - 0.05, num_steps).reshape(
            -1, 1
        )
        dead_loglik = jax.vmap(loglikelihood_fn)(positions.flatten())
        dead_logprior = jax.vmap(logprior_fn)(positions.flatten())

        # Sort by likelihood (as NS naturally produces)
        sorted_indices = jnp.argsort(dead_loglik)
        dead_loglik = dead_loglik[sorted_indices]
        positions = positions[sorted_indices]
        dead_logprior = dead_logprior[sorted_indices]

        # Birth likelihoods - start from prior
        dead_loglik_birth = jnp.full_like(dead_loglik, -jnp.inf)

        # Create NSInfo object
        mock_info = make_mock_nsinfo(
            positions, dead_loglik, dead_loglik_birth, dead_logprior
        )

        # Generate many evidence estimates for statistical testing
        n_evidence_samples = 500
        key = jax.random.key(789)
        keys = jax.random.split(key, n_evidence_samples)

        def single_evidence_estimate(rng_key):
            log_weights_matrix = utils.log_weights(rng_key, mock_info, shape=15)
            return jax.scipy.special.logsumexp(log_weights_matrix, axis=0)

        # Compute evidence estimates
        log_evidence_samples = jax.vmap(single_evidence_estimate)(keys)
        log_evidence_samples = log_evidence_samples.flatten()

        # Statistical validation
        mean_estimate = jnp.mean(log_evidence_samples)
        std_estimate = jnp.std(log_evidence_samples)

        # Check statistical consistency with 95% confidence interval
        # For mock data with simplified NS, expect some bias but should be in ballpark
        tolerance = 2.0 * std_estimate  # 95% CI
        bias = jnp.abs(mean_estimate - analytical_log_evidence)

        self.assertLess(
            bias,
            tolerance,
            f"Evidence estimate {mean_estimate} vs analytic {analytical_log_evidence} "
            f"differs by {bias}, which exceeds 2σ = {tolerance}",
        )

        # Also test that individual estimates are reasonable
        self.assertFalse(
            jnp.any(jnp.isnan(log_evidence_samples)),
            "No evidence estimates should be NaN",
        )
        self.assertFalse(
            jnp.any(jnp.isinf(log_evidence_samples)),
            "No evidence estimates should be infinite",
        )

        # Check that estimates are in a reasonable range
        self.assertGreater(
            mean_estimate, analytical_log_evidence - 1.0, "Mean estimate not too low"
        )
        self.assertLess(
            mean_estimate, analytical_log_evidence + 1.0, "Mean estimate not too high"
        )

    def test_uniform_prior_evidence(self):
        """Test evidence estimation for uniform prior with simple likelihood."""

        # Setup: Uniform prior on [0, 1], simple likelihood
        def logprior_fn(x):
            return jnp.where((x >= 0.0) & (x <= 1.0), 0.0, -jnp.inf)

        def loglikelihood_fn(x):
            # Simple quadratic likelihood peaked at 0.5
            return -10.0 * (x - 0.5) ** 2

        # Analytical evidence can be computed numerically for comparison
        # Z = integral_0^1 exp(-10(x-0.5)^2) dx ≈ sqrt(π/10) * erf(...)

        num_live = 50
        key = jax.random.key(456)

        # Initialize particles uniformly in [0, 1]
        positions = jax.random.uniform(key, (num_live,))
        init_state_fn = jax.vmap(make_init_state_fn(logprior_fn, loglikelihood_fn))
        state = base.init(positions, init_state_fn)

        # Check that initialization worked correctly
        self.assertTrue(jnp.all(state.particles.position >= 0.0))
        self.assertTrue(jnp.all(state.particles.position <= 1.0))
        self.assertFalse(jnp.any(jnp.isinf(state.particles.logdensity)))
        self.assertFalse(jnp.any(jnp.isnan(state.particles.loglikelihood)))

    def test_evidence_monotonicity(self):
        """Real NS steps value-check the production integrator: prior volume
        logX strictly decreases, evidence logZ accumulates, and every replaced
        particle beats the death contour."""
        num_live = 50
        algo = nss.as_top_level_api(
            gaussian_logprior, gaussian_loglikelihood, num_inner_steps=5
        )
        positions = jax.random.normal(jax.random.key(789), (num_live, 2))
        state = algo.init(positions)

        # init_integrator contract: full prior volume, no evidence accumulated yet
        self.assertAlmostEqual(float(state.integrator.logX), 0.0, places=6)
        self.assertEqual(float(state.integrator.logZ), float("-inf"))

        def step(state, key):
            new_state, info = algo.step(key, state)
            # contour post-condition: every live point beats the deleted contour
            contour = info.particles.loglikelihood.max()
            ok = jnp.all(new_state.particles.loglikelihood > contour)
            integ = new_state.integrator
            return new_state, (integ.logX, integ.logZ, ok)

        keys = jax.random.split(jax.random.key(0), 12)
        _, (logX, logZ, ok) = jax.lax.scan(step, state, keys)

        self.assertTrue(bool(jnp.all(jnp.diff(logX) < 0)))  # prior volume shrinks
        self.assertTrue(bool(jnp.all(jnp.diff(logZ) >= 0)))  # evidence accumulates
        self.assertTrue(bool(jnp.all(jnp.isfinite(logZ))))  # finite after death
        self.assertTrue(bool(ok.all()))  # likelihood constraint held

    def test_2d_gaussian_evidence(self):
        """End-to-end: NS recovers the analytic evidence of a 2D Gaussian prior
        x correlated-Gaussian likelihood, via the production integrator."""
        prior_mean, prior_cov = jnp.zeros(2), jnp.eye(2)
        like_mean = jnp.array([0.5, -0.5])
        like_cov = jnp.array([[1.0, 0.3], [0.3, 0.6]])

        def logprior(x):
            return stats.multivariate_normal.logpdf(x, prior_mean, prior_cov)

        def loglikelihood(x):
            return stats.multivariate_normal.logpdf(x, like_mean, like_cov)

        # Gaussian x Gaussian: Z = N(prior_mean; like_mean, prior_cov + like_cov)
        analytic_logZ = float(
            stats.multivariate_normal.logpdf(
                prior_mean, like_mean, prior_cov + like_cov
            )
        )

        n_live = 50
        key = jax.random.key(0)
        key, init_key = jax.random.split(key)
        positions = jax.random.multivariate_normal(
            init_key, prior_mean, prior_cov, (n_live,)
        )
        # num_delete=2 exercises the batch-delete path while keeping the compiled
        # kernel small (the kernel vmaps over num_delete replacements); the dynamic
        # termination criterion converges well within the 200-step cap for dim=2.
        algo = nss.as_top_level_api(
            logprior, loglikelihood, num_inner_steps=6, num_delete=2
        )
        state = algo.init(positions)

        step = jax.jit(algo.step)
        for _ in range(200):  # dynamic termination (logZ_live - logZ < -3), capped
            if float(state.integrator.logZ_live - state.integrator.logZ) < -3.0:
                break
            key, sub = jax.random.split(key)
            state, _ = step(sub, state)

        total_logZ = float(
            jnp.logaddexp(state.integrator.logZ, state.integrator.logZ_live)
        )
        self.assertAlmostEqual(total_logZ, analytic_logZ, delta=0.75)

    def test_swig_2d_gaussian_evidence(self):
        """End-to-end: the SwiG (slice-within-Gibbs) sampler recovers the analytic
        evidence of a 2D Gaussian prior x axis-aligned-Gaussian likelihood."""
        prior_mean, prior_cov = jnp.zeros(2), jnp.eye(2)
        like_mean = jnp.array([0.5, -0.5])
        like_cov = jnp.array([[1.0, 0.0], [0.0, 0.6]])  # axis-aligned: SwiG's regime

        def logprior(x):
            return stats.multivariate_normal.logpdf(x, prior_mean, prior_cov)

        def loglikelihood(x):
            return stats.multivariate_normal.logpdf(x, like_mean, like_cov)

        analytic_logZ = float(
            stats.multivariate_normal.logpdf(
                prior_mean, like_mean, prior_cov + like_cov
            )
        )

        n_live = 50
        key = jax.random.key(0)
        key, init_key = jax.random.split(key)
        positions = jax.random.multivariate_normal(
            init_key, prior_mean, prior_cov, (n_live,)
        )
        algo = nss.swig_as_top_level_api(
            logprior, loglikelihood, num_inner_steps=6, num_delete=2
        )
        state = algo.init(positions)

        step = jax.jit(algo.step)
        for _ in range(200):  # dynamic termination (logZ_live - logZ < -3), capped
            if float(state.integrator.logZ_live - state.integrator.logZ) < -3.0:
                break
            key, sub = jax.random.split(key)
            state, _ = step(sub, state)

        total_logZ = float(
            jnp.logaddexp(state.integrator.logZ, state.integrator.logZ_live)
        )
        self.assertAlmostEqual(total_logZ, analytic_logZ, delta=0.75)

    def test_evidence_integrator_constant_likelihood(self):
        """Telescoping regression: a constant likelihood makes the shells sum to
        the full prior volume, so the integrator must return Z = 1 (logZ_total = 0)
        for any num_live and num_delete. Anchoring shells on the post-deletion
        volume instead biases logZ low by ~1/n, which this catches."""

        def constant_particles(n):
            zeros = jnp.zeros(n)
            return base.StateWithLogLikelihood(
                position=jnp.zeros((n, 1)),
                logdensity=zeros,
                loglikelihood=zeros,  # logL = 0 everywhere
                loglikelihood_birth=jnp.full(n, -jnp.inf),
            )

        def swept_logZ(live, dead, n_iter):
            # accumulate n_iter constant-likelihood deletions in one compiled
            # scan (vs an un-jitted Python loop of eager update_integrator calls)
            def body(integ, _):
                return integrator.update_integrator(integ, live, dead), None

            integ, _ = jax.lax.scan(
                body, integrator.init_integrator(live), None, length=n_iter
            )
            return jnp.logaddexp(integ.logZ, integ.logZ_live)

        for num_live, num_delete in [(20, 1), (20, 4), (50, 1), (50, 5)]:
            # dead shells (logZ) + remaining live volume (logZ_live) = full prior
            total_logZ = float(
                swept_logZ(
                    constant_particles(num_live),
                    constant_particles(num_delete),
                    (num_live * 8) // num_delete,
                )
            )
            self.assertAlmostEqual(
                total_logZ,
                0.0,
                places=2,
                msg=f"constant-L evidence off for "
                f"num_live={num_live}, num_delete={num_delete}",
            )

    def test_reject_constrained_step_rw_evidence(self):
        """End-to-end: NS with a generic random-walk inner kernel (wrapped via
        from_mcmc.reject_constrained_step) recovers the analytic evidence of a 2D
        Gaussian prior x axis-aligned-Gaussian likelihood. Exercises the
        propose-then-reject path and ConstrainedMCMCInfo, which the slice
        samplers bypass."""
        prior_mean, prior_cov = jnp.zeros(2), jnp.eye(2)
        like_mean = jnp.array([0.5, -0.5])
        like_cov = jnp.array([[1.0, 0.0], [0.0, 0.6]])

        def logprior(x):
            return stats.multivariate_normal.logpdf(x, prior_mean, prior_cov)

        def loglikelihood(x):
            return stats.multivariate_normal.logpdf(x, like_mean, like_cov)

        analytic_logZ = float(
            stats.multivariate_normal.logpdf(
                prior_mean, like_mean, prior_cov + like_cov
            )
        )

        n_live = 50
        key = jax.random.key(0)
        key, init_key = jax.random.split(key)
        positions = jax.random.multivariate_normal(
            init_key, prior_mean, prior_cov, (n_live,)
        )
        # num_inner_steps=6 satisfies the >= max(5, 2*dim) = 5 floor for dim=2;
        # num_delete=2 keeps the vmapped replacement kernel compact.
        init_fn, kernel = build_rw_nested_sampler(
            logprior, loglikelihood, num_inner_steps=6, num_delete=2
        )
        state = init_fn(positions)

        step = jax.jit(kernel)
        info = None
        for _ in range(200):  # dynamic termination (logZ_live - logZ < -3), capped
            if float(state.integrator.logZ_live - state.integrator.logZ) < -3.0:
                break
            key, sub = jax.random.split(key)
            state, info = step(sub, state)

        total_logZ = float(
            jnp.logaddexp(state.integrator.logZ, state.integrator.logZ_live)
        )
        self.assertAlmostEqual(total_logZ, analytic_logZ, delta=1.0)
        # the inner update info is the propose-then-reject info, not a SliceInfo
        self.assertIsInstance(info.update_info, from_mcmc.ConstrainedMCMCInfo)

    def test_finalise_combines_dead_and_live(self):
        """finalise concatenates all dead particles with the final live set; per
        its contract the update_info covers the dead steps only, so it is shorter
        than particles by the number of live points (and is None when disabled)."""
        state, dead = _finalised_ns_run()
        n_live = state.particles.loglikelihood.shape[0]
        n_dead = sum(d.particles.loglikelihood.shape[0] for d in dead)

        final = utils.finalise(state, dead)
        self.assertEqual(final.particles.loglikelihood.shape[0], n_dead + n_live)
        ui_len = jax.tree_util.tree_leaves(final.update_info)[0].shape[0]
        self.assertEqual(ui_len, n_dead)  # dead steps only, no live entry

        final_no_ui = utils.finalise(state, dead, update_info=False)
        self.assertIsNone(final_no_ui.update_info)

    def test_sample_resamples_from_finalised(self):
        """sample draws (with replacement) from the finalised particle set by
        weight: the output has the requested leading dimension and every drawn
        point is one of the finalised particles."""
        state, dead = _finalised_ns_run()
        final = utils.finalise(state, dead)

        n_samples = 500
        samples = utils.sample(jax.random.key(99), final, shape=n_samples)
        chex.assert_shape(samples.position, (n_samples, 2))
        chex.assert_shape(samples.loglikelihood, (n_samples,))
        self.assertFalse(bool(jnp.any(jnp.isnan(samples.position))))
        # resampling, not generation: every drawn point is a finalised particle
        self.assertTrue(
            bool(
                jnp.all(jnp.isin(samples.loglikelihood, final.particles.loglikelihood))
            )
        )

    def test_nested_sampling_utils_statistical_properties(self):
        """Test statistical properties of nested sampling utility functions."""
        key = jax.random.key(101112)

        # Create realistic mock data
        n_dead = 100

        # Generate realistic loglikelihood sequence (increasing)
        base_loglik = jnp.linspace(-10, -1, n_dead)
        noise = jax.random.normal(key, (n_dead,)) * 0.1
        dead_loglik = jnp.sort(base_loglik + noise)

        # Create more realistic birth likelihoods that reflect actual NS behavior
        # Particles can be born at various levels, not just at previous death
        key, subkey = jax.random.split(key)
        birth_noise = jax.random.uniform(subkey, (n_dead,)) * 2.0 - 1.0  # [-1, 1]
        dead_loglik_birth = jnp.concatenate(
            [
                jnp.array([-jnp.inf]),  # First particle born from prior
                dead_loglik[:-1] + birth_noise[1:] * 0.5,  # Others with some variation
            ]
        )
        # Ensure birth likelihoods don't exceed death likelihoods
        dead_loglik_birth = jnp.minimum(dead_loglik_birth, dead_loglik - 0.01)

        mock_info = make_mock_nsinfo(
            jnp.zeros((n_dead, 2)), dead_loglik, dead_loglik_birth, jnp.zeros(n_dead)
        )

        # Test compute_num_live
        num_live = utils.compute_num_live(mock_info)
        chex.assert_shape(num_live, (n_dead,))

        # Basic sanity checks for number of live points
        # NOTE: num_live should NOT be monotonically decreasing in general NS!
        # It follows a sawtooth pattern as particles die and are replenished
        self.assertTrue(
            jnp.all(num_live >= 1), "Should always have at least 1 live point"
        )
        self.assertTrue(
            jnp.all(num_live <= 1000),  # Reasonable upper bound
            "Number of live points should be reasonable",
        )
        self.assertFalse(
            jnp.any(jnp.isnan(num_live)), "Number of live points should not be NaN"
        )

        # Test logX simulation
        n_samples = 50
        logX_seq, logdX_seq = utils.logX(key, mock_info, shape=n_samples)
        chex.assert_shape(logX_seq, (n_dead, n_samples))
        chex.assert_shape(logdX_seq, (n_dead, n_samples))

        # Log volumes should be decreasing
        self.assertTrue(
            jnp.all(logX_seq[1:] <= logX_seq[:-1]), "Log volumes should be decreasing"
        )

        # All log volume elements should be negative (since dX < X)
        finite_logdX = logdX_seq[jnp.isfinite(logdX_seq)]
        if len(finite_logdX) > 0:
            self.assertTrue(
                jnp.all(finite_logdX <= 0.0), "Log volume elements should be negative"
            )

        # Test log_weights function
        log_weights_matrix = utils.log_weights(key, mock_info, shape=n_samples)
        chex.assert_shape(log_weights_matrix, (n_dead, n_samples))

        # Weights should be finite for most particles
        finite_weights = jnp.isfinite(log_weights_matrix)
        self.assertGreater(
            jnp.sum(finite_weights),
            n_dead * n_samples * 0.5,
            "Most weights should be finite",
        )

    def test_gaussian_evidence_narrow_prior(self):
        """Test evidence estimation with narrow prior for challenging case."""

        # Setup: Gaussian likelihood with narrow uniform prior (more challenging)
        mu_true = 1.2
        sigma_true = 0.6
        prior_a, prior_b = 0.8, 1.6  # Narrow prior around the mean

        def logprior_fn(x):
            return jnp.where(
                (x >= prior_a) & (x <= prior_b), -jnp.log(prior_b - prior_a), -jnp.inf
            )

        def loglikelihood_fn(x):
            return -0.5 * ((x - mu_true) / sigma_true) ** 2 - 0.5 * jnp.log(
                2 * jnp.pi * sigma_true**2
            )

        # Analytic evidence
        from scipy.stats import norm

        analytical_evidence = (
            norm.cdf((prior_b - mu_true) / sigma_true)
            - norm.cdf((prior_a - mu_true) / sigma_true)
        ) / (prior_b - prior_a)
        analytical_log_evidence = jnp.log(analytical_evidence)

        # Generate mock NS data with higher resolution for narrow prior
        num_steps = 60
        key = jax.random.key(12345)

        # Dense sampling in the narrow prior region
        positions = jnp.linspace(prior_a + 0.01, prior_b - 0.01, num_steps).reshape(
            -1, 1
        )
        dead_loglik = jax.vmap(loglikelihood_fn)(positions.flatten())
        dead_logprior = jax.vmap(logprior_fn)(positions.flatten())

        # Sort by likelihood
        sorted_indices = jnp.argsort(dead_loglik)
        dead_loglik = dead_loglik[sorted_indices]
        positions = positions[sorted_indices]
        dead_logprior = dead_logprior[sorted_indices]

        # Birth likelihoods
        key, subkey = jax.random.split(key)
        birth_noise = jax.random.uniform(subkey, (num_steps,)) * 0.3 - 0.15
        dead_loglik_birth = jnp.concatenate(
            [jnp.array([-jnp.inf]), dead_loglik[:-1] + birth_noise[1:]]
        )
        dead_loglik_birth = jnp.minimum(dead_loglik_birth, dead_loglik - 0.01)

        mock_info = make_mock_nsinfo(
            positions, dead_loglik, dead_loglik_birth, dead_logprior
        )

        # Generate evidence estimates for statistical testing
        n_evidence_samples = 800
        key = jax.random.key(555)
        keys = jax.random.split(key, n_evidence_samples)

        def single_evidence_estimate(rng_key):
            log_weights_matrix = utils.log_weights(rng_key, mock_info, shape=15)
            return jax.scipy.special.logsumexp(log_weights_matrix, axis=0)

        log_evidence_samples = jax.vmap(single_evidence_estimate)(keys)
        log_evidence_samples = log_evidence_samples.flatten()

        # Statistical validation
        mean_estimate = jnp.mean(log_evidence_samples)
        std_estimate = jnp.std(log_evidence_samples)

        # 99% confidence interval test
        lower_bound = mean_estimate - 2.576 * std_estimate  # 99% CI
        upper_bound = mean_estimate + 2.576 * std_estimate

        self.assertGreater(
            analytical_log_evidence,
            lower_bound,
            f"Analytic evidence {analytical_log_evidence} below 99% CI lower bound {lower_bound}",
        )
        self.assertLess(
            analytical_log_evidence,
            upper_bound,
            f"Analytic evidence {analytical_log_evidence} above 99% CI upper bound {upper_bound}",
        )

    def test_evidence_integration_simple_case(self):
        """Test evidence calculation for a simple analytical case with constant likelihood."""
        # Test case: uniform prior on [0,2], constant likelihood
        # Evidence = ∫[0,2] (1/width) * exp(loglik_constant) dx = exp(loglik_constant)

        loglik_constant = -1.5
        prior_width = 2.0  # Prior on [0, 2]
        n_dead = 40

        # Analytic answer: evidence = ∫[0,2] (1/2) * exp(-1.5) dx = exp(-1.5)
        analytical_log_evidence = loglik_constant

        # Mock data: all particles have same likelihood (constant function)
        dead_loglik = jnp.full(n_dead, loglik_constant)
        dead_loglik_birth = jnp.full(n_dead, -jnp.inf)  # All from prior

        mock_info = make_mock_nsinfo(
            jnp.zeros((n_dead, 1)),
            dead_loglik,
            dead_loglik_birth,
            jnp.full(n_dead, -jnp.log(prior_width)),  # Uniform prior log density
        )

        # Generate many evidence estimates
        n_samples = 500
        key = jax.random.key(999)
        keys = jax.random.split(key, n_samples)

        def single_evidence_estimate(rng_key):
            log_weights_matrix = utils.log_weights(rng_key, mock_info, shape=25)
            return jax.scipy.special.logsumexp(log_weights_matrix, axis=0)

        log_evidence_samples = jax.vmap(single_evidence_estimate)(keys)
        log_evidence_samples = log_evidence_samples.flatten()

        mean_estimate = jnp.mean(log_evidence_samples)
        std_estimate = jnp.std(log_evidence_samples)

        # For constant likelihood case, should be very accurate
        # 95% confidence interval
        lower_bound = mean_estimate - 1.96 * std_estimate
        upper_bound = mean_estimate + 1.96 * std_estimate

        self.assertGreater(
            analytical_log_evidence,
            lower_bound,
            f"Analytic evidence {analytical_log_evidence} below 95% CI",
        )
        self.assertLess(
            analytical_log_evidence,
            upper_bound,
            f"Analytic evidence {analytical_log_evidence} above 95% CI",
        )

    def test_effective_sample_size_calculation(self):
        """Test effective sample size calculation."""
        key = jax.random.key(67890)

        # Create mock data with varying weights
        n_dead = 50
        dead_loglik = jax.random.uniform(key, (n_dead,)) * 5 - 10  # Range [-10, -5]
        dead_loglik_birth = jnp.full(n_dead, -jnp.inf)

        mock_info = make_mock_nsinfo(
            jnp.zeros((n_dead, 1)),
            jnp.sort(dead_loglik),  # Ensure increasing
            dead_loglik_birth,
            jnp.zeros(n_dead),
        )

        # Calculate ESS
        ess_value = utils.ess(key, mock_info)

        # ESS should be positive and reasonable
        self.assertIsInstance(ess_value, (float, jax.Array))
        self.assertGreater(ess_value, 0.0, "ESS should be positive")
        self.assertLessEqual(
            ess_value, n_dead, "ESS should not exceed number of samples"
        )
        self.assertFalse(jnp.isnan(ess_value), "ESS should not be NaN")


if __name__ == "__main__":
    absltest.main()
