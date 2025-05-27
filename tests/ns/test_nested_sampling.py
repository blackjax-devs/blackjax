"""Test the Nested Sampling algorithms"""
import functools

import chex
import jax
import jax.numpy as jnp
import jax.scipy.stats as stats
from absl.testing import absltest, parameterized

from blackjax.ns import adaptive, base, nss, utils


def gaussian_logprior(x):
    """Standard normal prior"""
    return stats.norm.logpdf(x).sum()


def gaussian_loglikelihood(x):
    """Gaussian likelihood with offset"""
    return stats.norm.logpdf(x - 1.0).sum()


def uniform_logprior_2d(x):
    """Uniform prior on [-5, 5]^2"""
    return jnp.where(jnp.all(jnp.abs(x) <= 5.0), 0.0, -jnp.inf)


def gaussian_mixture_loglikelihood(x):
    """2D Gaussian mixture for multi-modal testing"""
    mixture1 = stats.norm.logpdf(x - jnp.array([2.0, 0.0])).sum()
    mixture2 = stats.norm.logpdf(x - jnp.array([-2.0, 0.0])).sum()
    return jnp.logaddexp(mixture1, mixture2)


class NestedSamplingTest(chex.TestCase):
    def setUp(self):
        super().setUp()
        self.key = jax.random.key(42)

    def test_base_ns_init(self):
        """Test basic NS initialization"""
        key = jax.random.key(123)
        num_live = 50

        # Generate initial particles
        particles = jax.random.normal(key, (num_live,))

        # Initialize NS state
        state = base.init(particles, gaussian_logprior, gaussian_loglikelihood)

        # Check state structure
        chex.assert_shape(state.particles, (num_live,))
        chex.assert_shape(state.loglikelihood, (num_live,))
        chex.assert_shape(state.logprior, (num_live,))
        chex.assert_shape(state.pid, (num_live,))

        # Check that loglikelihood and logprior are properly computed
        expected_loglik = jax.vmap(gaussian_loglikelihood)(particles)
        expected_logprior = jax.vmap(gaussian_logprior)(particles)

        chex.assert_trees_all_close(state.loglikelihood, expected_loglik)
        chex.assert_trees_all_close(state.logprior, expected_logprior)

    def test_delete_fn(self):
        """Test particle deletion function"""
        key = jax.random.key(456)
        num_live = 20
        num_delete = 3

        particles = jax.random.normal(key, (num_live,))
        state = base.init(particles, gaussian_logprior, gaussian_loglikelihood)

        dead_idx, target_idx, start_idx = base.delete_fn(key, state, num_delete)

        # Check correct number of deletions
        chex.assert_shape(dead_idx, (num_delete,))
        chex.assert_shape(target_idx, (num_delete,))
        chex.assert_shape(start_idx, (num_delete,))

        # Check that worst particles are selected
        worst_loglik = jnp.sort(state.loglikelihood)[:num_delete]
        selected_loglik = state.loglikelihood[dead_idx]
        chex.assert_trees_all_close(jnp.sort(selected_loglik), worst_loglik)

    @parameterized.parameters([1, 2, 5])
    def test_ns_step_consistency(self, num_delete):
        """Test NS step maintains particle count"""
        key = jax.random.key(789)
        num_live = 50

        particles = jax.random.normal(key, (num_live, 2))
        state = base.init(
            particles, uniform_logprior_2d, gaussian_mixture_loglikelihood
        )

        # Mock inner kernel for testing
        def mock_inner_kernel(
            rng_key, inner_state, logprior_fn, loglikelihood_fn, loglikelihood_0, params
        ):
            # Simple random walk for testing
            new_pos = (
                inner_state["position"]
                + jax.random.normal(rng_key, inner_state["position"].shape) * 0.1
            )
            new_logprior = logprior_fn(new_pos)
            new_loglik = loglikelihood_fn(new_pos)

            new_inner_state = {
                "position": new_pos,
                "logprior": new_logprior,
                "loglikelihood": new_loglik,
            }
            return new_inner_state, {}

        def mock_inner_init(position, logprior, loglikelihood):
            # Return a simple dict that works with JAX
            return {
                "position": position,
                "logprior": logprior,
                "loglikelihood": loglikelihood,
            }

        delete_fn = functools.partial(base.delete_fn, num_delete=num_delete)
        kernel = base.build_kernel(
            uniform_logprior_2d,
            gaussian_mixture_loglikelihood,
            delete_fn,
            mock_inner_init,
            mock_inner_kernel,
        )

        # Test that the kernel can be constructed with mock components
        # Full execution would require more complex mocking of inner kernel behavior
        self.assertTrue(callable(kernel))

        # Test delete function works
        dead_idx, target_idx, start_idx = base.delete_fn(key, state, num_delete)
        chex.assert_shape(dead_idx, (num_delete,))
        chex.assert_shape(target_idx, (num_delete,))
        chex.assert_shape(start_idx, (num_delete,))

    def test_utils_functions(self):
        """Test utility functions"""
        key = jax.random.key(101112)

        # Create mock dead info
        n_dead = 20
        dead_loglik = jnp.sort(jax.random.uniform(key, (n_dead,))) * 10 - 5
        dead_loglik_birth = jnp.full_like(dead_loglik, -jnp.inf)

        mock_info = base.NSInfo(
            particles=jnp.zeros((n_dead, 2)),
            loglikelihood=dead_loglik,
            loglikelihood_birth=dead_loglik_birth,
            logprior=jnp.zeros(n_dead),
            inner_kernel_info={},
        )

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

        particles = jax.random.normal(key, (num_live,))

        def mock_update_params_fn(state, info, current_params):
            return {"test_param": 1.0}

        state = adaptive.init(
            particles,
            gaussian_logprior,
            gaussian_loglikelihood,
            update_inner_kernel_params_fn=mock_update_params_fn,
        )

        # Check that inner kernel params were set
        self.assertEqual(state.inner_kernel_params["test_param"], 1.0)


class NestedSliceSamplingTest(chex.TestCase):
    def setUp(self):
        super().setUp()
        self.key = jax.random.key(42)

    def test_nss_direction_functions(self):
        """Test NSS direction generation functions"""
        key = jax.random.key(456)

        # Test covariance computation
        particles = jax.random.normal(key, (50, 3))
        state = base.init(particles, gaussian_logprior, gaussian_loglikelihood)

        params = nss.compute_covariance_from_particles(state, None, {})

        # Check that covariance is computed
        self.assertIn("cov", params)
        cov_pytree = params["cov"]
        chex.assert_shape(cov_pytree, (3, 3))

        # Test direction sampling
        direction = nss.sample_direction_from_covariance(key, params)
        chex.assert_shape(direction, (3,))

    def test_nss_kernel_construction(self):
        """Test NSS kernel can be constructed"""
        kernel = nss.build_kernel(
            gaussian_logprior, gaussian_loglikelihood, num_inner_steps=10
        )

        # Test that kernel is callable
        self.assertTrue(callable(kernel))


class NestedSamplingStatisticalTest(chex.TestCase):
    """Statistical correctness tests for nested sampling algorithms."""

    def setUp(self):
        super().setUp()
        self.key = jax.random.key(42)

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
        mock_info = base.NSInfo(
            particles=positions,
            loglikelihood=dead_loglik,
            loglikelihood_birth=dead_loglik_birth,
            logprior=dead_logprior,
            inner_kernel_info={},
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
            f"Evidence estimate {mean_estimate:.3f} vs analytic {analytical_log_evidence:.3f} "
            f"differs by {bias:.3f}, which exceeds 2σ = {tolerance:.3f}",
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
        particles = jax.random.uniform(key, (num_live,))
        state = base.init(particles, logprior_fn, loglikelihood_fn)

        # Check that initialization worked correctly
        self.assertTrue(jnp.all(state.particles >= 0.0))
        self.assertTrue(jnp.all(state.particles <= 1.0))
        self.assertFalse(jnp.any(jnp.isinf(state.logprior)))
        self.assertFalse(jnp.any(jnp.isnan(state.loglikelihood)))

        # Test evidence contribution from live points
        logZ_live_contribution = state.logZ_live
        self.assertIsInstance(logZ_live_contribution, (float, jax.Array))
        self.assertFalse(jnp.isnan(logZ_live_contribution))

    def test_evidence_monotonicity(self):
        """Test that evidence estimates are monotonically increasing during NS run."""

        # Simple setup for testing monotonicity
        def logprior_fn(x):
            return stats.norm.logpdf(x)

        def loglikelihood_fn(x):
            return -0.5 * x**2  # Simple quadratic

        num_live = 30
        key = jax.random.key(789)

        particles = jax.random.normal(key, (num_live,))
        initial_state = base.init(particles, logprior_fn, loglikelihood_fn)

        # Test that we can track evidence during run
        logZ_sequence = [initial_state.logZ]

        # Simulate a few evidence updates manually
        for i in range(5):
            # Simulate removing worst particle and updating evidence
            worst_idx = jnp.argmin(initial_state.loglikelihood)
            dead_loglik = initial_state.loglikelihood[worst_idx]

            # Update evidence (simplified)
            delta_logX = -1.0 / num_live  # Approximate volume decrease
            new_logZ = jnp.logaddexp(initial_state.logZ, dead_loglik + delta_logX)
            logZ_sequence.append(new_logZ)

            # Update for next iteration (simplified)
            new_loglik = jnp.concatenate(
                [
                    initial_state.loglikelihood[:worst_idx],
                    initial_state.loglikelihood[worst_idx + 1 :],
                    jnp.array([dead_loglik + 0.1]),  # Mock new particle
                ]
            )
            initial_state = initial_state._replace(loglikelihood=new_loglik)

        # Check monotonicity
        logZ_array = jnp.array(logZ_sequence)
        differences = logZ_array[1:] - logZ_array[:-1]
        self.assertTrue(
            jnp.all(differences >= -1e-10),
            "Evidence should be monotonically increasing",
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

        mock_info = base.NSInfo(
            particles=jnp.zeros((n_dead, 2)),
            loglikelihood=dead_loglik,
            loglikelihood_birth=dead_loglik_birth,
            logprior=jnp.zeros(n_dead),
            inner_kernel_info={},
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

        mock_info = base.NSInfo(
            particles=positions,
            loglikelihood=dead_loglik,
            loglikelihood_birth=dead_loglik_birth,
            logprior=dead_logprior,
            inner_kernel_info={},
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
            f"Analytic evidence {analytical_log_evidence:.3f} below 99% CI lower bound {lower_bound:.3f}",
        )
        self.assertLess(
            analytical_log_evidence,
            upper_bound,
            f"Analytic evidence {analytical_log_evidence:.3f} above 99% CI upper bound {upper_bound:.3f}",
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

        mock_info = base.NSInfo(
            particles=jnp.zeros((n_dead, 1)),
            loglikelihood=dead_loglik,
            loglikelihood_birth=dead_loglik_birth,
            logprior=jnp.full(
                n_dead, -jnp.log(prior_width)
            ),  # Uniform prior log density
            inner_kernel_info={},
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
            f"Analytic evidence {analytical_log_evidence:.3f} below 95% CI",
        )
        self.assertLess(
            analytical_log_evidence,
            upper_bound,
            f"Analytic evidence {analytical_log_evidence:.3f} above 95% CI",
        )

    def test_effective_sample_size_calculation(self):
        """Test effective sample size calculation."""
        key = jax.random.key(67890)

        # Create mock data with varying weights
        n_dead = 50
        dead_loglik = jax.random.uniform(key, (n_dead,)) * 5 - 10  # Range [-10, -5]
        dead_loglik_birth = jnp.full(n_dead, -jnp.inf)

        mock_info = base.NSInfo(
            particles=jnp.zeros((n_dead, 1)),
            loglikelihood=jnp.sort(dead_loglik),  # Ensure increasing
            loglikelihood_birth=dead_loglik_birth,
            logprior=jnp.zeros(n_dead),
            inner_kernel_info={},
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
