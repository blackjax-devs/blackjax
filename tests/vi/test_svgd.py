import functools

import jax
import jax.numpy as jnp
import jax.scipy.stats as stats
import numpy as np
from absl.testing import absltest
from optax import adam

import blackjax
from blackjax.vi.svgd import (
    SVGDState,
    build_kernel,
    init,
    median_heuristic,
    rbf_kernel,
    update_median_heuristic,
)
from tests.fixtures import BlackJAXTest


class SVGDUnitTest(BlackJAXTest):
    """Unit tests for SVGD building blocks."""

    def setUp(self):
        super().setUp()
        self.optimizer = adam(1e-2)

    # ---- rbf_kernel ----

    def test_rbf_kernel_positive(self):
        """RBF kernel is always positive."""
        x = jnp.array([1.0, 2.0])
        y = jnp.array([3.0, -1.0])
        assert float(rbf_kernel(x, y)) > 0.0

    def test_rbf_kernel_same_point_equals_one(self):
        """k(x, x) = exp(0) = 1 for any x."""
        x = jnp.array([1.0, -2.0, 0.5])
        np.testing.assert_allclose(float(rbf_kernel(x, x)), 1.0, atol=1e-6)

    def test_rbf_kernel_symmetric(self):
        """k(x, y) == k(y, x)."""
        x = jnp.array([1.0, 0.0])
        y = jnp.array([0.0, 1.0])
        np.testing.assert_allclose(rbf_kernel(x, y), rbf_kernel(y, x), atol=1e-7)

    def test_rbf_kernel_larger_bandwidth_higher_value(self):
        """Larger length_scale → larger kernel value for the same distance."""
        x = jnp.array([0.0])
        y = jnp.array([1.0])
        k_small = float(rbf_kernel(x, y, length_scale=0.5))
        k_large = float(rbf_kernel(x, y, length_scale=2.0))
        assert k_large > k_small

    def test_rbf_kernel_pytree(self):
        """rbf_kernel handles PyTree inputs."""
        x = {"a": jnp.array([1.0]), "b": jnp.array([0.0, 0.0])}
        y = {"a": jnp.array([0.0]), "b": jnp.array([1.0, 1.0])}
        val = rbf_kernel(x, y)
        assert jnp.isfinite(val)
        assert float(val) > 0.0

    # ---- median_heuristic ----

    def test_median_heuristic_sets_length_scale(self):
        """median_heuristic updates kernel_parameters['length_scale']."""
        particles = jnp.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        params = {"length_scale": 1.0}
        updated = median_heuristic(params, particles)
        assert "length_scale" in updated
        assert jnp.isfinite(updated["length_scale"])

    def test_median_heuristic_changes_value(self):
        """median_heuristic changes length_scale from its initial value."""
        particles = jnp.array([[0.0], [5.0], [10.0]])
        params = {"length_scale": 1.0}
        updated = median_heuristic(params, particles)
        assert float(updated["length_scale"]) != 1.0

    # ---- init / SVGDState ----

    def test_init_returns_svgd_state(self):
        """init returns an SVGDState with correct particles."""
        particles = jnp.zeros((5, 3))
        params = {"length_scale": 1.0}
        state = init(particles, params, self.optimizer)
        self.assertIsInstance(state, SVGDState)
        np.testing.assert_array_equal(state.particles, particles)
        self.assertEqual(state.kernel_parameters["length_scale"], 1.0)

    # ---- build_kernel / step ----

    def test_build_kernel_step_output_shape(self):
        """One kernel step preserves particle shape."""
        particles = jax.random.normal(self.next_key(), (10, 2))
        params = {"length_scale": 1.0}
        state = init(particles, params, self.optimizer)

        def grad_logdensity(x):
            return -x  # gradient of -0.5*||x||^2

        kernel_fn = build_kernel(self.optimizer)
        new_state = kernel_fn(state, grad_logdensity, rbf_kernel)
        self.assertEqual(new_state.particles.shape, particles.shape)

    def test_build_kernel_step_moves_particles(self):
        """Particles change after a step with non-trivial gradient."""
        particles = jnp.ones((5, 2))
        params = {"length_scale": 1.0}
        state = init(particles, params, self.optimizer)

        def grad_logdensity(x):
            return jnp.array([1.0, 1.0])  # constant push

        kernel_fn = build_kernel(self.optimizer)
        new_state = kernel_fn(state, grad_logdensity, rbf_kernel)
        assert not jnp.allclose(new_state.particles, particles)

    # ---- update_median_heuristic ----

    def test_update_median_heuristic_returns_svgd_state(self):
        """update_median_heuristic returns SVGDState with updated length_scale."""
        particles = jnp.array([[0.0, 0.0], [3.0, 0.0], [0.0, 3.0]])
        params = {"length_scale": 1.0}
        state = init(particles, params, self.optimizer)
        new_state = update_median_heuristic(state)
        self.assertIsInstance(new_state, SVGDState)
        assert new_state.kernel_parameters["length_scale"] != 1.0

    # ---- jit ----

    def test_jit_compatible(self):
        """build_kernel step is JIT-compilable."""
        particles = jax.random.normal(self.next_key(), (6, 2))
        params = {"length_scale": 1.0}
        state = init(particles, params, self.optimizer)

        def grad_logdensity(x):
            return -x

        kernel_fn = jax.jit(build_kernel(self.optimizer), static_argnums=(1, 2))
        new_state = kernel_fn(state, grad_logdensity, rbf_kernel)
        self.assertEqual(new_state.particles.shape, particles.shape)


def svgd_training_loop(
    log_p,
    initial_position,
    initial_kernel_parameters,
    kernel,
    optimizer,
    *,
    num_iterations=500,
) -> SVGDState:
    svgd = blackjax.svgd(jax.grad(log_p), optimizer, kernel, update_median_heuristic)
    state = svgd.init(initial_position, initial_kernel_parameters)
    step = jax.jit(svgd.step)  # type: ignore[attr-defined]

    for _ in range(num_iterations):
        state = step(state)
    return state


class SvgdTest(BlackJAXTest):
    def test_recover_posterior(self):
        # TODO improve testing
        """Simple Normal mean test"""

        ndim = 2

        rng_key_chol, rng_key_observed, rng_key_init = jax.random.split(
            self.next_key(), 3
        )
        L = jnp.tril(jax.random.normal(rng_key_chol, (ndim, ndim)))
        true_mu = jnp.arange(ndim)
        true_cov = L @ L.T
        true_prec = jnp.linalg.pinv(true_cov)

        def logp_posterior_conjugate_normal_model(
            observed, prior_mu, prior_prec, true_prec
        ):
            n = observed.shape[0]
            posterior_cov = jnp.linalg.inv(prior_prec + n * true_prec)
            posterior_mu = (
                posterior_cov
                @ (
                    prior_prec @ prior_mu[:, None]
                    + n * true_prec @ observed.mean(0)[:, None]
                )
            )[:, 0]
            return posterior_mu

        def logp_unnormalized_posterior(x, observed, prior_mu, prior_prec, true_cov):
            logp = 0.0
            logp += stats.multivariate_normal.logpdf(x, prior_mu, prior_prec)
            logp += stats.multivariate_normal.logpdf(observed, x, true_cov).sum()
            return logp

        prior_mu = jnp.zeros(ndim)
        prior_prec = jnp.eye(ndim)

        # Simulate the data
        observed = jax.random.multivariate_normal(
            rng_key_observed, true_mu, true_cov, shape=(500,)
        )

        logp_model = functools.partial(
            logp_unnormalized_posterior,
            observed=observed,
            prior_mu=prior_mu,
            prior_prec=prior_prec,
            true_cov=true_cov,
        )

        num_particles = 50
        initial_particles = jax.random.multivariate_normal(
            rng_key_init, prior_mu, prior_prec, shape=(num_particles,)
        )

        out = svgd_training_loop(
            log_p=logp_model,
            initial_position=initial_particles,
            initial_kernel_parameters={"length_scale": 1.0},
            kernel=rbf_kernel,
            optimizer=adam(0.2),
            num_iterations=200,
        )

        posterior_mu = logp_posterior_conjugate_normal_model(
            observed, prior_mu, prior_prec, true_prec
        )

        self.assertAlmostEqual(
            jnp.linalg.norm(posterior_mu - out.particles.mean(0)), 0.0, delta=1.0
        )


if __name__ == "__main__":
    absltest.main()
