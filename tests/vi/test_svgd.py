import functools

import chex
import jax
import jax.numpy as jnp
import jax.scipy.stats as stats
from absl.testing import absltest
from optax import adam

import blackjax
from blackjax.vi.svgd import SVGDState, rbf_kernel, update_median_heuristic


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


class SvgdTest(chex.TestCase):
    def setUp(self):
        super().setUp()
        self.key = jax.random.key(1)

    def test_recover_posterior(self):
        # TODO improve testing
        """Simple Normal mean test"""

        ndim = 2

        rng_key_chol, rng_key_observed, rng_key_init = jax.random.split(self.key, 3)
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
            rng_key_observed, true_mu, true_cov, shape=(10_000,)
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
            num_iterations=500,
        )

        posterior_mu = logp_posterior_conjugate_normal_model(
            observed, prior_mu, prior_prec, true_prec
        )

        self.assertAlmostEqual(
            jnp.linalg.norm(posterior_mu - out.particles.mean(0)), 0.0, delta=1.0
        )


if __name__ == "__main__":
    absltest.main()
