import chex
import jax.numpy as jnp
import jax.scipy.stats as stats
import numpy as np


class SMCLinearRegressionTestCase(chex.TestCase):
    def logdensity_fn(self, log_scale, coefs, preds, x):
        """Linear regression"""
        scale = jnp.exp(log_scale)
        y = jnp.dot(x, coefs)
        logpdf = stats.norm.logpdf(preds, y, scale)
        return jnp.sum(logpdf)

    def particles_prior_loglikelihood(self):
        num_particles = 100

        x_data = np.random.normal(0, 1, size=(1000, 1))
        y_data = 3 * x_data + np.random.normal(size=x_data.shape)
        observations = {"x": x_data, "preds": y_data}

        logprior_fn = lambda x: stats.norm.logpdf(x["log_scale"]) + stats.norm.logpdf(
            x["coefs"]
        )
        loglikelihood_fn = lambda x: self.logdensity_fn(**x, **observations)

        log_scale_init = np.random.randn(num_particles)
        coeffs_init = np.random.randn(num_particles)
        init_particles = {"log_scale": log_scale_init, "coefs": coeffs_init}

        return init_particles, logprior_fn, loglikelihood_fn

    def assert_linear_regression_test_case(self, result):
        np.testing.assert_allclose(
            np.mean(np.exp(result.particles["log_scale"])), 1.0, rtol=1e-1
        )
        np.testing.assert_allclose(np.mean(result.particles["coefs"]), 3.0, rtol=1e-1)
