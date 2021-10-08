"""Test the accuracy of the MCMC kernels."""
import functools
import itertools

import chex
import jax
import jax.numpy as jnp
import jax.scipy.stats as stats
import numpy as np
from absl.testing import absltest, parameterized

import blackjax.diagnostics as diagnostics
import blackjax.hmc as hmc
import blackjax.nuts as nuts
import blackjax.rmh as rmh
import blackjax.stan_warmup as stan_warmup


def inference_loop(kernel, num_samples, rng_key, initial_state):
    def one_step(state, rng_key):
        state, _ = kernel(rng_key, state)
        return state, state

    keys = jax.random.split(rng_key, num_samples)
    _, states = jax.lax.scan(one_step, initial_state, keys)

    return states


regresion_test_cases = [
    {
        "algorithm": hmc,
        "initial_position": {"scale": 1.0, "coefs": 2.0},
        "parameters": {"num_integration_steps": 90},
        "num_warmup_steps": 3_000,
        "num_sampling_steps": 2_000,
    },
    {
        "algorithm": nuts,
        "initial_position": {"scale": 1.0, "coefs": 2.0},
        "parameters": {},
        "num_warmup_steps": 1_000,
        "num_sampling_steps": 500,
    },
]


class LinearRegressionTest(chex.TestCase):
    """Test sampling of a linear regression model."""

    def setUp(self):
        super().setUp()
        self.key = jax.random.PRNGKey(19)

    def regression_logprob(self, scale, coefs, preds, x):
        """Linear regression"""
        logpdf = 0
        logpdf += stats.expon.logpdf(scale, 1, 1)
        logpdf += stats.norm.logpdf(coefs, 3 * jnp.ones(x.shape[-1]), 2)
        y = jnp.dot(x, coefs)
        logpdf += stats.norm.logpdf(preds, y, scale)
        return jnp.sum(logpdf)

    @chex.all_variants(with_pmap=False)
    @parameterized.parameters(itertools.product(regresion_test_cases, [True, False]))
    def test_linear_regression(self, case, is_mass_matrix_diagonal):
        """Test the HMC kernel and the Stan warmup."""
        rng_key, init_key0, init_key1 = jax.random.split(self.key, 3)
        x_data = jax.random.normal(init_key0, shape=(1000, 1))
        y_data = 3 * x_data + jax.random.normal(init_key1, shape=x_data.shape)

        logposterior_fn_ = functools.partial(
            self.regression_logprob, x=x_data, preds=y_data
        )
        logposterior_fn = lambda x: logposterior_fn_(**x)

        warmup_key, inference_key = jax.random.split(rng_key, 2)
        initial_position = case["initial_position"]
        initial_state = case["algorithm"].new_state(initial_position, logposterior_fn)

        def kernel_factory(step_size, inverse_mass_matrix):
            return case["algorithm"].kernel(
                logposterior_fn, step_size, inverse_mass_matrix, **case["parameters"]
            )

        warmup_run = functools.partial(
            stan_warmup.run,
            kernel_factory=kernel_factory,
            num_steps=case["num_warmup_steps"],
            is_mass_matrix_diagonal=is_mass_matrix_diagonal,
        )
        state, (step_size, inverse_mass_matrix), _ = self.variant(warmup_run)(
            warmup_key, initial_state=initial_state
        )

        if is_mass_matrix_diagonal:
            assert inverse_mass_matrix.ndim == 1
        else:
            assert inverse_mass_matrix.ndim == 2

        kernel = kernel_factory(step_size, inverse_mass_matrix)
        states = inference_loop(
            kernel, case["num_sampling_steps"], inference_key, initial_state
        )

        coefs_samples = states.position["coefs"]
        scale_samples = states.position["scale"]

        np.testing.assert_allclose(np.mean(scale_samples), 1.0, atol=1e-1)
        np.testing.assert_allclose(np.mean(coefs_samples), 3.0, atol=1e-1)


normal_test_cases = [
    {
        "algorithm": hmc,
        "initial_position": jnp.array(100.0),
        "parameters": {
            "step_size": 0.1,
            "inverse_mass_matrix": jnp.array([0.1]),
            "num_integration_steps": 100,
        },
        "num_sampling_steps": 6000,
        "burnin": 5_000,
    },
    {
        "algorithm": nuts,
        "initial_position": jnp.array(100.0),
        "parameters": {"step_size": 0.1, "inverse_mass_matrix": jnp.array([0.1])},
        "num_sampling_steps": 6000,
        "burnin": 5_000,
    },
    {
        "algorithm": rmh,
        "initial_position": 1.0,
        "parameters": {"sigma": jnp.array([1.0])},
        "num_sampling_steps": 20_000,
        "burnin": 5_000,
    },
]


class UnivariateNormalTest(chex.TestCase):
    """Test sampling of a univariate Normal distribution."""

    def setUp(self):
        super().setUp()
        self.key = jax.random.PRNGKey(19)

    def normal_logprob(self, x):
        return stats.norm.logpdf(x, loc=1.0, scale=2.0)

    @chex.all_variants(with_pmap=False)
    @parameterized.parameters(normal_test_cases)
    def test_univariate_normal(
        self, algorithm, initial_position, parameters, num_sampling_steps, burnin
    ):
        initial_state = algorithm.new_state(initial_position, self.normal_logprob)

        kernel = algorithm.kernel(self.normal_logprob, **parameters)
        states = self.variant(
            functools.partial(inference_loop, kernel, num_sampling_steps)
        )(self.key, initial_state)

        samples = states.position[burnin:]

        np.testing.assert_allclose(np.mean(samples), 1.0, rtol=1e-1)
        np.testing.assert_allclose(np.var(samples), 4.0, rtol=1e-1)


mcse_test_cases = [
    {
        "algorithm": hmc,
        "parameters": {
            "step_size": 0.1,
            "num_integration_steps": 32,
        },
    },
    {
        "algorithm": nuts,
        "parameters": {"step_size": 0.07},
    },
]


class MonteCarloStandardErrorTest(chex.TestCase):
    """Test sampler correctness using Monte Carlo Central Limit Theorem."""

    def setUp(self):
        super().setUp()
        self.key = jax.random.PRNGKey(2351235)

    def generate_multivariate_target(self, rng=None):
        """Genrate a Multivariate Normal distribution as target."""
        if rng is None:
            loc = jnp.array([0.0, 3])
            scale = jnp.array([1.0, 2.0])
            rho = jnp.array(0.75)
        else:
            rng, loc_rng, scale_rng, rho_rng = jax.random.split(rng, 4)
            loc = jax.random.normal(loc_rng, [2]) * 10.0
            scale = jnp.abs(jax.random.normal(scale_rng, [2])) * 2.5
            rho = jax.random.uniform(rho_rng, [], minval=-1.0, maxval=1.0)

        cov = jnp.diag(scale ** 2)
        cov = cov.at[0, 1].set(rho * scale[0] * scale[1])
        cov = cov.at[1, 0].set(rho * scale[0] * scale[1])

        def logprob_fn(x):
            return stats.multivariate_normal.logpdf(x, loc, cov).sum()

        return logprob_fn, loc, scale, rho

    def mcse_test(self, samples, true_param, p_val=0.01):
        posterior_mean = jnp.mean(samples, axis=[0, 1])
        ess = diagnostics.effective_sample_size(samples, chain_axis=1, sample_axis=0)
        posterior_sd = jnp.std(samples, axis=0, ddof=1)
        avg_monte_carlo_standard_error = jnp.mean(posterior_sd, axis=0) / jnp.sqrt(ess)
        scaled_error = (
            jnp.abs(posterior_mean - true_param) / avg_monte_carlo_standard_error
        )
        np.testing.assert_array_less(scaled_error, stats.norm.ppf(1 - p_val))
        return scaled_error

    @parameterized.parameters(mcse_test_cases)
    def test_mcse(self, algorithm, parameters):
        """Test convergence using Monte Carlo CLT across multiple chains."""
        init_fn_key, pos_init_key, sample_key = jax.random.split(self.key, 3)
        logprob_fn, true_loc, true_scale, true_rho = self.generate_multivariate_target(
            None
        )
        num_chains = 10
        initial_positions = jax.random.normal(pos_init_key, [num_chains, 2])
        kernel = algorithm.kernel(
            logprob_fn, inverse_mass_matrix=true_scale, **parameters
        )
        initial_states = jax.vmap(algorithm.new_state, in_axes=(0, None))(
            initial_positions, logprob_fn
        )
        multi_chain_sample_key = jax.random.split(sample_key, num_chains)

        inference_loop_multiple_chains = jax.vmap(
            functools.partial(inference_loop, kernel, 2_000)
        )
        states = inference_loop_multiple_chains(multi_chain_sample_key, initial_states)

        posterior_samples = states.position[-1000:]
        posterior_delta = posterior_samples - true_loc
        posterior_variance = posterior_delta ** 2.0
        posterior_correlation = jnp.prod(posterior_delta, axis=-1, keepdims=True) / (
            true_scale[0] * true_scale[1]
        )

        _ = jax.tree_multimap(
            self.mcse_test,
            [posterior_samples, posterior_variance, posterior_correlation],
            [true_loc, true_scale ** 2, true_rho],
        )


if __name__ == "__main__":
    absltest.main()
