"""Test the accuracy of the MCMC kernels."""
import functools
import itertools

import chex
import jax
import jax.numpy as jnp
import jax.scipy.stats as stats
import numpy as np
from absl.testing import absltest, parameterized

import blackjax
import blackjax.diagnostics as diagnostics


def inference_loop(kernel, num_samples, rng_key, initial_state):
    def one_step(state, rng_key):
        state, _ = kernel(rng_key, state)
        return state, state

    keys = jax.random.split(rng_key, num_samples)
    _, states = jax.lax.scan(one_step, initial_state, keys)

    return states


def orbit_samples(orbits, weights, rng_key):
    def sample_orbit(orbit, weights, rng_key):
        sample = jax.random.choice(rng_key, orbit, p=weights)
        return sample

    keys = jax.random.split(rng_key, orbits.shape[0])
    samples = jax.vmap(sample_orbit)(orbits, weights, keys)

    return samples


regresion_test_cases = [
    {
        "algorithm": blackjax.hmc,
        "initial_position": {"scale": 1.0, "coefs": 2.0},
        "parameters": {"num_integration_steps": 90},
        "num_warmup_steps": 3_000,
        "num_sampling_steps": 2_000,
    },
    {
        "algorithm": blackjax.nuts,
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

        warmup = blackjax.window_adaptation(
            case["algorithm"],
            logposterior_fn,
            case["num_warmup_steps"],
            is_mass_matrix_diagonal,
            progress_bar=True,
            **case["parameters"],
        )
        state, kernel, _ = warmup.run(
            warmup_key,
            case["initial_position"],
        )

        states = inference_loop(
            kernel, case["num_sampling_steps"], inference_key, state
        )

        coefs_samples = states.position["coefs"]
        scale_samples = states.position["scale"]

        np.testing.assert_allclose(np.mean(scale_samples), 1.0, atol=1e-1)
        np.testing.assert_allclose(np.mean(coefs_samples), 3.0, atol=1e-1)


class SGMCMCTest(chex.TestCase):
    """Test sampling of a linear regression model."""

    def setUp(self):
        super().setUp()
        self.key = jax.random.PRNGKey(19)

    def logprior_fn(self, position):
        return -0.5 * jnp.dot(position, position) * 0.01

    def loglikelihood_fn(self, position, x):
        w = x - position
        return -0.5 * jnp.dot(w, w)

    def test_linear_regression(self):
        """Test the HMC kernel and the Stan warmup."""
        import blackjax.sgmcmc.gradients

        rng_key, data_key = jax.random.split(self.key, 2)

        data_size = 1000
        X_data = jax.random.normal(data_key, shape=(data_size, 5))

        schedule_fn = lambda _: 1e-3
        grad_fn = blackjax.sgmcmc.gradients.grad_estimator(
            self.logprior_fn, self.loglikelihood_fn, data_size
        )
        sgld = blackjax.sgld(grad_fn, schedule_fn)

        init_position = 1.0
        data_batch = X_data[:100, :]
        init_state = sgld.init(init_position, data_batch)

        _, rng_key = jax.random.split(rng_key)
        data_batch = X_data[100:200, :]
        _ = sgld.step(rng_key, init_state, data_batch)


normal_test_cases = [
    {
        "algorithm": blackjax.hmc,
        "initial_position": jnp.array(3.0),
        "parameters": {
            "step_size": 3.9,
            "inverse_mass_matrix": jnp.array([1.0]),
            "num_integration_steps": 30,
        },
        "num_sampling_steps": 6000,
        "burnin": 1_000,
    },
    {
        "algorithm": blackjax.nuts,
        "initial_position": jnp.array(3.0),
        "parameters": {"step_size": 4.0, "inverse_mass_matrix": jnp.array([1.0])},
        "num_sampling_steps": 6000,
        "burnin": 1_000,
    },
    {
        "algorithm": blackjax.orbital_hmc,
        "initial_position": jnp.array(100.0),
        "parameters": {
            "step_size": 0.1,
            "inverse_mass_matrix": jnp.array([0.1]),
            "period": 100,
        },
        "num_sampling_steps": 20_000,
        "burnin": 15_000,
    },
    {
        "algorithm": blackjax.rmh,
        "initial_position": 1.0,
        "parameters": {"sigma": jnp.array([1.0])},
        "num_sampling_steps": 20_000,
        "burnin": 5_000,
    },
    {
        "algorithm": blackjax.mala,
        "initial_position": 1.0,
        "parameters": {"step_size": 1e-1},
        "num_sampling_steps": 20_000,
        "burnin": 2_000,
    },
    {
        "algorithm": blackjax.elliptical_slice,
        "initial_position": 1.0,
        "parameters": {"cov": jnp.array([2.0**2]), "mean": 1.0},
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
        algo = algorithm(self.normal_logprob, **parameters)
        if algorithm == blackjax.elliptical_slice:
            algo = algorithm(lambda _: 1.0, **parameters)
        initial_state = algo.init(initial_position)

        kernel = algo.step
        states = self.variant(
            functools.partial(inference_loop, kernel, num_sampling_steps)
        )(self.key, initial_state)

        if algorithm == blackjax.orbital_hmc:
            _, orbit_key = jax.random.split(self.key)
            samples = orbit_samples(
                states.positions[burnin:], states.weights[burnin:], orbit_key
            )
        else:
            samples = states.position[burnin:]

        np.testing.assert_allclose(np.mean(samples), 1.0, rtol=1e-1)
        np.testing.assert_allclose(np.var(samples), 4.0, rtol=1e-1)


mcse_test_cases = [
    {
        "algorithm": blackjax.hmc,
        "parameters": {
            "step_size": 1.0,
            "num_integration_steps": 32,
        },
        "custom_gradients": False,
    },
    {
        "algorithm": blackjax.nuts,
        "parameters": {"step_size": 1.0},
        "custom_gradients": False,
    },
    {
        "algorithm": blackjax.nuts,
        "parameters": {"step_size": 1.0},
        "custom_gradients": True,
    },
]


class MonteCarloStandardErrorTest(chex.TestCase):
    """Test sampler correctness using Monte Carlo Central Limit Theorem."""

    def setUp(self):
        super().setUp()
        self.key = jax.random.PRNGKey(20220203)

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

        cov = jnp.diag(scale**2)
        cov = cov.at[0, 1].set(rho * scale[0] * scale[1])
        cov = cov.at[1, 0].set(rho * scale[0] * scale[1])

        def logprob_fn(x):
            return stats.multivariate_normal.logpdf(x, loc, cov).sum()

        return logprob_fn, loc, scale, rho

    def mcse_test(self, samples, true_param, p_val=0.01):
        posterior_mean = jnp.mean(samples, axis=[0, 1])
        ess = diagnostics.effective_sample_size(samples, chain_axis=0, sample_axis=1)
        posterior_sd = jnp.std(samples, axis=0, ddof=1)
        avg_monte_carlo_standard_error = jnp.mean(posterior_sd, axis=0) / jnp.sqrt(ess)
        scaled_error = (
            jnp.abs(posterior_mean - true_param) / avg_monte_carlo_standard_error
        )
        np.testing.assert_array_less(scaled_error, stats.norm.ppf(1 - p_val))
        return scaled_error

    @parameterized.parameters(mcse_test_cases)
    def test_mcse(self, algorithm, parameters, custom_gradients):
        """Test convergence using Monte Carlo CLT across multiple chains."""
        init_fn_key, pos_init_key, sample_key = jax.random.split(self.key, 3)
        logprob_fn, true_loc, true_scale, true_rho = self.generate_multivariate_target(
            None
        )
        if custom_gradients:
            logprob_grad_fn = jax.jacfwd(logprob_fn)
        else:
            logprob_grad_fn = None

        kernel = algorithm(
            logprob_fn,
            inverse_mass_matrix=true_scale,
            logprob_grad_fn=logprob_grad_fn,
            **parameters,
        )

        num_chains = 10
        initial_positions = jax.random.normal(pos_init_key, [num_chains, 2])
        initial_states = jax.vmap(kernel.init, in_axes=(0,))(initial_positions)
        multi_chain_sample_key = jax.random.split(sample_key, num_chains)

        inference_loop_multiple_chains = jax.vmap(
            functools.partial(inference_loop, kernel.step, 2_000)
        )
        states = inference_loop_multiple_chains(multi_chain_sample_key, initial_states)

        posterior_samples = states.position[:, -1000:]
        posterior_delta = posterior_samples - true_loc
        posterior_variance = posterior_delta**2.0
        posterior_correlation = jnp.prod(posterior_delta, axis=-1, keepdims=True) / (
            true_scale[0] * true_scale[1]
        )

        _ = jax.tree_map(
            self.mcse_test,
            [posterior_samples, posterior_variance, posterior_correlation],
            [true_loc, true_scale**2, true_rho],
        )


if __name__ == "__main__":
    absltest.main()
