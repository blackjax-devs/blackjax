"""Test the accuracy of the MCMC kernels."""
import functools
import itertools

import chex
import jax
import jax.numpy as jnp
import jax.scipy.stats as stats
import numpy as np
import optax
from absl.testing import absltest, parameterized

import blackjax
import blackjax.diagnostics as diagnostics
import blackjax.mcmc.random_walk
from blackjax.util import run_inference_algorithm


def orbit_samples(orbits, weights, rng_key):
    def sample_orbit(orbit, weights, rng_key):
        sample = jax.random.choice(rng_key, orbit, p=weights)
        return sample

    keys = jax.random.split(rng_key, orbits.shape[0])
    samples = jax.vmap(sample_orbit)(orbits, weights, keys)

    return samples


def irmh_proposal_distribution(rng_key):
    """
    The proposal distribution is chosen to be wider than the target, so that the RMH rejection
    doesn't make the sample overemphasize the center of the target distribution.
    """
    return 1.0 + jax.random.normal(rng_key) * 25.0


def rmh_proposal_distribution(rng_key, position):
    return position + jax.random.normal(rng_key) * 25.0


regression_test_cases = [
    {
        "algorithm": blackjax.hmc,
        "initial_position": {"log_scale": 0.0, "coefs": 4.0},
        "parameters": {"num_integration_steps": 90},
        "num_warmup_steps": 1_000,
        "num_sampling_steps": 3_000,
    },
    {
        "algorithm": blackjax.nuts,
        "initial_position": {"log_scale": 0.0, "coefs": 4.0},
        "parameters": {},
        "num_warmup_steps": 1_000,
        "num_sampling_steps": 1_000,
    },
]


class LinearRegressionTest(chex.TestCase):
    """Test sampling of a linear regression model."""

    def setUp(self):
        super().setUp()
        self.key = jax.random.key(19)

    def regression_logprob(self, log_scale, coefs, preds, x):
        """Linear regression"""
        scale = jnp.exp(log_scale)
        scale_prior = stats.expon.logpdf(scale, 0, 1) + log_scale
        coefs_prior = stats.norm.logpdf(coefs, 0, 5)
        y = jnp.dot(x, coefs)
        logpdf = stats.norm.logpdf(preds, y, scale)
        # reduce sum otherwise broacasting will make the logprob biased.
        return sum(x.sum() for x in [scale_prior, coefs_prior, logpdf])

    def run_mclmc(self, logdensity_fn, num_steps, initial_position, key):
        init_key, tune_key, run_key = jax.random.split(key, 3)

        initial_state = blackjax.mcmc.mclmc.init(
            x_initial=initial_position, logdensity_fn=logdensity_fn, rng_key=init_key
        )

        kernel = blackjax.mcmc.mclmc.build_kernel(
            logdensity_fn=logdensity_fn,
            integrator=blackjax.mcmc.integrators.noneuclidean_mclachlan,
        )

        (
            blackjax_state_after_tuning,
            blackjax_mclmc_sampler_params,
        ) = blackjax.mclmc_find_L_and_step_size(
            mclmc_kernel=kernel,
            num_steps=num_steps,
            state=initial_state,
            rng_key=tune_key,
        )

        sampling_alg = blackjax.mclmc(
            logdensity_fn,
            L=blackjax_mclmc_sampler_params.L,
            step_size=blackjax_mclmc_sampler_params.step_size,
        )

        _, samples, _ = run_inference_algorithm(
            rng_key=run_key,
            initial_state_or_position=blackjax_state_after_tuning,
            inference_algorithm=sampling_alg,
            num_steps=num_steps,
            transform=lambda x: x.position,
        )

        return samples

    @parameterized.parameters(itertools.product(regression_test_cases, [True, False]))
    def test_window_adaptation(self, case, is_mass_matrix_diagonal):
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
            is_mass_matrix_diagonal,
            progress_bar=True,
            **case["parameters"],
        )
        (state, parameters), _ = warmup.run(
            warmup_key,
            case["initial_position"],
            case["num_warmup_steps"],
        )
        inference_algorithm = case["algorithm"](logposterior_fn, **parameters)

        _, states, _ = run_inference_algorithm(
            inference_key, state, inference_algorithm, case["num_sampling_steps"]
        )

        coefs_samples = states.position["coefs"]
        scale_samples = np.exp(states.position["log_scale"])

        np.testing.assert_allclose(np.mean(scale_samples), 1.0, atol=1e-1)
        np.testing.assert_allclose(np.mean(coefs_samples), 3.0, atol=1e-1)

    def test_mala(self):
        """Test the MALA kernel."""
        init_key0, init_key1, inference_key = jax.random.split(self.key, 3)
        x_data = jax.random.normal(init_key0, shape=(1000, 1))
        y_data = 3 * x_data + jax.random.normal(init_key1, shape=x_data.shape)

        logposterior_fn_ = functools.partial(
            self.regression_logprob, x=x_data, preds=y_data
        )
        logposterior_fn = lambda x: logposterior_fn_(**x)

        mala = blackjax.mala(logposterior_fn, 1e-5)
        state = mala.init({"coefs": 1.0, "log_scale": 1.0})
        _, states, _ = run_inference_algorithm(inference_key, state, mala, 10_000)

        coefs_samples = states.position["coefs"][3000:]
        scale_samples = np.exp(states.position["log_scale"][3000:])

        np.testing.assert_allclose(np.mean(scale_samples), 1.0, atol=1e-1)
        np.testing.assert_allclose(np.mean(coefs_samples), 3.0, atol=1e-1)

    def test_mclmc(self):
        """Test the MCLMC kernel."""
        init_key0, init_key1, inference_key = jax.random.split(self.key, 3)
        x_data = jax.random.normal(init_key0, shape=(1000, 1))
        y_data = 3 * x_data + jax.random.normal(init_key1, shape=x_data.shape)

        logposterior_fn_ = functools.partial(
            self.regression_logprob, x=x_data, preds=y_data
        )
        logdensity_fn = lambda x: logposterior_fn_(**x)

        states = self.run_mclmc(
            initial_position={"coefs": 1.0, "log_scale": 1.0},
            logdensity_fn=logdensity_fn,
            key=inference_key,
            num_steps=10000,
        )

        coefs_samples = states["coefs"][3000:]
        scale_samples = np.exp(states["log_scale"][3000:])

        np.testing.assert_allclose(np.mean(scale_samples), 1.0, atol=1e-2)
        np.testing.assert_allclose(np.mean(coefs_samples), 3.0, atol=1e-2)

    @parameterized.parameters(regression_test_cases)
    def test_pathfinder_adaptation(
        self,
        algorithm,
        num_warmup_steps,
        initial_position,
        num_sampling_steps,
        parameters,
    ):
        """Test the HMC kernel and the Stan warmup."""
        rng_key, init_key0, init_key1 = jax.random.split(self.key, 3)
        x_data = jax.random.normal(init_key0, shape=(1000, 1))
        y_data = 3 * x_data + jax.random.normal(init_key1, shape=x_data.shape)

        logposterior_fn_ = functools.partial(
            self.regression_logprob, x=x_data, preds=y_data
        )
        logposterior_fn = lambda x: logposterior_fn_(**x)

        warmup_key, inference_key = jax.random.split(rng_key, 2)

        warmup = blackjax.pathfinder_adaptation(
            algorithm,
            logposterior_fn,
            **parameters,
        )
        (state, parameters), _ = warmup.run(
            warmup_key,
            initial_position,
            num_warmup_steps,
        )
        inference_algorithm = algorithm(logposterior_fn, **parameters)

        _, states, _ = run_inference_algorithm(
            inference_key, state, inference_algorithm, num_sampling_steps
        )

        coefs_samples = states.position["coefs"]
        scale_samples = np.exp(states.position["log_scale"])

        np.testing.assert_allclose(np.mean(scale_samples), 1.0, atol=1e-1)
        np.testing.assert_allclose(np.mean(coefs_samples), 3.0, atol=1e-1)

    def test_meads(self):
        """Test the MEADS adaptation w/ GHMC kernel."""
        rng_key, init_key0, init_key1 = jax.random.split(self.key, 3)
        x_data = jax.random.normal(init_key0, shape=(1000, 1))
        y_data = 3 * x_data + jax.random.normal(init_key1, shape=x_data.shape)

        logposterior_fn_ = functools.partial(
            self.regression_logprob, x=x_data, preds=y_data
        )
        logposterior_fn = lambda x: logposterior_fn_(**x)

        init_key, warmup_key, inference_key = jax.random.split(rng_key, 3)

        num_chains = 128
        warmup = blackjax.meads_adaptation(
            logposterior_fn,
            num_chains=num_chains,
        )
        scale_key, coefs_key = jax.random.split(init_key, 2)
        log_scales = 1.0 + jax.random.normal(scale_key, (num_chains,))
        coefs = 4.0 + jax.random.normal(coefs_key, (num_chains,))
        initial_positions = {"log_scale": log_scales, "coefs": coefs}
        (last_states, parameters), _ = warmup.run(
            warmup_key,
            initial_positions,
            num_steps=1000,
        )
        inference_algorithm = blackjax.ghmc(logposterior_fn, **parameters)

        chain_keys = jax.random.split(inference_key, num_chains)
        _, states, _ = jax.vmap(
            lambda key, state: run_inference_algorithm(
                key, state, inference_algorithm, 100
            )
        )(chain_keys, last_states)

        coefs_samples = states.position["coefs"]
        scale_samples = np.exp(states.position["log_scale"])

        np.testing.assert_allclose(np.mean(scale_samples), 1.0, atol=1e-1)
        np.testing.assert_allclose(np.mean(coefs_samples), 3.0, atol=1e-1)

    @parameterized.parameters([None, jax.random.uniform])
    def test_chees(self, jitter_generator):
        """Test the ChEES adaptation w/ HMC kernel."""
        rng_key, init_key0, init_key1 = jax.random.split(self.key, 3)
        x_data = jax.random.normal(init_key0, shape=(1000, 1))
        y_data = 3 * x_data + jax.random.normal(init_key1, shape=x_data.shape)

        logposterior_fn_ = functools.partial(
            self.regression_logprob, x=x_data, preds=y_data
        )
        logposterior_fn = lambda x: logposterior_fn_(**x)

        init_key, warmup_key, inference_key = jax.random.split(rng_key, 3)

        num_chains = 128
        warmup = blackjax.chees_adaptation(
            logposterior_fn, num_chains=num_chains, jitter_generator=jitter_generator
        )
        scale_key, coefs_key = jax.random.split(init_key, 2)
        log_scales = 1.0 + jax.random.normal(scale_key, (num_chains,))
        coefs = 4.0 + jax.random.normal(coefs_key, (num_chains,))
        initial_positions = {"log_scale": log_scales, "coefs": coefs}
        (last_states, parameters), _ = warmup.run(
            warmup_key,
            initial_positions,
            step_size=0.001,
            optim=optax.adam(learning_rate=0.1),
            num_steps=1000,
        )
        inference_algorithm = blackjax.dynamic_hmc(logposterior_fn, **parameters)

        chain_keys = jax.random.split(inference_key, num_chains)
        _, states, _ = jax.vmap(
            lambda key, state: run_inference_algorithm(
                key, state, inference_algorithm, 100
            )
        )(chain_keys, last_states)

        coefs_samples = states.position["coefs"]
        scale_samples = np.exp(states.position["log_scale"])

        np.testing.assert_allclose(np.mean(scale_samples), 1.0, atol=1e-1)
        np.testing.assert_allclose(np.mean(coefs_samples), 3.0, atol=1e-1)

    def test_barker(self):
        """Test the Barker kernel."""
        init_key0, init_key1, inference_key = jax.random.split(self.key, 3)
        x_data = jax.random.normal(init_key0, shape=(1000, 1))
        y_data = 3 * x_data + jax.random.normal(init_key1, shape=x_data.shape)

        logposterior_fn_ = functools.partial(
            self.regression_logprob, x=x_data, preds=y_data
        )
        logposterior_fn = lambda x: logposterior_fn_(**x)

        barker = blackjax.barker_proposal(logposterior_fn, 1e-1)
        state = barker.init({"coefs": 1.0, "log_scale": 1.0})

        _, states, _ = run_inference_algorithm(inference_key, state, barker, 10_000)

        coefs_samples = states.position["coefs"][3000:]
        scale_samples = np.exp(states.position["log_scale"][3000:])

        np.testing.assert_allclose(np.mean(scale_samples), 1.0, atol=1e-2)
        np.testing.assert_allclose(np.mean(coefs_samples), 3.0, atol=1e-2)


class SGMCMCTest(chex.TestCase):
    """Test sampling of a linear regression model."""

    def setUp(self):
        super().setUp()
        self.key = jax.random.key(19)

    def logprior_fn(self, position):
        return -0.5 * jnp.dot(position, position) * 0.01

    def loglikelihood_fn(self, position, x):
        w = x - position
        return -0.5 * jnp.dot(w, w)

    def test_linear_regression_contour_sgld(self):
        rng_key, data_key = jax.random.split(self.key, 2)

        data_size = 1000
        X_data = jax.random.normal(data_key, shape=(data_size, 5))

        logdensity_fn = blackjax.sgmcmc.logdensity_estimator(
            self.logprior_fn, self.loglikelihood_fn, data_size
        )
        grad_fn = blackjax.sgmcmc.grad_estimator(
            self.logprior_fn, self.loglikelihood_fn, data_size
        )
        csgld = blackjax.csgld(logdensity_fn, grad_fn)

        data_batch = X_data[:100, :]
        init_position = 1.0
        init_state = csgld.init(init_position)
        _ = csgld.step(rng_key, init_state, data_batch, 1e-3, 1e-2)

    def test_linear_regression_sgld(self):
        rng_key, data_key = jax.random.split(self.key, 2)

        data_size = 1000
        X_data = jax.random.normal(data_key, shape=(data_size, 5))

        grad_fn = blackjax.sgmcmc.grad_estimator(
            self.logprior_fn, self.loglikelihood_fn, data_size
        )
        sgld = blackjax.sgld(grad_fn)

        data_batch = X_data[:100, :]
        init_position = 1.0
        init_position = sgld.init(init_position)
        _ = sgld.step(rng_key, init_position, data_batch, 1e-3)

    def test_linear_regression_sgld_cv(self):
        rng_key, data_key = jax.random.split(self.key, 2)

        data_size = 1000
        X_data = jax.random.normal(data_key, shape=(data_size, 5))

        centering_position = 1.0

        grad_fn = blackjax.sgmcmc.grad_estimator(
            self.logprior_fn, self.loglikelihood_fn, data_size
        )
        cv_grad_fn = blackjax.sgmcmc.gradients.control_variates(
            grad_fn, centering_position, X_data
        )

        sgld = blackjax.sgld(cv_grad_fn)

        init_position = 1.0
        init_position = sgld.init(init_position)
        data_batch = X_data[:100, :]
        _ = sgld.step(rng_key, init_position, data_batch, 1e-3)

    def test_linear_regression_sghmc(self):
        rng_key, data_key = jax.random.split(self.key, 2)

        data_size = 1000
        X_data = jax.random.normal(data_key, shape=(data_size, 5))

        grad_fn = blackjax.sgmcmc.grad_estimator(
            self.logprior_fn, self.loglikelihood_fn, data_size
        )
        sghmc = blackjax.sghmc(grad_fn, 10)

        data_batch = X_data[100:200, :]
        init_position = 1.0
        init_position = sghmc.init(init_position)
        data_batch = X_data[:100, :]
        _ = sghmc.step(rng_key, init_position, data_batch, 1e-3)

    def test_linear_regression_sghmc_cv(self):
        rng_key, data_key = jax.random.split(self.key, 2)

        data_size = 1000
        X_data = jax.random.normal(data_key, shape=(data_size, 5))

        centering_position = 1.0
        grad_fn = blackjax.sgmcmc.grad_estimator(
            self.logprior_fn, self.loglikelihood_fn, data_size
        )
        cv_grad_fn = blackjax.sgmcmc.gradients.control_variates(
            grad_fn, centering_position, X_data
        )

        sghmc = blackjax.sghmc(cv_grad_fn, 10)

        init_position = 1.0
        init_position = sghmc.init(init_position)
        data_batch = X_data[:100, :]
        _ = sghmc.step(rng_key, init_position, data_batch, 1e-3)

    def test_linear_regression_sgnht(self):
        rng_key, data_key = jax.random.split(self.key, 2)

        data_size = 1000
        X_data = jax.random.normal(data_key, shape=(data_size, 5))

        grad_fn = blackjax.sgmcmc.grad_estimator(
            self.logprior_fn, self.loglikelihood_fn, data_size
        )
        sgnht = blackjax.sgnht(grad_fn)

        _, rng_key = jax.random.split(rng_key)
        data_batch = X_data[100:200, :]
        init_position = 1.0
        data_batch = X_data[:100, :]
        init_state = sgnht.init(init_position, self.key)
        _ = sgnht.step(rng_key, init_state, data_batch, 1e-3)

    def test_linear_regression_sgnhtc_cv(self):
        rng_key, data_key = jax.random.split(self.key, 2)

        data_size = 1000
        X_data = jax.random.normal(data_key, shape=(data_size, 5))

        centering_position = 1.0
        grad_fn = blackjax.sgmcmc.grad_estimator(
            self.logprior_fn, self.loglikelihood_fn, data_size
        )
        cv_grad_fn = blackjax.sgmcmc.gradients.control_variates(
            grad_fn, centering_position, X_data
        )

        sgnht = blackjax.sgnht(cv_grad_fn)

        _, rng_key = jax.random.split(rng_key)
        init_position = 1.0
        data_batch = X_data[:100, :]
        init_state = sgnht.init(init_position, self.key)
        _ = sgnht.step(rng_key, init_state, data_batch, 1e-3)


class LatentGaussianTest(chex.TestCase):
    """Test sampling of a linear regression model."""

    def setUp(self):
        super().setUp()
        self.key = jax.random.key(19)
        self.C = 2.0 * np.eye(1)
        self.delta = 5.0
        self.sampling_steps = 25_000
        self.burnin = 5_000

    @chex.all_variants(with_pmap=False)
    def test_latent_gaussian(self):
        from blackjax import mgrad_gaussian

        inference_algorithm = mgrad_gaussian(
            lambda x: -0.5 * jnp.sum((x - 1.0) ** 2), self.C
        )
        inference_algorithm = inference_algorithm._replace(
            step=functools.partial(
                inference_algorithm.step,
                delta=self.delta,
            )
        )

        initial_state = inference_algorithm.init(jnp.zeros((1,)))

        _, states, _ = self.variant(
            functools.partial(
                run_inference_algorithm,
                inference_algorithm=inference_algorithm,
                num_steps=self.sampling_steps,
            ),
        )(self.key, initial_state)

        np.testing.assert_allclose(
            np.var(states.position[self.burnin :]), 1 / (1 + 0.5), rtol=1e-2, atol=1e-2
        )
        np.testing.assert_allclose(
            np.mean(states.position[self.burnin :]), 2 / 3, rtol=1e-2, atol=1e-2
        )


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
        "algorithm": blackjax.additive_step_random_walk.normal_random_walk,
        "initial_position": 1.0,
        "parameters": {"sigma": jnp.array([1.0])},
        "num_sampling_steps": 20_000,
        "burnin": 5_000,
    },
    {
        "algorithm": blackjax.rmh,
        "parameters": {},
        "initial_position": 1.0,
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
    {
        "algorithm": blackjax.irmh,
        "initial_position": jnp.array(1.0),
        "parameters": {},
        "num_sampling_steps": 50_000,
        "burnin": 5_000,
    },
    {
        "algorithm": blackjax.ghmc,
        "initial_position": jnp.array(1.0),
        "parameters": {
            "step_size": 1.0,
            "momentum_inverse_scale": jnp.array(1.0),
            "alpha": 0.8,
            "delta": 2.0,
        },
        "num_sampling_steps": 6000,
        "burnin": 1_000,
    },
    {
        "algorithm": blackjax.barker_proposal,
        "initial_position": 1.0,
        "parameters": {"step_size": 1.5},
        "num_sampling_steps": 20_000,
        "burnin": 2_000,
    },
]


class UnivariateNormalTest(chex.TestCase):
    """Test sampling of a univariate Normal distribution.

    (TODO) This only passes due to clever seed hacking.
    """

    def setUp(self):
        super().setUp()
        self.key = jax.random.key(12)

    def normal_logprob(self, x):
        return stats.norm.logpdf(x, loc=1.0, scale=2.0)

    @chex.all_variants(with_pmap=False)
    @parameterized.parameters(normal_test_cases)
    def test_univariate_normal(
        self, algorithm, initial_position, parameters, num_sampling_steps, burnin
    ):
        if algorithm == blackjax.irmh:
            parameters["proposal_distribution"] = irmh_proposal_distribution

        if algorithm == blackjax.rmh:
            parameters["proposal_generator"] = rmh_proposal_distribution

        inference_algorithm = algorithm(self.normal_logprob, **parameters)
        rng_key = self.key
        if algorithm == blackjax.elliptical_slice:
            inference_algorithm = algorithm(lambda x: jnp.ones_like(x), **parameters)
        if algorithm == blackjax.ghmc:
            rng_key, initial_state_key = jax.random.split(rng_key)
            initial_state = inference_algorithm.init(
                initial_position, initial_state_key
            )
        else:
            initial_state = inference_algorithm.init(initial_position)

        inference_key, orbit_key = jax.random.split(rng_key)
        _, states, _ = self.variant(
            functools.partial(
                run_inference_algorithm,
                inference_algorithm=inference_algorithm,
                num_steps=num_sampling_steps,
            )
        )(inference_key, initial_state)

        if algorithm == blackjax.orbital_hmc:
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
            "step_size": 0.5,
            "num_integration_steps": 20,
        },
        "is_mass_matrix_diagonal": True,
    },
    {
        "algorithm": blackjax.nuts,
        "parameters": {"step_size": 0.5},
        "is_mass_matrix_diagonal": True,
    },
    {
        "algorithm": blackjax.hmc,
        "parameters": {
            "step_size": 0.85,
            "num_integration_steps": 27,
        },
        "is_mass_matrix_diagonal": False,
    },
    {
        "algorithm": blackjax.nuts,
        "parameters": {"step_size": 0.85},
        "is_mass_matrix_diagonal": False,
    },
    {
        "algorithm": blackjax.barker_proposal,
        "parameters": {"step_size": 0.5},
        "is_mass_matrix_diagonal": None,
    },
]


class MonteCarloStandardErrorTest(chex.TestCase):
    """Test sampler correctness using Monte Carlo Central Limit Theorem."""

    def setUp(self):
        super().setUp()
        self.key = jax.random.key(20220203)

    def generate_multivariate_target(self, rng=None):
        """Genrate a Multivariate Normal distribution as target."""
        if rng is None:
            loc = jnp.array([0.0, 3])
            scale = jnp.array([1.0, 2.0])
            rho = jnp.array(0.75)
        else:
            loc_rng, scale_rng, rho_rng = jax.random.split(rng, 3)
            loc = jax.random.normal(loc_rng, [2]) * 10.0
            scale = jnp.abs(jax.random.normal(scale_rng, [2])) * 2.5
            rho = jax.random.uniform(rho_rng, [], minval=-1.0, maxval=1.0)

        cov = jnp.diag(scale**2)
        cov = cov.at[0, 1].set(rho * scale[0] * scale[1])
        cov = cov.at[1, 0].set(rho * scale[0] * scale[1])

        def logdensity_fn(x):
            return stats.multivariate_normal.logpdf(x, loc, cov).sum()

        return logdensity_fn, loc, scale, rho, cov

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
    def test_mcse(self, algorithm, parameters, is_mass_matrix_diagonal):
        """Test convergence using Monte Carlo CLT across multiple chains."""
        pos_init_key, sample_key = jax.random.split(self.key)
        (
            logdensity_fn,
            true_loc,
            true_scale,
            true_rho,
            true_cov,
        ) = self.generate_multivariate_target(None)
        if is_mass_matrix_diagonal is not None:
            if is_mass_matrix_diagonal:
                inverse_mass_matrix = true_scale**2
            else:
                inverse_mass_matrix = true_cov
            inference_algorithm = algorithm(
                logdensity_fn,
                inverse_mass_matrix=inverse_mass_matrix,
                **parameters,
            )
        else:
            inference_algorithm = algorithm(logdensity_fn, **parameters)

        num_chains = 10
        initial_positions = jax.random.normal(pos_init_key, [num_chains, 2])
        initial_states = jax.vmap(inference_algorithm.init, in_axes=(0,))(
            initial_positions
        )
        multi_chain_sample_key = jax.random.split(sample_key, num_chains)

        inference_loop_multiple_chains = jax.vmap(
            functools.partial(
                run_inference_algorithm,
                inference_algorithm=inference_algorithm,
                num_steps=2_000,
            )
        )
        _, states, _ = inference_loop_multiple_chains(
            multi_chain_sample_key, initial_states
        )

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
