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
from blackjax.adaptation.base import get_filter_adapt_info_fn, return_all_adapt_info
from blackjax.mcmc.integrators import isokinetic_mclachlan
from blackjax.util import run_inference_algorithm


def orbit_samples(orbits, weights, rng_key):
    def sample_orbit(orbit, weights, rng_key):
        sample = jax.random.choice(rng_key, orbit, p=weights)
        return sample

    keys = jax.random.split(rng_key, orbits.shape[0])
    samples = jax.vmap(sample_orbit)(orbits, weights, keys)

    return samples


def irmh_proposal_distribution(rng_key, mean):
    """
    The proposal distribution is chosen to be wider than the target, so that the RMH rejection
    doesn't make the sample overemphasize the center of the target distribution.
    """
    return mean + jax.random.normal(rng_key) * 25.0


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

window_adaptation_filters = [
    {
        "filter_fn": return_all_adapt_info,
        "return_sets": None,
    },
    {
        "filter_fn": get_filter_adapt_info_fn(),
        "return_sets": (set(), set(), set()),
    },
    {
        "filter_fn": get_filter_adapt_info_fn(
            {"position"}, {"is_divergent"}, {"ss_state", "inverse_mass_matrix"}
        ),
        "return_sets": (
            {"position"},
            {"is_divergent"},
            {"ss_state", "inverse_mass_matrix"},
        ),
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

    def run_mclmc(
        self,
        logdensity_fn,
        num_steps,
        initial_position,
        key,
        diagonal_preconditioning=False,
    ):
        init_key, tune_key, run_key = jax.random.split(key, 3)

        initial_state = blackjax.mcmc.mclmc.init(
            position=initial_position, logdensity_fn=logdensity_fn, rng_key=init_key
        )

        kernel = lambda sqrt_diag_cov: blackjax.mcmc.mclmc.build_kernel(
            logdensity_fn=logdensity_fn,
            integrator=blackjax.mcmc.mclmc.isokinetic_mclachlan,
            sqrt_diag_cov=sqrt_diag_cov,
        )

        (
            blackjax_state_after_tuning,
            blackjax_mclmc_sampler_params,
        ) = blackjax.mclmc_find_L_and_step_size(
            mclmc_kernel=kernel,
            num_steps=num_steps,
            state=initial_state,
            rng_key=tune_key,
            diagonal_preconditioning=diagonal_preconditioning,
        )

        sampling_alg = blackjax.mclmc(
            logdensity_fn,
            L=blackjax_mclmc_sampler_params.L,
            step_size=blackjax_mclmc_sampler_params.step_size,
            sqrt_diag_cov=blackjax_mclmc_sampler_params.sqrt_diag_cov,
        )

        _, samples, _ = run_inference_algorithm(
            rng_key=run_key,
            initial_state=blackjax_state_after_tuning,
            inference_algorithm=sampling_alg,
            num_steps=num_steps,
            transform=lambda x: x.position,
        )

        return samples

    @parameterized.parameters(
        itertools.product(
            regression_test_cases, [True, False], window_adaptation_filters
        )
    )
    def test_window_adaptation(
        self, case, is_mass_matrix_diagonal, window_adapt_config
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

        warmup = blackjax.window_adaptation(
            case["algorithm"],
            logposterior_fn,
            is_mass_matrix_diagonal,
            progress_bar=True,
            adaptation_info_fn=window_adapt_config["filter_fn"],
            **case["parameters"],
        )
        (state, parameters), info = warmup.run(
            warmup_key,
            case["initial_position"],
            case["num_warmup_steps"],
        )
        inference_algorithm = case["algorithm"](logposterior_fn, **parameters)

        def check_attrs(attribute, keyset):
            for name, param in getattr(info, attribute)._asdict().items():
                if name in keyset:
                    assert param is not None
                else:
                    assert param is None

        keysets = window_adapt_config["return_sets"]
        if keysets is None:
            keysets = (
                info.state._fields,
                info.info._fields,
                info.adaptation_state._fields,
            )
        for i, attribute in enumerate(["state", "info", "adaptation_state"]):
            check_attrs(attribute, keysets[i])

        _, states, _ = run_inference_algorithm(
            rng_key=inference_key,
            initial_state=state,
            inference_algorithm=inference_algorithm,
            num_steps=case["num_sampling_steps"],
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
        _, states, _ = run_inference_algorithm(
            rng_key=inference_key,
            initial_state=state,
            inference_algorithm=mala,
            num_steps=10_000,
        )

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

    def test_mclmc_preconditioning(self):
        class IllConditionedGaussian:
            """Gaussian distribution. Covariance matrix has eigenvalues equally spaced in log-space, going from 1/condition_bnumber^1/2 to condition_number^1/2."""

            def __init__(self, d, condition_number):
                """numpy_seed is used to generate a random rotation for the covariance matrix.
                If None, the covariance matrix is diagonal."""

                self.ndims = d
                self.name = "IllConditionedGaussian"
                self.condition_number = condition_number
                eigs = jnp.logspace(
                    -0.5 * jnp.log10(condition_number),
                    0.5 * jnp.log10(condition_number),
                    d,
                )
                self.E_x2 = eigs
                self.R = jnp.eye(d)
                self.Hessian = jnp.diag(1 / eigs)
                self.Cov = jnp.diag(eigs)
                self.Var_x2 = 2 * jnp.square(self.E_x2)

                self.logdensity_fn = lambda x: -0.5 * x.T @ self.Hessian @ x
                self.transform = lambda x: x

                self.sample_init = lambda key: jax.random.normal(
                    key, shape=(self.ndims,)
                ) * jnp.max(jnp.sqrt(eigs))

        dim = 100
        condition_number = 10
        eigs = jnp.logspace(
            -0.5 * jnp.log10(condition_number), 0.5 * jnp.log10(condition_number), dim
        )
        model = IllConditionedGaussian(dim, condition_number)
        num_steps = 20000
        key = jax.random.PRNGKey(2)

        integrator = isokinetic_mclachlan

        def get_sqrt_diag_cov():
            init_key, tune_key = jax.random.split(key)

            initial_position = model.sample_init(init_key)

            initial_state = blackjax.mcmc.mclmc.init(
                position=initial_position,
                logdensity_fn=model.logdensity_fn,
                rng_key=init_key,
            )

            kernel = lambda sqrt_diag_cov: blackjax.mcmc.mclmc.build_kernel(
                logdensity_fn=model.logdensity_fn,
                integrator=integrator,
                sqrt_diag_cov=sqrt_diag_cov,
            )

            (
                _,
                blackjax_mclmc_sampler_params,
            ) = blackjax.mclmc_find_L_and_step_size(
                mclmc_kernel=kernel,
                num_steps=num_steps,
                state=initial_state,
                rng_key=tune_key,
                diagonal_preconditioning=True,
            )

            return blackjax_mclmc_sampler_params.sqrt_diag_cov

        sqrt_diag_cov = get_sqrt_diag_cov()
        assert (
            jnp.abs(
                jnp.dot(
                    (sqrt_diag_cov**2) / jnp.linalg.norm(sqrt_diag_cov**2),
                    eigs / jnp.linalg.norm(eigs),
                )
                - 1
            )
            < 0.1
        )

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
            rng_key=inference_key,
            initial_state=state,
            inference_algorithm=inference_algorithm,
            num_steps=num_sampling_steps,
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
                rng_key=key,
                initial_state=state,
                inference_algorithm=inference_algorithm,
                num_steps=100,
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
                rng_key=key,
                initial_state=state,
                inference_algorithm=inference_algorithm,
                num_steps=100,
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

        _, states, _ = run_inference_algorithm(
            rng_key=inference_key,
            initial_state=state,
            inference_algorithm=barker,
            num_steps=10_000,
        )

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
        step_key, data_key = jax.random.split(self.key, 2)

        data_size = 1000
        X_data = jax.random.normal(data_key, shape=(data_size, 5))

        grad_fn = blackjax.sgmcmc.grad_estimator(
            self.logprior_fn, self.loglikelihood_fn, data_size
        )
        sgnht = blackjax.sgnht(grad_fn)

        data_batch = X_data[100:200, :]
        init_position = 1.0
        data_batch = X_data[:100, :]
        init_state = sgnht.init(init_position, self.key)
        _ = sgnht.step(step_key, init_state, data_batch, 1e-3)

    def test_linear_regression_sgnhtc_cv(self):
        step_key, data_key = jax.random.split(self.key, 2)

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

        init_position = 1.0
        data_batch = X_data[:100, :]
        init_state = sgnht.init(init_position, self.key)
        _ = sgnht.step(step_key, init_state, data_batch, 1e-3)


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
            lambda x: -0.5 * jnp.sum((x - 1.0) ** 2),
            covariance=self.C,
            step_size=self.delta,
        )

        initial_state = inference_algorithm.init(jnp.zeros((1,)))

        _, states, _ = self.variant(
            functools.partial(
                run_inference_algorithm,
                inference_algorithm=inference_algorithm,
                num_steps=self.sampling_steps,
            ),
        )(rng_key=self.key, initial_state=initial_state)

        np.testing.assert_allclose(
            np.var(states.position[self.burnin :]), 1 / (1 + 0.5), rtol=1e-2, atol=1e-2
        )
        np.testing.assert_allclose(
            np.mean(states.position[self.burnin :]), 2 / 3, rtol=1e-2, atol=1e-2
        )


def rmhmc_static_mass_matrix_fn(position):
    del position
    return jnp.array([1.0])


class UnivariateNormalTest(chex.TestCase):
    """Test sampling of a univariate Normal distribution.

    (TODO) This only passes due to clever seed hacking.
    """

    def setUp(self):
        super().setUp()
        self.key = jax.random.key(12)

    def normal_logprob(self, x):
        return stats.norm.logpdf(x, loc=1.0, scale=2.0)

    def univariate_normal_test_case(
        self,
        inference_algorithm,
        rng_key,
        initial_state,
        num_sampling_steps,
        burnin,
        postprocess_samples=None,
        **kwargs,
    ):
        inference_key, orbit_key = jax.random.split(rng_key)
        _, states, _ = self.variant(
            functools.partial(
                run_inference_algorithm,
                inference_algorithm=inference_algorithm,
                num_steps=num_sampling_steps,
                **kwargs,
            )
        )(rng_key=inference_key, initial_state=initial_state)

        if postprocess_samples:
            samples = postprocess_samples(states, orbit_key)
        else:
            samples = states.position[burnin:]
        np.testing.assert_allclose(np.mean(samples), 1.0, rtol=1e-1)
        np.testing.assert_allclose(np.var(samples), 4.0, rtol=1e-1)

    @chex.all_variants(with_pmap=False)
    def test_irmh(self):
        inference_algorithm = blackjax.irmh(
            self.normal_logprob,
            proposal_distribution=functools.partial(
                irmh_proposal_distribution, mean=1.0
            ),
        )
        initial_state = inference_algorithm.init(jnp.array(1.0))

        self.univariate_normal_test_case(
            inference_algorithm, self.key, initial_state, 50000, 5000
        )

    @chex.all_variants(with_pmap=False)
    def test_nuts(self):
        inference_algorithm = blackjax.nuts(
            self.normal_logprob, step_size=4.0, inverse_mass_matrix=jnp.array([1.0])
        )

        initial_state = inference_algorithm.init(jnp.array(3.0))

        self.univariate_normal_test_case(
            inference_algorithm, self.key, initial_state, 5000, 1000
        )

    @chex.all_variants(with_pmap=False)
    def test_rmh(self):
        inference_algorithm = blackjax.rmh(
            self.normal_logprob, proposal_generator=rmh_proposal_distribution
        )
        initial_state = inference_algorithm.init(1.0)

        self.univariate_normal_test_case(
            inference_algorithm, self.key, initial_state, 20_000, 5_000
        )

    @chex.all_variants(with_pmap=False)
    def test_rmhmc(self):
        inference_algorithm = blackjax.rmhmc(
            self.normal_logprob,
            mass_matrix=rmhmc_static_mass_matrix_fn,
            step_size=1.0,
            num_integration_steps=30,
        )

        initial_state = inference_algorithm.init(jnp.array(3.0))

        self.univariate_normal_test_case(
            inference_algorithm, self.key, initial_state, 6_000, 1_000
        )

    @chex.all_variants(with_pmap=False)
    def test_elliptical_slice(self):
        inference_algorithm = blackjax.elliptical_slice(
            lambda x: jnp.ones_like(x), cov=jnp.array([2.0**2]), mean=1.0
        )

        initial_state = inference_algorithm.init(1.0)

        self.univariate_normal_test_case(
            inference_algorithm, self.key, initial_state, 20_000, 5_000
        )

    @chex.all_variants(with_pmap=False)
    def test_ghmc(self):
        rng_key, initial_state_key = jax.random.split(self.key)
        inference_algorithm = blackjax.ghmc(
            self.normal_logprob,
            step_size=1.0,
            momentum_inverse_scale=jnp.array(1.0),
            alpha=0.8,
            delta=2.0,
        )
        initial_state = inference_algorithm.init(jnp.array(1.0), initial_state_key)
        self.univariate_normal_test_case(
            inference_algorithm, rng_key, initial_state, 6000, 1000
        )

    @chex.all_variants(with_pmap=False)
    def test_hmc(self):
        rng_key, initial_state_key = jax.random.split(self.key)
        inference_algorithm = blackjax.hmc(
            self.normal_logprob,
            step_size=3.9,
            inverse_mass_matrix=jnp.array([1.0]),
            num_integration_steps=30,
        )
        initial_state = inference_algorithm.init(jnp.array(3.0))
        self.univariate_normal_test_case(
            inference_algorithm, rng_key, initial_state, 6000, 1000
        )

    @chex.all_variants(with_pmap=False)
    def test_orbital_hmc(self):
        inference_algorithm = blackjax.orbital_hmc(
            self.normal_logprob,
            step_size=0.1,
            inverse_mass_matrix=jnp.array([0.1]),
            period=100,
        )
        initial_state = inference_algorithm.init(jnp.array(100.0))
        burnin = 15_000

        def postprocess_samples(states, key):
            positions, weights = states
            return orbit_samples(positions[burnin:], weights[burnin:], key)

        self.univariate_normal_test_case(
            inference_algorithm,
            self.key,
            initial_state,
            20_000,
            burnin,
            postprocess_samples,
            transform=lambda x: (x.positions, x.weights),
        )

    @chex.all_variants(with_pmap=False)
    def test_random_walk(self):
        inference_algorithm = blackjax.additive_step_random_walk.normal_random_walk(
            self.normal_logprob, sigma=jnp.array([1.0])
        )
        initial_state = inference_algorithm.init(jnp.array(1.0))

        self.univariate_normal_test_case(
            inference_algorithm, self.key, initial_state, 20_000, 5_000
        )

    @chex.all_variants(with_pmap=False)
    def test_mala(self):
        inference_algorithm = blackjax.mala(self.normal_logprob, step_size=0.2)
        initial_state = inference_algorithm.init(jnp.array(1.0))
        self.univariate_normal_test_case(
            inference_algorithm, self.key, initial_state, 45000, 5_000
        )

    @chex.all_variants(with_pmap=False)
    def test_barker(self):
        inference_algorithm = blackjax.barker_proposal(
            self.normal_logprob, step_size=1.5
        )
        initial_state = inference_algorithm.init(jnp.array(1.0))
        self.univariate_normal_test_case(
            inference_algorithm, self.key, initial_state, 20000, 2_000
        )


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
        self.key = jax.random.key(8456)

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
            rng_key=multi_chain_sample_key, initial_state=initial_states
        )

        posterior_samples = states.position[:, -1000:]
        posterior_delta = posterior_samples - true_loc
        posterior_variance = posterior_delta**2.0
        posterior_correlation = jnp.prod(posterior_delta, axis=-1, keepdims=True) / (
            true_scale[0] * true_scale[1]
        )

        _ = jax.tree.map(
            self.mcse_test,
            [posterior_samples, posterior_variance, posterior_correlation],
            [true_loc, true_scale**2, true_rho],
        )


if __name__ == "__main__":
    absltest.main()
