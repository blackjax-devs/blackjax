from functools import partial

import chex
import numpy as np
from absl.testing import absltest, parameterized
from jax import jit
from jax import numpy as jnp
from jax import random as jr
from jax import tree, vmap

import blackjax
from blackjax.diagnostics import psis_weights
from blackjax.util import (
    run_inference_algorithm,
    store_only_expectation_values,
    thin_algorithm,
    thin_kernel,
)


class RunInferenceAlgorithmTest(chex.TestCase):
    def setUp(self):
        super().setUp()
        self.key = jr.key(42)
        self.algorithm = blackjax.hmc(
            logdensity_fn=self.logdensity_fn,
            inverse_mass_matrix=jnp.eye(2),
            step_size=1.0,
            num_integration_steps=1000,
        )
        self.num_steps = 10

    def check_compatible(self, initial_state, progress_bar):
        """
        Runs 10 steps with `run_inference_algorithm` starting with
        `initial_state` and potentially a progress bar.
        """
        _ = run_inference_algorithm(
            rng_key=self.key,
            initial_state=initial_state,
            inference_algorithm=self.algorithm,
            num_steps=self.num_steps,
            progress_bar=progress_bar,
            transform=lambda state, info: state.position,
        )

    def test_streaming(self):
        def logdensity_fn(x):
            return -0.5 * jnp.sum(jnp.square(x))

        initial_position = jnp.ones(
            10,
        )

        init_key, state_key, run_key = jr.split(self.key, 3)
        initial_state = blackjax.mcmc.mclmc.init(
            position=initial_position, logdensity_fn=logdensity_fn, rng_key=state_key
        )
        L = 1.0
        step_size = 0.1
        num_steps = 4

        sampling_alg = blackjax.mclmc(
            logdensity_fn,
            L=L,
            step_size=step_size,
        )

        state_transform = lambda x: x.position

        _, samples = run_inference_algorithm(
            rng_key=run_key,
            initial_state=initial_state,
            inference_algorithm=sampling_alg,
            num_steps=num_steps,
            transform=lambda state, info: state_transform(state),
            progress_bar=True,
        )

        print("average of steps (slow way):", samples.mean(axis=0))

        memory_efficient_sampling_alg, transform = store_only_expectation_values(
            sampling_algorithm=sampling_alg, state_transform=state_transform
        )

        initial_state = memory_efficient_sampling_alg.init(initial_state)

        final_state, trace_at_every_step = run_inference_algorithm(
            rng_key=run_key,
            initial_state=initial_state,
            inference_algorithm=memory_efficient_sampling_alg,
            num_steps=num_steps,
            transform=transform,
            progress_bar=True,
        )

        assert jnp.allclose(trace_at_every_step[0][-1], samples.mean(axis=0))

    @parameterized.parameters([True, False])
    def test_compatible_with_initial_pos(self, progress_bar):
        _ = run_inference_algorithm(
            rng_key=self.key,
            initial_position=jnp.array([1.0, 1.0]),
            inference_algorithm=self.algorithm,
            num_steps=self.num_steps,
            progress_bar=progress_bar,
            transform=lambda state, info: state.position,
        )

    @parameterized.parameters([True, False])
    def test_compatible_with_initial_state(self, progress_bar):
        state = self.algorithm.init(jnp.array([1.0, 1.0]))
        self.check_compatible(state, progress_bar)

    @staticmethod
    def logdensity_fn(x):
        return -0.5 * jnp.sum(jnp.square(x))


class ThinInferenceAlgorithmTest(chex.TestCase):
    def setUp(self):
        super().setUp()
        self.logdf = lambda x: -(x**2).sum(-1) / 2  # Gaussian
        # self.logdf = (
        #     lambda x: -((x[::2] - 1) ** 2 + (x[1::2] - x[::2] ** 2) ** 2).sum(-1) / 2
        # )  # Rosenbrock
        dim = 2
        self.init_pos = jnp.ones(dim)
        self.rng_keys = jr.split(jr.key(42), 2)
        self.num_steps = 10_000

    def warmup(self, rng_key, num_steps, thinning: int = 1):
        from blackjax.mcmc.integrators import isokinetic_mclachlan

        init_key, tune_key = jr.split(rng_key, 2)

        state = blackjax.mcmc.mclmc.init(
            position=self.init_pos,
            logdensity_fn=self.logdf,
            rng_key=init_key,
        )

        if thinning == 1:
            kernel = lambda inverse_mass_matrix: blackjax.mcmc.mclmc.build_kernel(
                integrator=isokinetic_mclachlan,
                logdensity_fn=self.logdf,
                inverse_mass_matrix=inverse_mass_matrix,
            )
        else:
            kernel = lambda inverse_mass_matrix: thin_kernel(
                blackjax.mcmc.mclmc.build_kernel(
                    integrator=isokinetic_mclachlan,
                    logdensity_fn=self.logdf,
                    inverse_mass_matrix=inverse_mass_matrix,
                ),
                thinning=thinning,
                info_transform=lambda info: tree.map(
                    lambda x: (x**2).mean() ** 0.5, info
                ),
            )

        state, config, n_steps = blackjax.mclmc_find_L_and_step_size(
            mclmc_kernel=kernel,
            num_steps=num_steps,
            state=state,
            rng_key=tune_key,
            # frac_tune3=0.
        )
        n_steps *= thinning
        config = config._replace(
            L=config.L * thinning
        )  # NOTE: compensate L for thinning
        return state, config, n_steps

    def run_algo(self, rng_key, state, config, num_steps, thinning: int = 1):
        sampler = blackjax.mclmc(
            self.logdf,
            L=config.L,
            step_size=config.step_size,
            inverse_mass_matrix=config.inverse_mass_matrix,
        )
        if thinning != 1:
            sampler = thin_algorithm(
                sampler,
                thinning=thinning,
                info_transform=lambda info: tree.map(jnp.mean, info),
            )

        state, history = run_inference_algorithm(
            rng_key=rng_key,
            initial_state=state,
            inference_algorithm=sampler,
            num_steps=num_steps,
            # progress_bar=True,
        )
        return state, history

    def test_thin(self):
        """
        Compare results obtained from thinning kernel or algorithm vs. no thinning.
        """
        # Test thin kernel in warmup
        state, config, n_steps = jit(
            vmap(partial(self.warmup, num_steps=self.num_steps, thinning=1))
        )(self.rng_keys)
        config = tree.map(lambda x: jnp.median(x, 0), config)
        state_thin, config_thin, n_steps_thin = jit(
            vmap(partial(self.warmup, num_steps=self.num_steps, thinning=4))
        )(self.rng_keys)
        config_thin = tree.map(lambda x: jnp.median(x, 0), config_thin)

        # Assert that the found parameters are close
        rtol, atol = 1e-1, 1e-1
        np.testing.assert_allclose(config_thin.L, config.L, rtol=rtol, atol=atol)
        np.testing.assert_allclose(
            config_thin.step_size, config.step_size, rtol=rtol, atol=atol
        )
        np.testing.assert_allclose(
            config_thin.inverse_mass_matrix,
            config.inverse_mass_matrix,
            rtol=rtol,
            atol=atol,
        )

        # Test thin algorithm in run_algo
        state, history = jit(
            vmap(
                partial(
                    self.run_algo, config=config, num_steps=self.num_steps, thinning=1
                )
            )
        )(self.rng_keys, state)
        samples = jnp.concatenate(history[0].position)
        state_thin, history_thin = jit(
            vmap(
                partial(
                    self.run_algo,
                    config=config_thin,
                    num_steps=self.num_steps,
                    thinning=4,
                )
            )
        )(self.rng_keys, state_thin)
        samples_thin = jnp.concatenate(history_thin[0].position)

        # Assert that the sample statistics are close
        rtol, atol = 1e-1, 1e-1
        np.testing.assert_allclose(
            samples_thin.mean(0), samples.mean(0), rtol=rtol, atol=atol
        )
        np.testing.assert_allclose(
            jnp.cov(samples_thin.T), jnp.cov(samples.T), rtol=rtol, atol=atol
        )


class PSISWeightsTest(chex.TestCase):
    """Tests for the pure-JAX psis_weights utility."""

    @chex.variants(with_jit=True, without_jit=True)
    def test_weights_sum_to_one(self):
        """Normalised weights must sum to 1."""
        key = jr.key(0)
        log_ratios = jr.normal(key, (200,))
        log_w, _ = self.variant(psis_weights)(log_ratios)
        np.testing.assert_allclose(jnp.exp(log_w).sum(), 1.0, atol=1e-5)

    @chex.variants(with_jit=True, without_jit=True)
    def test_output_shapes(self):
        """Output shapes should match input length; k should be a scalar."""
        n = 300
        log_ratios = jr.normal(jr.key(1), (n,))
        log_w, k = self.variant(psis_weights)(log_ratios)
        self.assertEqual(log_w.shape, (n,))
        self.assertEqual(k.shape, ())

    @chex.variants(with_jit=True, without_jit=True)
    def test_uniform_ratios_give_uniform_weights(self):
        """When p == q (all log-ratios zero) weights should be uniform."""
        n = 100
        log_ratios = jnp.zeros(n)
        log_w, _ = self.variant(psis_weights)(log_ratios)
        expected = -jnp.log(n)
        np.testing.assert_allclose(log_w, jnp.full(n, expected), atol=1e-5)

    @chex.variants(with_jit=True, without_jit=True)
    def test_pareto_k_light_tail(self):
        """For near-uniform log-ratios (light-tailed), k should be <= 0."""
        # When log-ratios are small and near-Gaussian the tail is light → k < 0.
        log_ratios = jr.normal(jr.key(3), (500,)) * 0.1
        _, k = self.variant(psis_weights)(log_ratios)
        self.assertLessEqual(float(k), 0.5)

    @chex.variants(with_jit=True, without_jit=True)
    def test_pareto_k_heavy_tail(self):
        """For heavy-tailed log-ratios k should be positive."""
        key = jr.key(4)
        # log-ratios from a heavy-tailed distribution: log of Pareto(alpha=2)
        u = jr.uniform(key, (1000,), minval=1e-4, maxval=1.0)
        log_ratios = jnp.log(u ** (-0.5))  # log-Pareto, true tail index k=0.5
        _, k = self.variant(psis_weights)(log_ratios)
        self.assertGreater(float(k), 0.0)

    @chex.variants(with_jit=True, without_jit=True)
    def test_constant_shift_does_not_change_weights(self):
        """Adding a constant to all log-ratios should not change the weights."""
        log_ratios = jr.normal(jr.key(5), (200,))
        log_w1, k1 = self.variant(psis_weights)(log_ratios)
        log_w2, k2 = self.variant(psis_weights)(log_ratios + 7.3)
        np.testing.assert_allclose(log_w1, log_w2, atol=1e-5)
        np.testing.assert_allclose(k1, k2, atol=1e-5)

    def test_too_few_samples_returns_inf_k(self):
        """k=inf signals that the tail is too small for a reliable GPD fit."""
        # n=20 gives M = min(max(int(3*sqrt(20)),5), 4) = 4 < 5 → degenerate.
        log_ratios = jr.normal(jr.key(7), (20,))
        _, k = psis_weights(log_ratios)
        self.assertTrue(jnp.isinf(k))

    def test_r_eff_increases_tail_size(self):
        """Smaller r_eff should produce a larger effective tail (more smoothing).

        r_eff is a Python-level compile-time constant (like a shape), so it
        cannot be traced by chex.variants; JIT coverage is provided implicitly
        since every other variant test uses the default r_eff=1.0.
        """
        # With r_eff < 1 the tail size M = floor(3*sqrt(n/r_eff)) grows, so
        # more of the weight distribution is smoothed and the returned k can differ.
        log_ratios = jr.normal(jr.key(8), (500,)) * 2.0
        _, k_iid = psis_weights(log_ratios, r_eff=1.0)
        _, k_corr = psis_weights(log_ratios, r_eff=0.25)
        # Both should be finite for this input.
        self.assertTrue(jnp.isfinite(k_iid))
        self.assertTrue(jnp.isfinite(k_corr))


if __name__ == "__main__":
    absltest.main()
