from functools import partial

import chex
import numpy as np
from absl.testing import absltest, parameterized
from jax import jit
from jax import numpy as jnp
from jax import random as jr
from jax import tree

import blackjax
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
    # logdf = lambda self, x: - (x**2).sum(-1) / 2 # Gaussian
    logdf = (
        lambda self, x: -((x[::2] - 1) ** 2 + (x[1::2] - x[::2] ** 2) ** 2).sum(-1) / 2
    )  # Rosenbrock
    d = 2
    init_pos = jnp.ones(d)
    rng_keys = jr.key(42)
    num_steps = 10_000

    def warmup(self, rng_key, num_steps, thinning: int = 1):
        from blackjax.mcmc.integrators import isokinetic_mclachlan

        init_key, tune_key = jr.split(rng_key, 2)

        state = blackjax.mcmc.mclmc.init(
            position=self.init_pos, logdensity_fn=self.logdf, rng_key=init_key
        )

        if thinning == 1:
            kernel = lambda inverse_mass_matrix: blackjax.mcmc.mclmc.build_kernel(
                logdensity_fn=self.logdf,
                integrator=isokinetic_mclachlan,
                inverse_mass_matrix=inverse_mass_matrix,
            )
        else:
            kernel = lambda inverse_mass_matrix: thin_kernel(
                blackjax.mcmc.mclmc.build_kernel(
                    logdensity_fn=self.logdf,
                    integrator=isokinetic_mclachlan,
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

    def run(self, rng_key, state, config, num_steps, thinning: int = 1):
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
        # Test thin kernel in warmup
        state, config, n_steps = jit(
            partial(self.warmup, num_steps=self.num_steps, thinning=1)
        )(self.rng_keys)
        state_thin, config_thin, n_steps_thin = jit(
            partial(self.warmup, num_steps=self.num_steps, thinning=4)
        )(self.rng_keys)

        rtol = 7e-1
        np.testing.assert_allclose(config_thin.L, config.L, rtol=rtol)
        np.testing.assert_allclose(config_thin.step_size, config.step_size, rtol=rtol)
        np.testing.assert_allclose(
            config_thin.inverse_mass_matrix, config.inverse_mass_matrix, rtol=rtol
        )

        # Test thin algorithm in run
        state, history = jit(
            partial(self.run, config=config, num_steps=self.num_steps, thinning=1)
        )(self.rng_keys, state)
        samples = history[0].position
        state_thin, history_thin = jit(
            partial(self.run, config=config_thin, num_steps=self.num_steps, thinning=4)
        )(self.rng_keys, state_thin)
        samples_thin = history_thin[0].position

        rtol, atol = 1e-1, 1e-1
        np.testing.assert_allclose(
            samples_thin.mean(0), samples.mean(0), rtol=rtol, atol=atol
        )
        np.testing.assert_allclose(
            jnp.cov(samples_thin.T), jnp.cov(samples.T), rtol=rtol, atol=atol
        )


if __name__ == "__main__":
    absltest.main()
