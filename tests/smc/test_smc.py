"""Test the generic SMC sampler"""
import functools

import chex
import jax
import jax.numpy as jnp
import jax.scipy.stats as stats
import numpy as np
from absl.testing import absltest

import blackjax
import blackjax.smc.resampling as resampling
from blackjax.smc.base import extend_params, init, step
from blackjax.smc.tempered import update_and_take_last
from blackjax.smc.waste_free import update_waste_free


def logdensity_fn(position):
    return jnp.sum(stats.norm.logpdf(position))


def _weighted_avg_and_std(values, weights):
    average = jnp.average(values, weights=weights)
    variance = jnp.average((values - average) ** 2, weights=weights)
    return average, jnp.sqrt(variance)


class SMCTest(chex.TestCase):
    def setUp(self):
        super().setUp()
        self.key = jax.random.key(42)

    @chex.variants(with_jit=True)
    def test_smc(self):
        num_mcmc_steps = 20
        num_particles = 5000

        same_for_all_params = dict(
            step_size=1e-2, inverse_mass_matrix=jnp.eye(1), num_integration_steps=50
        )
        hmc_kernel = functools.partial(
            blackjax.hmc.build_kernel(), **same_for_all_params
        )
        hmc_init = blackjax.hmc.init

        update_fn, _ = update_and_take_last(
            hmc_init, logdensity_fn, hmc_kernel, num_mcmc_steps, num_particles
        )
        init_key, sample_key = jax.random.split(self.key)

        # Initialize the state of the SMC sampler
        init_particles = 0.25 + jax.random.normal(init_key, shape=(num_particles,))
        state = init(init_particles, {})
        # Run the SMC sampler once
        new_state, info = self.variant(step, static_argnums=(2, 3, 4))(
            sample_key,
            state,
            update_fn,
            jax.vmap(logdensity_fn),
            resampling.systematic,
        )

        assert new_state.particles.shape == (num_particles,)
        mean, std = _weighted_avg_and_std(new_state.particles, state.weights)
        np.testing.assert_allclose(mean, 0.0, atol=1e-1)
        np.testing.assert_allclose(std, 1.0, atol=1e-1)

    @chex.variants(with_jit=True)
    def test_smc_waste_free(self):
        p = 500
        num_particles = 1000
        num_resampled = num_particles // p
        init_key, sample_key = jax.random.split(self.key)

        # Initialize the state of the SMC sampler
        init_particles = 0.25 + jax.random.normal(init_key, shape=(num_particles,))
        state = init(
            init_particles,
            {},
        )
        same_for_all_params = dict(
            step_size=1e-2, inverse_mass_matrix=jnp.eye(1), num_integration_steps=100
        )
        hmc_kernel = functools.partial(
            blackjax.hmc.build_kernel(), **same_for_all_params
        )
        hmc_init = blackjax.hmc.init

        waste_free_update_fn, _ = update_waste_free(
            hmc_init,
            logdensity_fn,
            hmc_kernel,
            num_particles,
            p=p,
            num_resampled=num_resampled,
        )

        # Run the SMC sampler once
        new_state, info = self.variant(step, static_argnums=(2, 3, 4, 5))(
            sample_key,
            state,
            waste_free_update_fn,
            jax.vmap(logdensity_fn),
            resampling.systematic,
            num_resampled,
        )
        assert new_state.particles.shape == (num_particles,)
        mean, std = _weighted_avg_and_std(new_state.particles, state.weights)
        np.testing.assert_allclose(mean, 0.0, atol=1e-1)
        np.testing.assert_allclose(std, 1.0, atol=1e-1)


class ExtendParamsTest(chex.TestCase):
    def test_extend_params(self):
        extended = extend_params(
            {
                "a": 50,
                "b": np.array([50]),
                "c": np.array([50, 60]),
                "d": np.array([[1, 2], [3, 4]]),
            },
        )
        np.testing.assert_allclose(extended["a"], np.ones((1,)) * 50)
        np.testing.assert_allclose(extended["b"], np.array([[50]]))
        np.testing.assert_allclose(extended["c"], np.array([[50, 60]]))
        np.testing.assert_allclose(
            extended["d"],
            np.array([[[1, 2], [3, 4]]]),
        )


if __name__ == "__main__":
    absltest.main()
