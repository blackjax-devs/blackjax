"""Test the generic SMC sampler"""
import chex
import jax
import jax.numpy as jnp
import jax.scipy.stats as stats
import numpy as np
from absl.testing import absltest

import blackjax
import blackjax.smc.resampling as resampling
from blackjax.smc.base import extend_params, init, step


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
        num_particles = 1000

        def update_fn(rng_key, position, update_params):
            hmc = blackjax.hmc(logdensity_fn, **update_params)
            state = hmc.init(position)

            def body_fn(state, rng_key):
                new_state, info = hmc.step(rng_key, state)
                return new_state, info

            keys = jax.random.split(rng_key, num_mcmc_steps)
            last_state, info = jax.lax.scan(body_fn, state, keys)
            return last_state.position, info

        init_key, sample_key = jax.random.split(self.key)

        # Initialize the state of the SMC sampler
        init_particles = 0.25 + jax.random.normal(init_key, shape=(num_particles,))
        same_for_all_params = dict(
            step_size=1e-2, inverse_mass_matrix=jnp.eye(1), num_integration_steps=50
        )

        state = init(
            init_particles,
            same_for_all_params,
        )

        # Run the SMC sampler once
        new_state, info = self.variant(step, static_argnums=(2, 3, 4))(
            sample_key,
            state,
            jax.vmap(update_fn, in_axes=(0, 0, None)),
            jax.vmap(logdensity_fn),
            resampling.systematic,
        )

        mean, std = _weighted_avg_and_std(new_state.particles, state.weights)
        np.testing.assert_allclose(0.0, mean, atol=1e-1)
        np.testing.assert_allclose(1.0, std, atol=1e-1)

    @chex.variants(with_jit=True)
    def test_smc_waste_free(self):
        num_mcmc_steps = 10
        num_particles = 1000
        num_resampled = num_particles // num_mcmc_steps

        def waste_free_update_fn(keys, particles, update_params):
            def one_particle_fn(rng_key, position, particle_update_params):
                hmc = blackjax.hmc(logdensity_fn, **particle_update_params)
                state = hmc.init(position)

                def body_fn(state, rng_key):
                    new_state, info = hmc.step(rng_key, state)
                    return new_state, (state, info)

                keys = jax.random.split(rng_key, num_mcmc_steps)
                _, (states, info) = jax.lax.scan(body_fn, state, keys)
                return states.position, info

            particles, info = jax.vmap(one_particle_fn, in_axes=(0, 0, None))(
                keys, particles, update_params
            )
            particles = particles.reshape((num_particles,))
            return particles, info

        init_key, sample_key = jax.random.split(self.key)

        # Initialize the state of the SMC sampler
        init_particles = 0.25 + jax.random.normal(init_key, shape=(num_particles,))
        state = init(
            init_particles,
            dict(
                step_size=1e-2,
                inverse_mass_matrix=jnp.eye(1),
                num_integration_steps=100,
            ),
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

        mean, std = _weighted_avg_and_std(new_state.particles, state.weights)
        np.testing.assert_allclose(0.0, mean, atol=1e-1)
        np.testing.assert_allclose(1.0, std, atol=1e-1)


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
