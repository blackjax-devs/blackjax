"""Test the behaviour of the coupled RWMH kernel"""

import jax
import jax.numpy as jnp
import jax.scipy.stats as stats
import numpy as np

import blackjax.coupled.rwmh as coupled_rwmh
import blackjax.rwmh as rwmh


def inference_loop(rng_key, kernel, initial_state, num_samples):
    def one_step(state, rng_key):
        state, _ = kernel(rng_key, state)
        return state, state

    keys = jax.random.split(rng_key, num_samples)
    _, states = jax.lax.scan(one_step, initial_state, keys)

    return states


def normal_potential_fn(x):
    return -stats.norm.logpdf(x, loc=1.0, scale=2.0).squeeze()


def test_coupling():
    rng_key = jax.random.PRNGKey(19)
    n_sampling_steps = 50_000

    potential = lambda x: normal_potential_fn(**x)

    initial_position_1 = {"x": 1.}
    initial_position_2 = {"x": -1.}

    initial_state = rwmh.new_state(initial_position_1, potential)
    initial_state_coupled = coupled_rwmh.new_state(initial_position_1, initial_position_2, potential)

    parameters = {"inverse_mass_matrix": jnp.array([10.]),
                  }

    rwmh_kernel = rwmh.kernel(potential, **parameters)
    coupled_rwmh_kernel = coupled_rwmh.kernel(potential, **parameters)

    states = inference_loop(rng_key, rwmh_kernel, initial_state, n_sampling_steps)
    coupled_states = inference_loop(rng_key, coupled_rwmh_kernel, initial_state_coupled, n_sampling_steps)

    samples = states.position["x"]
    coupled_states_1 = coupled_states.state_1.position["x"]
    coupled_states_2 = coupled_states.state_2.position["x"]

    np.testing.assert_array_almost_equal(samples, coupled_states_1)
    np.testing.assert_array_almost_equal(coupled_states_1[2000:],
                                         coupled_states_2[2000:])  # the two trajectories match
    np.testing.assert_array_less(coupled_states_2[:1000], coupled_states_1[:1000])  # burn-in phase
