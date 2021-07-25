"""Test the behaviour of the coupled RWMH kernel"""

import jax
import jax.numpy as jnp
import jax.scipy.stats as stats
import numpy as np

import blackjax.coupled.rwmh as coupled_rwmh
import blackjax.rwmh as rwmh
from blackjax.inference import metrics
from .utils import inference_loop

def normal_potential_fn(x):
    return -stats.norm.logpdf(x, loc=1.0, scale=2.0).squeeze()


def test_coupling():
    rng_key = jax.random.PRNGKey(19)
    n_sampling_steps = 10_000

    potential = lambda x: normal_potential_fn(**x)

    initial_position_1 = {"x": 1.}
    initial_position_2 = {"x": -1.}

    initial_state = rwmh.new_state(initial_position_1, potential)
    initial_state_coupled = coupled_rwmh.new_state(initial_position_1, initial_position_2, potential)

    parameters = {"inverse_mass_matrix": jnp.array([1e3]), }

    rwmh_kernel = rwmh.kernel(potential, **parameters)
    coupled_rwmh_kernel = coupled_rwmh.kernel(potential, **parameters)

    states = inference_loop(rng_key, rwmh_kernel, initial_state, n_sampling_steps)
    coupled_states = inference_loop(rng_key, coupled_rwmh_kernel, initial_state_coupled, n_sampling_steps)

    samples = states.position["x"]
    coupled_states_1 = coupled_states.state_1.position["x"]
    coupled_states_2 = coupled_states.state_2.position["x"]

    # np.testing.assert_array_almost_equal(samples, coupled_states_1)
    np.testing.assert_array_almost_equal(coupled_states_1[2_000:],
                                         coupled_states_2[2_000:])  # the two trajectories match
    np.testing.assert_array_less(coupled_states_2[:300], coupled_states_1[:300])  # burn-in phase


def test_maximal_coupling():
    position_1, position_2 = 1., -1.
    n_samples = 50_000

    approx_equal_proba = 0.5
    sigma = np.abs(position_1 - position_2) / (np.sqrt(2 * np.pi) * (1 - approx_equal_proba))

    inverse_mass_matrix = jnp.array([1 / sigma ** 2])
    keys = jax.random.split(jax.random.PRNGKey(42), n_samples)

    norm_1, norm_2 = jax.vmap(coupled_rwmh.reflected_gaussians, [0, None, None, None])(keys,
                                                                                       inverse_mass_matrix,
                                                                                       position_1,
                                                                                       position_2)

    np.testing.assert_almost_equal(np.mean(norm_1), position_1, decimal=2)
    np.testing.assert_almost_equal(np.mean(norm_2), position_2, decimal=2)

    np.testing.assert_almost_equal(np.var(norm_1), sigma ** 2, decimal=2)
    np.testing.assert_almost_equal(np.var(norm_2), sigma ** 2, decimal=2)

    np.testing.assert_almost_equal(np.sum((norm_1 - norm_2 < 1e-5)) / n_samples, approx_equal_proba, decimal=1)


def test_one_proposal():
    position_1, position_2 = 0.1, -0.1
    inverse_mass_matrix = jnp.array([1. / (0.1 ** 2)])
    key = jax.random.PRNGKey(42)

    momentum_generator, *_ = metrics.gaussian_euclidean(inverse_mass_matrix)

    norm_1, norm_2 = coupled_rwmh.reflected_gaussians(key, inverse_mass_matrix, position_1, position_2)
    other_norm_1 = momentum_generator(jax.random.split(key)[1], 0.1)
    np.testing.assert_array_almost_equal(norm_1, other_norm_1)
    print(norm_2)
