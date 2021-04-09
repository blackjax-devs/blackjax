import jax
from jax import numpy as jnp

from blackjax.adaptation.step_size import find_reasonable_step_size
import blackjax.hmc as hmc
from blackjax.hmc import HMCParameters


def test_find_reasonable_step_size():
    def potential_fn(x):
        return jnp.sum(0.5 * jnp.square(x))

    inv_mass_matrix = jnp.array([1.0])
    num_integration_steps = 30
    initial_position = jnp.array([3.0])
    initial_state = hmc.new_state(initial_position, potential_fn)

    def compute_step_size(target_accept):
        init, update, final, do_continue = find_reasonable_step_size(target_accept)

        def cond_fn(val):
            _, rss_state, _ = val
            return do_continue(rss_state)

        def one_step(val):
            rng_key, rss_state, state = val
            _, rng_key = jax.random.split(rng_key)
            params = HMCParameters(rss_state.step_size, num_integration_steps, inv_mass_matrix)
            kernel = hmc.kernel(potential_fn, params)
            state, info = kernel(rng_key, state)
            rss_state = update(rss_state, state, info)
            return (rng_key, rss_state, state)

        rss_state = init(1.)
        _, rss_state, _ = jax.lax.while_loop(cond_fn, one_step, (jax.random.PRNGKey(0), rss_state, initial_state))
        step_size = final(rss_state)

        return step_size

    epsilon_0 = compute_step_size(.95)
    epsilon_1 = compute_step_size(.01)

    assert epsilon_0 != 1.
    assert epsilon_0 != epsilon_1
