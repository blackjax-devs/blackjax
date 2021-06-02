import jax
import jax.numpy as jnp
import numpy as np

import blackjax.hmc as hmc
from blackjax.adaptation.step_size import find_reasonable_step_size
from blackjax.inference.base import new_hmc_state


def test_reasonable_step_size():
    def potential_fn(x):
        return jnp.sum(0.5 * x)

    rng_key = jax.random.PRNGKey(0)

    init_position = jnp.array([3.0])
    reference_state = new_hmc_state(init_position, potential_fn)

    inv_mass_matrix = jnp.array([1.0])
    generator = lambda step_size: hmc.kernel(
        potential_fn, step_size, inv_mass_matrix, 10
    )

    # Test that the algorithm actually does something
    epsilon_1 = find_reasonable_step_size(
        rng_key,
        generator,
        reference_state,
        0.01,
        0.95,
    )
    assert epsilon_1 != 1.0
    assert epsilon_1 != np.inf

    # Different target acceptance rate
    epsilon_2 = find_reasonable_step_size(
        rng_key,
        generator,
        reference_state,
        1.0,
        0.05,
    )
    assert epsilon_2.item != epsilon_1
