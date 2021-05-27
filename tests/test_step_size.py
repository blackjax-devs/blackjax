import functools as ft
from collections import namedtuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import blackjax.hmc as hmc
from blackjax.adaptation.step_size import dual_averaging, find_reasonable_step_size
from blackjax.inference.base import new_hmc_state


def test_reasonable_step_size():
    def potential_fn(x):
        return jnp.sum(0.5 * x)

    rng_key = jax.random.PRNGKey(0)

    init_position = jnp.array([3.0])
    reference_state = new_hmc_state(init_position, potential_fn)

    inv_mass_matrix = jnp.array([1.0])
    generator = ft.partial(
        hmc.kernel,
        potential_fn=potential_fn,
        num_integration_steps=10,
        inverse_mass_matrix=inv_mass_matrix,
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


def test_dual_averaging():
    """We test the dual averaging algorithm by searching for the point that
    minimizes the gradient of a simple function.
    """

    # we need to wrap the gradient in a namedtuple as we optimize for a target
    # acceptance probability in the context of HMC.
    to_hmc_info_mock = namedtuple("Gradient", ["acceptance_probability"])
    f = lambda x: (x - 1) ** 2

    # Our target gradient is 0. we increase the rate of convergence by
    # increasing the value of gamma (see documentation of the algorithm).
    init, update, final = dual_averaging(gamma=0.3, target=0)

    da_state = init(3)
    for _ in range(100):
        x = jnp.exp(da_state.log_step_size)
        g = -jax.grad(f)(x)
        da_state = update(da_state, to_hmc_info_mock(g))

    assert final(da_state) == pytest.approx(1.0, 1e-1)
