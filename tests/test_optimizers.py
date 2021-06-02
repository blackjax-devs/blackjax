import jax
import jax.numpy as jnp
import pytest

import blackjax.optimizers as optimizers


def test_dual_averaging():
    """We test the dual averaging algorithm by searching for the point that
    minimizes the gradient of a simple function.
    """

    # we need to wrap the gradient in a namedtuple as we optimize for a target
    # acceptance probability in the context of HMC.
    f = lambda x: (x - 1) ** 2

    # Our target gradient is 0. we increase the rate of convergence by
    # increasing the value of gamma (see documentation of the algorithm).
    init, update, final = optimizers.dual_averaging(gamma=0.3)

    da_state = init(3)
    for _ in range(100):
        x = jnp.exp(da_state.log_x)
        g = jax.grad(f)(x)
        da_state = update(da_state, g)

    assert final(da_state) == pytest.approx(1.0, 1e-1)
