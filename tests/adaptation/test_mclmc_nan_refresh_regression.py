"""A finite kernel restore must still refresh momentum during adaptation."""

import jax
import jax.numpy as jnp
import numpy as np

from blackjax.adaptation.mclmc_adaptation import handle_nans
from blackjax.mcmc import mclmc
from blackjax.util import generate_unit_vector


def test_refresh_after_finite_kernel_restore():
    logdensity = lambda x: jnp.where(x[0] > 0, -0.5 * jnp.sum(x**2), jnp.nan)
    previous = mclmc.init(jnp.array([0.01, 0.2]), logdensity, jax.random.key(0))
    previous = previous._replace(momentum=jnp.array([-1.0, 0.0]))
    next_state, info = mclmc.build_kernel()(
        jax.random.key(1), previous, logdensity, jnp.ones(2), 1e30, 1.0
    )
    assert not info.nonans
    assert jnp.isfinite(next_state.logdensity)
    key = jax.random.key(2)
    success, restored, cap, energy = handle_nans(
        previous, next_state, 1.0, jnp.inf, info.energy_change, info.nonans, key
    )
    assert not success
    np.testing.assert_array_equal(restored.position, previous.position)
    np.testing.assert_array_equal(restored.logdensity_grad, previous.logdensity_grad)
    assert restored.logdensity == previous.logdensity
    assert cap == 0.8
    assert energy == 0.0
    np.testing.assert_allclose(
        restored.momentum, generate_unit_vector(key, previous.position), rtol=1e-6
    )
