import jax
import jax.numpy as jnp
import numpy as np


def extend_to_all_particles(n_particles, tuning_strategy):
    """
    given a tuning strategy that returns a single parameter,
    that parameter gets extended to be applied to all particles
    """
    def extended(state, info):
        res = tuning_strategy(state, info)
        return jnp.repeat(res, n_particles)

    return extended

def extend_params_inner_kernel(n_particles, params):
    """
    Given a dictionary of params, repeats them for every single particle
    Shapes>
    scalar, 1000,
    1 . 1000, 1
    2,2 . 1000,2,2
    """
    def extend(param):
        if np.isscalar(param):
            return jnp.repeat(param, n_particles, axis=0)
        else:
            return jnp.repeat(jnp.atleast_1d(param)[np.newaxis, :], n_particles, axis=0)

    return jax.tree_map(extend, params)