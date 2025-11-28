from functools import partial
from typing import Any, Callable, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

import blackjax.mcmc.hmc as hmc
import blackjax.mcmc.integrators as integrators
import blackjax.mcmc.metrics as metrics
import blackjax.mcmc.proposal as proposal
import blackjax.mcmc.termination as termination
import blackjax.mcmc.trajectory as trajectory
from blackjax.base import SamplingAlgorithm
from blackjax.types import ArrayLikeTree, ArrayTree, PRNGKey
import blackjax
from blackjax.mcmc.integrators import (
    IntegratorState,
    isokinetic_mclachlan,
    with_isokinetic_maruyama,
)

def sample_noise_complex(shape, scale=1.0, rng_key=None):
    """
    TODO: Write a good doc for this general function
    TODO: Add type information
    """
    out = jax.random.normal(rng_key, shape=shape)*jnp.sqrt(scale) + 1j*jax.random.normal(rng_key, shape=shape)*jnp.sqrt(scale)
    return out


# class PseudofermionState(NamedTuple):
#     pf: ArrayTree

class GibbsState(NamedTuple):
    x: Any
    y: Any

    # make a property .position
    @property
    def position(self):
        return self.x.position


# x is the gauge field
# \varphi (or y) is the pseudofermion

# draw from: N(0, D(\phi))
# draw from \chi ~ N(0, I) and return D(\phi)\chi

# S \varphi D(\phi)D^T(\phi)\varphi

def build_kernel(kernel_1, kernel_2):
    def kernel(rng_key, state, logdensity_fn, L, step_size, inverse_mass_matrix):
        key1, key2 = jax.random.split(rng_key, 2)
        next_x_state, info = kernel_1(
            rng_key=key1,
            state=state.x,
            step_size=step_size,
            L=L,
            inverse_mass_matrix=inverse_mass_matrix,
            logdensity_fn=partial(logdensity_fn, pf=state.y),
        )
        next_y_state = kernel_2(key2, state.x)
        # st = {'x': next_x_state, 'y': next_y_state}
        st = GibbsState(x=next_x_state, y=next_y_state)
        return st, info
    return kernel

def init(position, logdensity_fn, pseudofermion, init_1, init_2, random_generator_arg  ):
    key_1, key_2 = jax.random.split(random_generator_arg, 2)
    state_b = init_1(position, partial(logdensity_fn, pf=pseudofermion), key_1 )
    state_pf = init_2(pseudofermion)

    return GibbsState(x=state_b, y=state_pf)
    # return {'x': state_b, 'y': state_pf}


def as_top_level_api(
    logdensity_fn: Callable,
    L,
    step_size,
    kernel_1,
    kernel_2,
    init_1,
    inverse_mass_matrix=1.0,
) -> SamplingAlgorithm:

    kernel = build_kernel(
        
        kernel_1=kernel_1,
        kernel_2=kernel_2,
    )

    pseudofermion = jnp.zeros(16*16*2, dtype=jnp.complex128)
    init_2 = lambda x: x
    def init_fn(position, rng_key):
        return init(position, logdensity_fn, pseudofermion=pseudofermion, init_1=init_1, init_2=init_2, rng_key=rng_key)

    def update_fn(rng_key, state):
        return kernel(rng_key, state, logdensity_fn, L, step_size, inverse_mass_matrix)

    return SamplingAlgorithm(init_fn, update_fn)


