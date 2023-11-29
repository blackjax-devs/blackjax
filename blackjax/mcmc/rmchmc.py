# Copyright 2020- The Blackjax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Public API for the MCLMC Kernel"""
from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp

from blackjax.base import SamplingAlgorithm
from blackjax.types import Array, ArrayLikeTree, PRNGKey

__all__ = ["RMCHMCState", "MCLMCInfo", "init", "build_kernel", "mclmc"]


from mclmc import Parameters, MCLMCInfo, full_refresh, update_position, update_momentum, minimal_norm


class RMCHMCState(NamedTuple):
    """State of the MCLMC algorithm."""
    
    t: float # time step (0., 1., 2., ....)
    x: Array # location in the sampling space
    l: float # - log p(x)
    g: Array # - grad log p(x)


def init(x_initial : ArrayLikeTree, logdensity_fn):

    grad_nlogp = jax.value_and_grad(lambda x : - logdensity_fn(x))
    l, g = grad_nlogp(x_initial)

    return RMCHMCState(0., x_initial, l, g)




def halton(t, max_bits=10):
    """for t= 0., 1., 2., ... it outputs halton sequence at that index (0.5, 0.25, 0.75, ...)
        taken from: https://github.com/tensorflow/probability/blob/main/discussion/snaper_hmc/SNAPER-HMC.ipynb"""
    float_index = jnp.asarray(t)
    bit_masks = 2**jnp.arange(max_bits, dtype=float_index.dtype)
    return jnp.einsum('i,i->', jnp.mod((float_index + 1) // bit_masks, 2), 0.5 / bit_masks)



def rescale(mu):
    """returns s, such that 
        round(U(0, 1) * s + 0.5)
       has expected value mu.    
    """
    k = jnp.floor(2 * mu -1)
    x = k * (mu - 0.5 *(k+1)) / (k + 1 - mu)
    return k + x
    

def trajectory_length(t, mu):
    s = rescale(mu)
    return jnp.rint(0.5 + halton(t) * s)



def proposal(hamiltonian_step, d):

    def prop(t, x, g, random_key, L, eps, sigma):
                
        #jiter the number of steps
        num_steps = jnp.rint(2 * halton(t) * L / eps).astype(int)
        
        #full momentum refreshment
        u = full_refresh(random_key, d)

        # do num_steps of the Hamiltonian dynamics

        def body(i, state):
            
            x, u, l, g, kinetic_energy = state
            xx, uu, ll, gg, kinetic_change = hamiltonian_step(x=x, u=u, g=g, eps=eps, sigma = sigma)

            return xx, uu, ll, gg, kinetic_energy + kinetic_change
        
        xx, uu, ll, gg, kinetic_change = jax.fori_loop(0, num_steps, body, (x, u, 0., g, 0.))
        
        return xx, ll, gg, kinetic_change

    return prop


def build_kernel(grad_nlogp, d, integrator, transform, params):

    L, eps, sigma = params

    hamiltonian_step, _ = integrator(T= update_position(grad_nlogp), V= update_momentum(d), d= d)
    get_proposal = proposal(hamiltonian_step, d)
    
    def kernel(rng_key : PRNGKey, state : RMCHMCState) -> tuple[RMCHMCState, MCLMCInfo]:
        
        key1, key2 = jax.random.split(rng_key)
            
        t, x, l, g = state
        xx, ll, gg, kinetic_change = get_proposal(t, x, g, key1, L, eps, sigma)
        de = kinetic_change + ll - l
        
        # accept/reject

        acc_prob = jnp.clip(jnp.exp(-de), 0, 1)
        accept = jax.random.bernoulli(key2, acc_prob)
        xx, ll, gg = jax.tree_util.tree_map(lambda new, old: jax.lax.select(accept, new, old), (xx, ll, gg), (x, l, g))
               
        return RMCHMCState(t + 1., xx, ll, gg), MCLMCInfo(transform(xx), ll, de)

    return kernel


class rmchmc:
    """todo: add documentation"""

    init = staticmethod(init)
    build_kernel = staticmethod(build_kernel)

    def __new__(  # type: ignore[misc]
        cls,
        logdensity_fn: Callable,
        d : int,
        transform : Callable,
        params : Parameters,
        *,
        integrator = minimal_norm,
    ) -> SamplingAlgorithm:
                
        grad_nlogp = jax.value_and_grad(lambda x : - logdensity_fn(x))

        kernel = cls.build_kernel(grad_nlogp, d, integrator, transform, params)

        def init_fn(position: ArrayLikeTree):
            return cls.init(position, logdensity_fn)

        def step_fn(rng_key: PRNGKey, state):
            return kernel(
                rng_key,
                state,
            )

        return SamplingAlgorithm(init_fn, step_fn)

