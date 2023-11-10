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
from typing import Callable, NamedTuple, Union

import jax
import jax.numpy as jnp

import blackjax.mcmc.integrators as integrators
import blackjax.mcmc.metrics as metrics
import blackjax.mcmc.proposal as proposal
import blackjax.mcmc.trajectory as trajectory
from blackjax.base import SamplingAlgorithm
from blackjax.types import Array, ArrayLikeTree, ArrayTree, PRNGKey

__all__ = ["MCLMCState", "MCLMCInfo", "init", "build_kernel", "mclmc"]

class Parameters(NamedTuple):
    """Tunable parameters
    """

    L: float
    eps: float
    sigma: Array


# MCLMCState = integrators.IntegratorState


class MCLMCState(NamedTuple):
    """State of the MCLMC algorithm.

    """

    x: Array
    u: Array
    l: float
    g: Array

class MCLMCInfo(NamedTuple):
    """Additional information on the MCLMC transition.

    This additional information can be used for debugging or computing
    diagnostics.
    """

    transformed_x: Array
    l: Array
    de: float

def init(x_initial : ArrayLikeTree, logdensity_fn, random_key):

        grad_nlogp = jax.value_and_grad(lambda x : - logdensity_fn(x))
        l, g = grad_nlogp(x_initial)

        u = random_unit_vector(random_key, d=x_initial.shape[0])

        return MCLMCState(x_initial, u, l, g)


def random_unit_vector(random_key,d):
    u = jax.random.normal(jax.random.PRNGKey(0), shape = (d, ))
    u /= jnp.sqrt(jnp.sum(jnp.square(u)))
    return u


def update_position(grad_nlogp):
  
  def update(eps, x, u):
    xx = x + eps * u
    ll, gg = grad_nlogp(xx)
    return xx, ll, gg
  
  return update

def update_momentum(d):
  """The momentum updating map of the esh dynamics (see https://arxiv.org/pdf/2111.02434.pdf)
  similar to the implementation: https://github.com/gregversteeg/esh_dynamics
  There are no exponentials e^delta, which prevents overflows when the gradient norm is large."""
  
  
  def update(eps, u, g):
      g_norm = jnp.sqrt(jnp.sum(jnp.square(g)))
      e = - g / g_norm
      ue = jnp.dot(u, e)
      delta = eps * g_norm / (d-1)
      zeta = jnp.exp(-delta)
      uu = e *(1-zeta)*(1+zeta + ue * (1-zeta)) + 2*zeta* u
      delta_r = delta - jnp.log(2) + jnp.log(1 + ue + (1-ue)*zeta**2)
      return uu/jnp.sqrt(jnp.sum(jnp.square(uu))), delta_r

  return update

def partially_refresh_momentum(d, sequential= True):
  """Adds a small noise to u and normalizes."""
    
    
  def rng_sequential(u, random_key, nu):
    z = nu * jax.random.normal(random_key, shape = (d, ))

    return (u + z) / jnp.sqrt(jnp.sum(jnp.square(u + z)))
  

#   def rng_parallel(u, random_key, nu):
#       key, subkey = jax.random.split(random_key)
#       noise = nu * jax.random.normal(subkey, shape= u.shape, dtype=u.dtype)

#       return (u + noise) / jnp.sqrt(jnp.sum(jnp.square(u + noise), axis = 1))[:, None], key


  return rng_sequential

def update(hamiltonian_dynamics, partially_refresh_momentum, d):
    
#   print("BAR 4")
  def step(x, u, g, random_key, L, eps, sigma):
      """One step of the generalized dynamics."""

      # Hamiltonian step
    #   print("BAR 3")
      xx, uu, ll, gg, kinetic_change = hamiltonian_dynamics(x=x, u=u, g=g, eps=eps, sigma = sigma)

      # Langevin-like noise
      nu = jnp.sqrt((jnp.exp(2 * eps / L) - 1.) / d)
      uu = partially_refresh_momentum(u= uu, random_key= random_key, nu= nu)

      return xx, uu, ll, gg, kinetic_change

  return step

def build_kernel(grad_nlogp, d, integrator, transform, params):

        L, eps, sigma = params

        hamiltonian_step, _ = integrator(T= update_position(grad_nlogp), 
                                                                V= update_momentum(d),
                                                                d= d)
        # print("BAR")
        move = update(hamiltonian_step, partially_refresh_momentum(d), d)
        # print("BAZ")
        
        def kernel(rng_key : PRNGKey, state : MCLMCState) -> tuple[MCLMCState, MCLMCInfo]:

            x, u, l, g = state

        
            xx, uu, ll, gg, kinetic_change = move(x, u, g, rng_key, L, eps, sigma)
            de = kinetic_change + ll - l
            return MCLMCState(xx, uu, ll, gg), MCLMCInfo(transform(xx), ll, de)

        return kernel

lambda_c = 0.1931833275037836 #critical value of the lambda parameter for the minimal norm integrator

def minimal_norm(d, T, V):

  def step(x, u, g, eps, sigma):
      """Integrator from https://arxiv.org/pdf/hep-lat/0505020.pdf, see Equation 20."""

      # V T V T V
      uu, r1 = V(eps * lambda_c, u, g * sigma)
      xx, ll, gg = T(eps, x, 0.5*uu*sigma)
      uu, r2 = V(eps * (1 - 2 * lambda_c), uu, gg * sigma)
      xx, ll, gg = T(eps, xx, 0.5*uu*sigma)
      uu, r3 = V(eps * lambda_c, uu, gg * sigma)

      #kinetic energy change
      kinetic_change = (r1 + r2 + r3) * (d-1)

      return xx, uu, ll, gg, kinetic_change
    
  return step, 2



class mclmc:
    """todo: add documentation"""

    init = staticmethod(init)
    build_kernel = staticmethod(build_kernel)

    def __new__(  # type: ignore[misc]
        cls,
        logdensity_fn: Callable,
        d : int,
        transform : Callable,
        params : Parameters,
        init_key,
        *,
        integrator = minimal_norm,
    ) -> SamplingAlgorithm:
                
        grad_nlogp = jax.value_and_grad(lambda x : - logdensity_fn(x))

        kernel = cls.build_kernel(grad_nlogp, d, integrator, transform, params)

        def init_fn(position: ArrayLikeTree):
            return cls.init(position, logdensity_fn, init_key)

        def step_fn(rng_key: PRNGKey, state):
            return kernel(
                rng_key,
                state,
            )

        return SamplingAlgorithm(init_fn, step_fn)

