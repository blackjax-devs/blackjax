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
#"""Public API for the MCLMC Kernel"""

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
from typing import Callable, NamedTuple, Any

from blackjax.base import SamplingAlgorithm
from blackjax.mcmc.integrators import IntegratorState, noneuclidean_mclachlan, noneuclidean_leapfrog
from blackjax.types import Array, ArrayLike, PRNGKey
from blackjax.util import pytree_size

from blackjax.mcmc import mclmc

from blackjax.mcmc.integrators import _normalized_flatten_array

#__all__ = ["MCLMCInfo", "init", "build_kernel", "mclmc"]


class Hyperparameters(NamedTuple):
    
    L: float
    step_size: float


class AdaptationState(NamedTuple):
    
    cond: bool
    steps: int
    eevpd: float
    history: Array
    
    hyperparameters: Any
    
    



def init(position, logdensity_fn):
    """initialize the chains based on the equipartition of the initial condition.
       We initialize the velocity along grad log p if E_ii > 1 and along -grad log p if E_ii < 1.
    """
    
    def _init(x):
        """u = g / gnorm"""
        l, g = jax.value_and_grad(logdensity_fn)(x)
        flat_g, unravel_fn = ravel_pytree(g)
        u = unravel_fn(_normalized_flatten_array(flat_g)[0]) # = grad logp/ |grad logp|
        return u, l, g

    velocity, logdensity, logdensity_grad = jax.vmap(_init)(position)
    
    # flip the velocity, depending on the equipartition condition
    to_sign = lambda equipartition: -2. * (equipartition < 1.) + 1.    
    velocity = jax.tree_util.tree_map(lambda x, u, g: to_sign(jnp.average(-x * g, axis = 0))* u, position, velocity, logdensity_grad)
    
    return IntegratorState(position, velocity, logdensity, logdensity_grad)
    
    

def init_adap(num_steps, delay_frac, position, d):
    
    delay_num = jnp.rint(delay_frac * num_steps).astype(int)
    
    #flat_pytree, unravel_fn = ravel_pytree(sequential_pytree)
    
    #sigma = unravel_fn(jnp.ones(flat_pytree.shape, dtype = flat_pytree.dtype))
    
    step_size = 0.01 * jnp.sqrt(d)
    L = computeL(position)
    hyp = Hyperparameters(L, step_size)
    
    history = jnp.concatenate((jnp.ones(1) * 1e50, jnp.ones(delay_num-1) * jnp.inf)) # loss history
    
    return AdaptationState(True, 0, 1e-3, history, hyp)


def equipartition_loss(fullrank = True):

    def hutchinson_realization(position, logdensity_grad, rng_key):
        #z ~ rademacher(shape of sequential x)
        #v = z - ensemble average(z.g[m] x[m])

        x, unravel_fn = ravel_pytree(position)
        z_flat = unravel_fn(jax.random.rademacher(rng_key, shape=x.shape, dtype=x.dtype))
        z = unravel_fn(z_flat)
        
        jax.tree_util.tree_map(lambda z, g: z * g)


    def _fullrank(position, logdensity_grad, rng_key):
        """loss = Tr[(1 - E)^T (1 - E)] / d^2
            where Eij = <xi gj> is the equipartition patrix.
            Loss is computed with the Hutchinson's trick."""
        
        raise ValueError("full rank equipartition not implemented yet")
        # z = (100, x.shape)
        # average(v^2) / d^2 
        
        chains = len(position)
        dim = pytree_size(position) // chains

        z = jax.vmap(generate_rademacher)(jax.random.split(rng_key, 100), position) # <z_i z_j> = delta_ij
        
        X = z - (g @ z.T).T @ position / chains
        return jnp.average(jnp.square(X)) / dim


    def _diagonal(position, logdensity_grad, rng_key):
        """Ei = E_ensemble (- grad log p_i x_i ). Ei is 1 if we have converged. 
        virial_loss = average over parameters (Ei)"""
        E = jax.tree_util.tree_map(lambda x, g: jnp.square(1 - jnp.average(-x * g, axis = 0)), position, logdensity_grad)
        
        return jnp.average(ravel_pytree(E)[0])
        

    return _fullrank if fullrank else _diagonal
  


def parallelize(func, chains):
    # we should also have pmap and a combination pmap(vmap())
    return jax.vmap(func)



def computeL(alpha, chains, position):
    
    x, unravel_fn = ravel_pytree(position)

    return alpha * jnp.sqrt(jnp.sum(jnp.square(x))/chains) #average over the ensemble, sum over dimensions


def no_nans(a):
    flat_a, unravel_fn = ravel_pytree(a)
    return jnp.all(jnp.isfinite(flat_a))


def nan_reject(nonans, old, new):
    """Equivalent to
        return new if nonans else old"""
    
    # TODO: jnp.nan_to_num(new)?
    return jax.lax.cond(nonans, lambda _: new, lambda _: old, operand=None)


def build_kernel1(sequential_mclmc_kerel, chains, equi, d, alpha = 1., C = 0.1):

    mclmc_kernel = parallelize(sequential_mclmc_kerel, chains)
    
    def kernel(state_all):

        state, adap_state, rng_key = state_all
        rng_key_new, key_kernel, key_hutchinson = jax.random.split(rng_key, 3)
        
        
        # apply one step of the dynamics
        _state, info = mclmc_kernel(key_kernel, state)
        eevpd = jnp.average(jnp.square(info.de)) / d
        
        
        # reject the new state if there were nans
        nonans = no_nans(_state.position)
        eevpd, state = nan_reject(nonans, (adap_state.eevpd, state), (eevpd, _state))


        # hyperparameter adaptation                                              
        bias = equi(_state, key_hutchinson) #estimate the bias from the equipartition loss

        eevpd_wanted = C * jnp.power(bias, 3./8.)
        
        eps_factor = jnp.select(nonans, jnp.power(eevpd_wanted / eevpd, 1./6.), 0.5)
        eps_factor = jnp.clip(eps_factor, 0.3, 3.)
        
        L = computeL(alpha, chains, state.position)
        
        hyp = Hyperparameters(L, adap_state.step_size * eps_factor)
        
        
        # determine if we want to finish this stage (= if loss is no longer decreassing)
        history = jnp.concatenate((jnp.ones(1) * bias, adap_state.history[:-1]))
        cond *= (history[-1] > history[0])

        
        return _state, AdaptationState(cond, adap_state.steps + 1, eevpd, history, hyp), rng_key_new
    
    
    def verbose_kernel(state_all):
        
        new_state_all = kernel(state_all)
                
        ### diagnostics ###
        equi_diag, key = self.equipartition_diagonal(x, g, key) # estimate the bias from the equipartition loss
        equi_full, key = self.equipartition_fullrank(x, g, key)
        
        btrue = ground_truth_bias(x) #actual bias
        
        varew = C * jnp.power(equi_full, 3./8.)
        
    
        
    return kernel



def stage1(logdensity_fn, num_steps, initial_position, chains, key):
    
    verbose = True
    delay_frac = 0.05
    C = 0.1
    alpha = 1. 
    fullrank = False
    
    
    # kernel    
    d = ravel_pytree(initial_position)[0].shape[0] //chains # number of dimensions

    sequential_kernel = mclmc.build_kernel(
        logdensity_fn=logdensity_fn,
        integrator= noneuclidean_leapfrog,
        transform=lambda x: x
    )
    
    kernel = build_kernel1(sequential_kernel, chains, equipartition_loss(fullrank), d, alpha, C)

    # initialize 
    init_key, run_key = jax.random.split(key, 3)
    
    state = init(position=initial_position, logdensity_fn=logdensity_fn, rng_key=init_key)
    adap_state = init_adap(num_steps, delay_frac, state.position, d)

    init = (state, adap_state, run_key)
    cond = lambda state_all: state_all[1][0]

    if verbose:
        diagnostics1 = []
        state_all = mywhile(cond, kernel, state)
    else:
        state_all = jax.lax.while_loop(cond, kernel, state)

    state, adap_state, key = state_all
    steps_left = num_steps - adap_state.steps

    
        
    # if self.integrator == 'MN':
        #hyp['eps'] *= jnp.sqrt(10.)




def mytest():

    chains = 10

    scale_x = jnp.array([0.1, 2., 10., 0.5])
    scale_y = 2.
    scale_z = 0.01

    key1, key2, key3, key21, key22, key23 = jax.random.split(jax.random.PRNGKey(0), 6)
    x = {'x': scale_x[None, :] * jax.random.normal(key1, shape = (chains, len(scale_x))), 'y': scale_y * jax.random.normal(key2, shape = (chains,)), 'z': scale_z * jax.random.normal(key3, shape = (chains,))}
    g = {'x': scale_x[None, :] * jax.random.normal(key21, shape = (chains, len(scale_x))), 'y': scale_y * jax.random.normal(key22, shape = (chains,)), 'z': scale_z * jax.random.normal(key23, shape = (chains,))}

    u = {'x': jnp.ones(shape = (chains, 3)), 'y': jnp.zeros(chains)}
    
    
    xreshaped = jax.numpy.reshape(ravel_pytree(x)[0], (chains,-1))
    
    print(xreshaped.shape)
    
    return
    #flat_x, unravel_fn = jax.vmap(ravel_pytree)(x)

    #print(flat_x.shape)
    equi = equipartition(fullrank= False)(x, g, jax.random.PRNGKey(10))
    print(equi)
    # logp = lambda x: -0.5 * (jnp.sum(jnp.square(x['x'])) + jnp.square(x['y']) + jnp.square(x['z'])
    # init(x, logp)




def mywhile(cond_fun, body_fun, init_val):
    val = init_val
    while cond_fun(val):
        val = body_fun(val)
    return val


mytest()