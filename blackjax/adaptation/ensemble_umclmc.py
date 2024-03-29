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

from blackjax.mcmc.integrators import IntegratorState, isokinetic_mclachlan, isokinetic_leapfrog
from blackjax.types import Array, ArrayLike, PRNGKey
from blackjax.util import pytree_size

from blackjax.mcmc import mclmc

from blackjax.mcmc.integrators import _normalized_flatten_array

#__all__ = ["Hyperparameters", "AdaptationState", "stage1"]


class Hyperparameters(NamedTuple):
    
    L: float
    step_size: float


class AdaptationState(NamedTuple):
    
    cond: bool
    steps: int
    eevpd: float
    eevpd_wanted: float
    history: Array
    
    hyperparameters: Any


class Parallelization():
    
    def __init__(self, pmap_chains= 1, vmap_chains= 1):
        
        self.num_chains = pmap_chains * vmap_chains
        
        if (pmap_chains != 1) and (vmap_chains != 1): # both vmap and pmap are to be applied
            self.pvmap = lambda func, in_axes: jax.pmap(jax.vmap(func, in_axes= in_axes), in_axes= in_axes)
            self.axis = (0, 1)
            self.shape = (pmap_chains, vmap_chains)
            self.flatten = lambda x: x.reshape(self.num_chains, *x.shape[2:])
            
        else:
            self.axis = 0
            self.shape = self.num_chains
            self.flatten = lambda x: x
            
            if vmap_chains == 1:
                self.pvmap = lambda func, in_axes: jax.pmap(func, in_axes= in_axes)
            else:
                self.pvmap = lambda func, in_axes: jax.vmap(func, in_axes= in_axes)
            
        
            

def to_dict(x):
    return {'L': x[:, 0], 
            'stepsize': x[:, 1],
            'eevpd wanted': x[:, 2],
            'eevpd observed': x[:, 3],
            'equi full': x[:, 4],
            'equi diag': x[:, 5],
            'summary': x[:, 6:]
            }
    

def init(position, logdensity_fn, parallelization):
    """initialize the chains based on the equipartition of the initial condition.
       We initialize the velocity along grad log p if E_ii > 1 and along -grad log p if E_ii < 1.
    """
    
    def _init(x):
        """u = g / gnorm"""
        l, g = jax.value_and_grad(logdensity_fn)(x)
        flat_g, unravel_fn = ravel_pytree(g)
        u = unravel_fn(_normalized_flatten_array(flat_g)[0]) # = grad logp/ |grad logp|
        return u, l, g

    velocity, logdensity, logdensity_grad = parallelization.pvmap(_init)(position)
    
    # flip the velocity, depending on the equipartition condition
    to_sign = lambda equipartition: -2. * (equipartition < 1.) + 1.    
    velocity = jax.tree_util.tree_map(lambda x, u, g: to_sign(jnp.average(-x * g, axis= parallelization.axis))* u, position, velocity, logdensity_grad)
    
    return IntegratorState(position, velocity, logdensity, logdensity_grad)
    
    

def init_adap(num_steps, parallelization, delay_frac, position, alpha, d):
    
    delay_num = (int)(jnp.rint(delay_frac * num_steps))
    #flat_pytree, unravel_fn = ravel_pytree(sequential_pytree)
    
    #sigma = unravel_fn(jnp.ones(flat_pytree.shape, dtype = flat_pytree.dtype))
    
    step_size = 0.01 * jnp.sqrt(d)
    L = computeL(alpha, parallelization, position)
    hyp = Hyperparameters(L, step_size)
    
    history = jnp.inf * jnp.ones(delay_num)
    #history = jnp.concatenate((jnp.ones(1) * 1e50, jnp.ones(delay_num-1) * jnp.inf)) # loss history
    
    return AdaptationState(True, 0, 1e-3, 1e-3, history, hyp)


# if fullrank:
#     print('Warning: full rank is not implemented for pytrees yet.')

# def hutchinson_realization(position, logdensity_grad, rng_key):
#     #z ~ rademacher(shape of sequential x)
#     #v = z - ensemble average(z.g[m] x[m])

#     x, unravel_fn = ravel_pytree(position)
#     z_flat = unravel_fn(jax.random.rademacher(rng_key, shape=x.shape, dtype=x.dtype))
#     z = unravel_fn(z_flat)
    
#     jax.tree_util.tree_map(lambda z, g: z * g)



def equipartition_fullrank(position, logdensity_grad, rng_key, parallelization):
    """loss = Tr[(1 - E)^T (1 - E)] / d^2
        where Eij = <xi gj> is the equipartition patrix.
        Loss is computed with the Hutchinson's trick."""
    
    _position = parallelization.flatten(position)
    
    d = ravel_pytree(_position)[0].shape[0] // parallelization.chains # number of dimensions

    z = jax.random.rademacher(rng_key, (100, d)) # <z_i z_j> = delta_ij
    X = z + (logdensity_grad @ z.T).T @ _position / parallelization.chains
    return jnp.average(jnp.square(X)) / d

    
    # chains = len(position)
    # dim = pytree_size(position) // chains

    # z = jax.vmap(generate_rademacher)(jax.random.split(rng_key, 100), position) # <z_i z_j> = delta_ij
    
    # X = z - (g @ z.T).T @ position / chains
    # return jnp.average(jnp.square(X)) / dim


def equipartition_diagonal(position, logdensity_grad, rng_key, parallelization):
    """Ei = E_ensemble (- grad log p_i x_i ). Ei is 1 if we have converged. 
    virial_loss = average over parameters (Ei)"""
    E = jax.tree_util.tree_map(lambda x, g: jnp.square(1 + jnp.average(x * g, axis= parallelization.axis)), position, logdensity_grad)
    
    return jnp.average(ravel_pytree(E)[0])
    



def computeL(alpha, parallelization, position):
    
    x, unravel_fn = ravel_pytree(position)

    return alpha * jnp.sqrt(jnp.sum(jnp.square(x))/parallelization.num_chains) #average over the ensemble, sum over dimensions



def no_nans(a):
    flat_a, unravel_fn = ravel_pytree(a)
    return jnp.all(jnp.isfinite(flat_a))


def nan_reject(nonans, old, new):
    """Equivalent to
        return new if nonans else old"""
    
    # TODO: jnp.nan_to_num(new)?
    return jax.lax.cond(nonans, lambda _: new, lambda _: old, operand=None)



def build_kernel1(sequential_mclmc_kerel, max_iter, parallelization, fullrank, d, alpha = 1., C = 0.1):

    mclmc_kernel = parallelization.pvmap(sequential_mclmc_kerel, (0, 0, None, None))
    
    equi = equipartition_fullrank if fullrank else equipartition_diagonal
    
    
    def kernel(state_all):

        state, adap_state, rng_key = state_all
        hyp = adap_state.hyperparameters
        rng_key_new, key_kernel, key_hutchinson = jax.random.split(rng_key, 3)
        keys_kernel = jax.random.split(key_kernel, parallelization.shape)
        
        # apply one step of the dynamics
        _state, info = mclmc_kernel(keys_kernel, state, hyp.L, hyp.step_size)
        eevpd = jnp.average(jnp.square(info.energy_change)) / d
        
        
        # reject the new state if there were nans
        nonans = no_nans(_state.position)
        eevpd, state = nan_reject(nonans, (adap_state.eevpd, state), (eevpd, _state))

        
        # hyperparameter adaptation                                              
        bias = equi(state.position, state.logdensity_grad, key_hutchinson, parallelization) #estimate the bias from the equipartition loss

        eevpd_wanted = C * jnp.power(bias, 3./8.)
        
        eps_factor = jnp.power(eevpd_wanted / eevpd, 1./6.)
        eps_factor = nonans * eps_factor + (1-nonans) * 0.5
        eps_factor = jnp.clip(eps_factor, 0.3, 3.)
        
        L = computeL(alpha, parallelization, state.position)
        
        hyp = Hyperparameters(L, hyp.step_size * eps_factor)
        
        # determine if we want to finish this stage (= if loss is no longer decreassing)
        history = jnp.concatenate((jnp.ones(1) * bias, adap_state.history[:-1]))
        decreasing = (history[-1] > history[0]) or (adap_state.steps < adap_state.history.shape[0])
        #cond = decreasing and (adap_state.steps < max_iter)
        cond = (adap_state.steps < max_iter)
    
        return _state, AdaptationState(cond, adap_state.steps + 1, eevpd, eevpd_wanted, history, hyp), rng_key_new
    
    
    return kernel


def kernel_with_observables(kernel, observables, parallelization):


    def _kernel(_state_all):
        
        state, adap, key = _state_all
        key1, key = jax.random.split(key)
        state_all = kernel((state, adap, key))    
        
        state, adap, key = state_all
        hyp = adap.hyperparameters
        
        equi_full = equipartition_fullrank(state.position, state.logdensity_grad, key1, parallelization)
        equi_diag = equipartition_diagonal(state.position, state.logdensity_grad, key1, parallelization)
        
        new_info = jnp.concatenate((jnp.array([hyp.L, hyp.step_size, adap.eevpd_wanted, adap.eevpd, equi_full, equi_diag]), 
                                    observables(state.position)))

        return state_all, new_info
        

    return _kernel



def stage1(logdensity_fn, num_steps, parallelization, initial_position, rng_key, 
            delay_frac = 0.05,
            C = 0.1,
            alpha = 1.,
            fullrank = False,
            observables= jnp.square):
    """observable: function taking position x and outputing O(x)."""
    

    d = ravel_pytree(initial_position)[0].shape[0] // parallelization.chains # number of dimensions
    
    # kernel    
    sequential_kernel = mclmc.build_kernel(logdensity_fn= logdensity_fn, integrator= isokinetic_leapfrog)
    
    max_iter = 500#num_steps // 4
    
    kernel = build_kernel1(sequential_kernel, max_iter, parallelization, fullrank, d, alpha, C)

    # initialize 
    state = init(position=initial_position, logdensity_fn=logdensity_fn, parallelization= parallelization)
    adap_state = init_adap(num_steps, parallelization, delay_frac, state.position, alpha, d)

    state_all = (state, adap_state, rng_key)
    cond = lambda state_all: state_all[1].cond
    
    if observables != None:
        
        num_info = 6 + len(observables(initial_position))
        _info = jnp.empty(shape = (max_iter, num_info))
        
        kernel = kernel_with_observables(kernel, observables, parallelization)
        counter = 0
        while cond(state_all):
            state_all, new_info = kernel(state_all)
            _info = _info.at[counter].set(new_info)
            counter += 1

        info = to_dict(_info[:counter])
    
        return state_all, info
    
    
    else:

        state_all = jax.lax.while_loop(cond, kernel, state_all)
        return state_all
