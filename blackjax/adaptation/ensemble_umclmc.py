
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

from blackjax.mcmc.integrators import IntegratorState, isokinetic_velocity_verlet
from blackjax.types import Array, ArrayLike
from blackjax.util import pytree_size
from blackjax.mcmc import mclmc
from blackjax.mcmc.integrators import _normalized_flatten_array
from blackjax.util import ensemble_execute_fn



def no_nans(a):
    flat_a, unravel_fn = ravel_pytree(a)
    return jnp.all(jnp.isfinite(flat_a))


def nan_reject(nonans, old, new):
    """Equivalent to
        return new if nonans else old"""
        
    return jax.lax.cond(nonans, lambda _: new, lambda _: old, operand=None)



def build_kernel(logdensity_fn):
    """MCLMC kernel (with nan rejection)"""
    
    kernel = mclmc.build_kernel(logdensity_fn= logdensity_fn, integrator= isokinetic_velocity_verlet)
    
    
    def sequential_kernel(key, state, adap):
        
        new_state, info = kernel(key, state, adap.L, adap.step_size)
        
        # reject the new state if there were nans
        nonans = no_nans(new_state)
        new_state = nan_reject(nonans, state, new_state)
        
        return new_state, {'nans': 1-nonans, 'energy_change': info.energy_change * nonans, 'logdensity': info.logdensity * nonans}

    
    return sequential_kernel


    
def initialize(rng_key, logdensity_fn, sample_init, num_chains, mesh):
    """initialize the chains based on the equipartition of the initial condition.
       We initialize the velocity along grad log p if E_ii > 1 and along -grad log p if E_ii < 1.
    """
    
    def sequential_init(key, x, args):
        """initialize the position using sample_init and the velocity along the gradient"""
        position = sample_init(key)
        logdensity, logdensity_grad = jax.value_and_grad(logdensity_fn)(position)
        flat_g, unravel_fn = ravel_pytree(logdensity_grad)
        velocity = unravel_fn(_normalized_flatten_array(flat_g)[0]) # = grad logp/ |grad logp|
        return IntegratorState(position, velocity, logdensity, logdensity_grad), None
    
    def summary_statistics_fn(state):
        """compute the diagonal elements of the equipartition matrix"""
        return -state.position * state.logdensity_grad
    
    def ensemble_init(key, state, signs):
        """flip the velocity, depending on the equipartition condition"""
        velocity = jax.tree_util.tree_map(lambda sign, u: sign * u, signs, state.momentum)
        return IntegratorState(state.position, velocity, state.logdensity, state.logdensity_grad), None

    key1, key2= jax.random.split(rng_key)    
    initial_state, equipartition = ensemble_execute_fn(sequential_init, key1, num_chains, mesh, summary_statistics_fn= summary_statistics_fn)
    signs = -2. * (equipartition < 1.) + 1.    
    initial_state, _ = ensemble_execute_fn(ensemble_init, key2, num_chains, mesh, x= initial_state, args= signs)
    
    return initial_state
    
    
def update_history(new_vals, history):
    return jnp.concatenate((new_vals[None, :], history[:-1]))

def update_history_scalar(new_val, history):
    return jnp.concatenate((new_val * jnp.ones(1), history[:-1]))

def contract_history(theta, weights):
  
    square_average = jnp.square(jnp.average(theta, weights= weights, axis= 0))
    average_square = jnp.average(jnp.square(theta), weights= weights, axis= 0)
      
    r = (average_square - square_average) / square_average
    
    return jnp.array([jnp.max(r), 
                      jnp.average(r)
                      ])


class History(NamedTuple):
    observables: Array
    stopping: Array
    weights: Array


class AdaptationState(NamedTuple):
    
    L: float
    inverse_mass_matrix: Any
    step_size: float
    
    step_count: int
    EEVPD: float
    EEVPD_wanted: float
    history: Any    
    

def equipartition_diagonal(state):
    """Ei = E_ensemble (- grad log p_i x_i ). Ei is 1 if we have converged. 
    equipartition_loss = average over parameters (Ei)"""
    return jax.tree_util.tree_map(lambda x, g: -x * g, state.position, state.logdensity_grad)



def equipartition_fullrank(state, rng_key):
    """loss = Tr[(1 - E)^T (1 - E)] / d^2
        where Eij = <xi gj> is the equipartition patrix.
        Loss is computed with the Hutchinson's trick."""

    x, unravel_fn = ravel_pytree(state.position)
    g, unravel_fn = ravel_pytree(state.logdensity_grad)
    d = len(x)
    
    def func(z):
        """z here has the same shape as position"""
        return (z + jnp.dot(z, g) * x)

    z = jax.random.rademacher(rng_key, (100, d)) # <z_i z_j> = delta_ij
    return jax.vmap(func)(z)


def equipartition_diagonal_loss(Eii):
    Eii_flat, unravel_fn = ravel_pytree(Eii)
    return jnp.average(jnp.square(1.- Eii_flat))


def equipartition_fullrank_loss(delta_z):
    d = delta_z.shape[-1]
    return jnp.average(jnp.square(delta_z)) / d


class Adaptation:
    
    def __init__(self, num_dims,
                 alpha= 1., C= 0.1, power = 3./8., r_end= 0.01,
                 bias_type= 0, save_num = 10,
                 observables= lambda x: 0., observables_for_bias= lambda x: 0., contract= lambda x: 0.
                 ):

        self.num_dims = num_dims
        self.alpha = alpha
        self.C = C
        self.power = power
        self.r_end = r_end
        self.observables = observables
        self.observables_for_bias = observables_for_bias 
        self.contract = contract
        self.bias_type = bias_type
        self.save_num = save_num
        #sigma = unravel_fn(jnp.ones(flat_pytree.shape, dtype = flat_pytree.dtype))
        
        r_save_num = save_num
        
        history = History(observables= jnp.zeros((r_save_num, num_dims)), 
                          stopping= jnp.full((save_num,), jnp.nan), 
                          weights= jnp.zeros(r_save_num))

        self.initial_state = AdaptationState(L= jnp.inf, # do not add noise for the first step
                                             inverse_mass_matrix= jnp.ones(num_dims),
                                             step_size= 0.01 * jnp.sqrt(num_dims),
                                             step_count= 0, 
                                             EEVPD=1e-3, EEVPD_wanted=1e-3,
                                             history=history)
        
        
    def summary_statistics_fn(self, state, info, rng_key):
    
        position_flat, unravel_fn = ravel_pytree(state.position)
        
        return {'equipartition_diagonal': equipartition_diagonal(state), 
                'equipartition_fullrank': equipartition_fullrank(state, rng_key), 
                'x': position_flat, 'xsq': jnp.square(position_flat), 
                'E': info['energy_change'], 'Esq': jnp.square(info['energy_change']),
                'rejection_rate_nans': info['nans'],
                'observables_for_bias': self.observables_for_bias(state.position),
                'observables': self.observables(state.position),
                'entropy': - info['logdensity']
                }
        
    
    def update(self, adaptation_state, Etheta):
        
        # combine the expectation values to get useful scalars
        equi_diag = equipartition_diagonal_loss(Etheta['equipartition_diagonal'])
        equi_full = equipartition_fullrank_loss(Etheta['equipartition_fullrank'])
        
        history_observables = update_history(Etheta['observables_for_bias'], adaptation_state.history.observables)        
        history_weights = update_history_scalar(1., adaptation_state.history.weights)
        fluctuations = contract_history(history_observables, history_weights)
        history_stopping = update_history_scalar(jax.lax.cond(adaptation_state.step_count > len(history_weights), lambda _: fluctuations[0], lambda _: jnp.nan, operand=None), 
                                                 adaptation_state.history.stopping)
        history = History(history_observables, history_stopping, history_weights)
        
        L = self.alpha * jnp.sqrt(jnp.sum(Etheta['xsq'] - jnp.square(Etheta['x']))) # average over the ensemble, sum over parameters (to get sqrt(d))
        inverse_mass_matrix = Etheta['xsq'] - jnp.square(Etheta['x'])
        EEVPD = (Etheta['Esq'] - jnp.square(Etheta['E'])) / self.num_dims        
        true_bias = self.contract(Etheta['observables_for_bias'])
        nans = (Etheta['rejection_rate_nans'] > 0.) #| (~jnp.isfinite(eps_factor))

        # hyperparameter adaptation
        # estimate bias
        bias = jnp.array([fluctuations[0], fluctuations[1], equi_full, equi_diag])[self.bias_type] # r_max, r_avg, equi_full, equi_diag
        EEVPD_wanted = self.C * jnp.power(bias, self.power)
        

        eps_factor = jnp.power(EEVPD_wanted / EEVPD, 1./6.)
        eps_factor = jnp.clip(eps_factor, 0.3, 3.)
        
        eps_factor = nan_reject(1-nans, 0.5, eps_factor) # reduce the stepsize if there were nans

        # determine if we want to finish this stage (i.e. if loss is no longer decreassing)
        #increasing = history.stopping[0] > history.stopping[-1] # will be false if some elements of history are still nan (have not been filled yet). Do not be tempted to simply change to while_cond = history[0] < history[-1]
        #while_cond = ~increasing
        while_cond = (fluctuations[0] > self.r_end) | (adaptation_state.step_count < self.save_num)
        
        info_to_be_stored = {'L': adaptation_state.L, 'step_size': adaptation_state.step_size, 
                             'EEVPD_wanted': EEVPD_wanted, 'EEVPD': EEVPD, 
                             'equi_diag': equi_diag, 'equi_full': equi_full, 'bias': true_bias,
                             'r_max': fluctuations[0], 'r_avg': fluctuations[1],
                             'while_cond': while_cond, 'entropy': Etheta['entropy'],
                             'observables': Etheta['observables']}
    
        adaptation_state_new = AdaptationState(L, 
                                               inverse_mass_matrix,
                                               adaptation_state.step_size * eps_factor, 
                                               adaptation_state.step_count + 1, 
                                               EEVPD, 
                                               EEVPD_wanted,
                                               history)
        
        return adaptation_state_new, info_to_be_stored
    
