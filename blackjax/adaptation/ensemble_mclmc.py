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

from typing import Callable, NamedTuple, Any

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
from blackjax.mcmc.integrators import IntegratorState, isokinetic_mclachlan, isokinetic_leapfrog, _normalized_flatten_array, with_isokinetic_maruyama
from blackjax.mcmc import mhmclmc
import blackjax.adaptation.ensemble_umclmc as umclmc

from blackjax.adaptation.step_size import dual_averaging_adaptation



class AdaptationState(NamedTuple):
    
    steps_per_sample: float
    Lpartial: float
    step_size: float
    
    da_state: Any
    

def to_dict(x):
    return {
            'steps_per_sample': x[..., 0],
            'stepsize': x[..., 1],
            'acc prob': x[..., 2],
            'equi full': x[..., 3],
            'equi diag': x[..., 4],
            'summary': x[..., 5:]
        }



def init(num_steps, steps_per_sample, adap_state, d, mclachlan):
    
    Lfull, step_size = adap_state.hyperparameters.L, adap_state.hyperparameters.step_size
    
    if mclachlan:
        integrator = isokinetic_mclachlan 
        grads_per_step = 2
        step_size *= jnp.sqrt(10.) #if we switched to the more accurate integrator we can use longer step size
        
    else:
        integrator = isokinetic_leapfrog
        grads_per_step = 1
        
    # Let's use the stepsize which will be optimal for the adjusted method. The energy variance after N steps scales as sigma^2 ~ N^2 eps^6 = eps^4 L^2
    # In the adjusted method we want sigma^2 = 2 mu = 2 * 0.41 = 0.82
    # With the current eps, we had sigma^2 = eevpd * d for N = 1. 
    # Combining the two we have eevpd * d / 0.82 = eps^6 / eps_new^4 L^2
    step_size *= jnp.power(0.82 / (d * adap_state.eevpd), 0.25) / jnp.sqrt(steps_per_sample)
    Lfull = steps_per_sample * step_size
    Lpartial = Lfull * 1.25
    #steps_left = (num_steps - adap_state.steps) // grads_per_step
    #steps_per_sample = (int)(jnp.max(jnp.array([Lfull / step_size, 1])))
    num_samples = (num_steps - adap_state.steps) // (grads_per_step * steps_per_sample)
    
    return integrator, num_samples, steps_per_sample, step_size, Lpartial



def stage2(logdensity_fn, integrator, num_samples, parallelization, 
           init_state, rng_key, 
           steps_per_sample, Lpartial, step_size,
           acc_prob_target= 0.7,
           observables= jnp.square):
    """observables: function taking position x and outputing O(x), can be vector valued."""
    
    
    da_init, da_update, da_final = dual_averaging_adaptation(target= acc_prob_target)
    da_state = da_init(step_size)
    adap = AdaptationState(steps_per_sample, Lpartial, step_size, da_state)
    
    def step(state_all, key):
        state, adap = state_all
        key1, key2 = jax.random.split(key)

        # dynamics
        sequential_kernel = mhmclmc.mhmclmc_proposal(with_isokinetic_maruyama(integrator(logdensity_fn)),
                                                     adap.step_size, adap.Lpartial, steps_per_sample)
        kernel = parallelization.pvmap(sequential_kernel, (0, 0))
        state, info, _ = kernel(jax.random.split(key1, parallelization.shape), state)

        # change the stepsize
        acc_prob = jnp.average(info.acceptance_rate)
        da_state = da_update(adap.da_state, acc_prob)
        step_size = jnp.exp(da_state.log_step_size)
    
        # additional information
        
        equi_full = umclmc.equipartition_fullrank(state.position, state.logdensity_grad, key2, parallelization)
        equi_diag = umclmc.equipartition_diagonal(state.position, state.logdensity_grad, key2, parallelization)
        new_info = jnp.concatenate((jnp.array([steps_per_sample, step_size, acc_prob, equi_full, equi_diag]), 
                                    observables(state.position)))
        
        return (state, AdaptationState(steps_per_sample, Lpartial, step_size, da_state)), new_info
   
    keys = jax.random.split(rng_key, num_samples)
    final_state, info = jax.lax.scan(step, init= (init_state, adap), xs = keys)

    #step_size = da_final(da_state)
    
    return final_state, to_dict(info)
    
    

def algorithm(logdensity_fn, num_steps, parallelization, 
              initial_position, rng_key,
              num_steps_per_sample, 
              mclachlan = True,
              observables= jnp.square):
    """chain_shape: an integer or tuple with two integers. If it is a tuple, """
    
    d = ravel_pytree(initial_position)[0].shape[0] // parallelization.num_chains # number of dimensions

    # burn-in with the unadjusted method    def stage1(logdensity_fn, num_steps, chains, initial_position, rng_key, 

    state_all, info1 = umclmc.stage1(logdensity_fn, num_steps, parallelization, initial_position, rng_key, observables= observables)

    state, adap_state, key = state_all
    
    # readjust the hyperparameters
    hyp = adap_state.hyperparameters
    adap_state = adap_state._replace(hyperparameters= hyp._replace(L= hyp.step_size * num_steps_per_sample))
    
    integrator, num_samples, steps_per_sample, step_size, Lpartial = init(num_steps, num_steps_per_sample, adap_state, d, mclachlan)

    # refine the results with the adjusted method
    state_final, info2 = stage2(logdensity_fn, integrator, num_samples, parallelization, 
                                state, key, 
                                steps_per_sample, Lpartial, step_size,
                                observables= observables)
    
    return info1, info2
    
    
