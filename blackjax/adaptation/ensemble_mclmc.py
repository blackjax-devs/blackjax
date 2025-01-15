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

from blackjax.util import run_eca
from blackjax.mcmc.integrators import generate_isokinetic_integrator, velocity_verlet_coefficients, mclachlan_coefficients, omelyan_coefficients
from blackjax.mcmc.hmc import HMCState
from blackjax.mcmc.adjusted_mclmc import build_kernel as build_kernel_malt
import blackjax.adaptation.ensemble_umclmc as umclmc
from blackjax.adaptation.ensemble_umclmc import equipartition_diagonal, equipartition_diagonal_loss, equipartition_fullrank, equipartition_fullrank_loss

from blackjax.adaptation.step_size import dual_averaging_adaptation, bisection_monotonic_fn

    

class AdaptationState(NamedTuple):
    steps_per_sample: float
    step_size: float
    epsadap_state: Any
    sample_count: int
    


def build_kernel(logdensity_fn, integrator, sqrt_diag_cov):
    """MCLMC kernel"""
    
    kernel = build_kernel_malt(logdensity_fn, integrator, sqrt_diag_cov= sqrt_diag_cov, L_proposal_factor = 1.25)
    
    def sequential_kernel(key, state, adap):
        return kernel(key, state, step_size= adap.step_size, num_integration_steps= adap.steps_per_sample)
    
    return sequential_kernel



class Adaptation:
    
    def __init__(self, adap_state, num_adaptation_samples,
                 steps_per_sample, acc_prob_target= 0.8,
                 observables = lambda x: 0.,
                 observables_for_bias = lambda x: 0., contract= lambda x: 0.):
        
        self.num_adaptation_samples= num_adaptation_samples
        self.observables = observables
        self.observables_for_bias = observables_for_bias
        self.contract = contract
        
        ### Determine the initial hyperparameters ###
        
        ## stepsize ##
        #if we switched to the more accurate integrator we can use longer step size
        #integrator_factor = jnp.sqrt(10.) if mclachlan else 1. 
        # Let's use the stepsize which will be optimal for the adjusted method. The energy variance after N steps scales as sigma^2 ~ N^2 eps^6 = eps^4 L^2
        # In the adjusted method we want sigma^2 = 2 mu = 2 * 0.41 = 0.82
        # With the current eps, we had sigma^2 = EEVPD * d for N = 1. 
        # Combining the two we have EEVPD * d / 0.82 = eps^6 / eps_new^4 L^2
        #adjustment_factor = jnp.power(0.82 / (num_dims * adap_state.EEVPD), 0.25) / jnp.sqrt(steps_per_sample)
        step_size = adap_state.step_size #* integrator_factor * adjustment_factor

        #steps_per_sample = (int)(jnp.max(jnp.array([Lfull / step_size, 1])))
                
        ### Initialize the dual averaging adaptation ###
        #da_init_fn, self.epsadap_update, _ = dual_averaging_adaptation(target= acc_prob_target)
        #epsadap_state = da_init_fn(step_size)
        
        ### Initialize the bisection for finding the step size
        epsadap_state, self.epsadap_update = bisection_monotonic_fn(acc_prob_target)
        
        self.initial_state = AdaptationState(steps_per_sample, step_size, epsadap_state, 0)
        
        
    def summary_statistics_fn(self, state, info, rng_key):
     
        return {'acceptance_probability': info.acceptance_rate,
                #'inv_acceptance_probability': 1./info.acceptance_rate,
                'equipartition_diagonal': equipartition_diagonal(state), 
                'equipartition_fullrank': equipartition_fullrank(state, rng_key), 
                'observables': self.observables(state.position),
                'observables_for_bias': self.observables_for_bias(state.position)
                }
        

    def update(self, adaptation_state, Etheta):
        
        # combine the expectation values to get useful scalars
        acc_prob = Etheta['acceptance_probability']
        #acc_prob = 1./Etheta['inv_acceptance_probability']
        equi_diag = equipartition_diagonal_loss(Etheta['equipartition_diagonal'])
        equi_full = equipartition_fullrank_loss(Etheta['equipartition_fullrank'])
        true_bias = self.contract(Etheta['observables_for_bias'])
        
        
        info_to_be_stored = {'L': adaptation_state.step_size * adaptation_state.steps_per_sample, 'steps_per_sample': adaptation_state.steps_per_sample, 'step_size': adaptation_state.step_size, 
                             'acc_prob': acc_prob,
                             'equi_diag': equi_diag, 'equi_full': equi_full, 'bias': true_bias,
                             'observables': Etheta['observables']
                             }

        # hyperparameter adaptation
        
        # Dual Averaging
        # adaptation_phase = adaptation_state.sample_count < self.num_adaptation_samples  
        
        # def update(_):
        #     da_state = self.epsadap_update(adaptation_state.epsadap_state, acc_prob)
        #     step_size = jnp.exp(da_state.log_step_size)
        #     return da_state, step_size
        
        # def dont_update(_):
        #     da_state = adaptation_state.epsadap_state
        #     return da_state, jnp.exp(da_state.log_step_size_avg)
        
        # epsadap_state, step_size = jax.lax.cond(adaptation_phase, update, dont_update, operand=None)
        
        # Bisection        
        epsadap_state, step_size = self.epsadap_update(adaptation_state.epsadap_state, adaptation_state.step_size, acc_prob)
        
        return AdaptationState(adaptation_state.steps_per_sample, step_size, epsadap_state, adaptation_state.sample_count + 1), info_to_be_stored



def bias(model):
    """should be transfered to benchmarks/"""
    
    def observables(position):
        return jnp.square(model.transform(position))
    
    def contract(sampler_E_x2):
        bsq = jnp.square(sampler_E_x2 - model.E_x2) / model.Var_x2
        return jnp.array([jnp.max(bsq), jnp.average(bsq)])
    
    return observables, contract



def while_steps_num(cond):
    if jnp.all(cond):
        return len(cond)
    else:
        return jnp.argmin(cond) + 1


def emaus(model, num_steps1, num_steps2, num_chains, mesh, rng_key,
          alpha= 1.9, bias_type= 0, save_frac= 0.2, C= 0.1, power= 3./8., early_stop= True, r_end= 5e-3,# stage1 parameters
          diagonal_preconditioning= True, integrator_coefficients= None, steps_per_sample= 10, acc_prob= None, 
          observables = lambda x: None, 
          ensemble_observables= None
          ):
    
    observables_for_bias, contract = bias(model)
    key_init, key_umclmc, key_mclmc = jax.random.split(rng_key, 3)
    
    # initialize the chains
    initial_state = umclmc.initialize(key_init, model.logdensity_fn, model.sample_init, num_chains, mesh)
    
    ### burn-in with the unadjusted method ###
    kernel = umclmc.build_kernel(model.logdensity_fn)
    save_num= (int)(jnp.rint(save_frac * num_steps1))
    adap = umclmc.Adaptation(model.ndims, alpha= alpha, bias_type= bias_type, save_num= save_num, C=C, power= power, r_end = r_end,
                             observables= observables, observables_for_bias= observables_for_bias, contract= contract)
    final_state, final_adaptation_state, info1 = run_eca(key_umclmc, initial_state, kernel, adap, num_steps1, num_chains, mesh, ensemble_observables)
    
    if early_stop: # here I am cheating a bit, because I am not sure if it is possible to do a while loop in jax and save something at every step. Therefore I rerun burn-in with exactly the same parameters and stop at the point where the orignal while loop would have stopped. The release implementation should not have that.
        
        num_steps_while = while_steps_num((info1[0] if ensemble_observables != None else info1)['while_cond'])
        #print(num_steps_while, save_num)
        final_state, final_adaptation_state, info1 = run_eca(key_umclmc, initial_state, kernel, adap, num_steps_while, num_chains, mesh, ensemble_observables)

    ### refine the results with the adjusted method ###
    _acc_prob = acc_prob
    if integrator_coefficients == None:
        high_dims = model.ndims > 200
        _integrator_coefficients = omelyan_coefficients if high_dims else mclachlan_coefficients
        if acc_prob == None:
            _acc_prob = 0.9 if high_dims else 0.7
        
    else:
        _integrator_coefficients = integrator_coefficients
        if acc_prob == None:
            _acc_prob = 0.9
            
        
    integrator = generate_isokinetic_integrator(_integrator_coefficients)
    gradient_calls_per_step= len(_integrator_coefficients) // 2 #scheme = BABAB..AB scheme has len(scheme)//2 + 1 Bs. The last doesn't count because that gradient can be reused in the next step.

    if diagonal_preconditioning:
        sqrt_diag_cov= final_adaptation_state.sqrt_diag_cov
        
        # scale the stepsize so that it reflects averag scale change of the preconditioning
        average_scale_change = jnp.sqrt(jnp.average(jnp.square(sqrt_diag_cov)))
        final_adaptation_state = final_adaptation_state._replace(step_size= final_adaptation_state.step_size / average_scale_change)

    else:
        sqrt_diag_cov= 1.
    
    kernel = build_kernel(model.logdensity_fn, integrator, sqrt_diag_cov= sqrt_diag_cov)
    initial_state= HMCState(final_state.position, final_state.logdensity, final_state.logdensity_grad)
    num_samples = num_steps2 // (gradient_calls_per_step * steps_per_sample)
    num_adaptation_samples = num_samples//2 # number of samples after which the stepsize is fixed.
    
    adap = Adaptation(final_adaptation_state, num_adaptation_samples, steps_per_sample, _acc_prob, 
                      observables= observables, observables_for_bias= observables_for_bias, contract= contract)
    
    final_state, final_adaptation_state, info2 = run_eca(key_mclmc, initial_state, kernel, adap, num_samples, num_chains, mesh, ensemble_observables)
    
    return info1, info2, gradient_calls_per_step, _acc_prob
    
    