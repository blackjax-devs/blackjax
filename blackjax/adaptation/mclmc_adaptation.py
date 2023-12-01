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
"""Algorithms to adapt the MCLMC kernel parameters, namely step size and L.

"""

from typing import NamedTuple
import warnings

from chex import PRNGKey
import jax
import jax.numpy as jnp
from scipy.fftpack import next_fast_len
from blackjax.diagnostics import effective_sample_size #type: ignore

from blackjax.mcmc.integrators import IntegratorState, noneuclidean_mclachlan
from blackjax.mcmc.mclmc import build_kernel, init
import jax
import jax.numpy as jnp
from typing import NamedTuple


from typing import NamedTuple

from blackjax.util import pytree_size

class MCLMCAdaptationState(NamedTuple):
    """Represents the tunable parameters for MCLMC adaptation.

    Attributes:
        L (float): The momentum decoherent rate for the MCLMC algorithm.
        step_size (float): The step size used for the MCLMC algorithm.
    """
    
    L: float
    step_size: float

def mclmc_find_L_and_step_size(kernel, num_steps, state, frac_tune1=0.1,
    frac_tune2=0.1,
    frac_tune3=0.1):
    """
    Finds the optimal value of L (step size) for the MCLMC algorithm.

    Args:
        kernel: The kernel function used for the MCMC algorithm.
        num_steps: The number of MCMC steps that will subsequently be run, after tuning
        state: The initial state of the MCMC algorithm.
        frac_tune1: The fraction of tuning for the first step of the adaptation.
        frac_tune2: The fraction of tuning for the second step of the adaptation.
        frac_tune3: The fraction of tuning for the third step of the adaptation.

    Returns:
        state: The final state of the MCMC algorithm.
        params: The final hyperparameters of the MCMC algorithm.
    """
    dim = state.position.shape[0]
    params = MCLMCAdaptationState(jnp.sqrt(dim), 
                                        jnp.sqrt(dim) * 0.25, 
                                        )
    varEwanted = 5e-4

    
    state, params = make_L_step_size_adaptation(kernel, dim, jnp.array([frac_tune1, frac_tune2]), varEwanted, 1.5, 150)(state, params, num_steps)

    if frac_tune3 != 0:

        state, params = make_adaptation_L(kernel, frac=frac_tune3, Lfactor=0.4)(state,params, num_steps)
        

        
    return state, params



def make_L_step_size_adaptation(kernel, d, frac, 
           varEwanted = 1e-3, sigma_xi = 1.5, neff = 150):
    """Adapts the stepsize and L of the MCLMC kernel. Designed for the unadjusted MCLMC"""

    gamma_forget = (neff - 1.0) / (neff + 1.0)
    
    
    def predictor(state_old, state, adaptive_state):
        """does one step with the dynamics and updates the prediction for the optimal stepsize
            Designed for the unadjusted MCHMC"""
        
        W, F, step_size_max = adaptive_state

        # stateamics
        # state_new, energy_change = stateamics(state_old, state)
        state_new, info = kernel(rng_key = jax.random.PRNGKey(0), state=state_old, L=state.L, step_size=state.step_size)
        energy_change = info.dE
        # step updating
        success, x, u, l, g, step_size_max, energy_change = handle_nans(state_old.position, state_old.momentum, state_old.logdensity, state_old.logdensity_grad, 
                                                                      state_new.position, state_new.momentum, state_new.logdensity, state_new.logdensity_grad, 
                                                                      state.step_size, step_size_max, energy_change)

        
        # Warning: var = 0 if there were nans, but we will give it a very small weight
        xi = (jnp.square(energy_change) / (d * varEwanted)) + 1e-8  # 1e-8 is added to avoid divergences in log xi
        w = jnp.exp(-0.5 * jnp.square(jnp.log(xi) / (6.0 * sigma_xi)))  # the weight reduces the impact of stepsizes which are much larger on much smaller than the desired one.

        F = gamma_forget * F + w * (xi/jnp.power(state.step_size, 6.0))
        W = gamma_forget * W + w
        step_size = jnp.power(F/W, -1.0/6.0) #We use the Var[E] = O(eps^6) relation here.
        step_size = (step_size < step_size_max) * step_size + (step_size > step_size_max) * step_size_max  # if the proposed stepsize is above the stepsize where we have seen divergences
        state_new = MCLMCAdaptationState(state.L, step_size)
        
        return IntegratorState(x, u, l, g), state_new, state_new, (W, F, step_size_max), success


    def update_kalman(x, state, outer_weight, success, step_size):
        """kalman filter to estimate the size of the posterior"""
        W, F1, F2 = state
        w = outer_weight * step_size * success
        zero_prevention = 1-outer_weight
        F1 = (W*F1 + w*x) / (W + w + zero_prevention)  # Update <f(x)> with a Kalman filter
        F2 = (W*F2 + w*jnp.square(x)) / (W + w + zero_prevention)  # Update <f(x)> with a Kalman filter
        W += w
        return (W, F1, F2)


    adap0 = (0., 0., jnp.inf)
    _step = predictor
        
        
    def step(state, outer_weight):
        """does one step of the dynamics and updates the estimate of the posterior size and optimal stepsize"""
        state, params, _, adaptive_state, kalman_state = state
        state, params, params_final, adaptive_state, success = _step(state, params, adaptive_state)
        kalman_state = update_kalman(state.position, kalman_state, outer_weight, success, params.step_size)

        return (state, params, params_final, adaptive_state, kalman_state), None


    def L_step_size_adaptation(state, params, num_steps):
        
        num_steps1, num_steps2 = jnp.rint(num_steps * frac).astype(int)
            
        # we use the last num_steps2 to compute the diagonal preconditioner
        outer_weights = jnp.concatenate((jnp.zeros(num_steps1), jnp.ones(num_steps2)))

        #initial state
        
        kalman_state = (0., jnp.zeros(d), jnp.zeros(d))

        # run the steps
        kalman_state = jax.lax.scan(step, init= (state, params, params, adap0, kalman_state), xs= outer_weights, length= num_steps1 + num_steps2)[0]
        state, _, params, _, kalman_state_output = kalman_state
        
        L = params.L
        # determine L
        if num_steps2 != 0.:
            _, F1, F2 = kalman_state_output
            variances = F2 - jnp.square(F1)
            L = jnp.sqrt(jnp.sum(variances))


        return state, MCLMCAdaptationState(L, params.step_size)

    return L_step_size_adaptation




def make_adaptation_L(kernel, frac, Lfactor):
    """determine L by the autocorrelations (around 10 effective samples are needed for this to be accurate)"""
    

    def sample_full(num_steps, state, params):

        def step(state, _):
            state, _ = kernel(rng_key=jax.random.PRNGKey(0), state=state, L=params.L, step_size=params.step_size)
            
            return state, state.position

        return jax.lax.scan(step, init=state, xs=None, length=num_steps)


    def adaptation_L(state, params, num_steps):
        steps = jnp.rint(num_steps * frac).astype(int)
        
        state, samples = sample_full(steps, state, params)
        num_samples = samples.shape[0]
        ESS = 0.5 * effective_sample_size(jnp.array([samples, samples]))
        ess_per_sample = ESS / num_samples
        ESS =  1.0 / jnp.average(1 / ess_per_sample)

        Lnew = Lfactor * params.step_size / ESS # = 0.4 * length corresponding to one effective sample

        return state, MCLMCAdaptationState(Lnew, params.step_size)


    return adaptation_L


def handle_nans(x, u, l, g, xx, uu, ll, gg, step_size, step_size_max, dK):
    """if there are nans, let's reduce the stepsize, and not update the state. The function returns the old state in this case."""
    
    nonans = jnp.all(jnp.isfinite(xx))
    x, u, l, g, step_size, dk = jax.tree_util.tree_map(lambda new, old: jax.lax.select(nonans, jnp.nan_to_num(new), old), 
                                                       (xx, uu, ll, gg, step_size_max, dK), 
                                                       (x, u, l, g, step_size * 0.8, 0.))
    
    return nonans, x, u, l, g, step_size, dk
    
 