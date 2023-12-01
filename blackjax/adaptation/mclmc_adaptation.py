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


class MCLMCAdaptationState(NamedTuple):
    """Tunable parameters for MCLMC"""

    L: float
    step_size: float

def ess_corr(x):
    num_samples = x.shape[0]
    ess = 0.5 * effective_sample_size(jnp.array([x, x]))
    ess_per_sample = ess / num_samples
    return 1.0 / jnp.average(1 / ess_per_sample)



def mclmc_find_L_and_step_size(kernel, num_steps, initial_state, frac_tune1 = 0.1,
    frac_tune2 = 0.1,
    frac_tune3 = 0.1):

    dim = initial_state.position.shape[0]
    dyn = initial_state
    hyp = MCLMCAdaptationState(jnp.sqrt(dim), 
                                        jnp.sqrt(dim) * 0.25, 
                                        )

    
    varEwanted = 5e-4
    tune12p = tune12(kernel, dim, False, jnp.array([frac_tune1, frac_tune2]), varEwanted, 1.5, 150)

    tune3p = tune3(kernel, frac_tune3, 0.4)

    if frac_tune3 != 0.:
        tune3p = tune3(kernel, frac= frac_tune3, Lfactor= 0.4)
        schedule = [tune12p, tune3p]
    else:
        schedule = [tune12p, ]

    dyn, hyp = run(dyn, hyp, schedule, num_steps)
    return dyn, hyp

def run(dyn, hyp, schedule, num_steps):
    
    _dyn, _hyp = dyn, hyp
    
    for program in schedule:
        _dyn, _hyp = program(_dyn, _hyp, num_steps)
        
    return _dyn, _hyp
 
 

 
def nan_reject(x, u, l, g, xx, uu, ll, gg, eps, eps_max, dK):
    """if there are nans, let's reduce the stepsize, and not update the state. The function returns the old state in this case."""
    
    nonans = jnp.all(jnp.isfinite(xx))
    _x, _u, _l, _g, _eps, _dk = jax.tree_util.tree_map(lambda new, old: jax.lax.select(nonans, jnp.nan_to_num(new), old), 
                                                       (xx, uu, ll, gg, eps_max, dK), 
                                                       (x, u, l, g, eps * 0.8, 0.))
    
    return nonans, _x, _u, _l, _g, _eps, _dk
    
 


def tune12(kernel, d,
           diag_precond, frac, 
           varEwanted = 1e-3, sigma_xi = 1.5, neff = 150):
    
    print("Starting tune12")
           
    gamma_forget = (neff - 1.0) / (neff + 1.0)
    
    
    def predictor(dyn_old, hyp, adaptive_state):
        """does one step with the dynamics and updates the prediction for the optimal stepsize
            Designed for the unadjusted MCHMC"""
        
        W, F, eps_max = adaptive_state

        # dynamics
        # dyn_new, energy_change = dynamics(dyn_old, hyp)
        dyn_new, info = kernel(rng_key = jax.random.PRNGKey(0), state=dyn_old, L=hyp.L, step_size=hyp.step_size)
        energy_change = info.dE
        # step updating
        success, x, u, l, g, eps_max, energy_change = nan_reject(dyn_old.position, dyn_old.momentum, dyn_old.logdensity, dyn_old.logdensity_grad, 
                                                                      dyn_new.position, dyn_new.momentum, dyn_new.logdensity, dyn_new.logdensity_grad, 
                                                                      hyp.step_size, eps_max, energy_change)

        dyn = IntegratorState(x, u, l, g)
        
        # Warning: var = 0 if there were nans, but we will give it a very small weight
        xi = (jnp.square(energy_change) / (d * varEwanted)) + 1e-8  # 1e-8 is added to avoid divergences in log xi
        w = jnp.exp(-0.5 * jnp.square(jnp.log(xi) / (6.0 * sigma_xi)))  # the weight reduces the impact of stepsizes which are much larger on much smaller than the desired one.

        F = gamma_forget * F + w * (xi/jnp.power(hyp.step_size, 6.0))
        W = gamma_forget * W + w
        eps = jnp.power(F/W, -1.0/6.0) #We use the Var[E] = O(eps^6) relation here.
        eps = (eps < eps_max) * eps + (eps > eps_max) * eps_max  # if the proposed stepsize is above the stepsize where we have seen divergences
        hyp_new = MCLMCAdaptationState(hyp.L, eps)
        
        return dyn, hyp_new, hyp_new, (W, F, eps_max), success


    def update_kalman(x, state, outer_weight, success, eps):
        """kalman filter to estimate the size of the posterior"""
        W, F1, F2 = state
        w = outer_weight * eps * success
        zero_prevention = 1-outer_weight
        F1 = (W*F1 + w*x) / (W + w + zero_prevention)  # Update <f(x)> with a Kalman filter
        F2 = (W*F2 + w*jnp.square(x)) / (W + w + zero_prevention)  # Update <f(x)> with a Kalman filter
        W += w
        return (W, F1, F2)


    adap0 = (0., 0., jnp.inf)
    _step = predictor
        
        
    def step(state, outer_weight):
        """does one step of the dynamcis and updates the estimate of the posterior size and optimal stepsize"""
        dyn, hyp, _, adaptive_state, kalman_state = state
        dyn, hyp, hyp_final, adaptive_state, success = _step(dyn, hyp, adaptive_state)
        kalman_state = update_kalman(dyn.position, kalman_state, outer_weight, success, hyp.step_size)

        return (dyn, hyp, hyp_final, adaptive_state, kalman_state), None


    def func(_dyn, _hyp, num_steps):
        
        num_steps1, num_steps2 = jnp.rint(num_steps * frac).astype(int)
            
        # we use the last num_steps2 to compute the diagonal preconditioner
        outer_weights = jnp.concatenate((jnp.zeros(num_steps1), jnp.ones(num_steps2)))

        #initial state
        
        kalman_state = (0., jnp.zeros(d), jnp.zeros(d))

        # run the steps
        state = jax.lax.scan(step, init= (_dyn, _hyp, _hyp, adap0, kalman_state), xs= outer_weights, length= num_steps1 + num_steps2)[0]
        dyn, _, hyp, adap, kalman_state = state
        
        L = hyp.L
        # determine L
        if num_steps2 != 0.:
            _, F1, F2 = kalman_state
            variances = F2 - jnp.square(F1)
            L = jnp.sqrt(jnp.sum(variances))

            # # optionally we do the diagonal preconditioning (and readjust the stepsize)
            # if diag_precond:

            #     # diagonal preconditioning
            #     sigma = jnp.sqrt(variances)
            #     L = jnp.sqrt(d)

            #     #readjust the stepsize
            #     steps = num_steps2 // 3 #we do some small number of steps
            #     state = jax.lax.scan(step, init= state, xs= jnp.ones(steps), length= steps)[0]
            #     dyn, _, hyp, adap, kalman_state = state
            # else:
            #     sigma = hyp.sigma
        
        # jax.debug.print(" \n\n\nPARAMS:\n{x}", x=(dyn,MCLMCAdaptationState(L, hyp.step_size) ))
        return dyn, MCLMCAdaptationState(L, hyp.step_size)

    return func




def tune3(kernel, frac, Lfactor):
    """determine L by the autocorrelations (around 10 effective samples are needed for this to be accurate)"""
    

    def sample_full(num_steps, _dyn, hyp):
        """Stores full x for each step. Used in tune2."""

        def _step(state, useless):
            dyn_old = state
            # dyn_new, _ = step(dyn_old, hyp)
            dyn_new, _ = kernel(rng_key=jax.random.PRNGKey(0), state=dyn_old, L=hyp.L, step_size=hyp.step_size)
            
            return dyn_new, dyn_new.position

        return jax.lax.scan(_step, init=_dyn, xs=None, length=num_steps)


    def func(dyn, hyp, num_steps):
        steps = jnp.rint(num_steps * frac).astype(int)
        
        dyn, X = sample_full(steps, dyn, hyp)
        ESS = ess_corr(X) # num steps / effective sample size
        Lnew = Lfactor * hyp.step_size / ESS # = 0.4 * length corresponding to one effective sample

        return dyn, MCLMCAdaptationState(Lnew, hyp.step_size)


    return func

