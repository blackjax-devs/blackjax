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
from scipy.fftpack import next_fast_len #type: ignore

from blackjax.mcmc.integrators import IntegratorState, noneuclidean_mclachlan
from blackjax.mcmc.mclmc import build_kernel, init


class MCLMCAdaptationState(NamedTuple):
    """Tunable parameters for MCLMC"""

    L: float
    step_size: float


def ess_corr(x):
    """Taken from: https://blackjax-devs.github.io/blackjax/diagnostics.html
    shape(x) = (num_samples, d)"""

    input_array = jnp.array(
        [
            x,
        ]
    )

    num_chains = 1  # input_array.shape[0]
    num_samples = input_array.shape[1]

    mean_across_chain = input_array.mean(axis=1, keepdims=True)
    # Compute autocovariance estimates for every lag for the input array using FFT.
    centered_array = input_array - mean_across_chain
    m = next_fast_len(2 * num_samples)
    ifft_ary = jnp.fft.rfft(centered_array, n=m, axis=1)
    ifft_ary *= jnp.conjugate(ifft_ary)
    autocov_value = jnp.fft.irfft(ifft_ary, n=m, axis=1)
    autocov_value = (
        jnp.take(autocov_value, jnp.arange(num_samples), axis=1) / num_samples
    )
    mean_autocov_var = autocov_value.mean(0, keepdims=True)
    mean_var0 = (
        jnp.take(mean_autocov_var, jnp.array([0]), axis=1)
        * num_samples
        / (num_samples - 1.0)
    )
    weighted_var = mean_var0 * (num_samples - 1.0) / num_samples

    weighted_var = jax.lax.cond(
        num_chains > 1,
        lambda _: weighted_var + mean_across_chain.var(axis=0, ddof=1, keepdims=True),
        lambda _: weighted_var,
        operand=None,
    )

    # Geyer's initial positive sequence
    num_samples_even = num_samples - num_samples % 2
    mean_autocov_var_tp1 = jnp.take(
        mean_autocov_var, jnp.arange(1, num_samples_even), axis=1
    )
    rho_hat = jnp.concatenate(
        [
            jnp.ones_like(mean_var0),
            1.0 - (mean_var0 - mean_autocov_var_tp1) / weighted_var,
        ],
        axis=1,
    )

    rho_hat = jnp.moveaxis(rho_hat, 1, 0)
    rho_hat_even = rho_hat[0::2]
    rho_hat_odd = rho_hat[1::2]

    mask0 = (rho_hat_even + rho_hat_odd) > 0.0
    carry_cond = jnp.ones_like(mask0[0])
    max_t = jnp.zeros_like(mask0[0], dtype=int)

    def positive_sequence_body_fn(state, mask_t):
        t, carry_cond, max_t = state
        next_mask = carry_cond & mask_t
        next_max_t = jnp.where(next_mask, jnp.ones_like(max_t) * t, max_t)
        return (t + 1, next_mask, next_max_t), next_mask

    (*_, max_t_next), mask = jax.lax.scan(
        positive_sequence_body_fn, (0, carry_cond, max_t), mask0
    )
    indices = jnp.indices(max_t_next.shape)
    indices = tuple([max_t_next + 1] + [indices[i] for i in range(max_t_next.ndim)])
    rho_hat_odd = jnp.where(mask, rho_hat_odd, jnp.zeros_like(rho_hat_odd))
    # improve estimation
    mask_even = mask.at[indices].set(rho_hat_even[indices] > 0)
    rho_hat_even = jnp.where(mask_even, rho_hat_even, jnp.zeros_like(rho_hat_even))

    # Geyer's initial monotone sequence
    def monotone_sequence_body_fn(rho_hat_sum_tm1, rho_hat_sum_t):
        update_mask = rho_hat_sum_t > rho_hat_sum_tm1
        next_rho_hat_sum_t = jnp.where(update_mask, rho_hat_sum_tm1, rho_hat_sum_t)
        return next_rho_hat_sum_t, (update_mask, next_rho_hat_sum_t)

    rho_hat_sum = rho_hat_even + rho_hat_odd
    _, (update_mask, update_value) = jax.lax.scan(
        monotone_sequence_body_fn, rho_hat_sum[0], rho_hat_sum
    )

    rho_hat_even_final = jnp.where(update_mask, update_value / 2.0, rho_hat_even)
    rho_hat_odd_final = jnp.where(update_mask, update_value / 2.0, rho_hat_odd)

    # compute effective sample size
    ess_raw = num_chains * num_samples
    tau_hat = (
        -1.0
        + 2.0 * jnp.sum(rho_hat_even_final + rho_hat_odd_final, axis=0)
        - rho_hat_even_final[indices]
    )

    tau_hat = jnp.maximum(tau_hat, 1 / jnp.log10(ess_raw))
    ess = ess_raw / tau_hat

    neff = ess.squeeze() / num_samples
    return 1.0 / jnp.average(1 / neff)

import jax
import jax.numpy as jnp
from typing import NamedTuple




def mclmc_find_L_and_step_size(kernel, num_steps, initial_state):

    d = initial_state.position.shape[0]
    dyn = initial_state
    hyp = MCLMCAdaptationState(jnp.sqrt(d), 
                                        jnp.sqrt(d) * 0.25, 
                                        )

    frac_tune1 = 0.1
    frac_tune2 = 0.1
    frac_tune3 = 0.1
    varEwanted = 5e-4
    tune12p = tune12(kernel, d, False, jnp.array([frac_tune1, frac_tune2]), varEwanted, 1.5, 150)

    tune3p = tune3(kernel, frac_tune3, 0.4)

    if frac_tune3 != 0.:
        tune3p = tune3(kernel, frac= frac_tune3, Lfactor= 0.4)
        schedule = [tune12p, tune3p]
    else:
        schedule = [tune12p, ]

    dyn, hyp = run(dyn, hyp, schedule, num_steps)
    return dyn, hyp

# all tuning functions are wrappers, recieving some parameters and returning a function
# func(dyn, hyp, num_total_steps) -> (dyn, hyp) 



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

