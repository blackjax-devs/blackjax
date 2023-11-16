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
"""Step size adaptation"""
from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp
from scipy.fft import next_fast_len

from blackjax.mcmc.hmc import HMCState
from blackjax.mcmc.mclmc import MCLMCState
from blackjax.optimizers.dual_averaging import dual_averaging
from blackjax.types import PRNGKey

__all__ = [
    "DualAveragingAdaptationState",
    "dual_averaging_adaptation",
    "find_reasonable_step_size",
]


# -------------------------------------------------------------------
#                        DUAL AVERAGING
# -------------------------------------------------------------------


class DualAveragingAdaptationState(NamedTuple):
    """State carried through the dual averaging procedure.

    log_step_size
        The logarithm of the current value of the step size.
    log_step_size_avg
        The time-weighted average of the values that the logarithm of the step
        size has taken so far.
    step
        The current iteration step.
    avg_err
        The time average of the value of the quantity :math:`H_t`, the
        difference between the target acceptance rate and the current
        acceptance rate.
    mu
        Arbitrary point the values of log_step_size are shrunk towards. Chose
        to be :math:`\\log(10 \\epsilon_0)` where :math:`\\epsilon_0` is chosen
        in this context to be the step size given by the
        `find_reasonable_step_size` procedure.

    """

    log_step_size: float
    log_step_size_avg: float
    step: int
    avg_error: float
    mu: float


def dual_averaging_adaptation(
    target: float, t0: int = 10, gamma: float = 0.05, kappa: float = 0.75
) -> tuple[Callable, Callable, Callable]:
    """Tune the step size in order to achieve a desired target acceptance rate.

    Let us note :math:`\\epsilon` the current step size, :math:`\\alpha_t` the
    metropolis acceptance rate at time :math:`t` and :math:`\\delta` the desired
    aceptance rate. We define:

    .. math:
        H_t = \\delta - \\alpha_t

    the error at time t. We would like to find a procedure that adapts the
    value of :math:`\\epsilon` such that :math:`h(x) =\\mathbb{E}\\left[H_t|\\epsilon\\right] = 0`

    Following :cite:p:`nesterov2009primal`, the authors of :cite:p:`hoffman2014no` proposed the following update scheme. If
    we note :math:`x = \\log \\epsilon` we follow:

    .. math:
        x_{t+1} \\LongLeftArrow \\mu - \\frac{\\sqrt{t}}{\\gamma} \\frac{1}{t+t_0} \\sum_{i=1}^t H_i
        \\overline{x}_{t+1} \\LongLeftArrow x_{t+1}\\, t^{-\\kappa}  + \\left(1-t^\\kappa\\right)\\overline{x}_t

    :math:`\\overline{x}_{t}` is guaranteed to converge to a value such that
    :math:`h(\\overline{x}_t)` converges to 0, i.e. the Metropolis acceptance
    rate converges to the desired rate.

    See reference :cite:p:`hoffman2014no` (section 3.2.1) for a detailed discussion.

    Parameters
    ----------
    t0: float >= 0
        Free parameter that stabilizes the initial iterations of the algorithm.
        Large values may slow down convergence. Introduced in :cite:p:`hoffman2014no` with a default
        value of 10.
    gamma:
        Controls the speed of convergence of the scheme. The authors of :cite:p:`hoffman2014no` recommend
        a value of 0.05.
    kappa: float in [0.5, 1]
        Controls the weights of past steps in the current update. The scheme will
        quickly forget earlier step for a small value of `kappa`. Introduced
        in :cite:p:`hoffman2014no`, with a recommended value of .75
    target:
        Target acceptance rate.

    Returns
    -------
    init
        A function that initializes the state of the dual averaging scheme.
    update
        A function that updates the state of the dual averaging scheme.

    """
    da_init, da_update, da_final = dual_averaging(t0, gamma, kappa)

    def init(inital_step_size: float) -> DualAveragingAdaptationState:
        """Initialize the state of the dual averaging scheme.

        The parameter :math:`\\mu` is set to :math:`\\log(10 \\epsilon_1)`
        where :math:`\\epsilon_1` is the initial value of the step size.
        """
        return DualAveragingAdaptationState(*da_init(inital_step_size))

    def update(
        da_state: DualAveragingAdaptationState, acceptance_rate: float
    ) -> DualAveragingAdaptationState:
        """Update the state of the Dual Averaging adaptive algorithm.

        Parameters
        ----------
        da_state:
            The current state of the dual averaging algorithm.
        acceptance_rate: float in [0, 1]
            The current metropolis acceptance rate.

        Returns
        -------
        The updated state of the dual averaging algorithm.

        """
        gradient = target - acceptance_rate
        return DualAveragingAdaptationState(*da_update(da_state, gradient))

    def final(da_state: DualAveragingAdaptationState) -> float:
        return jnp.exp(da_state.log_step_size_avg)

    return init, update, final


# -------------------------------------------------------------------
#                 REASONABLE FIRST STEP SIZE
# -------------------------------------------------------------------


class ReasonableStepSizeState(NamedTuple):
    """State carried through the search for a reasonable first step size.

    rng_key
        Key used by JAX's random number generator.
    direction: {-1, 1}
        Determines whether the step size should be increased or decreased during the
        previous step search. If direction = 1 it will be increased, otherwise decreased.
    previous_direction
        The previous direction. It is necessary to carry it because the choice of step
        size is made at the end of the search update.
    step_size
        The current step size in the search.

    """

    rng_key: PRNGKey
    direction: int
    previous_direction: int
    step_size: float


def find_reasonable_step_size(
    rng_key: PRNGKey,
    kernel_generator: Callable[[float], Callable],
    reference_state: HMCState,
    initial_step_size: float,
    target_accept: float = 0.65,
) -> float:
    """Find a reasonable initial step size during warmup.

    While the dual averaging scheme is guaranteed to converge to a reasonable
    value for the step size starting from any value, choosing a good first
    value can speed up the convergence. This heuristics doubles and halves the
    step size until the acceptance probability of the HMC proposal crosses the
    target value :cite:p:`hoffman2014no`.

    Parameters
    ----------
    rng_key
       Key used by JAX's random number generator.
    kernel_generator
        A function that takes a step size as an input and returns the corresponding
        sampling kernel.
    reference_hmc_state
        The location (HMC state) where this first step size must be found. This function
        never advances the chain.
    inverse_mass_matrix
        The inverse mass matrix relative to which the step size must be found.
    initial_step_size
        The first step size used to start the search.
    target_accept
        Once that value of the metropolis acceptance probability is reached we
        estimate that we have found a "reasonable" first step size.

    Returns
    -------
    float
        A reasonable first value for the step size.

    """
    fp_limit = jnp.finfo(jax.lax.dtype(initial_step_size))

    def do_continue(rss_state: ReasonableStepSizeState) -> bool:
        """Decides whether the search should continue.

        The search stops when it crosses the `target_accept` threshold, i.e.
        when the current direction is opposite to the previous direction.

        Note
        ----
        Per JAX's documentation :cite:p:`jax_finfo` the `jnp.finfo` object is cached so we do not
        occur any performance penalty when calling it repeatedly inside this
        function.

        """
        _, direction, previous_direction, step_size = rss_state

        not_too_large = (step_size < fp_limit.max) | (direction <= 0)
        not_too_small = (step_size > fp_limit.tiny) | (direction >= 0)
        is_step_size_not_extreme = not_too_large & not_too_small
        has_acceptance_rate_not_crossed_threshold = (previous_direction == 0) | (
            direction == previous_direction
        )
        return is_step_size_not_extreme & has_acceptance_rate_not_crossed_threshold

    def update(rss_state: ReasonableStepSizeState) -> ReasonableStepSizeState:
        """Perform one step of the step size search."""
        rng_key, direction, _, step_size = rss_state
        rng_key, subkey = jax.random.split(rng_key)

        step_size = (2.0**direction) * step_size
        kernel = kernel_generator(step_size)
        _, info = kernel(subkey, reference_state)

        new_direction = jnp.where(target_accept < info.acceptance_rate, 1, -1)
        return ReasonableStepSizeState(rng_key, new_direction, direction, step_size)

    rss_state = ReasonableStepSizeState(rng_key, 0, 0, initial_step_size)
    rss_state = jax.lax.while_loop(do_continue, update, rss_state)

    return rss_state.step_size










#### mclmc

class Parameters(NamedTuple):
    """Tunable parameters"""

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

    ### my part (combine all dimensions): ###
    neff = ess.squeeze() / num_samples
    return 1.0 / jnp.average(1 / neff)


# ?
# tuning()
num_steps = 100
initial_params = Parameters(1.9913111, 0.6458658)

def nan_reject(x, u, l, g, xx, uu, ll, gg, eps, eps_max, dK):
        """if there are nans, let's reduce the stepsize, and not update the state. The function returns the old state in this case."""
        
        nonans = jnp.all(jnp.isfinite(xx))

        return nonans, *jax.tree_util.tree_map(lambda new, old: jax.lax.select(nonans, jnp.nan_to_num(new), old), (xx, uu, ll, gg, eps_max, dK), (x, u, l, g, eps * 0.8, 0.))


def dynamics_adaptive(dynamics, state, L, sigma):
        """One step of the dynamics with the adaptive stepsize"""

        x, u, l, g, E, Feps, Weps, eps_max, key = state

        eps = jnp.power(Feps/Weps, -1.0/6.0) #We use the Var[E] = O(eps^6) relation here.
        eps = (eps < eps_max) * eps + (eps > eps_max) * eps_max  # if the proposed stepsize is above the stepsize where we have seen divergences

        # grad_logp = jax.value_and_grad(logdensity_fn)
        

        # print(L_given, eps, "\n\n\n")
        # m = build_kernel(grad_logp, minimal_norm, lambda x:x, L_given, eps, sigma)
        # dynamics = lambda x,u,l,g, key : m(jax.random.PRNGKey(0), MCLMCState(x,u,l,g))

        # dynamics
        
        # xx, uu, ll, gg, kinetic_change, key = dynamics(x, u, g, key, L, eps, sigma)
        # jax.debug.print("🤯 {x} x 🤯", x=(x,u,l,g, E, Feps, Weps, eps_max))
        jax.debug.print("🤯 {x} L eps 🤯", x=(L, eps, sigma))
        jax.debug.print("🤯 {x} x u 🤯", x=(x,u, g))
        state, info = dynamics(jax.random.PRNGKey(0), MCLMCState(x, u, -l, -g), L=L, step_size=eps)
        
        xx, uu, ll, gg = state
        ll, gg = -ll, -gg
        # jax.debug.print("🤯 {x} xx uu 🤯", x=(xx,uu))
        kinetic_change = info.kinetic_change

        varEwanted = 5e-4
        sigma_xi= 1.5        
        neff = 150 # effective number of steps used to determine the stepsize in the adaptive step
        gamma = (neff - 1.0) / (neff + 1.0) # forgeting factor in the adaptive step


        # step updating
        # jax.debug.print("🤯 {x} L eps 🤯", x=(x, u, l, g, xx, uu, ll, gg, eps, eps_max, kinetic_change))
        success, xx, uu, ll, gg, eps_max, kinetic_change = nan_reject(x, u, l, g, xx, uu, ll, gg, eps, eps_max, kinetic_change)


        DE = info.dE  # energy difference
        # jax.debug.print("🤯 {x} DE 🤯", x=(DE, kinetic_change))
        EE = E + DE  # energy
        # Warning: var = 0 if there were nans, but we will give it a very small weight
        xi = ((DE ** 2) / (xx.shape[0] * varEwanted)) + 1e-8  # 1e-8 is added to avoid divergences in log xi
        w = jnp.exp(-0.5 * jnp.square(jnp.log(xi) / (6.0 * sigma_xi)))  # the weight which reduces the impact of stepsizes which are much larger on much smaller than the desired one.
        Feps = gamma * Feps + w * (xi/jnp.power(eps, 6.0))  # Kalman update the linear combinations
        Weps = gamma * Weps + w

        return xx, uu, ll, gg, EE, Feps, Weps, eps_max, key, eps * success

def tune12(kernel,x, u, l, g, random_key, L_given, eps, sigma_given, num_steps1, num_steps2):
        """cheap hyperparameter tuning"""

        # mclmc = blackjax.mclmc(
        # logdensity_fn=logdensity_fn, transform=lambda x: x, L=params.L, step_size=params.step_size, inverse_mass_matrix=params.inverse_mass_matrix
        # )
        

        sigma = sigma_given
        
        def step(state, outer_weight):
            """one adaptive step of the dynamics"""
            # x,u,l,g = state
            # E, Feps, Weps, eps_max = 1.0,1.0,1.0,1.0
            x, u, l, g, E, Feps, Weps, eps_max, key, eps = dynamics_adaptive(kernel, state[0], L, sigma)
            W, F1, F2 = state[1]
            w = outer_weight * eps
            zero_prevention = 1-outer_weight
            F1 = (W*F1 + w*x) / (W + w + zero_prevention)  # Update <f(x)> with a Kalman filter
            F2 = (W*F2 + w*jnp.square(x)) / (W + w + zero_prevention)  # Update <f(x)> with a Kalman filter
            W += w

            return ((x, u, l, g, E, Feps, Weps, eps_max, key), (W, F1, F2)), eps

        L = L_given

        # we use the last num_steps2 to compute the diagonal preconditioner
        outer_weights = jnp.concatenate((jnp.zeros(num_steps1), jnp.ones(num_steps2)))

        #initial state
        state = ((x, u, l, g, 0., jnp.power(eps, -6.0) * 1e-5, 1e-5, jnp.inf, random_key), (0., jnp.zeros(len(x)), jnp.zeros(len(x))))
        # run the steps
        state, eps = jax.lax.scan(step, init=state, xs= outer_weights, length= num_steps1 + num_steps2)
        # determine L
        if num_steps2 != 0.:
            F1, F2 = state[1][1], state[1][2]
            variances = F2 - jnp.square(F1)
            sigma2 = jnp.average(variances)

            L = jnp.sqrt(sigma2 * x.shape[0])

        xx, uu, ll, gg, key = state[0][0], state[0][1], state[0][2], state[0][3], state[0][-1] # the final state
        return L, eps[-1], sigma, xx, uu, ll, gg, key #return the tuned hyperparameters and the final state

def tune3(kernel, x, u, l, g, rng_key, L, eps, sigma, num_steps):
    """determine L by the autocorrelations (around 10 effective samples are needed for this to be accurate)"""
    print(L, eps, sigma, x,u, "initial params")



    
    
    keys = jnp.array([jax.random.PRNGKey(0)]*num_steps)

    state, info = jax.lax.scan(
        lambda s, k: (kernel(k, s, L, eps)), MCLMCState(x,u,-l,-g), keys
    )

    # state, info = kernel(jax.random.PRNGKey(0), MCLMCState(x, u, l, g), L=L, step_size=eps)
    # xx,uu,ll,gg = state
    X = info.transformed_x
    
        # sample_full(num_steps, x, u, l, g, random_key, L, eps, sigma)
    ESS = ess_corr(X)
    Lfactor = 0.4
    Lnew = Lfactor * eps / ESS # = 0.4 * correlation length
    print(ESS, "ess", X, Lfactor, eps)
    return Lnew, state

def tune(kernel, num_steps: int, rng_key: PRNGKey) -> Parameters:

    num_tune_step_ratio_1 = 0.1
    num_tune_step_ratio_2 = 0.1


    x, u, l, g, L, eps, sigma = jnp.array([0.1, 0.1]), jnp.array([-0.6755803,   0.73728645]), 0.010000001, jnp.array([0.1, 0.1]), 1.4142135, 0.56568545, jnp.array([1., 1.])

    L, eps, sigma, x, u, l, g, key = tune12(kernel, x, u, l, g, rng_key, L, eps, sigma, int(num_steps * num_tune_step_ratio_1), int(num_steps * num_tune_step_ratio_1))
    print("L, eps post tune12", L, eps)

    L, state = tune3(kernel, x, u, l, g, key, L, eps, sigma, int(num_steps * num_tune_step_ratio_2))
    print("L post tune3", L)
    return L, eps, state