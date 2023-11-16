from typing import NamedTuple
from chex import Array
import jax
import jax.numpy as jnp
from scipy.fftpack import next_fast_len  # type: ignore

import blackjax
from blackjax.base import SamplingAlgorithm
from blackjax.mcmc.integrators import minimal_norm
from blackjax.mcmc.mclmc import MCLMCState, build_kernel
# from blackjax.diagnostics import effective_sample_size
from blackjax.types import PRNGKey

class Parameters(NamedTuple):
    """Tunable parameters"""

    L: float
    step_size: float


def logdensity_fn(x):
    return -0.5 * jnp.sum(jnp.square(x))


def run_sampling_algorithm(
    sampling_algorithm: SamplingAlgorithm, num_steps: int, initial_val, rng_key
):
    # keys = jax.random.split(rng_key, num_steps)
    keys = jnp.array([jax.random.PRNGKey(0)]*num_steps)
    state = sampling_algorithm.init(initial_val)
    print("\n\n", state.position, "\n\n")
    print("\n\n", state.momentum, "\n\n")
    _, info = jax.lax.scan(
        lambda s, k: (sampling_algorithm.step(k, s)), state, keys
    )
    return info


key = jax.random.PRNGKey(0)
main_key, tune_key = jax.random.split(key)


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

def tune12(x, u, l, g, random_key, L_given, eps, sigma_given, num_steps1, num_steps2):
        """cheap hyperparameter tuning"""

        # mclmc = blackjax.mclmc(
        # logdensity_fn=logdensity_fn, transform=lambda x: x, L=params.L, step_size=params.step_size, inverse_mass_matrix=params.inverse_mass_matrix
        # )
        

        sigma = sigma_given
        gr = jax.value_and_grad(logdensity_fn)
        dynamics = build_kernel(grad_logp=gr, 
                                        integrator=minimal_norm, transform=lambda x:x)

        def step(state, outer_weight):
            """one adaptive step of the dynamics"""
            # x,u,l,g = state
            # E, Feps, Weps, eps_max = 1.0,1.0,1.0,1.0
            x, u, l, g, E, Feps, Weps, eps_max, key, eps = dynamics_adaptive(dynamics, state[0], L, sigma)
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

def tune3(x, u, l, g, rng_key, L, eps, sigma, num_steps):
    """determine L by the autocorrelations (around 10 effective samples are needed for this to be accurate)"""
    print(L, eps, sigma, x,u, "initial params")



    gr = jax.value_and_grad(logdensity_fn)
    kernel = build_kernel(grad_logp=gr, 
                                        integrator=minimal_norm, transform=lambda x:x)
    
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

def tune(num_steps: int, params: Parameters, rng_key: PRNGKey) -> Parameters:



    x, u, l, g, key, L, eps, sigma, steps1, steps2 = jnp.array([0.1, 0.1]), jnp.array([-0.6755803,   0.73728645]), 0.010000001, jnp.array([0.1, 0.1]), jax.random.PRNGKey(0), 1.4142135, 0.56568545, jnp.array([1., 1.]), 10, 10

    L, eps, sigma, x, u, l, g, key = tune12(x, u, l, g, key, L, eps, sigma, steps1, steps2)
    print("L, eps post tune12", L, eps)

    


    
    steps3 = int(num_steps * 0.1)
    L, state = tune3(x, u, l, g, key, L, eps, sigma, steps3)
    print("L post tune3", L)
    return L, eps, state


L, eps, state = (tune(num_steps=100, params=initial_params, rng_key=tune_key))
print("L, eps post tuning", L, eps)
raise Exception
mclmc = blackjax.mcmc.mclmc.mclmc(
    logdensity_fn=logdensity_fn,
    transform=lambda x: x,
    # L=0.56568545, step_size=1.4142135, inverse_mass_matrix=jnp.array([1.0, 1.0]
    step_size=0.56568545, L=1.4142135,
)


out = run_sampling_algorithm(
    sampling_algorithm=mclmc,
    num_steps=100,
    initial_val=jnp.array([0.1, 0.1]),
    rng_key=main_key,
)

print(jnp.mean(out.transformed_x, axis=0))

# print(logdensity_fn(jnp.array([0.1, 0.1])))
# print(out)

assert jnp.array_equal(jnp.mean(out.transformed_x, axis=0), [-1.2130139,  1.5367734])


