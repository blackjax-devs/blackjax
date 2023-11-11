import jax
import jax.numpy as jnp

import blackjax
from blackjax.base import SamplingAlgorithm
from blackjax.diagnostics import effective_sample_size
from blackjax.mcmc.mclmc import Parameters
from blackjax.types import PRNGKey
from scipy.fftpack import next_fast_len #type: ignore


def logdensity_fn(x):
    return -0.5 * jnp.sum(jnp.square(x - 5))


# Initialize the state
initial_position = jnp.array([1.0, 1.0])


dim = 2 







def run_sampling_algorithm(
    sampling_algorithm: SamplingAlgorithm, num_steps: int, initial_val, rng_key
):
    keys = jax.random.split(rng_key, num_steps + 1)
    state = sampling_algorithm.init(initial_val, keys[0])
    _, info = jax.lax.scan(
        lambda s, k: (sampling_algorithm.step(k, s)), state, keys[1:]
    )
    return info


key = jax.random.PRNGKey(0)
main_key, tune_key = jax.random.split(key)

def ess_corr(x):
    """Taken from: https://blackjax-devs.github.io/blackjax/diagnostics.html
        shape(x) = (num_samples, d)"""

    input_array = jnp.array([x, ])

    num_chains = 1#input_array.shape[0]
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
    mean_var0 = (jnp.take(mean_autocov_var, jnp.array([0]), axis=1) * num_samples / (num_samples - 1.0))
    weighted_var = mean_var0 * (num_samples - 1.0) / num_samples
    weighted_var = jax.lax.cond(
        num_chains > 1,
        lambda _: weighted_var+ mean_across_chain.var(axis=0, ddof=1, keepdims=True),
        lambda _: weighted_var,
        operand=None,
    )

    # Geyer's initial positive sequence
    num_samples_even = num_samples - num_samples % 2
    mean_autocov_var_tp1 = jnp.take(mean_autocov_var, jnp.arange(1, num_samples_even), axis=1)
    rho_hat = jnp.concatenate([jnp.ones_like(mean_var0), 1.0 - (mean_var0 - mean_autocov_var_tp1) / weighted_var,], axis=1,)

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
    tau_hat = (-1.0
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
num_steps = 10000
initial_params = Parameters(1.3852125, 1.0604926, jnp.array([1., 1.]))
# Parameters(L=jnp.sqrt(dim),step_size=0.4*jnp.sqrt(dim), inverse_mass_matrix=jnp.array([1.0, 1.0]))
def tune(num_steps : int, params : Parameters, rng_key : PRNGKey) -> Parameters:

    # steps1 = (int)(num_steps * 0.1)
    # steps2 = (int)(num_steps * 0.1)
    # def tune12(self, x, u, l, g, random_key, L_given, eps, sigma_given, num_steps1, num_steps2):
    #     """cheap hyperparameter tuning"""
        

    #     def step(state, outer_weight):
    #         """one adaptive step of the dynamics"""
    #         x, u, l, g, E, Feps, Weps, eps_max, key, eps = self.dynamics_adaptive(state[0], L, sigma)
    #         W, F1, F2 = state[1]
    #         w = outer_weight * eps
    #         zero_prevention = 1-outer_weight
    #         F1 = (W*F1 + w*x) / (W + w + zero_prevention)  # Update <f(x)> with a Kalman filter
    #         F2 = (W*F2 + w*jnp.square(x)) / (W + w + zero_prevention)  # Update <f(x)> with a Kalman filter
    #         W += w

    #         return ((x, u, l, g, E, Feps, Weps, eps_max, key), (W, F1, F2)), eps

    #     L = L_given

    #     # we use the last num_steps2 to compute the diagonal preconditioner
    #     outer_weights = jnp.concatenate((jnp.zeros(num_steps1), jnp.ones(num_steps2)))

    #     #initial state
    #     state = ((x, u, l, g, 0., jnp.power(eps, -6.0) * 1e-5, 1e-5, jnp.inf, random_key), (0., jnp.zeros(len(x)), jnp.zeros(len(x))))
    #     # run the steps
    #     state, eps = jax.lax.scan(step, init=state, xs= outer_weights, length= num_steps1 + num_steps2)
    #     # determine L
    #     if num_steps2 != 0.:
    #         F1, F2 = state[1][1], state[1][2]
    #         variances = F2 - jnp.square(F1)
    #         sigma2 = jnp.average(variances)

    #         # optionally we do the diagonal preconditioning (and readjust the stepsize)
    #         if self.diagonal_preconditioning:

    #             # diagonal preconditioning
    #             sigma = jnp.sqrt(variances)
    #             L = jnp.sqrt(self.Target.d)

    #             #readjust the stepsize
    #             steps = num_steps2 // 3 #we do some small number of steps
    #             state, eps = jax.lax.scan(step, init= state, xs= jnp.ones(steps), length= steps)
    #         else:
    #             L = jnp.sqrt(sigma2 * self.Target.d)

    #     xx, uu, ll, gg, key = state[0][0], state[0][1], state[0][2], state[0][3], state[0][-1] # the final state
    #     return L, eps[-1], sigma, xx, uu, ll, gg, key #return the tuned hyperparameters and the final state


    # params = 

    mclmc = blackjax.mcmc.mclmc.mclmc(
        logdensity_fn=logdensity_fn,
        dim=dim,
        transform=lambda x: x,
        params=params
    )
    out = run_sampling_algorithm(
        sampling_algorithm=mclmc,
        num_steps= int(num_steps * 0.1),
        initial_val=jnp.array([0.1, 0.1]),
        rng_key=rng_key,
    )
    Lfactor = 0.4
    # ESS = effective_sample_size(out.transformed_x)
    ESS = ess_corr(out.transformed_x)
    # neff = ESS / num_steps
    # ESS = 1.0 / jnp.average(1 / neff)
    # print(f"Ess is {ESS}")
    # print(f"Ess is {ESS2}")
    # print(out.transformed_x)
    Lnew = Lfactor * initial_params.step_size / ESS
    return Parameters(L=Lnew, step_size=params.step_size, inverse_mass_matrix=params.inverse_mass_matrix)

print(tune(num_steps=10000, params=initial_params, rng_key=tune_key))


mclmc = blackjax.mcmc.mclmc.mclmc(
    logdensity_fn=logdensity_fn,
    dim=dim,
    transform=lambda x: x,
    params=Parameters(
        L=0.56568545, step_size=1.4142135, inverse_mass_matrix=jnp.array([1.0, 1.0])
    ),
)

out = run_sampling_algorithm(
    sampling_algorithm=mclmc,
    num_steps=10000,
    initial_val=jnp.array([0.1, 0.1]),
    rng_key=main_key,
)

print(jnp.mean(out.transformed_x, axis=0))

assert jnp.array_equal(jnp.mean(out.transformed_x, axis=0), [5.0377626, 4.9752364])



