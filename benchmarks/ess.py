from inference_gym import using_jax as gym
import jax
import jax.numpy as jnp
import numpy as np

import blackjax
from blackjax.util import run_inference_algorithm


def get_num_latents(target):
  return int(sum(map(np.prod, list(jax.tree_flatten(target.event_shape)[0]))))

def err(f_true, var_f, contract = jnp.max):
    """Computes the error b^2 = (f - f_true)^2 / var_f
        Args:
            f: E_sampler[f(x)], can be a vector
            f_true: E_true[f(x)]
            var_f: Var_true[f(x)]
            contract: how to combine a vector f in a single number, can be for example jnp.average or jnp.max
            
        Returns:
            contract(b^2)
    """    
    
    def _err(f):
        bsq = jnp.square(f - f_true) / var_f
        return contract(bsq)
    
    return jax.vmap(_err)



def grads_to_low_error(err_t, low_error= 0.01, grad_evals_per_step= 1):
    """Uses the error of the expectation values to compute the effective sample size neff
        b^2 = 1/neff"""
    
    cutoff_reached = err_t[-1] < low_error
    return find_crossing(err_t, low_error) * grad_evals_per_step, cutoff_reached
    
    
    
def ess(err_t, neff= 100, grad_evals_per_step = 1):
    
    low_error = 1./neff
    cutoff_reached = err_t[-1] < low_error
    crossing = find_crossing(err_t, low_error)
    print("time to reach cutoff", crossing)
    
    return (neff / (crossing * grad_evals_per_step)) * cutoff_reached



def find_crossing(array, cutoff):
    """the smallest M such that array[m] < cutoff for all m > M"""

    def step(carry, element):
        """carry = (, 1 if (array[i] > cutoff for all i < current index) else 0"""
        above_threshold = element > cutoff
        never_been_below = carry[1] * above_threshold  #1 if (array[i] > cutoff for all i < current index) else 0
        return (carry[0] + never_been_below, never_been_below), above_threshold

    state, track = jax.lax.scan(step, init=(0, 1), xs=array, length=len(array))

    return state[0]
    #return jnp.sum(track) #total number of indices for which array[m] < cutoff



def cumulative_avg(samples):
    return jnp.cumsum(samples, axis = 0) / jnp.arange(1, samples.shape[0] + 1)[:, None]

def run_mclmc(logdensity_fn, num_steps, initial_position):
        init_key, tune_key, run_key = jax.random.split(jax.random.PRNGKey(2), 3)

        initial_state = blackjax.mcmc.mclmc.init(
            position=initial_position, logdensity_fn=logdensity_fn, rng_key=init_key
        )

        kernel = blackjax.mcmc.mclmc.build_kernel(
            logdensity_fn=logdensity_fn,
            integrator=blackjax.mcmc.integrators.isokinetic_mclachlan,
        )

        (
            blackjax_state_after_tuning,
            blackjax_mclmc_sampler_params,
        ) = blackjax.mclmc_find_L_and_step_size(
            mclmc_kernel=kernel,
            num_steps=num_steps,
            state=initial_state,
            rng_key=tune_key,
        )

        sampling_alg = blackjax.mclmc(
            logdensity_fn,
            L=blackjax_mclmc_sampler_params.L,
            step_size=blackjax_mclmc_sampler_params.step_size,
        )

        _, samples, _ = run_inference_algorithm(
            rng_key=run_key,
            initial_state_or_position=blackjax_state_after_tuning,
            inference_algorithm=sampling_alg,
            num_steps=num_steps,
            transform=lambda x: x.position,
        )

        return samples

def inference_loop_multiple_chains(
    rng_key, initial_states, tuned_params, log_prob_fn, num_samples, num_chains
):
    kernel = blackjax.nuts.build_kernel()

    def step_fn(key, state, **params):
        return kernel(key, state, log_prob_fn, **params)

    def one_step(states, rng_key):
        keys = jax.random.split(rng_key, num_chains)
        states, infos = jax.vmap(step_fn)(keys, states, **tuned_params)
        return states, (states, infos)

    keys = jax.random.split(rng_key, num_samples)
    _, (states, infos) = jax.lax.scan(one_step, initial_states, keys)

    return (states, infos)

def run_hmc(
    rng_key, initial_states, tuned_params, logdensity_fn, num_samples):
    
    warmup = blackjax.window_adaptation(blackjax.nuts, logdensity_fn)

    # we use 4 chains for sampling
    rng_key, init_key, warmup_key = jax.random.split(rng_key, 3)
    # init_keys = jax.random.split(init_key, num_chains)
    # init_params = jax.vmap(init_param_fn)(init_keys)

    (initial_states, tuned_params), _ = warmup.run(warmup_key, init_params, 1000)


    states, infos = run_inference_algorithm(
        rng_key=rng_key,
        initial_state_or_position=initial_states,
        inference_algorithm=blackjax.nuts,
        num_steps=num_samples,
        transform=lambda x: x.position,
    )

def gauss_nlogp(x, Sigma):
    """sigma = [Sigma[0, 0], Simga[1, 1], Sigma[1, 2]]"""
    det, H = invert_cov(Sigma)
    return 0.5 * x.T @ H @ x + jnp.log(2 * jnp.pi * jnp.sqrt(det))
    
def invert_cov(Sigma):
    det = Sigma[0] * Sigma[1] - Sigma[2]**2
    H = jnp.array([[Sigma[1], - Sigma[2]], [-Sigma[2], Sigma[0]]]) / det
    return det, H

def banana(x):
    a, b = 2., 0.2
    y = jnp.array([x[0]/a,  a*x[1] + a*b*(x[0]**2 + a**2) - 4.])
    
    return gauss_nlogp(y, jnp.array([1., 1., 0.5]))


def benchmark(model, sampler):
    n = 10000

    # model = gym.targets.IllConditionedGaussian()
    identity_fn = model.sample_transformations['identity']
    print('True mean', identity_fn.ground_truth_mean)
    print('True std', identity_fn.ground_truth_standard_deviation)




    # def logdensity_fn(x):
    #     return -0.5 * jnp.sum(jnp.square(x))

    logdensity_fn = model.unnormalized_log_prob
    # logdensity_fn = banana

    d = get_num_latents(model)
    # d = 100
    initial_position = jnp.zeros(d,)

    samples = (sampler(logdensity_fn, n, initial_position)) # num_chains=max(1000//d, d)
    print("Empirical mean", samples.mean(axis=0))
    print("Empirical std", samples.std(axis=0))

    # raise Exception

    
    # example usage
    
    # in reality we would generate the samples with some sampler
    # samples = jnp.square(jax.random.normal(jax.random.PRNGKey(42), shape = (n, d)))  
    f = cumulative_avg(samples)

    favg, fvar = identity_fn.ground_truth_mean, identity_fn.ground_truth_standard_deviation**2
    
    # error after using some number of samples
    err_t = err(favg, fvar, jnp.average)(f)
    
    # effective sample size
    ess_per_sample = ess(err_t, grad_evals_per_step=2)
    
    print("Effective sample size / sample: {0:.3}".format(ess_per_sample))
    return ess_per_sample

import matplotlib.pyplot as plt

# Define the models and samplers
models = [gym.targets.IllConditionedGaussian(), gym.targets.Banana()]
samplers = [run_mclmc]

# Create an empty list to store the results
results = []

# Run the benchmark for each model and sampler
for model in models:
    for sampler in samplers:
        result = benchmark(model, sampler)
        results.append(result)

# Create a list of colors for the samplers
colors = ['blue', 'red', 'green', 'orange']

results2 = [1,1]

# Create a barplot for results
plt.bar(range(len(results)), results, label='Results')

# Create a barplot for results2 with a different color
plt.bar(range(len(results2)), results2, label='Results2', color='orange')

# Add labels to the x-axis
plt.xticks(range(len(results)), ['Model 1 - Sampler 1', 'Model 2 - Sampler 1'])

# Add a title to the plot
plt.title('Benchmark Results')

# Add a legend to indicate the difference between results and results2
plt.legend()

# Show the plot
plt.show()


# Create a barplot of the results
# plt.bar(range(len(results)), results, label='1')
# plt.bar(range(len(results)), [1,1], label='2')



# Add labels to the x-axis
plt.xticks(range(len(results)), ['Model 1 - Sampler 1', 'Model 2 - Sampler 1'])

# Add a title to the plot
plt.title('Benchmark Results')

# Show the plot
plt.show()

