from collections import defaultdict
from inference_gym import using_jax as gym
import jax
import jax.numpy as jnp
import numpy as np
from sampling_algorithms import samplers
from inference_models import models
import blackjax
from blackjax.util import run_inference_algorithm


import matplotlib.pyplot as plt


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
    # print(len(err_t), "len err t")

    print("crossing", crossing)
    # print((err_t)[-100:], "le")
    
    return (neff / (crossing * grad_evals_per_step)) * cutoff_reached



def find_crossing(array, cutoff):
    """the smallest M such that array[m] < cutoff for all m > M"""

    def step(carry, element):
        """carry = (, 1 if (array[i] > cutoff for all i < current index) else 0"""
        above_threshold = element > cutoff
        # jax.debug.print("{x}", x=(carry))
        never_been_below = carry[1] * above_threshold  #1 if (array[i] > cutoff for all i < current index) else 0
        return (carry[0] + never_been_below, never_been_below), above_threshold

    state, track = jax.lax.scan(step, init=(0, 1), xs=array, length=len(array))
    

    return state[0]
    #return jnp.sum(track) #total number of indices for which array[m] < cutoff



def cumulative_avg(samples):
    return jnp.cumsum(samples, axis = 0) / jnp.arange(1, samples.shape[0] + 1)[:, None]



def benchmark(model, sampler):

    # print(find_crossing(jnp.array([0.4, 0.2, 0.3, 0.4, 0.5, 0.2, 0.2]), 0.3))
    # raise Exception

    n = 20000

    # model = gym.targets.IllConditionedGaussian()
    identity_fn = model.sample_transformations['identity']
    # print('True mean', identity_fn.ground_truth_mean)
    # print('True std', identity_fn.ground_truth_standard_deviation)

    logdensity_fn = model.unnormalized_log_prob
    # logdensity_fn = banana

    d = get_num_latents(model)
    # d = 100
    initial_position = jnp.zeros(d,)
    keys = jax.random.split(jax.random.PRNGKey(0), 2)

    samples = jnp.mean(jax.vmap(lambda pos, key: sampler(logdensity_fn, n, pos, key))(jnp.zeros((2, d)), keys), axis=0)
    # samples = sampler(logdensity_fn, n, initial_position, jax.random.PRNGKey(0))
    print("samples shape", samples.shape)
    # (sampler(logdensity_fn, n, initial_position)) # num_chains=max(1000//d, d)
    # print("Empirical mean", samples.mean(axis=0))
    # print("Empirical std", samples.std(axis=0))


    favg, fvar = identity_fn.ground_truth_mean, identity_fn.ground_truth_standard_deviation**2
    
    # error after using some number of samples
    err_t = err(favg, fvar, jnp.average)(cumulative_avg(samples))
    
    # effective sample size
    ess_per_sample = ess(err_t, grad_evals_per_step=2)
    
    print("Effective sample size / sample: {0:.3}".format(ess_per_sample))
    return ess_per_sample




# Define the models and samplers
# models = {'icg' : gym.targets.IllConditionedGaussian(), 'banana' : gym.targets.Banana()}

# Create an empty list to store the results
results = defaultdict(float)

# Run the benchmark for each model and sampler
for model in models:
    for sampler in samplers:
        result = benchmark(models[model], samplers[sampler])
        results[(model, sampler)] = result

print(results)

raise Exception

# Extract the models and samplers from the results dictionary
models = [model for model, _ in results.keys()]
samplers = [sampler for _, sampler in results.keys()]

# Extract the corresponding results
results_values = list(results.values())

# Create a figure with two subplots
fig, axs = plt.subplots(1, 2, figsize=(10, 5))

# Plot the results in the first subplot
axs[0].bar(range(len(results)), results_values)
axs[0].set_xticks(range(len(results)))
axs[0].set_xticklabels(['{} - {}'.format(model, sampler) for model, sampler in zip(models, samplers)], rotation=90)
axs[0].set_title('Benchmark Results')

# Plot the results in the second subplot
axs[1].bar(range(len(results)), results_values, color='orange')
axs[1].set_xticks(range(len(results)))
axs[1].set_xticklabels(['{} - {}'.format(model, sampler) for model, sampler in zip(models, samplers)], rotation=90)
axs[1].set_title('Benchmark Results')

# Adjust the layout of the subplots
plt.tight_layout()

# Show the plot
plt.show()



# # Create a list of colors for the samplers
# colors = ['blue', 'red', 'green', 'orange']

# results2 = [1,1]

# # Create a barplot for results
# plt.bar(range(len(results)), results, label='Results')

# # Create a barplot for results2 with a different color
# plt.bar(range(len(results2)), results2, label='Results2', color='orange')

# # Add labels to the x-axis
# plt.xticks(range(len(results)), ['Model 1 - Sampler 1', 'Model 2 - Sampler 1'])

# # Add a title to the plot
# plt.title('Benchmark Results')

# # Add a legend to indicate the difference between results and results2
# plt.legend()

