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
        # print(bsq.shape, "shape ASDFADSF \n\n")
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

    indices = jnp.argwhere(array > cutoff)
    return jnp.max(indices)+1

def cumulative_avg(samples):
    return jnp.cumsum(samples, axis = 0) / jnp.arange(1, samples.shape[0] + 1)[:, None]



def benchmark(model, sampler):

    # print(find_crossing(jnp.array([0.4, 0.2, 0.3, 0.4, 0.5, 0.2, 0.2]), 0.3))
    # print(cumulative_avg(jnp.array([[1., 2.], [1.,2.]]).T))
    # raise Exception

    n = 20000

    identity_fn = model.sample_transformations['identity']
    # print('True mean', identity_fn.ground_truth_mean)
    # print('True std', identity_fn.ground_truth_standard_deviation)
    # print("Empirical mean", samples.mean(axis=0))
    # print("Empirical std", samples.std(axis=0))

    logdensity_fn = model.unnormalized_log_prob
    d = get_num_latents(model)
    initial_position = jnp.zeros(d,)
    samples = sampler(logdensity_fn, n, initial_position, jax.random.PRNGKey(0))
    # print(samples[-1], samples[0], "samps", samples.shape)

    favg, fvar = identity_fn.ground_truth_mean, identity_fn.ground_truth_standard_deviation**2
    err_t = err(favg, fvar, jnp.average)(cumulative_avg(samples))
    # print(err_t[-1], "benchmark err_t[0]")
    ess_per_sample = ess(err_t, grad_evals_per_step=2)
    
    return ess_per_sample

def benchmark_chains(model, sampler):

    n = 100000

    identity_fn = model.sample_transformations['identity']
    logdensity_fn = model.unnormalized_log_prob
    d = get_num_latents(model)
    batch = np.ceil(1000 / d).astype(int)
    keys = jax.random.split(jax.random.PRNGKey(1), batch)
    # keys = jnp.array([jax.random.PRNGKey(0)])

    samples = jax.vmap(lambda pos, key: sampler(logdensity_fn, n, pos, key))(jnp.zeros((batch, d)), keys)
    # print(samples[0][-1], samples[0][0], "samps chain", samples.shape)
    favg, fvar = identity_fn.ground_truth_mean, identity_fn.ground_truth_standard_deviation**2
    full = lambda arr : err(favg, fvar, jnp.average)(cumulative_avg(arr))
    err_t = jnp.mean(jax.vmap(full)(samples), axis=0)
    # err_t = jax.vmap(full)(samples)[1]
    # print(err_t[-1], "benchmark chains err_t[0]")
    ess_per_sample = ess(err_t, grad_evals_per_step=2)
    
    return ess_per_sample




# Define the models and samplers
# models = {'icg' : gym.targets.IllConditionedGaussian(), 'banana' : gym.targets.Banana()}

# Create an empty list to store the results
results = defaultdict(float)

# Run the benchmark for each model and sampler
for model in models:
    for sampler in samplers:
        # result = benchmark(models[model], samplers[sampler])
        result = benchmark_chains(models[model], samplers[sampler])
        # print(result, result2, "results")
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

