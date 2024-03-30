from collections import defaultdict
from functools import partial
import math
import os
from statistics import mean, median
import jax
import jax.numpy as jnp

# os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=' + str(128)
# num_cores = jax.local_device_count()
# print(num_cores, jax.lib.xla_bridge.get_backend().platform)

import itertools

import numpy as np

from blackjax.benchmarks.mcmc.sampling_algorithms import samplers
from blackjax.benchmarks.mcmc.inference_models import StandardNormal, models
from blackjax.mcmc.integrators import generate_euclidean_integrator, generate_isokinetic_integrator, isokinetic_mclachlan, mclachlan_coefficients, name_integrator, omelyan_coefficients, velocity_verlet, velocity_verlet_coefficients, yoshida_coefficients



def get_num_latents(target):
  return target.ndims
#   return int(sum(map(np.prod, list(jax.tree_flatten(target.event_shape)[0]))))


def err(f_true, var_f, contract):
    """Computes the error b^2 = (f - f_true)^2 / var_f
        Args:
            f: E_sampler[f(x)], can be a vector
            f_true: E_true[f(x)]
            var_f: Var_true[f(x)]
            contract: how to combine a vector f in a single number, can be for example jnp.average or jnp.max
            
        Returns:
            contract(b^2)
    """    
    
    return jax.vmap(lambda f: contract(jnp.square(f - f_true) / var_f))



def grads_to_low_error(err_t, grad_evals_per_step= 1, low_error= 0.01):
    """Uses the error of the expectation values to compute the effective sample size neff
        b^2 = 1/neff"""
    
    cutoff_reached = err_t[-1] < low_error
    return find_crossing(err_t, low_error) * grad_evals_per_step, cutoff_reached
    
        
def calculate_ess(err_t, grad_evals_per_step, neff= 100):
    
    grads_to_low, cutoff_reached = grads_to_low_error(err_t, grad_evals_per_step, 1./neff)
    
    return (neff / grads_to_low) * cutoff_reached, grads_to_low*(1/cutoff_reached), cutoff_reached


def find_crossing(array, cutoff):
    """the smallest M such that array[m] < cutoff for all m > M"""

    b = array > cutoff
    indices = jnp.argwhere(b)
    if indices.shape[0] == 0:
        print("\n\n\nNO CROSSING FOUND!!!\n\n\n", array, cutoff)
        return 1

    return jnp.max(indices)+1


def cumulative_avg(samples):
    return jnp.cumsum(samples, axis = 0) / jnp.arange(1, samples.shape[0] + 1)[:, None]



def benchmark_chains(model, sampler, key, n=10000, batch=None, contract = jnp.average,):

    
    d = get_num_latents(model)
    if batch is None:
        batch = np.ceil(1000 / d).astype(int)
    key, init_key = jax.random.split(key, 2)
    keys = jax.random.split(key, batch)

    init_keys = jax.random.split(init_key, batch)
    init_pos = jax.vmap(model.sample_init)(init_keys)

    # samples, params, avg_num_steps_per_traj = jax.pmap(lambda pos, key: sampler(model.logdensity_fn, n, pos, model.transform, key))(init_pos, keys)
    samples, params, grad_calls_per_traj = jax.vmap(lambda pos, key: sampler(logdensity_fn=model.logdensity_fn, num_steps=n, initial_position= pos,transform= model.transform, key=key))(init_pos, keys)
    # avg_grad_calls_per_traj = jnp.mean(jnp.where(jnp.isnan(grad_calls_per_traj), 1, grad_calls_per_traj), axis=0)
    avg_grad_calls_per_traj = jnp.nanmean(grad_calls_per_traj, axis=0)
    try:
        print(jnp.nanmean(params.step_size,axis=0), jnp.nanmean(params.L,axis=0))
    except: pass
    # print("grad calls", avg_grad_calls_per_traj)
    
    full = lambda arr : err(model.E_x2, model.Var_x2, contract)(cumulative_avg(arr))
    # err_t = jnp.mean(jax.vmap(full)(samples**2), axis=0)
    err_t = jax.vmap(full)(samples**2)
    err_t_median = jnp.median(err_t, axis=0)
    # raise Exception
    # print(err_t.shape)
    # foo = jax.vmap(lambda x: calculate_ess(x, grad_evals_per_step=avg_grad_calls_per_traj))(err_t)
    # print(foo.shape)
    # outs = [calculate_ess(b, grad_evals_per_step=avg_grad_calls_per_traj) for b in err_t]
    # print(outs[:10])
    # esses = [i[0].item() for i in outs if not math.isnan(i[0].item())]
    # grad_calls = [i[1].item() for i in outs if not math.isnan(i[1].item())]
    # print(grad_calls)
    # raise Exception

    esses, grad_calls, _ = calculate_ess(err_t_median, grad_evals_per_step=avg_grad_calls_per_traj)

    # print(mean(esses), median(esses))
    # print(mean(grad_calls), median(grad_calls))
    
    # return grads_to_low_error(err_t, avg_grad_calls_per_traj)[0]
    # ess_per_sample = calculate_ess(err_t, grad_evals_per_step=avg_grad_calls_per_traj)
    return esses, grad_calls
    # , err_t[-1], params




def run_benchmarks(batch_size):

    results = defaultdict(tuple)

    sampling_algs = ["mhmclmc", "nuts", "mclmc", ]
    coeffs = [velocity_verlet_coefficients, mclachlan_coefficients, yoshida_coefficients, omelyan_coefficients]
    models = [StandardNormal(d) for d in np.ceil(np.logspace(np.log10(10), np.log10(100000), 5)).astype(int)]

    # for model, sampler in itertools.product(models, sampling_algs):
    for variables in itertools.product(models, sampling_algs, coeffs):

        model, sampler, coefficients = variables
        print(f"\nModel: {model.name}, Sampler: {sampler}\n Coefficients: {coefficients}\n") 
        # sampler_to_integrator_type = {
        #     "mclmc": generate_isokinetic_integrator,
        #     "mhmclmc": generate_isokinetic_integrator,
        #     "nuts": generate_euclidean_integrator,
        # }

        # Model = models[model][0]
        key = jax.random.PRNGKey(2)
        for i in range(1):
            key1, key = jax.random.split(key)
            # integrator = sampler_to_integrator_type[sampler](coefficients)
            ess, grad_calls = benchmark_chains(model, partial(samplers[sampler], coefficients=coefficients),key1, n=1000, batch=1 + 1000//model.ndims, contract=jnp.average)
            #print(f"ESS: {result.item()}")
            print(f"grads to low bias: {grad_calls}")
            # results.append(result[1])
            results[((model.name, model.ndims), sampler, name_integrator(coefficients))] = (ess, grad_calls) 

        # import matplotlib.pyplot as plt

        # # ... existing code ...

        # # Plot the second_elements in a scatterplot
        # plt.scatter([0.5]*len(results), results)
        # plt.xlabel("Iteration")
        # plt.ylabel("Second Element of Results")
        # plt.title("Scatterplot of Second Element of Results")
        # plt.savefig("scatterplot_mclmc.png")  # Save the plot as scatterplot.png
        # plt.show()
            
    print(results)
            
    import pandas as pd

    df = pd.Series(results).reset_index()
    df.columns = ["model", "sampler", "coeffs", "result"] 
    df.result = df.result.apply(lambda x: x[1].item())
    df.model = df.model.apply(lambda x: x[1])
    df.to_csv("results.csv", index=False)

    return results

if __name__ == "__main__":
    run_benchmarks(batch_size=100)


