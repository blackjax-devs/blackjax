import os
import jax
import jax.numpy as jnp

# os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=' + str(128)
# num_cores = jax.local_device_count()
# print(num_cores, jax.lib.xla_bridge.get_backend().platform)

import itertools

import numpy as np

from blackjax.benchmarks.mcmc.sampling_algorithms import samplers
from blackjax.benchmarks.mcmc.inference_models import models



def get_num_latents(target):
  return target.ndims
#   return int(sum(map(np.prod, list(jax.tree_flatten(target.event_shape)[0]))))


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
    
    return jax.vmap(lambda f: contract(jnp.square(f - f_true) / var_f))



def grads_to_low_error(err_t, grad_evals_per_step= 1, low_error= 0.01):
    """Uses the error of the expectation values to compute the effective sample size neff
        b^2 = 1/neff"""
    
    cutoff_reached = err_t[-1] < low_error
    return find_crossing(err_t, low_error) * grad_evals_per_step, cutoff_reached
    
        
def calculate_ess(err_t, grad_evals_per_step, neff= 100):
    
    grads_to_low, cutoff_reached = grads_to_low_error(err_t, grad_evals_per_step, 1./neff)
    
    return (neff / grads_to_low) * cutoff_reached, grads_to_low*(1/cutoff_reached)


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



def benchmark_chains(model, sampler, n=10000, batch=None, contract = jnp.average):

    
    d = get_num_latents(model)
    if batch is None:
        batch = np.ceil(1000 / d).astype(int)
    key, init_key = jax.random.split(jax.random.PRNGKey(44), 2)
    keys = jax.random.split(key, batch)

    init_keys = jax.random.split(init_key, batch)
    init_pos = jax.vmap(model.sample_init)(init_keys)

    # samples, params, avg_num_steps_per_traj = jax.pmap(lambda pos, key: sampler(model.logdensity_fn, n, pos, model.transform, key))(init_pos, keys)
    samples, params, grad_calls_per_traj = jax.vmap(lambda pos, key: sampler(model.logdensity_fn, n, pos, model.transform, key))(init_pos, keys)
    avg_grad_calls_per_traj = jnp.mean(grad_calls_per_traj, axis=0)
    
    full = lambda arr : err(model.E_x2, model.Var_x2, contract)(cumulative_avg(arr))
    err_t = jnp.mean(jax.vmap(full)(samples**2), axis=0)
    
    return grads_to_low_error(err_t, avg_grad_calls_per_traj)[0]
    ess_per_sample = calculate_ess(err_t, grad_evals_per_step=avg_grad_calls_per_traj)
    return ess_per_sample
    # , err_t[-1], params




def run_benchmarks():

    for model, sampler in itertools.product(models, samplers):

        print(f"\nModel: {model}, Sampler: {sampler}\n")

        Model = models[model][0]
        result = benchmark_chains(Model, samplers[sampler], n=models[model][1][sampler], batch=200)
        #print(f"ESS: {result.item()}")
        print(f"grads to low bias: " + str(result[1]))


if __name__ == "__main__":
    
    run_benchmarks()


