from collections import defaultdict
import itertools
import operator
import jax
import numpy as np

from benchmark import benchmark_chains, cumulative_avg, err, ess, get_num_latents
import blackjax
from blackjax.adaptation.mclmc_adaptation import MCLMCAdaptationState
from blackjax.mcmc.mhmclmc import rescale
from blackjax.util import run_inference_algorithm
import jax.numpy as jnp 
from sampling_algorithms import samplers
from inference_models import models

def sampler_mhmclmc_with_tuning(step_size, L, frac_tune2, frac_tune3):

    def s(logdensity_fn, num_steps, initial_position, key):

        init_key, tune_key, key = jax.random.split(key, 3)

        initial_state = blackjax.mcmc.mhmclmc.init(
        position=initial_position, logdensity_fn=logdensity_fn, random_generator_arg=init_key
        )
        kernel = lambda rng_key, state, avg_num_integration_steps, step_size: blackjax.mcmc.mhmclmc.build_kernel(
                integrator=blackjax.mcmc.integrators.isokinetic_mclachlan,
                integration_steps_fn = lambda key : jnp.ceil(jax.random.uniform(key) * rescale(avg_num_integration_steps)),
                # integration_steps_fn = lambda key: avg_num_integration_steps, 
            )(
                rng_key=rng_key, 
                state=state, 
                step_size=step_size, 
                logdensity_fn=logdensity_fn)

        jax.debug.print("params before tuning {x}", x=MCLMCAdaptationState(L=L, step_size=step_size))
        (
            blackjax_state_after_tuning,
            blackjax_mclmc_sampler_params,
        ) = blackjax.adaptation.mclmc_adaptation.mhmclmc_find_L_and_step_size(
            mclmc_kernel=kernel,
            num_steps=num_steps,
            state=initial_state,
            rng_key=tune_key,
            frac_tune2=frac_tune2,
            frac_tune3=frac_tune3,
            params=MCLMCAdaptationState(L=L, step_size=step_size)
        )

        jax.debug.print("params {x}", x=blackjax_mclmc_sampler_params)
        # jax.debug.print("acceptance rate {x}", x=blackjax_mclmc_sampler_params)

        # L = blackjax_mclmc_sampler_params.L
        # step_size = blackjax_mclmc_sampler_params.step_size

        num_steps_per_traj = blackjax_mclmc_sampler_params.L/blackjax_mclmc_sampler_params.step_size
        alg = blackjax.mcmc.mhmclmc.mhmclmc(
        logdensity_fn=logdensity_fn,
        step_size=blackjax_mclmc_sampler_params.step_size,
        integration_steps_fn = lambda k : jnp.ceil(jax.random.uniform(k) * rescale(num_steps_per_traj)) ,
        # integration_steps_fn = lambda k: num_steps_per_traj ,
        # integration_steps_fn = lambda _ : 5,
        # integration_steps_fn = lambda key: jnp.ceil(jax.random.poisson(key, L/step_size )) ,

        )
        
        _, out, info = run_inference_algorithm(
        rng_key=key,
        initial_state_or_position=blackjax_state_after_tuning,
        inference_algorithm=alg,
        num_steps=num_steps, 
        transform=lambda x: x.position, 
        progress_bar=True)

        print(info.acceptance_rate.mean(), "acceptance probability\n\n\n\n")
        # print(out.var(axis=0), "acceptance probability")

        return out, blackjax_mclmc_sampler_params, num_steps_per_traj

    return s

def sampler_mhmclmc(step_size, L):

    def s(logdensity_fn, num_steps, initial_position, key):

        num_steps_per_traj = L/step_size
        alg = blackjax.mcmc.mhmclmc.mhmclmc(
        logdensity_fn=logdensity_fn,
        step_size=step_size,
        integration_steps_fn = lambda k : jnp.ceil(jax.random.uniform(k) * rescale(num_steps_per_traj)) ,
        )
        
        
        _, out, info = run_inference_algorithm(
        rng_key=key,
        initial_state_or_position=initial_position,
        inference_algorithm=alg,
        num_steps=num_steps, 
        transform=lambda x: x.position, 
        progress_bar=False)

        # print(info.acceptance_rate.mean(), "acceptance probability\n\n\n\n")
        # print(out.var(axis=0), "acceptance probability")

        return out, MCLMCAdaptationState(L=L, step_size=step_size), num_steps_per_traj

    return s


# Empirical mean [ 2.6572839e-05 -4.0523437e-06]
# Empirical std [0.07159886 0.07360378]

def grid_search(n=10000, model='icg'):

    print(f"\nModel: {model}")

    results = defaultdict(float)
            
    # result, _, center_L, center_step_size = benchmark_chains(models[model], samplers["mclmc"], n=100000, batch=10, favg=models[model].E_x2, fvar=models[model].Var_x2)

    print(f"initial params found by MCLMC {center_step_size, center_L} (with ESS {result.item()})")
    
    print("\nBeginning grid search:\n")
    for i in range(3):
        for step_size, L in itertools.product(np.logspace(np.log10(center_step_size/2), np.log10(center_step_size*2), 9), np.logspace(np.log10(center_L/2), np.log10(center_L*2),9)):
        
        # result, bias = benchmark_chains(models[model], sampler_mhmclmc_with_tuning(step_size, L), n=n00, batch=1)
        # result, bias = benchmark_chains(models[model], samplers['mclmc'], n=n00, batch=10, favg=models[model].E_x2, fvar=models[model].Var_x2)
        # results[(model, "mclmc")] = result.item()


        # result, bias = benchmark_chains(models[model], sampler_mhmclmc_with_tuning(jnp.sqrt(models[model].ndims)/4, jnp.sqrt(models[model].ndims), frac_tune2=0.1, frac_tune3=0.1), n=n, batch=10,favg=models[model].E_x2, fvar=models[model].Var_x2)
        # result, bias = benchmark_chains(models[model], sampler_mhmclmc_with_tuning(step_size=3.4392192, L=2.7043579, frac_tune2=0.1, frac_tune3=0.1), n=n, batch=10,favg=models[model].E_x2, fvar=models[model].Var_x2)
            
            result, bias, _, _ = benchmark_chains(models[model], sampler_mhmclmc(step_size=step_size, L=L), n=n, batch=100,favg=models[model].E_x2, fvar=models[model].Var_x2)
            # result, bias = benchmark_chains(models[model], samplers[sampler], n=10000, batch=200, favg=models[model].E_x2, fvar=models[model].Var_x2)
            results[(step_size, L)] = result.item()
        
        val, (step_size, L) = max([(results[r], r) for r in results], key=operator.itemgetter(0))

        center_L, center_step_size = L, step_size

        print(f"best params on iteration {i} are {step_size, L} with ESS {val}")


        # for step_size, L in make_grid(center_L=21.48713, center_step_size= 2.2340074):
        #     result, bias = benchmark_chains(models[model], sampler_mhmclmc(step_size=step_size, L=L), n=10000, batch=100,favg=models[model].E_x2, fvar=models[model].Var_x2)
        #     # result, bias = benchmark_chains(models[model], samplers["mhmclmc"], n=1000000, batch=10)
            # results[(model, "mhmclmc")] = result.item()

    # for step in grid:



    return results

def make_grid(center_L, center_step_size):
    return itertools.product(np.linspace(center_step_size-1, center_step_size+1, 10), np.linspace(center_L-1, center_L+1, 10))

if __name__ == "__main__":

    # grid_search(n=2500, model='banana')
    # grid_search(n=2500, model='icg')
    # grid_search(n=2500, model='normal')

    out = benchmark_chains(models['icg'], sampler_mhmclmc(step_size=4.475385912886005, L=2.2708939161637853), n=10000, batch=100,favg=models['icg'].E_x2, fvar=models['icg'].Var_x2)
    # print(out)
    # pass
# print(grid_search())

# for model in ["simple"]:
#     for sampler in ["mhmclmc", "mclmc"]:
#         # result, bias = benchmark_chains(models[model], sampler_mhmclmc_with_tuning(step_size, L), n=1000000, batch=1)
#         # result, bias = benchmark_chains(models[model], samplers["mhmclmc"], n=1000000, batch=10)
#         result, bias = benchmark_chains(models[model], samplers[sampler], n=100000, batch=1)
        
#         results[(model, sampler)] = result, bias
# print(results)
