from collections import defaultdict
import itertools
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

def sampler_mhmclmc_with_tuning(step_size, L):

    def s(logdensity_fn, num_steps, initial_position, key):

        init_key, tune_key, key = jax.random.split(key, 3)

        initial_state = blackjax.mcmc.mhmclmc.init(
        position=initial_position, logdensity_fn=logdensity_fn, random_generator_arg=init_key
        )
        kernel = lambda rng_key, state, avg_num_integration_steps, step_size: blackjax.mcmc.mhmclmc.build_kernel(
                integrator=blackjax.mcmc.integrators.isokinetic_mclachlan,
                # integration_steps_fn = lambda key: jnp.round(jax.random.uniform(key) * rescale(avg_num_integration_steps + 0.5)), 
                integration_steps_fn = lambda key: avg_num_integration_steps, 
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
            # frac_tune2=0,
            # frac_tune3=0,
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
        # integration_steps_fn = lambda k: jnp.round(jax.random.uniform(k) * rescale(num_steps_per_traj+ 0.5)) ,
        integration_steps_fn = lambda k: num_steps_per_traj ,
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

        return out, num_steps_per_traj

    return s

def sampler_mhmclmc(step_size, L):

    def s(logdensity_fn, num_steps, initial_position, key):


       

        num_steps_per_traj = L/step_size
        alg = blackjax.mcmc.mhmclmc.mhmclmc(
        logdensity_fn=logdensity_fn,
        step_size=step_size,
        integration_steps_fn = lambda k: jnp.round(jax.random.uniform(k) * rescale(num_steps_per_traj+ 0.5)) ,
        # integration_steps_fn = lambda _ : 5,
        # integration_steps_fn = lambda key: jnp.ceil(jax.random.poisson(key, L/step_size )) ,

        )
        
        _, out, info = run_inference_algorithm(
        rng_key=key,
        initial_state_or_position=initial_position,
        inference_algorithm=alg,
        num_steps=num_steps, 
        transform=lambda x: x.position, 
        progress_bar=True)

        print(info.acceptance_rate.mean(), "acceptance probability\n\n\n\n")
        # print(out.var(axis=0), "acceptance probability")

        return out, num_steps_per_traj

    return s

results = defaultdict(float)

# Empirical mean [ 2.6572839e-05 -4.0523437e-06]
# Empirical std [0.07159886 0.07360378]

for model in ["Banana"]:
    # for step_size, L in itertools.product([16.866055/10], [16.866055]):
    for sampler in samplers:
        # result, bias = benchmark_chains(models[model], sampler_mhmclmc_with_tuning(step_size, L), n=1000000, batch=1)
        result, bias = benchmark_chains(models[model], samplers[sampler], n=10000, batch=200, favg= jnp.array([100.0, 19.0]), fvar =jnp.array([20000.0, 4600.898]))
        # result, bias = benchmark_chains(models[model], sampler_mhmclmc(1e-2, 2), n=10000, batch=1, favg= jnp.array([100.0, 19.0]), fvar =jnp.array([20000.0, 4600.898]))
        # result, bias = benchmark_chains(models[model], samplers["mhmclmc"], n=1000000, batch=10)
        results[(model, sampler)] = result.item()
print(results)

# for model in ["simple"]:
#     for sampler in ["mhmclmc", "mclmc"]:
#         # result, bias = benchmark_chains(models[model], sampler_mhmclmc_with_tuning(step_size, L), n=1000000, batch=1)
#         # result, bias = benchmark_chains(models[model], samplers["mhmclmc"], n=1000000, batch=10)
#         result, bias = benchmark_chains(models[model], samplers[sampler], n=100000, batch=1)
        
#         results[(model, sampler)] = result, bias
# print(results)
