from collections import defaultdict
import itertools
import operator
import jax
import numpy as np

from benchmark import benchmark_chains, cumulative_avg, err, calculate_ess, get_num_latents, grads_to_low_error, gridsearch_tune, run_mhmclmc_no_tuning
import blackjax
from blackjax.adaptation.mclmc_adaptation import MCLMCAdaptationState
from blackjax.mcmc.integrators import calls_per_integrator_step, mclachlan_coefficients
from blackjax.mcmc.mhmclmc import rescale
from blackjax.util import run_inference_algorithm
import jax.numpy as jnp 
from sampling_algorithms import run_mclmc, run_mhmclmc, samplers
from inference_models import Brownian, IllConditionedGaussian, models
from blackjax.adaptation.mclmc_adaptation import MCLMCAdaptationState, target_acceptance_rate_of_order


def sampler_mhmclmc_with_tuning(step_size, L, frac_tune2, frac_tune3):

    def s(logdensity_fn, num_steps, initial_position, transform, key):

        init_key, tune_key, key = jax.random.split(key, 3)

        initial_state = blackjax.mcmc.mhmclmc.init(
        position=initial_position, logdensity_fn=logdensity_fn, random_generator_arg=init_key
        )
        integrator = blackjax.mcmc.integrators.isokinetic_mclachlan
        kernel = lambda rng_key, state, avg_num_integration_steps, step_size: blackjax.mcmc.mhmclmc.build_kernel(
                integrator=integrator,
                integration_steps_fn = lambda key : jnp.ceil(jax.random.uniform(key) * rescale(avg_num_integration_steps)),
                # integration_steps_fn = lambda key: avg_num_integration_steps, 
            )(
                rng_key=rng_key, 
                state=state, 
                step_size=step_size, 
                logdensity_fn=logdensity_fn)

        # jax.debug.print("params before tuning {x}", x=MCLMCAdaptationState(L=L, step_size=step_size))
        (
            blackjax_state_after_tuning,
            blackjax_mclmc_sampler_params,
        ) = blackjax.adaptation.mclmc_adaptation.mhmclmc_find_L_and_step_size(
            mclmc_kernel=kernel,
            num_steps=num_steps,
            state=initial_state,
            rng_key=tune_key,
            target=target_acceptance_rate_of_order[mhmclmc_integrator_order[integrator]],
            frac_tune2=frac_tune2,
            frac_tune3=frac_tune3,
            params=MCLMCAdaptationState(L=L, step_size=step_size, std_mat=1.)
        )

        # jax.debug.print("params {x}", x=blackjax_mclmc_sampler_params)
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
        transform=lambda x: transform(x.position), 
        progress_bar=True)

        print(info.acceptance_rate.mean(), "acceptance probability\n\n\n\n")
        # print(out.var(axis=0), "acceptance probability")

        return out, blackjax_mclmc_sampler_params, num_steps_per_traj * calls_per_integrator_step(coefficients)

    return s



# Empirical mean [ 2.6572839e-05 -4.0523437e-06]
# Empirical std [0.07159886 0.07360378]



def grid_search(n, model):
    

    print(f"\nModel: {model}")

    

    if True:
        batch = 10
        init_key, sample_key = jax.random.split(jax.random.PRNGKey(1), 2)
        init_keys = jax.random.split(init_key, batch)
        init_pos = jax.vmap(model.sample_init)(init_keys)
        sample_keys = jax.random.split(sample_key, batch)
                
        avg_num_steps_per_traj = 2
        samples, params, _ = jax.vmap(lambda pos, key: samplers["mclmc"](mclachlan_coefficients, model.logdensity_fn, n*10, pos, model.transform, key))(init_pos, sample_keys)

        # avg_num_steps_per_traj = 1
        # samples, params, _ = jax.vmap(lambda pos, key: samplers["nuts"](model.logdensity_fn, 1000, pos, model.transform, key))(init_pos, sample_keys)
        
        full = lambda arr : err(model.E_x2, model.Var_x2, jnp.max)(cumulative_avg(arr))
        err_t = jnp.mean(jax.vmap(full)(samples**2), axis=0)
        
        ess_val, grads_to_low_error, _ = calculate_ess(err_t, avg_num_steps_per_traj)
        print(ess_val, grads_to_low_error)



    center_L, center_step_size = params.L.mean(), params.step_size.mean()
    print(f"initial params found by MCLMC are step size {center_step_size} and L {center_L}, with grad calls {grads_to_low_error}")
    # center_L, center_step_size = 0.5755017, 0.7676609
    # center_L, center_step_size = 0.7639537453651428, 0.5901154279708862
    # center_L, center_step_size = 2.079161234577297, 0.3441933917149635
    # center_L, center_step_size = 1.3701616525650024, 0.44564130902290344
    print(f"Initial params hard coded as L {center_L} and step size as {center_step_size}")

    # nuts result

    
    # print("\nBeginning grid search:\n")
    # grid_size = 5
    # batch = 100
    # best params on iteration 0 are stepsize 5.103655551427525 and L 5.408820389035896 with Grad Calls until Convergence 216.19784545898438
    # center_L, center_step_size = gridsearch_tune(iterations=0, grid_size=grid_size, model=model, sampler=sampler_mhmclmc, batch=batch, num_steps=n, center_L=center_L, center_step_size=center_step_size)


    tune_key, init_key, init_pos_key, run_key = jax.random.split(jax.random.PRNGKey(0), 4)
    initial_position = model.sample_init(init_pos_key)

    initial_state = blackjax.mcmc.mhmclmc.init(
    position=initial_position, logdensity_fn=model.logdensity_fn, random_generator_arg=init_key
    )

    kernel = lambda rng_key, state, avg_num_integration_steps, step_size: blackjax.mcmc.mhmclmc.build_kernel(
                integrator=blackjax.mcmc.integrators.isokinetic_mclachlan,
                integration_steps_fn = lambda k : jnp.ceil(jax.random.uniform(k) * rescale(avg_num_integration_steps))
            )(
                rng_key=rng_key, 
                state=state, 
                step_size=step_size, 
                logdensity_fn=model.logdensity_fn)

    (
        blackjax_state_after_tuning,
        blackjax_mclmc_sampler_params,
    ) = blackjax.adaptation.mclmc_adaptation.mhmclmc_find_L_and_step_size(
        mclmc_kernel=kernel,
        num_steps=n,
        state=initial_state,
        rng_key=tune_key,
        target=0.65,
        frac_tune3=0,
        frac_tune2=0,
        params = MCLMCAdaptationState(L=center_L, step_size=center_step_size, std_mat=1.),
        # params = MCLMCAdaptationState(L=10., step_size=3.3454525677773526, std_mat=1.),
        # params = MCLMCAdaptationState(L=16., step_size=1., std_mat=1.),
        # params = MCLMCAdaptationState(L=10., step_size=5.103655551427525, std_mat=1.),
    )
    
    print(f"initial params are L {center_L} and step_size {center_step_size}")
    print(f"params found by mhmclmc tuning are L {blackjax_mclmc_sampler_params.L} and step_size {blackjax_mclmc_sampler_params.step_size}")


    ess, grad_calls_until_convergence = benchmark_chains(model, run_mhmclmc_no_tuning(coefficients=mclachlan_coefficients, step_size=blackjax_mclmc_sampler_params.step_size, L=blackjax_mclmc_sampler_params.L), jax.random.PRNGKey(0), n=n, batch = batch, contract=jnp.max) # batch=1000//model.ndims)
    print(f"ess from tuning is {ess} and num grad calls is {grad_calls_until_convergence}")
    
    
    # # L, step_size = 2.6195790055693493, 0.4336564994563942
    # L, step_size = blackjax_mclmc_sampler_params.L, blackjax_mclmc_sampler_params.step_size
    # ess, grad_calls_until_convergence = benchmark_chains(model, sampler_mhmclmc(step_size=step_size, L=L),  keys[-1], n=n, batch = 200) # batch=1000//model.ndims)
    # print("final grads", grad_calls_until_convergence)

    # step_size = blackjax_mclmc_sampler_params.step_size
    # L = blackjax_mclmc_sampler_params.L

    # jax.debug.print("{x} num_steps, L, step_size", x=(jnp.ceil(L/step_size), L, step_size))


    # alg = blackjax.mcmc.mhmclmc.mhmclmc(
    #     logdensity_fn=model.logdensity_fn,
    #     step_size=step_size,
    #     integration_steps_fn = lambda key: jnp.round(jax.random.uniform(key) * rescale(L/step_size + 0.5)) ,
    #     # integrator=integrator,
    #     # integration_steps_fn = lambda key: jnp.ceil(jax.random.poisson(key, L/step_size )) ,

    # )

    # _, out, info = run_inference_algorithm(
    #     rng_key=run_key,
    #     initial_state_or_position=blackjax_state_after_tuning,
    #     inference_algorithm=alg,
    #     num_steps=num_steps, 
    #     transform=lambda x: transform(x.position), 
    #     progress_bar=True)
    

    

    # return results


if __name__ == "__main__":


    # for i in range(100):
    #     ess, grad_calls_until_convergence = benchmark_chains(Brownian(), sampler_mhmclmc(step_size=0.4336564994563942, L=2.6195790055693493),  jax.random.PRNGKey(i), n=10000, batch = 10) # batch=1000//model.ndims)

    #     print (f"ess from tuning is {ess} and num grad calls is {grad_calls_until_convergence}")
    # raise Exception




    # for i in range(100):

    #     # ess, grad_calls_until_convergence, _ = benchmark_chains(Brownian(), sampler_mhmclmc_with_tuning(step_size=0.1, L=40.8, frac_tune2=0.1, frac_tune3=0),  jax.random.PRNGKey(i), n=50000, batch = 10) # batch=1000//model.ndims)

    #     ess, grad_calls_until_convergence, _ = benchmark_chains(Brownian(), sampler_mhmclmc(step_size=0.5901154279708862, L=0.7639537453651428),  jax.random.PRNGKey(i), n=10000, batch = 1) # batch=1000//model.ndims)
    #     print(f"ess from tuning is {ess} and num grad calls is {grad_calls_until_convergence}")
    #     print(f"L from ESS (0.4 * step_size/ESS): {0.4 * 0.4/ess}")

    # model=IllConditionedGaussian(10, 2)
    # ess, grad_calls_until_convergence = benchmark_chains(model, run_mclmc, n=2500, batch =10) # batch=1000//model.ndims)
    # print(ess)
    # raise Exception

#     benchmarks(5000)

    # grid_search(n=2500, model=IllConditionedGaussian(10, 2))
    grid_search(n=10000, model=Brownian())
    # grid_search(n=2500, model='icg')
    # grid_search(n=2500, model='normal')

    # m = models['icg']
    # initial_position = m.sample(jax.random.PRNGKey(0))
    # _, blackjax_mclmc_sampler_params, _ = sampler_mhmclmc_with_tuning(L=4.291135699906666, step_size=1.005, frac_tune2=0, frac_tune3=0)(lambda x: -m.nlogp(x), 100000, initial_position, jax.random.PRNGKey(0))
    # print(blackjax_mclmc_sampler_params)

    # out = benchmark_chains(models['icg'], sampler_mhmclmc(step_size=4.475385912886005, L=2.2708939161637853), n=100, batch=10,favg=models['icg'].E_x2, fvar=models['icg'].Var_x2)
    # print(out)
    # pass
# print(grid_search())

# for model in ["simple"]:
#     for sampler in ["mhmclmc", "mclmc"]:
#         # result, bias = benchmark_chains(model, sampler_mhmclmc_with_tuning(step_size, L), n=1000000, batch=1)
#         # result, bias = benchmark_chains(models[model], samplers["mhmclmc"], n=1000000, batch=10)
#         result, bias = benchmark_chains(models[model], samplers[sampler], n=100000, batch=1)
        
#         results[(model, sampler)] = result, bias
# print(results)


# 0.3441933917149635 and L 2.079161234577297