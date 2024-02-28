

import jax
import jax.numpy as jnp
import blackjax
from blackjax.mcmc.mhmclmc import rescale
from blackjax.util import run_inference_algorithm
import blackjax

__all__ = ["samplers"]




def run_nuts(
    logdensity_fn, num_steps, initial_position, key):
    
    warmup = blackjax.window_adaptation(blackjax.nuts, logdensity_fn)

    # we use 4 chains for sampling
    rng_key, warmup_key = jax.random.split(key, 2)


    (state, params), _ = warmup.run(warmup_key, initial_position, num_steps // 10)

    nuts = blackjax.nuts(logdensity_fn=logdensity_fn, step_size=params['step_size'], inverse_mass_matrix= params['inverse_mass_matrix'])

    final_state, state_history, info_history = run_inference_algorithm(
        rng_key=rng_key,
        initial_state_or_position=state,
        inference_algorithm=nuts,
        num_steps=num_steps,
        transform=lambda x: x.position,
    )

    # print("INFO\n\n",info_history.num_integration_steps)

    return state_history, params, info_history.num_integration_steps.mean() 

def run_mclmc(logdensity_fn, num_steps, initial_position, key):
    init_key, tune_key, run_key = jax.random.split(key, 3)

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

    # jax.debug.print("params {x}", x=blackjax_mclmc_sampler_params)

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

    avg_steps_per_traj = 1
    return samples, blackjax_mclmc_sampler_params, avg_steps_per_traj


def run_mhmclmc(logdensity_fn, num_steps, initial_position, key):


    init_key, tune_key, run_key = jax.random.split(key, 3)


    initial_state = blackjax.mcmc.mhmclmc.init(
        position=initial_position, logdensity_fn=logdensity_fn, random_generator_arg=init_key
    )

    kernel = lambda rng_key, state, avg_num_integration_steps, step_size: blackjax.mcmc.mhmclmc.build_kernel(
                integrator=blackjax.mcmc.integrators.isokinetic_mclachlan,
                integration_steps_fn = lambda k : jnp.ceil(jax.random.uniform(k) * rescale(avg_num_integration_steps))
            )(
                rng_key=rng_key, 
                state=state, 
                step_size=step_size, 
                logdensity_fn=logdensity_fn)

    (
        blackjax_state_after_tuning,
        blackjax_mclmc_sampler_params,
    ) = blackjax.adaptation.mclmc_adaptation.mhmclmc_find_L_and_step_size(
        mclmc_kernel=kernel,
        num_steps=num_steps,
        state=initial_state,
        rng_key=tune_key,
        # frac_tune2=0,
        frac_tune3=0,
        # params=MCLMCAdaptationState(L=16.765137, step_size=1.005)
    )

    # raise Exception


    # step_size = 1.0784992
    # L = 1.7056025
    step_size = blackjax_mclmc_sampler_params.step_size
    L = blackjax_mclmc_sampler_params.L

    jax.debug.print("{x} num_steps, L, step_size", x=(jnp.ceil(L/step_size), L, step_size))


    alg = blackjax.mcmc.mhmclmc.mhmclmc(
        logdensity_fn=logdensity_fn,
        step_size=step_size,
        integration_steps_fn = lambda key: jnp.round(jax.random.uniform(key) * rescale(L/step_size + 0.5)) ,
        # integration_steps_fn = lambda key: jnp.ceil(jax.random.poisson(key, L/step_size )) ,

    )

    _, out, info = run_inference_algorithm(
        rng_key=run_key,
        initial_state_or_position=blackjax_state_after_tuning,
        inference_algorithm=alg,
        num_steps=num_steps, 
        transform=lambda x: x.position, 
        progress_bar=True)
    
    
    jax.debug.print("ACCEPTANCE {x}", x = (info.acceptance_rate.shape, jnp.mean(info.acceptance_rate,)))
    
    # jax.debug.print("THING\n\n {x}",x=jnp.mean(info.num_integration_steps))
    # raise Exception
    return out, blackjax_mclmc_sampler_params, L/step_size

# we should do at least: mclmc, nuts, unadjusted hmc, mhmclmc, langevin

samplers = {'mclmc' : run_mclmc, 'mhmclmc': run_mhmclmc, 'nuts' : run_nuts}
