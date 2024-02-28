

import jax
import jax.numpy as jnp
import blackjax
from blackjax.adaptation.mclmc_adaptation import MCLMCAdaptationState
from blackjax.mcmc.mhmclmc import rescale
from blackjax.util import run_inference_algorithm
import blackjax

__all__ = ["samplers"]




# def run_hmc(
#     rng_key, initial_states, tuned_params, logdensity_fn, num_samples):
    
#     warmup = blackjax.window_adaptation(blackjax.nuts, logdensity_fn)

#     # we use 4 chains for sampling
#     rng_key, init_key, warmup_key = jax.random.split(rng_key, 3)
#     # init_keys = jax.random.split(init_key, num_chains)
#     # init_params = jax.vmap(init_param_fn)(init_keys)

#     (initial_states, tuned_params), _ = warmup.run(warmup_key, init_params, 1000)


#     states, infos = run_inference_algorithm(
#         rng_key=rng_key,
#         initial_state_or_position=initial_states,
#         inference_algorithm=blackjax.nuts,
#         num_steps=num_samples,
#         transform=lambda x: x.position,
#     )

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

samplers = {'mclmc' : run_mclmc, 'mhmclmc': run_mhmclmc}
