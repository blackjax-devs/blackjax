import jax
import jax.numpy as jnp
import blackjax
from blackjax.util import run_inference_algorithm
import blackjax

from blackjax.adaptation.unadjusted_alba import unadjusted_alba
from blackjax.adaptation.unadjusted_step_size import robnik_step_size_tuning
from blackjax.adaptation.unadjusted_alba import unadjusted_alba
import math
from blackjax.mcmc.adjusted_mclmc_dynamic import make_random_trajectory_length_fn
from functools import partial

def las(logdensity_fn, key, ndims, num_steps1, num_steps2, num_chains, diagonal_preconditioning=True):

    # begin by running unadjusted alba tuning for umclmc


    init_key, tune_key, run_key = jax.random.split(key, 3)
    initial_position = jax.random.normal(init_key, (ndims,))

    integrator = blackjax.mcmc.integrators.isokinetic_mclachlan
        
    num_alba_steps = 10000
    warmup = unadjusted_alba(
        algorithm=blackjax.mclmc, 
        logdensity_fn=logdensity_fn, integrator=integrator, 
        target_eevpd=5e-4, 
        v=jnp.sqrt(ndims), 
        num_alba_steps=num_alba_steps,
        preconditioning=diagonal_preconditioning,
        alba_factor=0.4,
        )
    

    # run warmup
    (blackjax_state_after_tuning, blackjax_mclmc_sampler_params), adaptation_info = warmup.run(tune_key, initial_position, 20000)

    ess_per_sample = blackjax_mclmc_sampler_params['ESS']
    print(ess_per_sample, "ESS")
    # get_final_sample = lambda state, info: (model.default_event_space_bijector(state.position), info)

    num_steps = math.ceil(200 // ess_per_sample)

    alg = blackjax.mclmc(
            logdensity_fn=logdensity_fn,
            L=blackjax_mclmc_sampler_params['L'],
            step_size=blackjax_mclmc_sampler_params['step_size'],
            inverse_mass_matrix=blackjax_mclmc_sampler_params['inverse_mass_matrix'],
            integrator=integrator,
        )

    final_output, history = run_inference_algorithm(
            rng_key=key,
            initial_state=blackjax_state_after_tuning,
            inference_algorithm=alg,
            num_steps=num_steps,
            transform=(lambda a, b: a),
            progress_bar=False,
        )
    
    samples = history.position

    print(samples.shape)

    subsamples = samples[::math.ceil(1/ess_per_sample)]

    integration_steps_fn = make_random_trajectory_length_fn(True)

    num_steps_per_traj = blackjax_mclmc_sampler_params['L'] / blackjax_mclmc_sampler_params['step_size']

    

    # initial_states = blackjax.adjusted_mclmc_dynamic.init(
    #         position=history.position[-1],
    #         logdensity_fn=logdensity_fn,
    #         random_generator_arg=jax.random.key(0),
    #     )

    initial_states = jax.lax.map(lambda x: blackjax.adjusted_mclmc_dynamic.init(x, logdensity_fn, jax.random.key(0)), xs=subsamples)

    print(initial_states, "initial_states")

    def f(step_size_and_positions):
        
        return jax.lax.map(lambda x: run_inference_algorithm(
            rng_key=key,
            initial_state=blackjax.adjusted_mclmc_dynamic.init(x, logdensity_fn, jax.random.key(0)),
            inference_algorithm=blackjax.adjusted_mclmc_dynamic(
                logdensity_fn=logdensity_fn,
                step_size=step_size_and_positions[0],
                integration_steps_fn=integration_steps_fn(num_steps_per_traj),
                integrator=blackjax.mcmc.integrators.isokinetic_velocity_verlet,
                inverse_mass_matrix=blackjax_mclmc_sampler_params['inverse_mass_matrix'],
                L_proposal_factor=jnp.inf,
            ),
            num_steps=1,
            transform=(lambda a, b: (a,b)),
            progress_bar=False,
        ), xs=step_size_and_positions[1])
    g = lambda x: (blackjax_mclmc_sampler_params['step_size'],x[0].position)



    step_size, position = feedback(f,g, 10, (blackjax_mclmc_sampler_params['step_size'], subsamples))

    # results = f((1.0, initial_states))
    # step_size = g(results)
    # print(step_size, "step_size")

    # history, final_output = results

    # print(history[0].position.shape)
    # print(history[1].acceptance_rate.mean())

    return position

# a ~ (stepsize, position), b ~ (results)
# type: forall a, b: (a -> b) -> (b -> a) -> Int -> (a -> b)
def feedback(f,g, n, state_a):
    for i in range(n):
        state_b = f(state_a)
        # print(state_b, "state_b")
        state_a = g(state_b)
        # print(state_a, "state_a")
    return state_a
        
