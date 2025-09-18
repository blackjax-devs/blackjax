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

# unbelievable that this is not in the standard library
def compose(f, g):
    return lambda x: f(g(x))


def las(logdensity_fn, key, ndims, num_steps1, num_steps2, num_chains, diagonal_preconditioning=True):

    init_key, tune_key, run_key = jax.random.split(key, 3)
    initial_position = jax.random.normal(init_key, (ndims,))

    ### Phase 1: unadjusted ###

    integrator = blackjax.mcmc.integrators.isokinetic_mclachlan
        
    # burn-in and adaptation
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

    (blackjax_state_after_tuning, blackjax_mclmc_sampler_params), adaptation_info = warmup.run(tune_key, initial_position, 20000)

    # sampling
    ess_per_sample = blackjax_mclmc_sampler_params['ESS']

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


    ### Phase 2: adjusted ###

    subsamples = samples[::math.ceil(1/ess_per_sample)]

    integration_steps_fn = make_random_trajectory_length_fn(True)

    initial_states = jax.lax.map(lambda x: blackjax.adjusted_mclmc_dynamic.init(x, logdensity_fn, jax.random.key(0)), xs=subsamples)

    def make_mams_step(key):
        def mams_step(step_size_positions_info):

            step_size, positions, info = step_size_positions_info
            num_steps_per_traj = blackjax_mclmc_sampler_params['L'] / step_size

            alg = blackjax.adjusted_mclmc_dynamic(
                    logdensity_fn=logdensity_fn,
                    step_size=step_size,
                    integration_steps_fn=integration_steps_fn(num_steps_per_traj),
                    integrator=blackjax.mcmc.integrators.isokinetic_velocity_verlet,
                    inverse_mass_matrix=blackjax_mclmc_sampler_params['inverse_mass_matrix'],
                    L_proposal_factor=jnp.inf,
                )
            
            new_states, infos = jax.lax.map(lambda x: alg.step(
                rng_key=key,
                state=blackjax.adjusted_mclmc_dynamic.init(x, logdensity_fn, key),
            ), xs=positions)
            return (step_size, new_states.position, infos)

        return mams_step
        
        
    def tuning_step(old_step_size_positions_info):

        old_step_size, old_positions, old_infos = old_step_size_positions_info
        acc_rate = old_infos.acceptance_rate.mean()
        
        new_step_size = jax.lax.cond(acc_rate < 0.8, lambda: old_step_size * 0.5, lambda: old_step_size * 2.0)
        
        return (new_step_size, old_positions, old_infos)

    step = lambda key: compose(tuning_step, make_mams_step(key))

    _, _, infos = make_mams_step(jax.random.key(0))((blackjax_mclmc_sampler_params['step_size'], subsamples, None))
    
    positions = subsamples
    step_size = blackjax_mclmc_sampler_params['step_size']

    (step_size, position, infos), (step_sizes, positions, infos) = jax.lax.scan(lambda state, key: (step(key)(state), step(key)(state)), (step_size, subsamples, infos), jax.random.split(jax.random.key(0), 10))

    return samples, positions, infos, num_steps

# type: forall a, b: (a -> b) -> (b -> a) -> Int -> (a -> b)
# e.g.: a ~ (stepsize, position), b ~ (state)
def feedback(f,g, n, state_a):
    for i in range(n):
        state_b = f(state_a)
        # print(state_b, "state_b")
        state_a = g(state_b)
        # print(state_a, "state_a")
    return state_a
        
