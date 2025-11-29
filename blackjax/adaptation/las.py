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
from blackjax.adaptation.step_size import bisection_monotonic_fn
from blackjax.util import thin_algorithm

# unbelievable that this is not in the standard library
def compose(f, g):
    return lambda x: f(g(x))


def las(logdensity_fn, num_chains, key, ndims, num_adjusted_steps, diagonal_preconditioning=True, target_acceptance_rate=0.8, target_eevpd=5e-4):

    init_key, tune_key, unadjusted_key, adjusted_key = jax.random.split(key, 4)
    initial_position = jax.random.normal(init_key, (ndims,))

    ### Phase 1: unadjusted ###

    integrator = blackjax.mcmc.integrators.isokinetic_mclachlan

    jax.debug.print("target_eevpd {x}", x=target_eevpd)
        
    # burn-in and adaptation
    num_alba_steps = 10000 // 3
    warmup = unadjusted_alba(
        algorithm=blackjax.mclmc, 
        logdensity_fn=logdensity_fn, integrator=integrator, 
        target_eevpd=target_eevpd, 
        # target_acceptance_rate=target_acceptance_rate,
        v=jnp.sqrt(ndims), 
        num_alba_steps=num_alba_steps,
        preconditioning=diagonal_preconditioning,
        alba_factor=0.4,
        )

    (blackjax_state_after_tuning, blackjax_mclmc_sampler_params), adaptation_info = warmup.run(tune_key, initial_position, 10000)

    # sampling
    ess_per_sample = blackjax_mclmc_sampler_params['ESS']

    num_steps = math.ceil(num_chains // ess_per_sample)

    # load from absolute path in storage
    import numpy as np
    metadata = np.load('/global/cfs/cdirs/m4031/reubenh/inverse_mass_matrix.npy')

    alg = blackjax.mclmc(
            logdensity_fn=logdensity_fn,
            L=metadata['L'],
            step_size=metadata['step_size'],
            inverse_mass_matrix=metadata['inverse_mass_matrix'],
            integrator=integrator,
        )

    thinning_rate = math.ceil(1/ess_per_sample)
    jax.debug.print("thinning_rate {x}", x=thinning_rate)

    # alg = thin_algorithm(
    #                 alg,
    #                 thinning=thinning_rate,
    #                 info_transform=lambda info: jax.tree.map(jnp.mean, info),
    #                 )

    # jax.debug.print("alg {x}", x=alg)

    final_output, history = run_inference_algorithm(
            rng_key=unadjusted_key,
            initial_state=blackjax_state_after_tuning,
            inference_algorithm=alg,
            # num_steps=num_steps//thinning_rate,
            num_steps=num_steps,
            transform=(lambda a, b: a),
            progress_bar=False,
        )
    return history.position
    samples = history.position


    ### Phase 2: adjusted ###

    # subsamples = samples[:th]
    subsamples = samples

    integration_steps_fn = make_random_trajectory_length_fn(True)


    # initial_states = jax.lax.map(lambda x: blackjax.adjusted_mclmc_dynamic.init(x, logdensity_fn, jax.random.key(0)), xs=subsamples)
    # initial_states = jax.lax.map(lambda x: blackjax.adjusted_mclmc_dynamic.init(x, logdensity_fn, jax.random.key(0)), xs=subsamples)

    def make_mams_step(key):
        def mams_step(inp):
            # init_key, run_key = jax.random.split(key, 2)


            step_size, positions, info, step_size_adaptation_state = inp
            jax.debug.print("step_size {step_size}", step_size=step_size)
            
            keys = jax.random.split(key, positions.shape[0])
            num_steps_per_traj = blackjax_mclmc_sampler_params['L'] / step_size
            # num_steps_per_traj = 1

            alg = blackjax.adjusted_mclmc_dynamic(
                    logdensity_fn=logdensity_fn,
                    step_size=step_size,
                    integration_steps_fn=integration_steps_fn(num_steps_per_traj),
                    integrator=blackjax.mcmc.integrators.isokinetic_velocity_verlet,
                    inverse_mass_matrix=blackjax_mclmc_sampler_params['inverse_mass_matrix'],
                    L_proposal_factor=jnp.inf,
                )
            
            # run_keys = jax.random.split(run_key, positions.shape[0])

            def step_fn(pos_key):
                pos, key = pos_key
                init_key, run_key = jax.random.split(key, 2)
                return alg.step(
                    rng_key=run_key,
                    state=blackjax.adjusted_mclmc_dynamic.init(pos, logdensity_fn, init_key),
                )
            
            new_states, infos = jax.lax.map(step_fn, xs=(positions,keys))
            # jax.debug.print("infos adaptation step: {infos}", infos=jnp.sum(infos.is_accepted))
            return (step_size, new_states.position, infos, step_size_adaptation_state)

        return mams_step
        
    epsadap_update = bisection_monotonic_fn(target_acceptance_rate)
    step_size_adaptation_state_initial = (jnp.array([-jnp.inf, jnp.inf]), False)
        
    def tuning_step(inp):

        old_step_size, old_positions, old_infos, step_size_adaptation_state = inp
        acc_rate = old_infos.acceptance_rate.mean()

        
        step_size_adaptation_state, new_step_size = epsadap_update(
            step_size_adaptation_state,
            old_step_size,
            acc_rate,
        )
        
        return (new_step_size, old_positions, old_infos, step_size_adaptation_state)
        # return (0, old_positions, old_infos, step_size_adaptation_state)

    # step = lambda key: compose(tuning_step, make_mams_step(key))

    def step_fn(inp, key):
        tuned_params = tuning_step(inp)
        out =  make_mams_step(key)(tuned_params)
        return out, out


    
    initial_adjusted_key, adjusted_key = jax.random.split(adjusted_key, 2)

    _, _, infos, _ = make_mams_step(initial_adjusted_key)((0.0, subsamples, None, step_size_adaptation_state_initial))
    # _, _, infos, _ = make_mams_step(initial_adjusted_key)((0.0, subsamples, None, step_size_adaptation_state_initial))
    # _, _, infos, _ = make_mams_step(initial_adjusted_key)((0.0, subsamples, None, step_size_adaptation_state_initial))
    
    
    positions = subsamples
    step_size = blackjax_mclmc_sampler_params['step_size']

    _, (step_sizes, positions, infos, step_size_adaptation_state) = jax.lax.scan(step_fn, (step_size, subsamples, infos, step_size_adaptation_state_initial), jax.random.split(adjusted_key, num_adjusted_steps))

    return samples, positions, infos, num_steps, step_size_adaptation_state, step_sizes

# type: forall a, b: (a -> b) -> (b -> a) -> Int -> (a -> b)
# e.g.: a ~ (stepsize, position), b ~ (state)
# def feedback(f,g, n, state_a):
#     for i in range(n):
#         state_b = f(state_a)
#         # print(state_b, "state_b")
#         state_a = g(state_b)
#         # print(state_a, "state_a")
#     return state_a
        
