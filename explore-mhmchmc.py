import jax
import jax.numpy as jnp
from benchmarks.mcmc.sampling_algorithms import samplers
import blackjax
from blackjax.mcmc.mhmclmc import mhmclmc, rescale
from blackjax.mcmc.hmc import hmc
from blackjax.mcmc.dynamic_hmc import dynamic_hmc
from blackjax.mcmc.integrators import isokinetic_mclachlan
from blackjax.util import run_inference_algorithm





init_key, tune_key, run_key = jax.random.split(jax.random.PRNGKey(0), 3)

def logdensity_fn(x):
    return -0.5 * jnp.sum(jnp.square(x))

initial_position = jnp.ones(10,)

def run_hmc(initial_position):


    alg = blackjax.mcmc.hmc.hmc(
        logdensity_fn=logdensity_fn,
        inverse_mass_matrix=jnp.eye(2),
        step_size=1e-3,
        num_integration_steps=1000,
    )

    _, out, info = run_inference_algorithm(
        rng_key=jax.random.PRNGKey(0),
        initial_state_or_position=initial_position,
        inference_algorithm=alg,
        num_steps=num_steps,  
        progress_bar=True)
    
    print(info.acceptance_rate)
    
    return out


def run_mclmc(logdensity_fn, num_steps, initial_position):
    key = jax.random.PRNGKey(0)
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

    print(blackjax_mclmc_sampler_params)


# compare_static_dynamic()


# out = run_mclmc(logdensity_fn, num_steps, initial_position)
# print(out.position.mean(axis=0) )


# out = run_hmc(initial_position)
out = samplers["mhmclmc"](logdensity_fn=logdensity_fn, num_steps=5000, initial_position=initial_position, key=jax.random.PRNGKey(0))
print(out.mean(axis=0) )




# sampling_alg = blackjax.mclmc(
#             logdensity_fn,
#             L=blackjax_mclmc_sampler_params.L,
#             step_size=blackjax_mclmc_sampler_params.step_size,
#         )

# _, samples, _ = run_inference_algorithm(
#     rng_key=run_key,
#     initial_state_or_position=blackjax_state_after_tuning,
#     inference_algorithm=sampling_alg,
#     num_steps=num_steps,
#     transform=lambda x: x.position,
# )

# print(samples.mean(axis=0))


# mhmhlmc_find_L_and_step_size
# mhmhlmc_find_L_and_step_size

# kernel = blackjax.mcmc.mclmc.build_kernel(
#     logdensity_fn=logdensity_fn,
#     integrator=blackjax.mcmc.integrators.isokinetic_mclachlan,
# )




# (
#     blackjax_state_after_tuning,
#     blackjax_mclmc_sampler_params,
# ) = blackjax.mclmc_find_L_and_step_size(
#     mclmc_kernel=kernel,
#     num_steps=10000,
#     state=initial_state,
#     rng_key=jax.random.PRNGKey(0),
# )

# alg = dynamic_mhmclmc(
#     logdensity_fn=logdensity_fn,
#     step_size=blackjax_mclmc_sampler_params.step_size,
# )

# # alg = dynamic_hmc(
# #     logdensity_fn=logdensity_fn,
# #     inverse_mass_matrix=jnp.eye(2),
# #     step_size=1.0,
# #     next_random_arg_fn  = lambda key: jax.random.split(key)[1],
# #     integration_steps_fn = lambda key: jax.random.randint(key, (), 1, 10),
# # )

# out = run_inference_algorithm(
#     rng_key=jax.random.PRNGKey(10),
#     initial_state_or_position=jnp.ones((2,)),
#     inference_algorithm=alg,
#     num_steps=100,  
#     progress_bar=True)



# print(out[1].position.mean())


# print(out[2].acceptance_rate[-1])

# print(out[2].is_accepted)

# # plot a scatterplot of out[1].position
# import matplotlib.pyplot as plt

# plt.scatter(out[1].position[:, 0], out[1].position[:, 1])
# plt.xlabel('x')
# plt.ylabel('y')
# plt.title('Scatter Plot of out[1].position')
# plt.show()



















# # alg = mhmclmc(
# #     logdensity_fn=logdensity_fn,
# #     step_size=1.0,
# #     num_integration_steps=100,
# # )



# # alg_dynamic = dynamic_hmc(
# #     logdensity_fn=logdensity_fn,
# #     inverse_mass_matrix=jnp.eye(2),
# #     step_size=1.0,
# #     next_random_arg_fn  = lambda key: jax.random.split(key)[1],
# #     integration_steps_fn = lambda key: jax.random.randint(key, (), 1, 10),
# # )

# # alg = hmc(
# #     logdensity_fn=logdensity_fn,
# #     inverse_mass_matrix=jnp.eye(2),
# #     step_size=1.0,
# #     num_integration_steps=1000,
# # )

# # out = alg_static.step(jax.random.PRNGKey(0), alg_static.init(jnp.array([1.0, 1.0])))
# # out2 = alg_dynamic.step(jax.random.PRNGKey(0), alg_dynamic.init(jnp.array([1.0, 1.0]), random_generator_arg= jax.random.PRNGKey(0)))

# # print(out[0])
# # print(out2[0])
# # raise Exception

# initial_position = jnp.array([1.0, 1.0])
# initial_state = blackjax.mcmc.mclmc.init(
#     position=initial_position, logdensity_fn=logdensity_fn, rng_key=jax.random.PRNGKey(0)
# )


# kernel = blackjax.mcmc.mclmc.build_kernel(
#     logdensity_fn=logdensity_fn,
#     integrator=blackjax.mcmc.integrators.isokinetic_mclachlan,
# )

# (
#     blackjax_state_after_tuning,
#     blackjax_mclmc_sampler_params,
# ) = blackjax.mclmc_find_L_and_step_size(
#     mclmc_kernel=kernel,
#     num_steps=10000,
#     state=initial_state,
#     rng_key=jax.random.PRNGKey(0),
# )

# print(blackjax_mclmc_sampler_params)

# alg = blackjax.mcmc.mclmc.mclmc(logdensity_fn=logdensity_fn, step_size=blackjax_mclmc_sampler_params.step_size, L=blackjax_mclmc_sampler_params.L)

# out = run_inference_algorithm(
#     rng_key=jax.random.PRNGKey(0),
#     initial_state_or_position=blackjax_state_after_tuning,
#     inference_algorithm=alg,
#     num_steps=100000,  
#     progress_bar=True)

# print(out[1].position.mean())

# # raise Exception

# # print(out[0])
# # state = blackjax.mcmc.mclmc.init(jnp.array([1.0, 1.0]), logdensity_fn, jax.random.PRNGKey(0))


# # print(state)


# # integrator = isokinetic_mclachlan(logdensity_fn=logdensity_fn)
# # print(jax.lax.scan(f=lambda s, _: integrator(s, step_size=1e-3),init=state, xs=None, length=1000))
# # # raise Exception
# # def run_integrator(state, step_size):
# #     return jax.lax.scan(f=lambda s, _: (integrator(s, step_size=step_size)), init=state, xs=None, length=1000)
        
# #         # integrator(state, step_size=1e-3)

# # (position, momentum, logdensity, logdensitygrad), kinetic_change = run_integrator(state, step_size=1e-3)

# # print(kinetic_change)
# # alg.init(jnp.array([1.0, 1.0]))

# # alg.init(position=jnp.array([1.0, 1.0]))

# # out = alg.step(jax.random.PRNGKey(0), alg.init(position=jnp.array([1.0, 1.0])))




# # print(out)