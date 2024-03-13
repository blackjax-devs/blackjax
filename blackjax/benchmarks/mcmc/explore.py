import jax

from datetime import date
from blackjax.benchmarks.mcmc.benchmark import benchmark_chains

from blackjax.benchmarks.mcmc.inference_models import IllConditionedGaussian

rng_key = jax.random.key(int(date.today().strftime("%Y%m%d")))

import blackjax
import numpy as np
import jax.numpy as jnp
from sampling_algorithms import samplers
from inference_models import StandardNormal, models

def run_mclmc(logdensity_fn, num_steps, initial_position, key, transform, std_mat, L, step_size):
    init_key, tune_key, run_key = jax.random.split(key, 3)

    # create an initial state for the sampler
    initial_state = blackjax.mcmc.mclmc.init(
        position=initial_position, logdensity_fn=logdensity_fn, rng_key=init_key
    )


    # use the quick wrapper to build a new kernel with the tuned parameters
    sampling_alg = blackjax.mclmc(
        logdensity_fn,
        L=L,
        step_size=step_size,
        std_mat=std_mat,
    )

    # run the sampler
    _, samples, _ = blackjax.util.run_inference_algorithm(
        rng_key=run_key,
        initial_state_or_position=initial_state,
        inference_algorithm=sampling_alg,
        num_steps=num_steps,
        transform=transform,
        progress_bar=True,
    )

    return samples, None, 1


def run_mclmc_with_tuning(logdensity_fn, num_steps, initial_position, key, transform):
    init_key, tune_key, run_key = jax.random.split(key, 3)

    # create an initial state for the sampler
    initial_state = blackjax.mcmc.mclmc.init(
        position=initial_position, logdensity_fn=logdensity_fn, rng_key=init_key
    )

    kernel = blackjax.mcmc.mclmc.build_kernel(
        logdensity_fn=logdensity_fn,
        integrator=blackjax.mcmc.integrators.isokinetic_mclachlan,
        std_mat=jnp.ones((initial_position.shape[0],)),
    )

    # find values for L and step_size
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



    # use the quick wrapper to build a new kernel with the tuned parameters
    sampling_alg = blackjax.mclmc(
        logdensity_fn,
        L=blackjax_mclmc_sampler_params.L,
        step_size=blackjax_mclmc_sampler_params.step_size,
        std_mat=blackjax_mclmc_sampler_params.std_mat,
    )

    # run the sampler
    _, samples, _ = blackjax.util.run_inference_algorithm(
        rng_key=run_key,
        initial_state_or_position=initial_state,
        inference_algorithm=sampling_alg,
        num_steps=num_steps,
        transform=transform,
        progress_bar=True,
    )

    return samples
# run the algorithm on a high dimensional gaussian, and show two of the dimensions

# sigma = .5

sample_key, rng_key = jax.random.split(rng_key)
# samples = run_mclmc(
#     logdensity_fn=lambda x: -0.5 * jnp.sum(jnp.square(x)),
#     num_steps=100000,
#     initial_position=jnp.ones((2,)),
#     key=sample_key,
#     std_mat=jnp.ones((2,))*sigma,
#     # std_mat=None,
#     transform=lambda x: x.position, # x.position[:2],
# )
# print(samples.var(axis=0))

# den = lambda x: jax.scipy.stats.norm.logpdf(x, loc=0., scale=jnp.sqrt(sigma)).sum()
# print(IllConditionedGaussian(2, 2).E_x2)
# samples = run_mclmc_with_tuning(
#     logdensity_fn=lambda x : - IllConditionedGaussian(2, 2).nlogp(x),
#     num_steps=1000000,
#     initial_position=jnp.ones((2,)),
#     key=sample_key,
#     transform=lambda x: x.position[:2],
# )
# # print(samples.var(axis=0))
# m = IllConditionedGaussian(10, 5)
# sampler = lambda logdensity_fn, num_steps, initial_position, key: run_mclmc(logdensity_fn=logdensity_fn, num_steps=num_steps, initial_position=initial_position, key=key, transform=lambda x:x.position, 
#     #  std_mat=jnp.ones((10,))
#      std_mat=jnp.sqrt(m.E_x2)
#      , L=2.6576319, step_size=3.40299)
# print(m.E_x2, "var")

# # sampler = 'mclmc'
# # samplers[sampler]
# result, bias, _ = benchmark_chains(m, sampler, n=5000, batch=1000//m.ndims,favg=m.E_x2, fvar=m.Var_x2)

# print(result)


# m = StandardNormal(10)
# sampler = lambda logdensity_fn, num_steps, initial_position, key: run_mclmc(logdensity_fn=logdensity_fn, num_steps=num_steps, initial_position=initial_position, key=key, transform=lambda x:x.position, 
#      std_mat=jnp.ones((10,))
#      , L=2.6576319, step_size=3.40299)
# # print(m.E_x2, "var")

# # sampler = 'mclmc'
# # samplers[sampler]
# result, bias, _ = benchmark_chains(m, sampler, n=5000, batch=1000//m.ndims,favg=m.E_x2, fvar=m.Var_x2)

# print(result)

