import sys
sys.path.append('./')
from collections import defaultdict
from functools import partial
import math
import operator
import os
import pprint
from statistics import mean, median
import jax
import jax.numpy as jnp
import pandas as pd
import scipy
from typing import Callable, Union
from chex import PRNGKey
import jax
import jax.numpy as jnp
import blackjax
from blackjax.adaptation.mclmc_adaptation import MCLMCAdaptationState
# from blackjax.adaptation.window_adaptation import da_adaptation
from blackjax.mcmc.integrators import generate_euclidean_integrator, generate_isokinetic_integrator, mclachlan, yoshida, velocity_verlet, omelyan, isokinetic_mclachlan, isokinetic_velocity_verlet, isokinetic_yoshida, isokinetic_omelyan
from blackjax.util import run_inference_algorithm
import blackjax
from blackjax.util import pytree_size

class StandardNormal():
    """Standard Normal distribution in d dimensions"""

    def __init__(self, d):
        self.ndims = d
        self.E_x2 = jnp.ones(d)
        self.Var_x2 = 2 * self.E_x2
        self.name = 'StandardNormal'
        

    def logdensity_fn(self, x):
        """- log p of the target distribution"""
        return -0.5 * jnp.sum(jnp.square(x), axis= -1)


    def transform(self, x):
        return x

    def sample_init(self, key):
        return jax.random.normal(key, shape = (self.ndims, ))



def run_mclmc(logdensity_fn, num_steps, initial_position, transform, key, preconditioning):

    integrator = isokinetic_mclachlan
    init_key, tune_key, run_key = jax.random.split(key, 3)


    initial_state = blackjax.mcmc.mclmc.init(
        position=initial_position, logdensity_fn=logdensity_fn, rng_key=init_key
    )

    
    kernel = lambda sqrt_diag_cov : blackjax.mcmc.mclmc.build_kernel(
        logdensity_fn=logdensity_fn,
        integrator=integrator,
        sqrt_diag_cov=sqrt_diag_cov,
    )

    (
        blackjax_state_after_tuning,
        blackjax_mclmc_sampler_params,
    ) = blackjax.mclmc_find_L_and_step_size(
        mclmc_kernel=kernel,
        num_steps=num_steps,
        state=initial_state,
        rng_key=tune_key,
        diagonal_preconditioning=preconditioning,
    )

    # sampling_alg = blackjax.mclmc(
    #     logdensity_fn,
    #     L=blackjax_mclmc_sampler_params.L,
    #     step_size=blackjax_mclmc_sampler_params.step_size,
    #     sqrt_diag_cov=blackjax_mclmc_sampler_params.sqrt_diag_cov,
    #     integrator = integrator,
    # )


    # _, samples, _ = run_inference_algorithm(
    #     rng_key=run_key,
    #     initial_state=blackjax_state_after_tuning,
    #     inference_algorithm=sampling_alg,
    #     num_steps=num_steps,
    #     transform=lambda x: transform(x.position),
    #     progress_bar=False,
    # )
    
    
    # return samples.mean(axis=0)

    return blackjax_state_after_tuning.position


model = StandardNormal(2)

key = jax.random.PRNGKey(0)


map = jax.pmap # change to pmap

sampler = partial(run_mclmc, preconditioning=True)

key1, init_key = jax.random.split(key, 2)
keys = jax.random.split(key1, 1)

init_keys = jax.random.split(init_key, 1)
init_pos = map(model.sample_init)(init_keys) # [batch_size, dim_model]

result = map(lambda pos, key: sampler(logdensity_fn=model.logdensity_fn, num_steps=1000, initial_position= pos,transform= model.transform, key=key))(init_pos, keys)

print(f'Result with {str(map)} is {result}')
