from blackjax.adaptation.step_size import (
    dual_averaging_adaptation,
)
from blackjax.mcmc.adjusted_mclmc_dynamic import rescale
from blackjax.base import AdaptationAlgorithm
from blackjax.types import ArrayLikeTree, PRNGKey
import jax
import jax.numpy as jnp
from typing import Callable
import blackjax
from blackjax.adaptation.unadjusted_alba import unadjusted_alba



def make_random_trajectory_length_fn(random_trajectory_length : bool):
    if random_trajectory_length:
        integration_steps_fn = lambda avg_num_integration_steps: lambda k: jnp.ceil(
            jax.random.uniform(k) * rescale(avg_num_integration_steps)
        ).astype('int32')
    else:
        integration_steps_fn = lambda avg_num_integration_steps: lambda _: jnp.ceil(
            avg_num_integration_steps
        ).astype('int32')
    return integration_steps_fn

def da_adaptation(
    algorithm,
    logdensity_fn: Callable,
    integration_steps_fn: Callable,
    inverse_mass_matrix,
    initial_step_size: float = 1.0,
    target_acceptance_rate: float = 0.80,
    integrator=blackjax.mcmc.integrators.velocity_verlet, 
):
    
    da_init, da_update, da_final = dual_averaging_adaptation(target_acceptance_rate)
    kernel = algorithm.build_kernel(integrator=integrator)
    
    def step(state, key):

        adaptation_state, kernel_state = state

        new_kernel_state, info = kernel(
            rng_key=key,
            state=kernel_state,
            logdensity_fn=logdensity_fn,
            step_size=jnp.exp(adaptation_state.log_step_size),
            inverse_mass_matrix=inverse_mass_matrix,
            integration_steps_fn=integration_steps_fn,
        )

        new_adaptation_state = da_update(
            adaptation_state,
            info.acceptance_rate,
        )

        return (
            (new_adaptation_state, new_kernel_state),
            info,
        )

    def run(rng_key: PRNGKey, position: ArrayLikeTree, num_steps: int = 1000):


        init_key, rng_key = jax.random.split(rng_key)
        
        init_kernel_state = algorithm.init(position=position, logdensity_fn=logdensity_fn, random_generator_arg=init_key)

        keys = jax.random.split(rng_key, num_steps)
        init_state = da_init(initial_step_size), init_kernel_state
        (adaptation_state, kernel_state), info = jax.lax.scan(
            step,
            init_state,
            keys,
        )
        step_size = da_final(adaptation_state)
        return (
            kernel_state,
            {
                "step_size": step_size,
                "inverse_mass_matrix": inverse_mass_matrix,
            },
            info,
        )

    return AdaptationAlgorithm(run)


def alba_adjusted(
    unadjusted_algorithm,
    logdensity_fn: Callable,
    target_eevpd,
    v,
    adjusted_algorithm,
    num_dimensions: int,
    integrator,
    target_acceptance_rate: float = 0.80,
    num_alba_steps: int = 500,
    alba_factor: float = 0.4,
    **extra_parameters,
    ):

    unadjusted_warmup = unadjusted_alba(
        algorithm= unadjusted_algorithm,
        logdensity_fn=logdensity_fn,
        target_eevpd=target_eevpd,
        v=v,
        integrator=integrator,
        num_alba_steps=num_alba_steps,
        alba_factor=alba_factor, **extra_parameters)
    
    def run(rng_key: PRNGKey, position: ArrayLikeTree, num_steps: int = 1000):
        
        unadjusted_warmup_key, adjusted_warmup_key = jax.random.split(rng_key)

        (state, params), adaptation_info = unadjusted_warmup.run(unadjusted_warmup_key, position, num_steps)

        avg_num_integration_steps = params["L"] / params["step_size"]

        integration_steps_fn = lambda k: jnp.ceil(
                    jax.random.uniform(k) * rescale(avg_num_integration_steps)
                )

        adjusted_warmup = da_adaptation(
            algorithm=adjusted_algorithm,
            logdensity_fn=logdensity_fn,
            integration_steps_fn=integration_steps_fn,
            initial_step_size=params["step_size"],
            target_acceptance_rate=target_acceptance_rate,
            inverse_mass_matrix=params["inverse_mass_matrix"],
            integrator=integrator, **extra_parameters)
        
        state, params, adaptation_info = adjusted_warmup.run(adjusted_warmup_key, state.position, num_steps)
        params["L"] = adaptation_info.num_integration_steps.mean()*params["step_size"]
        return state, params, adaptation_info
    
    return AdaptationAlgorithm(run)

    