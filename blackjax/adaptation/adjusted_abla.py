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
        integration_steps_fn = lambda avg_num_integration_steps: lambda k: jnp.where(jnp.ceil(
            jax.random.uniform(k) * rescale(avg_num_integration_steps)
        )==0, 1, jnp.ceil(
            jax.random.uniform(k) * rescale(avg_num_integration_steps))).astype('int32')
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
    initial_L: float = 1.0,
    integrator=blackjax.mcmc.integrators.velocity_verlet, 
    L_proposal_factor=jnp.inf,
):
    
    da_init, da_update, da_final = dual_averaging_adaptation(target_acceptance_rate)
    kernel = algorithm.build_kernel(integrator=integrator, L_proposal_factor=L_proposal_factor)

    # initial_L = jnp.clip(initial_L, min=initial_step_size+0.01)

    
    def step(state, key):

        (adaptation_state, kernel_state), L = state
        new_kernel_state, info = kernel(
            rng_key=key,
            state=kernel_state,
            logdensity_fn=logdensity_fn,
            step_size=jnp.exp(adaptation_state.log_step_size),
            inverse_mass_matrix=inverse_mass_matrix,
            integration_steps_fn=integration_steps_fn(L/jnp.exp(adaptation_state.log_step_size)),
        )
        
        new_adaptation_state = da_update(
            adaptation_state,
            info.acceptance_rate,
        )


        return (
            ((new_adaptation_state, new_kernel_state), L),
            info,
        )

    def run(rng_key: PRNGKey, position: ArrayLikeTree, num_steps: int = 1000):


        init_key, rng_key = jax.random.split(rng_key)
        
        init_kernel_state = algorithm.init(position=position, logdensity_fn=logdensity_fn, random_generator_arg=init_key)

        keys = jax.random.split(rng_key, num_steps)
        init_state = da_init(initial_step_size), init_kernel_state
        ((adaptation_state, kernel_state), L), info = jax.lax.scan(
            step,
            (init_state, initial_L),
            keys,

        )
        step_size = da_final(adaptation_state)
        return (
            kernel_state,
            {
                "step_size": step_size,
                "inverse_mass_matrix": inverse_mass_matrix,
                "L": L,
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
    integrator,
    target_acceptance_rate: float = 0.80,
    num_alba_steps: int = 500,
    alba_factor: float = 0.4,
    preconditioning: bool = True,
    L_proposal_factor=jnp.inf,
    **extra_parameters,
    ):

    unadjusted_warmup = unadjusted_alba(
        algorithm= unadjusted_algorithm,
        logdensity_fn=logdensity_fn,
        target_eevpd=target_eevpd,
        v=v,
        integrator=integrator,
        num_alba_steps=num_alba_steps,
        alba_factor=alba_factor,
        preconditioning=preconditioning,
        **extra_parameters)
    
    def run(rng_key: PRNGKey, position: ArrayLikeTree, num_steps: int = 1000):
        
        unadjusted_warmup_key, adjusted_warmup_key = jax.random.split(rng_key)

        num_unadjusted_steps = 20000

        (state, params), adaptation_info = unadjusted_warmup.run(unadjusted_warmup_key, position, num_unadjusted_steps)

        # jax.debug.print("unadjusted params: {params}", params=(params["L"], params["step_size"]))
        # jax.debug.print("unadjusted params: {params}", params=params)

        integration_steps_fn = make_random_trajectory_length_fn(random_trajectory_length=True)

        adjusted_warmup = da_adaptation(
            algorithm=adjusted_algorithm,
            logdensity_fn=logdensity_fn,
            integration_steps_fn=integration_steps_fn,
            initial_L=params["L"],
            initial_step_size=params["step_size"],
            target_acceptance_rate=target_acceptance_rate,
            inverse_mass_matrix=params["inverse_mass_matrix"],
            integrator=integrator, L_proposal_factor=L_proposal_factor, **extra_parameters)
        
        
        
        state, params, adaptation_info = adjusted_warmup.run(adjusted_warmup_key, state.position, num_steps)
        # jax.debug.print("adjusted params: {params}", params=(params["L"], params["step_size"]))
        # raise Exception("stop")
        # return None
        return state, params, adaptation_info
    
    return AdaptationAlgorithm(run)

    