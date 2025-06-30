import jax
import jax.numpy as jnp

from typing import Callable, NamedTuple

import blackjax.mcmc as mcmc
from blackjax.adaptation.base import AdaptationResults, return_all_adapt_info
from blackjax.adaptation.mass_matrix import (
    MassMatrixAdaptationState,
    mass_matrix_adaptation,
)
from blackjax.base import AdaptationAlgorithm
from blackjax.progress_bar import gen_scan_fn
from blackjax.types import Array, ArrayLikeTree, PRNGKey
from blackjax.util import pytree_size
from blackjax.adaptation.window_adaptation import build_schedule
from jax.flatten_util import ravel_pytree
from blackjax.diagnostics import effective_sample_size
from blackjax.adaptation.unadjusted_step_size import robnik_step_size_tuning, RobnikStepSizeTuningState

class AlbaAdaptationState(NamedTuple):
    ss_state: RobnikStepSizeTuningState  # step size
    imm_state: MassMatrixAdaptationState  # inverse mass matrix
    step_size: float
    inverse_mass_matrix: Array
    L : float

def base(
    is_mass_matrix_diagonal: bool,
    v,
    target_eevpd,
) -> tuple[Callable, Callable, Callable]:
    
    mm_init, mm_update, mm_final = mass_matrix_adaptation(is_mass_matrix_diagonal)

    # step_size_init, step_size_update, step_size_final = dual_averaging_adaptation(target_eevpd)
    step_size_init, step_size_update, step_size_final = robnik_step_size_tuning(desired_energy_var=target_eevpd)

    def init(
        position: ArrayLikeTree,
    ) -> AlbaAdaptationState:
        
        num_dimensions = pytree_size(position)
        imm_state = mm_init(num_dimensions)

        ss_state = step_size_init(initial_step_size=jnp.sqrt(num_dimensions)/5, num_dimensions=num_dimensions)

        return AlbaAdaptationState(
            ss_state,
            imm_state,
            ss_state.step_size,
            imm_state.inverse_mass_matrix,
            L = jnp.sqrt(num_dimensions)/v
        )

    def fast_update(
        position: ArrayLikeTree,
        value: float,
        warmup_state: AlbaAdaptationState,
    ) -> AlbaAdaptationState:
        """Update the adaptation state when in a "fast" window.

        Only the step size is adapted in fast windows. "Fast" refers to the fact
        that the optimization algorithms are relatively fast to converge
        compared to the covariance estimation with Welford's algorithm

        """

        del position


        new_ss_state =  step_size_update(warmup_state.ss_state, value)
        new_step_size = new_ss_state.step_size # jnp.exp(new_ss_state.log_step_size)
        
        return AlbaAdaptationState(
            new_ss_state,
            warmup_state.imm_state,
            new_step_size,
            warmup_state.inverse_mass_matrix,
            L = warmup_state.L
        )

    def slow_update(
        position: ArrayLikeTree,
        value: float,
        warmup_state: AlbaAdaptationState,
    ) -> AlbaAdaptationState:
    
        new_imm_state = mm_update(warmup_state.imm_state, position)
        new_ss_state = step_size_update(warmup_state.ss_state, value)
        new_step_size = new_ss_state.step_size # jnp.exp(new_ss_state.log_step_size)

        return AlbaAdaptationState(
            new_ss_state, new_imm_state, new_step_size, warmup_state.inverse_mass_matrix, L = warmup_state.L
        )

    def slow_final(warmup_state: AlbaAdaptationState) -> AlbaAdaptationState:

        new_imm_state = mm_final(warmup_state.imm_state)
        new_ss_state = step_size_init(step_size_final(warmup_state.ss_state), warmup_state.ss_state.num_dimensions)
        new_step_size = new_ss_state.step_size # jnp.exp(new_ss_state.log_step_size)

        new_L = jnp.sqrt(warmup_state.ss_state.num_dimensions)/v # 

        return AlbaAdaptationState(
            new_ss_state,
            new_imm_state,
            new_step_size,
            new_imm_state.inverse_mass_matrix,
            L = new_L
        )

    def update(
        adaptation_state: AlbaAdaptationState,
        adaptation_stage: tuple,
        position: ArrayLikeTree,
        value: float,
    ) -> AlbaAdaptationState:
        """Update the adaptation state and parameter values.

        Parameters
        ----------
        adaptation_state
            Current adptation state.
        adaptation_stage
            The current stage of the warmup: whether this is a slow window,
            a fast window and if we are at the last step of a slow window.
        position
            Current value of the model parameters.
        value
            Value of the acceptance rate for the last mcmc step.

        Returns
        -------
        The updated adaptation state.

        """
        stage, is_middle_window_end = adaptation_stage

        warmup_state = jax.lax.switch(
            stage,
            (fast_update, slow_update),
            position,
            value,
            adaptation_state,
        )

        warmup_state = jax.lax.cond(
            is_middle_window_end,
            slow_final,
            lambda x: x,
            warmup_state,
        )

        return warmup_state

    def final(warmup_state: AlbaAdaptationState) -> tuple[float, Array]:
        """Return the final values for the step size and mass matrix."""
        step_size = warmup_state.ss_state.step_size 
        # step_size = jnp.exp(warmup_state.ss_state.log_step_size_avg)
        inverse_mass_matrix = warmup_state.imm_state.inverse_mass_matrix
        L = warmup_state.L
        return step_size, L, inverse_mass_matrix

    return init, update, final

def unadjusted_alba(
    algorithm,
    logdensity_fn: Callable,
    target_eevpd,
    v,
    is_mass_matrix_diagonal: bool = True,
    progress_bar: bool = False,
    adaptation_info_fn: Callable = return_all_adapt_info,
    integrator=mcmc.integrators.velocity_verlet,
    num_alba_steps: int = 500,
    alba_factor: float = 0.4,
    **extra_parameters,
) -> AdaptationAlgorithm:
    

    mcmc_kernel = algorithm.build_kernel(integrator)

    adapt_init, adapt_step, adapt_final = base(
        is_mass_matrix_diagonal,
        target_eevpd=target_eevpd,
        v=v,
    )

    def one_step(carry, xs):
        _, rng_key, adaptation_stage = xs
        state, adaptation_state = carry

        new_state, info = mcmc_kernel(
            rng_key=rng_key,
            state=state,
            logdensity_fn=logdensity_fn,
            step_size=adaptation_state.step_size,
            inverse_mass_matrix=adaptation_state.inverse_mass_matrix,
            L=adaptation_state.L,
            **extra_parameters,
        )
        new_adaptation_state = adapt_step(
            adaptation_state,
            adaptation_stage,
            new_state.position,
            info.energy_change,
        )

        return (
            (new_state, new_adaptation_state),
            adaptation_info_fn(new_state, info, new_adaptation_state),
        )

    def run(rng_key: PRNGKey, position: ArrayLikeTree, num_steps: int = 1000):
        init_key, rng_key, alba_key = jax.random.split(rng_key, 3)
        init_state = algorithm.init(position=position, logdensity_fn=logdensity_fn, random_generator_arg=init_key)
        init_adaptation_state = adapt_init(position)

        if progress_bar:
            print("Running window adaptation")
        scan_fn = gen_scan_fn(num_steps-num_alba_steps, progress_bar=progress_bar)
        start_state = (init_state, init_adaptation_state)
        keys = jax.random.split(rng_key, num_steps-num_alba_steps)
        schedule = build_schedule(num_steps-num_alba_steps)
        last_state, info = scan_fn(
            one_step,
            start_state,
            (jnp.arange(num_steps-num_alba_steps), keys, schedule),
        )

        last_chain_state, last_warmup_state, *_ = last_state
        step_size, L, inverse_mass_matrix = adapt_final(last_warmup_state)

        ###
        ### ALBA TUNING
        ###
        keys = jax.random.split(alba_key, num_alba_steps)
        mcmc_kernel = algorithm.build_kernel(integrator)
        def step(state, key):
            next_state, _ = mcmc_kernel(
                rng_key=key,
                state=state,
                logdensity_fn=logdensity_fn,
                L=L,
                step_size=step_size,
                inverse_mass_matrix=inverse_mass_matrix,
            )

            return next_state, next_state.position
        
        if num_alba_steps > 0:
            _, samples = jax.lax.scan(step, last_chain_state, keys)
            flat_samples = jax.vmap(lambda x: ravel_pytree(x)[0])(samples)
            ess = effective_sample_size(flat_samples[None, ...])

            L=alba_factor * step_size * jnp.mean(num_alba_steps / ess)
        

        parameters = {
            "step_size": step_size,
            "inverse_mass_matrix": inverse_mass_matrix,
            "L": L,
            **extra_parameters,
        }

        return (
            AdaptationResults(
                last_chain_state,
                parameters,
            ),
            info,
        )

    return AdaptationAlgorithm(run)



