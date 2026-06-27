from functools import partial
from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp

from blackjax.ns.adaptive import build_kernel as build_adaptive_kernel
from blackjax.ns.base import delete_fn as default_delete_fn


class MCMCUpdateInfo(NamedTuple):
    """Thin layer to hold all the info pertaining to the update step."""

    mcmc_states: NamedTuple
    mcmc_infos: NamedTuple


class ConstrainedMCMCInfo(NamedTuple):
    """Info for a constrained MCMC proposal.

    Attributes
    ----------
    info
        The underlying MCMC info (e.g., RWInfo for random walk).
    is_accepted
        True if both the MCMC proposal was accepted AND the proposed
        point is above the likelihood threshold.
    """

    info: NamedTuple
    is_accepted: jnp.ndarray


def update_with_mcmc_take_last(
    constrained_mcmc_step_fn,
    num_mcmc_steps,
    num_delete,
):
    """An update strategy for NS that uses MCMC to update the particles.
    For now we will not keep the states as they will be too large to store.
    Similar to the update_and_take_last from SMC.

    Parameters
    ----------
    constrained_mcmc_step_fn
        Wrapped MCMC step function that enforces the NS likelihood constraint.
    num_mcmc_steps
        Number of MCMC proposals per particle.
    num_delete
        Number of particles to replace per step.
    """

    def update_function(rng_key, state, loglikelihood_0, **step_parameters):
        choice_key, sample_key = jax.random.split(rng_key)
        particles = state.particles

        # Select start particles from survivors
        weights = (particles.loglikelihood > loglikelihood_0).astype(jnp.float32)
        weights = jnp.where(weights.sum() > 0.0, weights, jnp.ones_like(weights))
        start_idx = jax.random.choice(
            choice_key,
            len(weights),
            shape=(num_delete,),
            p=weights / weights.sum(),
            replace=True,
        )
        start_state = jax.tree.map(lambda x: x[start_idx], particles)

        shared_mcmc_step_fn = partial(
            constrained_mcmc_step_fn,
            loglikelihood_0=loglikelihood_0,
            **step_parameters,
        )

        def mcmc_kernel(rng_key, state):
            keys = jax.random.split(rng_key, num_mcmc_steps)

            def body_fn(state, rng_key):
                new_state, info = shared_mcmc_step_fn(rng_key, state)
                return new_state, info

            final_state, infos = jax.lax.scan(body_fn, state, keys)
            return final_state, infos

        sample_keys = jax.random.split(sample_key, num_delete)
        return jax.vmap(mcmc_kernel)(sample_keys, start_state)

    return update_function


def build_kernel(
    init_state_fn: Callable,
    logdensity_fn: Callable,
    mcmc_init_fn: Callable,
    mcmc_step_fn: Callable,
    num_inner_steps: int,
    update_inner_kernel_params_fn: Callable,
    num_delete: int = 1,
    delete_fn: Callable = default_delete_fn,
) -> Callable:
    """Builds a Nested Sampling kernel wrapping any MCMC algorithm.

    Parameters
    ----------
    init_state_fn
        Function to initialize a NS particle state from a position.
    logdensity_fn
        Log-density function (typically the prior log-probability).
    mcmc_init_fn
        Function to initialize MCMC state from position and logdensity_fn.
    mcmc_step_fn
        MCMC step function with signature (rng_key, state, logdensity_fn, **params).
    num_inner_steps
        Number of MCMC steps per particle replacement.
    update_inner_kernel_params_fn
        Function to update MCMC kernel parameters adaptively.
    num_delete
        Number of particles to replace per NS iteration.
    delete_fn
        Function to select which particles to delete.
    """

    def constrained_mcmc_step_fn(rng_key, state, loglikelihood_0, **params):
        """Single constrained MCMC step that respects the likelihood threshold.

        Proposes a move, accepts if both the MCMC acceptance criterion is
        satisfied AND the proposed point is above the likelihood threshold.
        If rejected, stays at current position.
        """
        mcmc_state = mcmc_init_fn(state.position, logdensity_fn)
        new_mcmc_state, mcmc_info = mcmc_step_fn(
            rng_key, mcmc_state, logdensity_fn, **params
        )
        proposed_state = init_state_fn(
            new_mcmc_state.position, loglikelihood_birth=loglikelihood_0
        )
        within_contour = proposed_state.loglikelihood > loglikelihood_0
        proposal_accepted = getattr(mcmc_info, "is_accepted", True)
        is_accepted = proposal_accepted & within_contour
        new_state = jax.lax.cond(
            is_accepted,
            lambda _: proposed_state,
            lambda _: state,
            operand=None,
        )
        info = ConstrainedMCMCInfo(mcmc_info, is_accepted)
        return new_state, info

    inner_kernel = update_with_mcmc_take_last(
        constrained_mcmc_step_fn, num_inner_steps, num_delete
    )

    delete_fn = partial(delete_fn, num_delete=num_delete)

    kernel = build_adaptive_kernel(
        delete_fn,
        inner_kernel,
        update_inner_kernel_params_fn=update_inner_kernel_params_fn,
    )
    return kernel
