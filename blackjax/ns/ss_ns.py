# Copyright 2024- Will Handley & David Yallup
from functools import partial
from typing import Callable, Dict, NamedTuple, Optional, Tuple

import jax
import jax.numpy as jnp
from jaxopt._src.loop import while_loop

import blackjax.ns.base as base
from blackjax.base import SamplingAlgorithm
from blackjax.ns.base import NSInfo, NSState
from blackjax.ns.base import init as init_base
from blackjax.smc.inner_kernel_tuning import StateWithParameterOverride
from blackjax.types import Array, ArrayLikeTree, ArrayTree, PRNGKey
from blackjax.ns.vectorized_slice import build_kernel as build_kernel_slice
from blackjax.ns.vectorized_slice import init as init_slice

__all__ = ["init", "as_top_level_api", "build_kernel"]


class NSState(NamedTuple):
    """State of the Nested Sampler."""

    particles: ArrayTree
    logL: Array  # The log-likelihood of the particles
    logL_birth: (
        Array  # The hard likelihood threshold of each particle at birth
    )
    logL_star: float  # The current hard likelihood threshold
    logX: float = 0.0  # The current log-volume estiamte
    logZ_live: float = -jnp.inf  # The current evidence estimate
    logZ: float = -jnp.inf  # The accumulated evidence estimate


class NSInfo(NamedTuple):
    """Additional information on the NS step."""

    particles: ArrayTree
    logL: Array  # The log-likelihood of the particles
    logL_birth: (
        Array  # The hard likelihood threshold of each particle at birth
    )
    update_info: NamedTuple

def init_base(particles: ArrayLikeTree, loglikelihood_fn):
    logL_star = -jnp.inf
    num_particles = jax.tree_util.tree_flatten(particles)[0][0].shape[0]
    logL_birth = logL_star * jnp.ones(num_particles)
    logL = loglikelihood_fn(particles)
    return NSState(particles, logL, logL_birth, logL_star)


def init(position, loglikelihood_fn, initial_parameter_value):
    return StateWithParameterOverride(
        init_base(position, loglikelihood_fn), initial_parameter_value
    )


def build_kernel(
    logprior_fn: Callable,
    loglikelihood_fn: Callable,
    delete_fn: Callable,
    parameter_update_fn: Callable[[NSState, NSInfo], Dict[str, ArrayTree]],
    num_mcmc_steps: int = 10,
    **extra_parameters,
) -> Callable:
    r"""Build a Nested Sampling by running a creation and deletion step.
    Parameters
    ----------
    logL_fn: Callable
        A function that assigns a weight to the particles.
    create_fn: Callable
        Function that takes an array of keys and particles and returns
        new particles.
    delete_fn: Callable
        Function that takes an array of keys and particles and deletes some
        particles.
    Returns
    -------
    A callable that takes a rng_key and a NSState that contains the current state
    of the chain and that returns a new state of the chain along with
    information about the transition.
    """

    def kernel(
        rng_key: PRNGKey,
        state: base.NSState,
        **extra_step_parameters,
    ) -> tuple[base.NSState, base.NSInfo]:
        rng_key, delete_fn_key = jax.random.split(rng_key)
        val, dead_idx, live_idx = delete_fn(
            delete_fn_key, state.sampler_state.logL
        )

        logL0 = val.min()
        dead_particles = jax.tree.map(
            lambda x: x[dead_idx], state.sampler_state.particles
        )
        dead_logL = state.sampler_state.logL[dead_idx]
        dead_logL_birth = state.sampler_state.logL_birth[dead_idx]

        new_pos = state.sampler_state.particles[live_idx]
        new_logl = state.sampler_state.logL[live_idx]

        kernel = build_kernel_slice(
            state.parameter_override["cov"],
            logL0,
            logprior_fn,
            loglikelihood_fn,
        )

        def mcmc_step(carry, xs):
            state, k = carry
            k, subk = jax.random.split(k, 2)
            state, info = kernel(subk, state)
            return (state, k), info

        rng_key, sample_key = jax.random.split(rng_key)
        mcmc_state = init_slice(new_pos, logprior_fn, new_logl)
        (new_state, rng_key), new_state_info = jax.lax.scan(
            mcmc_step, (mcmc_state, sample_key), length=num_mcmc_steps
        )

        logL_births = logL0 * jnp.ones(dead_idx.shape)
        particles = state.sampler_state.particles.at[dead_idx].set(
            new_state.position
        )
        logL = state.sampler_state.logL.at[dead_idx].set(
            new_state.loglikelihood
        )
        logL_birth = state.sampler_state.logL_birth.at[dead_idx].set(
            logL_births
        )
        logL_star = state.sampler_state.logL.min()

        delta_log_xi = (
            -dead_idx.shape[0] / state.sampler_state.particles.shape[0]
        )
        log_delta_xi = state.sampler_state.logX + jnp.log(
            1 - jnp.exp(delta_log_xi)
        )
        delta_logz_dead = logL0 + log_delta_xi

        # logX = jnp.logaddexp(state.sampler_state.logX, delta_xi)
        logX = state.sampler_state.logX + delta_log_xi
        logZ_dead = jnp.logaddexp(state.sampler_state.logZ, delta_logz_dead)
        logZ_live = logL_star + logX

        state = NSState(
            particles,
            logL,
            logL_birth,
            logL_star,
            logX=logX,
            logZ=logZ_dead,
            logZ_live=logZ_live,
        )
        info = NSInfo(
            dead_particles,
            dead_logL,
            dead_logL_birth,
            new_state_info
        )
        new_parameter_override = parameter_update_fn(state, info)
        return StateWithParameterOverride(state, new_parameter_override), info

    return kernel


def delete_fn(key, logL, n_delete):
    val, dead_idx = jax.lax.top_k(-logL, n_delete)
    weights = jnp.array(logL > -val.min(), dtype=jnp.float32)
    live_idx = jax.random.choice(
        key,
        weights.shape[0],
        shape=(n_delete,),
        p=weights / weights.sum(),
        replace=True,
    )
    return -val, dead_idx, live_idx


def contour_fn(logL, lstar):
    return logL <= lstar


def as_top_level_api(
    logprior_fn: Callable,
    loglikelihood_fn: Callable,
    parameter_update_fn: Callable[[NSState, NSInfo], Dict[str, ArrayTree]],
    initial_parameters: dict,
    num_mcmc_steps: int = 10,
    n_delete: int = 1,
    **extra_parameters,
) -> SamplingAlgorithm:
    """Implements the (basic) user interface for the Adaptive Tempered SMC kernel.

    Parameters
    ----------
    logL_fn: Callable
        A function that assigns a weight to the particles.
    create_fn: Callable
        Function that takes an array of keys and particles and returns
        new particles.
    delete_fn: Callable
        Function that takes an array of keys and particles and deletes some
        particles.
    Returns
    -------
    A ``SamplingAlgorithm``.

    """
    delete_func = partial(delete_fn, n_delete=n_delete)

    kernel = build_kernel(
        logprior_fn,
        loglikelihood_fn,
        delete_func,
        parameter_update_fn,
        num_mcmc_steps,
        **extra_parameters,
    )

    def init_fn(position: ArrayLikeTree, rng_key=None):
        del rng_key
        return init(position, loglikelihood_fn, initial_parameters)

    def step_fn(rng_key: PRNGKey, state, **extra_parameters):
        return kernel(rng_key, state, **extra_parameters)

    return SamplingAlgorithm(init_fn, step_fn)