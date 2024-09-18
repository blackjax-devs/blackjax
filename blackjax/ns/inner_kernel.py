# Copyright 2024- Will Handley & David Yallup
from typing import Callable, NamedTuple, Optional
from typing import Callable, Dict, NamedTuple, Tuple
import jax
import jax.numpy as jnp

import blackjax.ns.base as base
from blackjax.base import SamplingAlgorithm
from blackjax.types import Array, ArrayLikeTree, ArrayTree, PRNGKey
from blackjax.ns.base import NSInfo, NSState
from blackjax.ns.base import init as init_base
from functools import partial
from blackjax.smc.inner_kernel_tuning import StateWithParameterOverride

__all__ = ["init", "as_top_level_api", "build_kernel"]


class NSState(NamedTuple):
    """State of the Nested Sampler."""

    particles: ArrayTree
    logL: Array  # The log-likelihood of the particles
    logL_birth: (
        Array  # The hard likelihood threshold of each particle at birth
    )
    logL_star: float  # The current hard likelihood threshold


class NSInfo(NamedTuple):
    """Additional information on the NS step."""

    particles: ArrayTree
    logL: Array  # The log-likelihood of the particles
    logL_birth: (
        Array  # The hard likelihood threshold of each particle at birth
    )


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
    contour_fn: Callable,
    mcmc_step_fn: Callable,
    mcmc_init_fn: Callable,
    mcmc_parameter_update_fn: Callable[
        [NSState, NSInfo], Dict[str, ArrayTree]
    ],
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

        logL_birth = state.sampler_state.logL_star
        val, dead_idx = delete_fn(state.sampler_state.logL)

        dead_particles = jax.tree.map(
            lambda x: x[dead_idx], state.sampler_state.particles
        )
        dead_logL = state.sampler_state.logL[dead_idx]
        dead_logL_birth = state.sampler_state.logL_birth[dead_idx]

        shared_mcmc_parameters = {}
        unshared_mcmc_parameters = {}
        for k, v in extra_step_parameters.items():
            if v.shape[0] == 1:
                shared_mcmc_parameters[k] = v[0, ...]
            else:
                unshared_mcmc_parameters[k] = v

        shared_mcmc_step_fn = partial(
            mcmc_step_fn, logdensity=logprior_fn, **state.parameter_override
        )

        contour_check_fn = lambda x: x <= -val.min()

        def particle_map(xs):
            xs,rng = xs
            state = mcmc_init_fn(xs, logprior_fn)

            def chain_scan(carry, xs):
                """Middle loop to scan over required MCMC steps."""

                def cond_fun(carry):
                    # _, _, logL, MHaccept = carry
                    _, _, logL = carry

                    return contour_check_fn(logL)  #& jnp.logical_not(MHaccept)

                def inner_chain(carry):
                    """Inner most while to check steps are in contour"""
                    # key, state, _, _ = carry
                    key, state, _ = carry
                    rng_key, subkey = jax.random.split(key)
                    new_state, info = shared_mcmc_step_fn(subkey, state)
                    logL = loglikelihood_fn(new_state.position)
                    # return rng_key, new_state, logL, info.is_accepted
                    return rng_key, new_state, logL

                state, _ = carry
                rng_key, step_key = jax.random.split(xs[0])
                # _, state, logL, _ = jax.lax.while_loop(
                #     cond_fun, inner_chain, (step_key, state, -jnp.inf, False)
                # )
                _, state, logL = jax.lax.while_loop(
                    cond_fun, inner_chain, (step_key, state, -jnp.inf)
                )
                return (state, logL), (rng_key, state, logL)

            (fs, fl), (rng, s, l) = jax.lax.scan(
                chain_scan, (state, -jnp.inf), (rng, jnp.zeros(rng.shape[0]))
            )
            return fs.position, fl

        scan_keys = jax.random.split(
            rng_key, (*dead_idx.shape, num_mcmc_steps)
        )

        # particle_map((dead_particles[0], scan_keys[0]))

        new_pos,new_logl = jax.pmap(particle_map)((dead_particles, scan_keys))
        logL_births = -val.min() * jnp.ones(dead_idx.shape)

        particles = state.sampler_state.particles.at[dead_idx].set(
            new_pos.squeeze()
        )
        logL = state.sampler_state.logL.at[dead_idx].set(new_logl.squeeze())
        logL_birth = state.sampler_state.logL_birth.at[dead_idx].set(
            logL_births
        )
        logL_star = state.sampler_state.logL.min()

        state = NSState(
            particles,
            logL,
            logL_birth,
            logL_star,
        )
        info = NSInfo(dead_particles, dead_logL, dead_logL_birth)
        new_parameter_override = mcmc_parameter_update_fn(state, info)
        return StateWithParameterOverride(state, new_parameter_override), info

    return kernel


def delete_fn(logL, n_delete):
    val, idx = jax.lax.top_k(-logL, n_delete)
    return val, idx


def contour_fn(logL, lstar):
    return logL <= lstar


def as_top_level_api(
    logprior_fn: Callable,
    loglikelihood_fn: Callable,
    mcmc_step_fn: Callable,
    mcmc_init_fn: Callable,
    mcmc_parameter_update_fn: Callable[
        [NSState, NSInfo], Dict[str, ArrayTree]
    ],
    mcmc_initial_parameters: dict,
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
        contour_fn,
        mcmc_step_fn,
        mcmc_init_fn,
        mcmc_parameter_update_fn,
        num_mcmc_steps,
        **extra_parameters,
    )

    def init_fn(position: ArrayLikeTree, rng_key=None):
        del rng_key
        return init(position, loglikelihood_fn, mcmc_initial_parameters)

    def step_fn(rng_key: PRNGKey, state, **extra_parameters):
        return kernel(rng_key, state, **extra_parameters)

    return SamplingAlgorithm(init_fn, step_fn)
