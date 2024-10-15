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
        val, dead_idx = delete_fn(state.sampler_state.logL)
        logL0 = val.min()

        dead_particles = jax.tree.map(
            lambda x: x[dead_idx], state.sampler_state.particles
        )
        dead_logL = state.sampler_state.logL[dead_idx]
        dead_logL_birth = state.sampler_state.logL_birth[dead_idx]

        rng_key, choice_key = jax.random.split(rng_key)

        # particle_map((dead_particles[0], scan_keys[0]))
        idx = jax.random.choice(
            choice_key,
            state.sampler_state.particles.shape[0],
            shape=(dead_particles.shape[0],),
        )

        # TODO loop this
        logpi = logprior_fn(state.sampler_state.particles[idx])
        rng_key, vertical_slice_key = jax.random.split(rng_key)
        logpi0 = logpi * jnp.log(jnp.random.uniform(vertical_slice_key, shape=(idx.shape[0],)))

        rng_key, horizontal_slice_key = jax.random.split(rng_key)
        new_pos, new_logl = horizontal_slice_proposal(
            horizontal_slice_key,
            state.sampler_state.particles[idx],
            state.parameter_override["cov"],
            loglikelihood_fn,
            logL0,
            logprior_fn,
            logpi0,
        )
        logL_births = logL0 * jnp.ones(dead_idx.shape)

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
        new_parameter_override = parameter_update_fn(state, info)
        return StateWithParameterOverride(state, new_parameter_override), info

    return kernel


def horizontal_slice_proposal(key, x0, cov, logL, logL0, logpi, logpi0):
    # Ensure cov_ and x0 are JAX arrays
    cov = jnp.asarray(cov)
    x0 = jnp.atleast_2d(x0)

    # Random direction
    key, subkey = jax.random.split(key)
    n = jax.random.normal(subkey, shape=(x0.shape))  # Standard normal samples

    # Compute Mahalanobis norms and normalize n
    n_cov = jnp.dot(n, cov)  # Shape (nlive, D)
    n_sqr = jnp.sum(n * n_cov, axis=1)  # Shape (nlive,)
    norms = jnp.sqrt(n_sqr)
    n = n / norms[:, None]  # Normalize to unit Mahalanobis length

    # Initial bounds
    key, subkey = jax.random.split(key)
    w = jax.random.uniform(subkey, shape=(x0.shape[0],))
    l = x0 + w[:, None] * n
    r = x0 + (w[:, None] - 1) * n

    # Expand l
    def expand_l(carry):
        l, within = carry
        l = l + within[:, None] * n
        within = jnp.logical_and(logL(l) > logL0, logpi0 > logpi(l))
        return l, within

    def cond_fun_l(carry):
        within = carry[1]
        print(within)
        return jnp.any(within)

    within = jnp.ones((x0.shape[0], 1), dtype=bool)
    carry = (l, within)
    l, l_exp = jax.lax.while_loop(cond_fun_l, expand_l, carry)

    # Expand r
    def expand_r(carry):
        r, within = carry
        r = r - within[:, None] * n
        within = jnp.logical_and(logL(r) > logL0, logpi0 > logpi(r))
        return r, within

    def cond_fun_r(carry):
        within = carry[1]
        return jnp.any(within)

    within = jnp.ones((x0.shape[0], 1), dtype=bool)
    carry = (r, within)
    r, r_exp = jax.lax.while_loop(cond_fun_r, expand_r, carry)

    # Shrink
    def shrink_step(carry):
        l, r, _, _, key, within = carry
        key, subkey = jax.random.split(key)
        u = jax.random.uniform(subkey, shape=(x0.shape[0],))
        x1 = l + u[:, None] * (r - l)
        logLx1 = logL(x1)
        within_new = jnp.logical_and(logLx1 > logL0, logpi0 > logpi(x1))[..., None]
        s = (jnp.sum((x1 - x0) * (r - l), axis=-1) > 0)[:, None]
        condition_l = (~within_new[:, 0]) & (~s[:, 0])
        l = jnp.where(condition_l[:, None], x1, l)
        condition_r = (~within_new[:, 0]) & s[:, 0]
        r = jnp.where(condition_r[:, None], x1, r)
        return l, r, x1, logLx1, key, within_new

    def cond_fun(carry):
        within = carry[-1]
        return ~jnp.all(within)

    within = jnp.zeros((x0.shape[0], 1), dtype=bool)
    carry = (l, r, x0, jnp.zeros(x0.shape[0]), key, within)
    shrink_step(carry)
    l, r, x1, logl, key, within = jax.lax.while_loop(
        cond_fun, shrink_step, carry
    )

    return x1, logl


def delete_fn(logL, n_delete):
    val, idx = jax.lax.top_k(-logL, n_delete)
    return -val, idx


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
