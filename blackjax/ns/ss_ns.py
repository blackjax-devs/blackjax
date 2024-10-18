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
    logL_birth: (Array)  # The hard likelihood threshold of each particle at birth
    logL_star: float  # The current hard likelihood threshold
    logX: float = 0.0  # The current log-volume estiamte
    logZ_live: float = -jnp.inf  # The current evidence estimate
    logZ: float = -jnp.inf  # The accumulated evidence estimate


class NSInfo(NamedTuple):
    """Additional information on the NS step."""

    particles: ArrayTree
    logL: Array  # The log-likelihood of the particles
    logL_birth: (Array)  # The hard likelihood threshold of each particle at birth


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
        val, dead_idx, live_idx = delete_fn(delete_fn_key, state.sampler_state.logL)

        logL0 = val.min()
        dead_particles = jax.tree.map(
            lambda x: x[dead_idx], state.sampler_state.particles
        )
        dead_logL = state.sampler_state.logL[dead_idx]
        dead_logL_birth = state.sampler_state.logL_birth[dead_idx]

        def mcmc_step(i, carry):
            rng_key, new_pos, new_logl = carry

            rng_key, vertical_slice_key = jax.random.split(rng_key)
            logpi = logprior_fn(new_pos)
            logpi0 = logpi + jnp.log(
                jax.random.uniform(vertical_slice_key, shape=(live_idx.shape[0],))
            )

            rng_key, horizontal_slice_key = jax.random.split(rng_key)
            new_pos, new_logl = horizontal_slice_proposal(
                horizontal_slice_key,
                new_pos,
                state.parameter_override["cov"],
                loglikelihood_fn,
                logL0,
                logprior_fn,
                logpi0,
            )
            return rng_key, new_pos, new_logl

        new_pos = state.sampler_state.particles[live_idx]
        new_logl = state.sampler_state.logL[live_idx]
        rng_key, new_pos, new_logl = jax.lax.fori_loop(
            0, num_mcmc_steps, mcmc_step, (rng_key, new_pos, new_logl)
        )

        logL_births = logL0 * jnp.ones(dead_idx.shape)
        particles = state.sampler_state.particles.at[dead_idx].set(new_pos.squeeze())
        logL = state.sampler_state.logL.at[dead_idx].set(new_logl.squeeze())
        logL_birth = state.sampler_state.logL_birth.at[dead_idx].set(logL_births)
        logL_star = state.sampler_state.logL.min()

        delta_log_xi = -dead_idx.shape[0] / state.sampler_state.particles.shape[0]
        log_delta_xi = state.sampler_state.logX + jnp.log(1-jnp.exp(delta_log_xi))
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
    n = jax.random.multivariate_normal(subkey, jnp.zeros(x0.shape[1]), cov, shape=(x0.shape[0],))  # Standard normal samples

    # Compute Mahalanobis norms and normalize n
    invcov = jnp.linalg.inv(cov)
    norm = jnp.sqrt(jnp.einsum("...i,...ij,...j", n, invcov, n))
    n = n / norm[..., None]

    # Initial bounds
    key, subkey = jax.random.split(key)
    w = jax.random.uniform(subkey, shape=(x0.shape[0],))
    l = x0 + w[:, None] * n
    r = x0 + (w[:, None] - 1) * n

    # Expand l
    def expand_l(carry):
        l, within = carry
        l = l + within[:, None] * n
        within = jnp.logical_and(logL(l) > logL0, logpi(l) > logpi0)
        return l, within

    def cond_fun_l(carry):
        within = carry[1]
        return jnp.any(within)

    within = jnp.ones(x0.shape[0], dtype=bool)
    carry = (l, within)
    l, l_exp = jax.lax.while_loop(cond_fun_l, expand_l, carry)

    # Expand r
    def expand_r(carry):
        r, within = carry
        r = r - within[:, None] * n
        within = jnp.logical_and(logL(r) > logL0, logpi(r) > logpi0)
        return r, within

    def cond_fun_r(carry):
        within = carry[1]
        return jnp.any(within)

    within = jnp.ones(x0.shape[0], dtype=bool)
    carry = (r, within)
    r, r_exp = jax.lax.while_loop(cond_fun_r, expand_r, carry)

    # Shrink
    def shrink_step(carry):
        l, r, _, _, key, within = carry
        key, subkey = jax.random.split(key)
        u = jax.random.uniform(subkey, shape=(x0.shape[0],))
        x1 = l + u[:, None] * (r - l)
        logLx1 = logL(x1)
        within_new = jnp.logical_and(logLx1 > logL0, logpi(x1) > logpi0)
        s = jnp.sum((x1 - x0) * (r - l), axis=-1) > 0
        condition_l = (~within_new) & (~s)
        l = jnp.where(condition_l[:, None], x1, l)
        condition_r = (~within_new) & s
        r = jnp.where(condition_r[:, None], x1, r)
        return l, r, x1, logLx1, key, within_new

    def cond_fun(carry):
        within = carry[-1]
        return ~jnp.all(within)

    within = jnp.zeros(x0.shape[0], dtype=bool)
    carry = (l, r, x0, jnp.zeros(x0.shape[0]), key, within)
    l, r, x1, logl, key, within = jax.lax.while_loop(cond_fun, shrink_step, carry)

    return x1, logl


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
