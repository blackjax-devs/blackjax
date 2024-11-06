# Copyright 2024- Will Handley & David Yallup
from functools import partial
from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp

from blackjax import SamplingAlgorithm
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
    update_info: NamedTuple


def init(particles: ArrayLikeTree, loglikelihood_fn):
    logL_star = -jnp.inf
    num_particles = jax.tree_util.tree_flatten(particles)[0][0].shape[0]
    logL_birth = logL_star * jnp.ones(num_particles)
    logL = loglikelihood_fn(particles)
    return NSState(particles, logL, logL_birth, logL_star)


def build_kernel(
    logprior_fn: Callable,
    loglikelihood_fn: Callable,
    delete_fn: Callable,
    mcmc_step_fn: Callable,
    mcmc_init_fn: Callable,
    num_mcmc_steps: int = 10,
) -> Callable:
    r"""Build a Nested Sampling by running a creation and deletion step.
    This base version does not tune the inner kernel parameters. Consequently,
    it will rapidly become inefficient.

    Parameters
        Parameters
    ----------
    logprior_fn : Callable
        A function that computes the log prior probability.
    loglikelihood_fn : Callable
        A function that computes the log likelihood.
    delete_fn : Callable
        Function that takes an array of keys and particles and deletes some
        particles.
    parameter_update_fn : Callable[[NSState, NSInfo], Dict[str, ArrayTree]]
        Function that updates the parameters of the inner kernel.
    num_mcmc_steps : int, optional
        Number of MCMC steps to perform, by default 10.

    Returns
    -------
    Callable
        A function that takes a rng_key and a NSState that contains the current state
        of the chain and returns a new state of the chain along with
        information about the transition.
    """

    def kernel(
        rng_key: PRNGKey,
        state: NSState,
        mcmc_parameters: dict,
    ) -> tuple[NSState, NSInfo]:
        num_particles = jnp.shape(jax.tree_leaves(state.particles)[0])[0]
        rng_key, delete_fn_key = jax.random.split(rng_key)
        val, dead_idx, live_idx = delete_fn(delete_fn_key, state.logL)

        logL0 = val.max()
        dead_particles = jax.tree.map(lambda x: x[dead_idx], state.particles)
        dead_logL = state.logL[dead_idx]
        dead_logL_birth = state.logL_birth[dead_idx]

        new_pos = jax.tree.map(lambda x: x[live_idx], state.particles)
        new_logl = state.logL[live_idx]

        kernel = mcmc_step_fn(logprior_fn, loglikelihood_fn, logL0, **mcmc_parameters)

        def mcmc_step(carry, xs):
            state, k = carry
            k, subk = jax.random.split(k, 2)
            state, info = kernel(subk, state)
            return (state, k), info

        rng_key, sample_key = jax.random.split(rng_key)

        mcmc_state = mcmc_init_fn(new_pos, logprior_fn, new_logl)

        (new_state, rng_key), new_state_info = jax.lax.scan(
            mcmc_step, (mcmc_state, sample_key), length=num_mcmc_steps
        )

        logL_births = logL0 * jnp.ones(dead_idx.shape)
        # particles = state.particles.at[dead_idx].set(new_state.position)
        particles = jax.tree_util.tree_map(
            lambda p, n: p.at[dead_idx].set(n),
            state.particles,
            new_state.position,
        )
        logL = state.logL.at[dead_idx].set(new_state.loglikelihood)
        logL_birth = state.logL_birth.at[dead_idx].set(logL_births)
        logL_star = state.logL.min()

        delta_log_xi = -dead_idx.shape[0] / num_particles
        log_delta_xi = state.logX + jnp.log(1 - jnp.exp(delta_log_xi))
        delta_logz_dead = state.logL_star + log_delta_xi

        # logX = jnp.logaddexp(state.sampler_state.logX, delta_xi)
        logX = state.logX + delta_log_xi
        logZ_dead = jnp.logaddexp(state.logZ, delta_logz_dead)
        logZ_live = logL0 + logX

        new_state = NSState(
            particles,
            logL,
            logL_birth,
            logL_star,
            logX=logX,
            logZ=logZ_dead,
            logZ_live=logZ_live,
        )
        info = NSInfo(dead_particles, dead_logL, dead_logL_birth, new_state_info)
        return new_state, info

    return kernel


def delete_fn(key, logL, n_delete):
    """Analogous to resampling functions in SMC, defines the likelihood level and associated particles to delete.
    As well as resampling live particles to then evolve.

     Parameters:
    -----------
    key : jax.random.PRNGKey
        A PRNG key used for random number generation.
    logL : jnp.ndarray
        Array of log-likelihood values for the current set of particles.
    n_delete : int
        Number of particles to delete and resample.

    Returns:
    --------
    val : jnp.ndarray
        log likelihood threshold of live particles.
    dead_idx : jnp.ndarray
        Indices of particles to be deleted.
    live_idx : jnp.ndarray
        Indices of resampled particles to evolve.
    """
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


def as_top_level_api(
    logprior_fn: Callable,
    loglikelihood_fn: Callable,
    mcmc_step_fn: Callable,
    mcmc_init_fn: Callable,
    mcmc_parameters: dict,
    num_mcmc_steps: int = 10,
    n_delete: int = 1,
) -> SamplingAlgorithm:
    """Implements the (basic) user interface for the Adaptive Nested Sampling kernel.
    Parameters
    ----------
    logprior_fn: Callable
        A function that computes the log prior probability.
    loglikelihood_fn: Callable
        A function that computes the log likelihood.
    parameter_update_fn: Callable
        A function that updates the parameters given the current state and info.
    initial_parameters: dict
        Initial parameters for the inner kernel.
    num_mcmc_steps: int, optional
        Number of MCMC steps to perform. Default is 10.
    n_delete: int, optional
        Number of particles to delete in each iteration. Default is 1.
    **extra_parameters
        Additional parameters for the algorithm.

    Returns
    -------
    SamplingAlgorithm
        A sampling algorithm object.
    """
    delete_func = partial(delete_fn, n_delete=n_delete)

    kernel = build_kernel(
        logprior_fn,
        loglikelihood_fn,
        delete_func,
        mcmc_step_fn,
        mcmc_init_fn,
        num_mcmc_steps,
    )

    def init_fn(position: ArrayLikeTree, rng_key=None):
        del rng_key
        return init(position, loglikelihood_fn)

    def step_fn(rng_key: PRNGKey, state):
        return kernel(rng_key, state, mcmc_parameters)

    return SamplingAlgorithm(init_fn, step_fn)
