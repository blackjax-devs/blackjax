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
    logL_birth: Array  # The hard likelihood threshold of each particle at birth
    logL_star: float  # The current hard likelihood threshold
    pid: Array = Array  # particle ID
    logX: float = 0.0  # The current log-volume estiamte
    logZ_live: float = -jnp.inf  # The current evidence estimate
    logZ: float = -jnp.inf  # The accumulated evidence estimate


class NSInfo(NamedTuple):
    """Additional information on the NS step."""

    particles: ArrayTree
    logL: Array  # The log-likelihood of the particles
    logL_birth: Array  # The hard likelihood threshold of each particle at birth
    update_info: NamedTuple


def init(particles: ArrayLikeTree, loglikelihood_fn, logL_star=-jnp.inf) -> NSState:
    logL = jax.vmap(loglikelihood_fn)(particles)
    logL_birth = logL_star * jnp.ones_like(logL)
    pid = jnp.arange(len(logL))
    return NSState(particles, logL, logL_birth, logL_star, pid)


def build_kernel(
    logprior_fn: Callable,
    loglikelihood_fn: Callable,
    delete_fn: Callable,
    mcmc_build_kernel: Callable,
    mcmc_init_fn: Callable,
    num_mcmc_steps: int,
) -> Callable:
    r"""Build a Nested Sampling by running a vectorized delete and creation of particles.
    This base version does not tune the inner kernel parameters. Consequently,
    it can be inefficient.

    Parameters
    ----------
    logprior_fn : Callable
        A function that computes the log prior probability.
    loglikelihood_fn : Callable
        A function that computes the log likelihood.
    delete_fn : Callable
        Function that takes an array of log likelihoods and marks particles for deletion and updates.
    mcmc_build_kernel:
        A function that builds an mcmc kernel
    mcmc_init_fn
        A callable that initializes the inner kernel
    num_mcmc_steps: int
        Number of MCMC steps to perform. Recommended is 5 times the dimension of the parameter space.

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

        # Delete, and grab all the dead information
        rng_key, delete_fn_key = jax.random.split(rng_key)
        dead_idx, live_idx = delete_fn(delete_fn_key, state)

        dead_particles = jax.tree.map(lambda x: x[dead_idx], state.particles)
        dead_logL = state.logL[dead_idx]
        dead_logL_birth = state.logL_birth[dead_idx]
        logL0 = dead_logL.max()
        num_deleted = len(dead_idx)

        # Resample the live particles
        kernel = mcmc_build_kernel(**mcmc_parameters)
        rng_key, sample_key = jax.random.split(rng_key)

        def logdensity_fn(x):
            return jnp.where(loglikelihood_fn(x) > logL0, logprior_fn(x), -jnp.inf)

        def num_mcmc_steps_kernel(rng_key, particles):
            def body_fn(state, rng_key):
                new_state, info = kernel(rng_key, state, logdensity_fn)
                return new_state, info

            init = mcmc_init_fn(particles, logprior_fn)
            keys = jax.random.split(rng_key, num_mcmc_steps)
            last_state, info = jax.lax.scan(body_fn, init, keys)
            return last_state, info

        new_particles = jax.tree.map(lambda x: x[live_idx], state.particles)
        sample_keys = jax.random.split(sample_key, num_deleted)
        new_state, new_state_info = jax.vmap(num_mcmc_steps_kernel)(
            sample_keys, new_particles
        )

        # Update the particles
        particles = jax.tree_util.tree_map(
            lambda p, n: p.at[dead_idx].set(n),
            state.particles,
            new_state.position,
        )
        new_state_loglikelihood = jax.vmap(loglikelihood_fn)(new_state.position)
        logL = state.logL.at[dead_idx].set(new_state_loglikelihood)
        logL_births = logL0 * jnp.ones(num_deleted)
        logL_birth = state.logL_birth.at[dead_idx].set(logL_births)
        logL_star = state.logL.min()
        pid = state.pid.at[dead_idx].set(state.pid[live_idx])

        # Update the logX and logZ
        num_particles = len(state.logL)
        num_live = jnp.arange(num_particles, num_particles - num_deleted, -1)
        delta_log_xi = -1 / num_live
        log_delta_xi = (
            state.logX + jnp.cumsum(delta_log_xi) + jnp.log(1 - jnp.exp(delta_log_xi))
        )
        delta_logz_dead = dead_logL + log_delta_xi

        logX = state.logX + delta_log_xi.sum()
        logZ_dead = jnp.logaddexp(
            state.logZ, jax.scipy.special.logsumexp(delta_logz_dead)
        )
        logZ_live = jax.scipy.special.logsumexp(logL) - jnp.log(num_particles) + logX

        # Update the state
        new_state = NSState(
            particles,
            logL,
            logL_birth,
            logL_star,
            pid,
            logX=logX,
            logZ=logZ_dead,
            logZ_live=logZ_live,
        )
        info = NSInfo(dead_particles, dead_logL, dead_logL_birth, new_state_info)
        return new_state, info

    return kernel


def delete_fn(key, state, n_delete):
    """Analogous to resampling functions in SMC, defines the likelihood level and associated particles to delete.
    As well as resampling live particles to then evolve.

     Parameters:
    -----------
    key : jax.random.PRNGKey
        A PRNG key used for random number generation.
    state : NSState
        The current state of the Nested Sampler.
    n_delete : int
        Number of particles to delete and resample.

    Returns:
    --------
    dead_idx : jnp.ndarray
        Indices of particles to be deleted.
    live_idx : jnp.ndarray
        Indices of resampled particles to evolve.
    """
    logL = state.logL
    neg_dead_logL, dead_idx = jax.lax.top_k(-logL, n_delete)
    weights = jnp.array(logL > -neg_dead_logL.min(), dtype=jnp.float32)
    live_idx = jax.random.choice(
        key,
        len(weights),
        shape=(n_delete,),
        p=weights / weights.sum(),
        replace=True,
    )
    return dead_idx, live_idx


def as_top_level_api(
    logprior_fn: Callable,
    loglikelihood_fn: Callable,
    mcmc_build_kernel: Callable,
    mcmc_init_fn: Callable,
    mcmc_parameters: dict,
    num_mcmc_steps: int,
    n_delete: int = 1,
) -> SamplingAlgorithm:
    """Implements the user interface for the Nested Sampling kernel.

    Parameters
    ----------
    logprior_fn : Callable
        A function that computes the log prior probability.
    loglikelihood_fn : Callable
        A function that computes the log likelihood.
    mcmc_build_kernel:
        A function that builds an mcmc kernel
    mcmc_init_fn
        A callable that initializes the inner kernel
    num_mcmc_steps: int
        Number of MCMC steps to perform. Recommended is 5 times the dimension of the parameter space.
    n_delete: int, optional
        Number of particles to delete in each iteration. Default is 1.

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
        mcmc_build_kernel,
        mcmc_init_fn,
        num_mcmc_steps,
    )

    def init_fn(particles: ArrayLikeTree, rng_key=None):
        del rng_key
        return init(particles, loglikelihood_fn)

    def step_fn(rng_key: PRNGKey, state):
        return kernel(rng_key, state, mcmc_parameters)

    return SamplingAlgorithm(init_fn, step_fn)
