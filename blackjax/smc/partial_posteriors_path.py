from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp

from blackjax.smc.from_mcmc import build_kernel as smc_from_mcmc
from blackjax.types import Array, ArrayTree


class PartialPosteriorsSMCState(NamedTuple):
    """Current state for the tempered SMC algorithm.

    particles: PyTree
        The particles' positions.
    weights: for

    """

    particles: ArrayTree
    weights: Array
    selector: Array


def init(particles, num_datapoints):
    num_particles = jax.tree_util.tree_flatten(particles)[0][0].shape[0]
    weights = jnp.ones(num_particles) / num_particles
    return PartialPosteriorsSMCState(particles, weights, jnp.zeros(num_datapoints))


def partial_posteriors_kernel(
    mcmc_step_fn: Callable,
    mcmc_init_fn: Callable,
    resampling_fn: Callable,
    num_mcmc_steps: int,
    mcmc_parameters: ArrayTree,
    partial_logposterior_factory: Callable[[Array], Callable],
):
    """Build the Partial Posteriors (data tempering) SMC kernel.
    The distribution's trajectory includes increasingly adding more
    datapoints to the likelihood.
     Parameters
     ----------
     mcmc_step_fn
         A function that computes the log density of the prior distribution
     mcmc_init_fn
         A function that returns the probability at a given
         position.
     resampling_fn
         A random function that resamples generated particles based of weights
     num_mcmc_steps
         Number of iterations in the MCMC chain.
     mcmc_parameters
         A dictionary of parameters to be used by the inner MCMC kernels
     partial_logposterior_factory:
         A callable that given an array of 0 and 1, returns a function logposterior(x).
         The array represents which values to include in the logposterior calculation. The logposterior
         must be jax compilable.

     Returns
     -------
     A callable that takes a rng_key and PartialPosteriorsSMCState and selectors for
     the current and previous posteriors, and takes a data-tempered SMC state.
    """
    delegate = smc_from_mcmc(mcmc_step_fn, mcmc_init_fn, resampling_fn)

    def step(key, state: PartialPosteriorsSMCState, selector):
        key, lp1, lp2 = jax.random.split(key, 3)

        logposterior_fn = partial_logposterior_factory(selector)
        previous_logposterior_fn = partial_logposterior_factory(state.selector)

        def log_weights_fn(x):
            return logposterior_fn(x) - previous_logposterior_fn(x)

        state, info = delegate(
            key, state, num_mcmc_steps, mcmc_parameters, logposterior_fn, log_weights_fn
        )

        return PartialPosteriorsSMCState(state.particles, state.weights, selector), info

    return step
