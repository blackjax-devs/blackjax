from typing import Callable, NamedTuple, Optional, Tuple

import jax
import jax.numpy as jnp

from blackjax import SamplingAlgorithm, smc
from blackjax.smc.base import update_and_take_last
from blackjax.smc.from_mcmc import build_kernel as smc_from_mcmc
from blackjax.types import Array, ArrayLikeTree, ArrayTree, PRNGKey


class PartialPosteriorsSMCState(NamedTuple):
    """Current state for the tempered SMC algorithm.

    particles: PyTree
        The particles' positions.
    weights:
        Weights of the particles, so that they represent a probability distribution
    data_mask:
        A 1D boolean array to indicate which datapoints to include
        in the computation of the observed likelihood.
    """

    particles: ArrayTree
    weights: Array
    data_mask: Array


def init(particles: ArrayLikeTree, num_datapoints: int) -> PartialPosteriorsSMCState:
    """num_datapoints are the number of observations that could potentially be
    used in a partial posterior. Since the initial data_mask is all 0s, it
    means that no likelihood term will be added (only prior).
    """
    num_particles = jax.tree_util.tree_flatten(particles)[0][0].shape[0]
    weights = jnp.ones(num_particles) / num_particles
    return PartialPosteriorsSMCState(particles, weights, jnp.zeros(num_datapoints))


def build_kernel(
    mcmc_step_fn: Callable,
    mcmc_init_fn: Callable,
    resampling_fn: Callable,
    num_mcmc_steps: Optional[int],
    mcmc_parameters: ArrayTree,
    partial_logposterior_factory: Callable[[Array], Callable],
    update_strategy=update_and_take_last,
) -> Callable:
    """Build the Partial Posteriors (data tempering) SMC kernel.
    The distribution's trajectory includes increasingly adding more
    datapoints to the likelihood. See Section 2.2 of https://arxiv.org/pdf/2007.11936
    Parameters
    ----------
    mcmc_step_fn
        A function that computes the log density of the prior distribution
    mcmc_init_fn
        A function that returns the probability at a given position.
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
    delegate = smc_from_mcmc(mcmc_step_fn, mcmc_init_fn, resampling_fn, update_strategy)

    def step(
        key, state: PartialPosteriorsSMCState, data_mask: Array
    ) -> Tuple[PartialPosteriorsSMCState, smc.base.SMCInfo]:
        logposterior_fn = partial_logposterior_factory(data_mask)

        previous_logposterior_fn = partial_logposterior_factory(state.data_mask)

        def log_weights_fn(x):
            return logposterior_fn(x) - previous_logposterior_fn(x)

        state, info = delegate(
            key, state, num_mcmc_steps, mcmc_parameters, logposterior_fn, log_weights_fn
        )

        return (
            PartialPosteriorsSMCState(state.particles, state.weights, data_mask),
            info,
        )

    return step


def as_top_level_api(
    mcmc_step_fn: Callable,
    mcmc_init_fn: Callable,
    mcmc_parameters: dict,
    resampling_fn: Callable,
    num_mcmc_steps,
    partial_logposterior_factory: Callable,
    update_strategy=update_and_take_last,
) -> SamplingAlgorithm:
    """A factory that wraps the kernel into a SamplingAlgorithm object.
    See build_kernel for full documentation on the parameters.
    """

    kernel = build_kernel(
        mcmc_step_fn,
        mcmc_init_fn,
        resampling_fn,
        num_mcmc_steps,
        mcmc_parameters,
        partial_logposterior_factory,
        update_strategy,
    )

    def init_fn(position: ArrayLikeTree, num_observations, rng_key=None):
        del rng_key
        return init(position, num_observations)

    def step(key: PRNGKey, state: PartialPosteriorsSMCState, data_mask: Array):
        return kernel(key, state, data_mask)

    return SamplingAlgorithm(init_fn, step)  # type: ignore[arg-type]
