from functools import partial
from typing import Callable

import jax

from blackjax import smc
from blackjax.smc.base import SMCState, update_and_take_last
from blackjax.types import Array, PRNGKey


def unshared_parameters_and_step_fn(
    mcmc_parameters: dict,
    mcmc_step_fn: Callable,
) -> tuple[dict, Callable]:
    """Split MCMC parameters into shared and unshared parameters.

    The shared dictionary represents the parameters common to all chains, and
    the unshared are different per chain. Binds the step function using the
    shared parameters.

    Parameters
    ----------
    mcmc_parameters: dict
        Dictionary of MCMC parameters. Parameters with shape[0] == 1 are
        considered shared across all chains.
    mcmc_step_fn: Callable
        MCMC step function.

    Returns
    -------
    unshared_mcmc_parameters: dict
        Parameters that differ per chain.
    shared_mcmc_step_fn: Callable
        MCMC step function with shared parameters bound.
    """
    shared_mcmc_parameters = {}
    unshared_mcmc_parameters = {}
    for k, v in mcmc_parameters.items():
        if v.shape[0] == 1:
            shared_mcmc_parameters[k] = v[0, ...]
        else:
            unshared_mcmc_parameters[k] = v
    shared_mcmc_step_fn = partial(mcmc_step_fn, **shared_mcmc_parameters)
    return unshared_mcmc_parameters, shared_mcmc_step_fn


def build_kernel(
    mcmc_step_fn: Callable,
    mcmc_init_fn: Callable,
    resampling_fn: Callable,
    update_strategy: Callable = update_and_take_last,
) -> Callable:
    """Build an SMC step function from MCMC kernels.

    Builds MCMC kernels from the input parameters, which may change across iterations.
    Moreover, it defines the way such kernels are used to update the particles. This
    layer adapts an API defined in terms of kernels (mcmc_step_fn and mcmc_init_fn)
    into an API that depends on an update function over the set of particles.

    Parameters
    ----------
    mcmc_step_fn: Callable
        MCMC step function.
    mcmc_init_fn: Callable
        Function that initializes an MCMC state from a position.
    resampling_fn: Callable
        Resampling function (from blackjax.smc.resampling).
    update_strategy: Callable
        Strategy to update particles using MCMC kernels, by default
        'update_and_take_last' from blackjax.smc.base.

    Returns
    -------
    step: Callable
        A callable that takes a rng_key and a state with .particles and .weights
        and returns a base.SMCState and base.SMCInfo pair.
    """

    def step(
        rng_key: PRNGKey,
        state: smc.base.SMCState,
        num_mcmc_steps: int | Array,
        mcmc_parameters: dict,
        logposterior_fn: Callable,
        log_weights_fn: Callable,
    ) -> tuple[smc.base.SMCState, smc.base.SMCInfo]:
        unshared_mcmc_parameters, shared_mcmc_step_fn = unshared_parameters_and_step_fn(
            mcmc_parameters, mcmc_step_fn
        )

        update_fn, num_resampled = update_strategy(
            mcmc_init_fn,
            logposterior_fn,
            shared_mcmc_step_fn,
            n_particles=state.weights.shape[0],
            num_mcmc_steps=num_mcmc_steps,
        )

        return smc.base.step(
            rng_key,
            SMCState(state.particles, state.weights, unshared_mcmc_parameters),
            update_fn,
            jax.vmap(log_weights_fn),
            resampling_fn,
            num_resampled,
        )

    return step
