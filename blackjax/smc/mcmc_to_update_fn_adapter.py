from functools import partial
from typing import Callable

from blackjax.smc.base import SMCState, update_and_take_last
from blackjax.types import PRNGKey
from blackjax import smc
import jax


def build_kernel(
    mcmc_step_fn: Callable,
    mcmc_init_fn: Callable,
    resampling_fn,
    update_strategy: Callable = update_and_take_last,
    ):
    """Builds a SMC step that constructs MCMC kernels from the input parameters,
    which may change across iterations. Moreover, it defines the way
    such kernels are used to update the particles. This layer
    adapts an API defined in terms of kernels (mcmc_step_fn and mcmc_init_fn) into an API
    that depends on an update function over the set of particles.
    ----------

    update_strategy


    Returns
    -------
    A callable that takes a rng_key and a TemperedSMCState that contains the current state
    of the chain and that returns a new state of the chain along with
    information about the transition.

    """

    def step(
        rng_key: PRNGKey,
        state,
        num_mcmc_steps: int,
        mcmc_parameters: dict,
        logposterior_fn: Callable,
        log_weights_fn: Callable,
    ) -> tuple[smc.base.SMCState, smc.base.SMCInfo]:

        shared_mcmc_parameters = {}
        unshared_mcmc_parameters = {}
        for k, v in mcmc_parameters.items():
            if v.shape[0] == 1:
                shared_mcmc_parameters[k] = v[0, ...]
            else:
                unshared_mcmc_parameters[k] = v

        shared_mcmc_step_fn = partial(mcmc_step_fn, **shared_mcmc_parameters)

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
