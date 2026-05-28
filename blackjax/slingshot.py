from typing import Callable
import jax
import jax.numpy as jnp

import blackjax.mcmc.slingshot as mcmc_slingshot
from blackjax.base import SamplingAlgorithm

__all__ = ["slingshot"]

def slingshot(
    logdensity_fn: Callable,
    step_size: float,
    num_proposals: int,
    cholesky: jnp.ndarray = None,
) -> SamplingAlgorithm:
    """User-facing interface factory for the exact Slingshot MP-MCMC sampler."""
    kernel = mcmc_slingshot.kernel()

    def init_fn(position: jax.Array) -> mcmc_slingshot.SlingshotState:
        return mcmc_slingshot.init(position, logdensity_fn)

    def step_fn(
        rng_key: jax.random.PRNGKey, state: mcmc_slingshot.SlingshotState
    ) -> tuple[mcmc_slingshot.SlingshotState, mcmc_slingshot.SlingshotInfo]:
        return kernel(
            rng_key,
            state,
            logdensity_fn,
            step_size,
            num_proposals,
            cholesky,
        )

    return SamplingAlgorithm(init_fn, step_fn)