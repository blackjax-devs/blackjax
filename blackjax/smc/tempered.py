# Copyright 2020- The Blackjax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Callable, NamedTuple, Optional

import jax
import jax.numpy as jnp

import blackjax.smc as smc
import blackjax.smc.from_mcmc as smc_from_mcmc
from blackjax.base import SamplingAlgorithm
from blackjax.smc.base import update_and_take_last
from blackjax.types import Array, ArrayLikeTree, ArrayTree, PRNGKey

__all__ = ["TemperedSMCState", "init", "build_kernel", "as_top_level_api"]


class TemperedSMCState(NamedTuple):
    """Current state for the tempered SMC algorithm.

    particles: PyTree
        The particles' positions.
    lmbda: float
        Current value of the tempering parameter.

    """

    particles: ArrayTree
    weights: Array
    lmbda: float


def init(particles: ArrayLikeTree):
    # Infer the number of particles from the size of the leading dimension of
    # the first leaf of the inputted PyTree.
    num_particles = jax.tree_util.tree_flatten(particles)[0][0].shape[0]
    weights = jnp.ones(num_particles) / num_particles
    return TemperedSMCState(particles, weights, 0.0)


def build_kernel(
    logprior_fn: Callable,
    loglikelihood_fn: Callable,
    mcmc_step_fn: Callable,
    mcmc_init_fn: Callable,
    resampling_fn: Callable,
    update_strategy: Callable = update_and_take_last,
    update_particles_fn: Optional[Callable] = None,
) -> Callable:
    """Build the base Tempered SMC kernel.

    Tempered SMC uses tempering to sample from a distribution given by

    .. math::
        p(x) \\propto p_0(x) \\exp(-V(x)) \\mathrm{d}x

    where :math:`p_0` is the prior distribution, typically easy to sample from
    and for which the density is easy to compute, and :math:`\\exp(-V(x))` is an
    unnormalized likelihood term for which :math:`V(x)` is easy to compute
    pointwise.

    Parameters
    ----------
    logprior_fn
        A function that computes the log density of the prior distribution
    loglikelihood_fn
        A function that returns the probability at a given
        position.
    mcmc_step_fn
        A function that creates a mcmc kernel from a log-probability density function.
    mcmc_init_fn: Callable
        A function that creates a new mcmc state from a position and a
        log-probability density function.
    resampling_fn
        A random function that resamples generated particles based of weights
    num_mcmc_iterations
        Number of iterations in the MCMC chain.

    Returns
    -------
    A callable that takes a rng_key and a TemperedSMCState that contains the current state
    of the chain and that returns a new state of the chain along with
    information about the transition.

    """
    update_particles = (
        smc_from_mcmc.build_kernel(
            mcmc_step_fn, mcmc_init_fn, resampling_fn, update_strategy
        )
        if update_particles_fn is None
        else update_particles_fn
    )

    def kernel(
        rng_key: PRNGKey,
        state: TemperedSMCState,
        num_mcmc_steps: int,
        lmbda: float,
        mcmc_parameters: dict,
    ) -> tuple[TemperedSMCState, smc.base.SMCInfo]:
        """Move the particles one step using the Tempered SMC algorithm.

        Parameters
        ----------
        rng_key
            JAX PRNGKey for randomness
        state
            Current state of the tempered SMC algorithm
        lmbda
            Current value of the tempering parameter
        mcmc_parameters
            The parameters of the MCMC step function.  Parameters with leading dimension
            length of 1 are shared amongst the particles.

        Returns
        -------
        state
            The new state of the tempered SMC algorithm
        info
            Additional information on the SMC step

        """
        delta = lmbda - state.lmbda

        def log_weights_fn(position: ArrayLikeTree) -> float:
            return delta * loglikelihood_fn(position)

        def tempered_logposterior_fn(position: ArrayLikeTree) -> float:
            logprior = logprior_fn(position)
            tempered_loglikelihood = state.lmbda * loglikelihood_fn(position)
            return logprior + tempered_loglikelihood

        smc_state, info = update_particles(
            rng_key,
            state,
            num_mcmc_steps,
            mcmc_parameters,
            tempered_logposterior_fn,
            log_weights_fn,
        )

        tempered_state = TemperedSMCState(
            smc_state.particles, smc_state.weights, state.lmbda + delta
        )

        return tempered_state, info

    return kernel


def as_top_level_api(
    logprior_fn: Callable,
    loglikelihood_fn: Callable,
    mcmc_step_fn: Callable,
    mcmc_init_fn: Callable,
    mcmc_parameters: dict,
    resampling_fn: Callable,
    num_mcmc_steps: Optional[int] = 10,
    update_strategy=update_and_take_last,
    update_particles_fn=None,
) -> SamplingAlgorithm:
    """Implements the (basic) user interface for the Adaptive Tempered SMC kernel.

    Parameters
    ----------
    logprior_fn
        The log-prior function of the model we wish to draw samples from.
    loglikelihood_fn
        The log-likelihood function of the model we wish to draw samples from.
    mcmc_step_fn
        The MCMC step function used to update the particles.
    mcmc_init_fn
        The MCMC init function used to build a MCMC state from a particle position.
    mcmc_parameters
        The parameters of the MCMC step function.  Parameters with leading dimension
        length of 1 are shared amongst the particles.
    resampling_fn
        The function used to resample the particles.
    num_mcmc_steps
        The number of times the MCMC kernel is applied to the particles per step.

    Returns
    -------
    A ``SamplingAlgorithm``.

    """

    kernel = build_kernel(
        logprior_fn,
        loglikelihood_fn,
        mcmc_step_fn,
        mcmc_init_fn,
        resampling_fn,
        update_strategy,
        update_particles_fn,
    )

    def init_fn(position: ArrayLikeTree, rng_key=None):
        del rng_key
        return init(position)

    def step_fn(rng_key: PRNGKey, state, lmbda):
        return kernel(
            rng_key,
            state,
            num_mcmc_steps,
            lmbda,
            mcmc_parameters,
        )

    return SamplingAlgorithm(init_fn, step_fn)  # type: ignore[arg-type]
