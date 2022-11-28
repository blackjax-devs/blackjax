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
from typing import Callable, NamedTuple, Tuple

import blackjax.smc as smc
from blackjax.types import PRNGKey, PyTree

__all__ = ["TemperedSMCState", "init", "kernel"]


class TemperedSMCState(NamedTuple):
    """Current state for the tempered SMC algorithm.

    particles: PyTree
        The particles' positions.
    lmbda: float
        Current value of the tempering parameter.

    """

    particles: PyTree
    lmbda: float


def init(position: PyTree):
    return TemperedSMCState(position, 0.0)


def kernel(
    logprior_fn: Callable,
    loglikelihood_fn: Callable,
    mcmc_kernel_factory: Callable,
    make_mcmc_state: Callable,
    resampling_fn: Callable,
    num_mcmc_iterations: int,
) -> Callable:
    """Build the base Tempered SMC kernel.

    Tempered SMC uses tempering to sample from a distribution given by

    .. math::
        p(x) \\propto p_0(x) \\exp(-V(x)) \\mathrm{d}x

    where :math:`p_0` is the prior distribution, typically easy to sample from and for which the density
    is easy to compute, and :math:`\\exp(-V(x))` is an unnormalized likelihood term for which :math:`V(x)` is easy
    to compute pointwise.

    Parameters
    ----------
    logprior_fn
        A function that computes the log density of the prior distribution
    loglikelihood_fn
        A function that returns the probability at a given
        position.
    mcmc_kernel_factory
        A function that creates a mcmc kernel from a log-probability density function.
    make_mcmc_state: Callable
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
    kernel = smc.base.kernel(
        mcmc_kernel_factory, make_mcmc_state, resampling_fn, num_mcmc_iterations
    )

    def one_step(
        rng_key: PRNGKey, state: TemperedSMCState, lmbda: float
    ) -> Tuple[TemperedSMCState, smc.base.SMCInfo]:
        """Move the particles one step using the Tempered SMC algorithm.

        Parameters
        ----------
        rng_key
            JAX PRNGKey for randomness
        state
            Current state of the tempered SMC algorithm
        lmbda
            Current value of the tempering parameter

        Returns
        -------
        state
            The new state of the tempered SMC algorithm
        info
            Additional information on the SMC step

        """
        delta = lmbda - state.lmbda

        def log_weights_fn(position: PyTree) -> float:
            return delta * loglikelihood_fn(position)

        def tempered_logposterior_fn(position: PyTree) -> float:
            logprior = logprior_fn(position)
            tempered_loglikelihood = state.lmbda * loglikelihood_fn(position)
            return logprior + tempered_loglikelihood

        smc_state, info = kernel(
            rng_key, state.particles, tempered_logposterior_fn, log_weights_fn
        )
        state = TemperedSMCState(smc_state, state.lmbda + delta)

        return state, info

    return one_step
