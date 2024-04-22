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
"""Public API for the Stochastic gradient Hamiltonian Monte Carlo kernel."""
from typing import Callable

import jax

import blackjax.sgmcmc.diffusions as diffusions
from blackjax.base import SamplingAlgorithm
from blackjax.types import ArrayLikeTree, ArrayTree, PRNGKey
from blackjax.util import generate_gaussian_noise

__all__ = ["init", "build_kernel", "as_top_level_api"]


def init(position: ArrayLikeTree) -> ArrayLikeTree:
    return position


def build_kernel(alpha: float = 0.01, beta: float = 0) -> Callable:
    """Stochastic gradient Hamiltonian Monte Carlo (SgHMC) algorithm."""
    integrator = diffusions.sghmc(alpha, beta)

    def kernel(
        rng_key: PRNGKey,
        position: ArrayLikeTree,
        grad_estimator: Callable,
        minibatch: ArrayLikeTree,
        step_size: float,
        num_integration_steps: int,
        temperature: float = 1.0,
    ) -> ArrayTree:
        def body_fn(state, rng_key):
            position, momentum = state
            logdensity_grad = grad_estimator(position, minibatch)
            position, momentum = integrator(
                rng_key, position, momentum, logdensity_grad, step_size, temperature
            )
            return ((position, momentum), position)

        momentum = generate_gaussian_noise(rng_key, position)
        keys = jax.random.split(rng_key, num_integration_steps)
        (position, momentum), _ = jax.lax.scan(body_fn, (position, momentum), keys)

        return position

    return kernel


def as_top_level_api(
    grad_estimator: Callable,
    num_integration_steps: int = 10,
    alpha: float = 0.01,
    beta: float = 0,
) -> SamplingAlgorithm:
    """Implements the (basic) user interface for the SGHMC kernel.

    The general sghmc kernel builder (:meth:`blackjax.sgmcmc.sghmc.build_kernel`, alias
    `blackjax.sghmc.build_kernel`) can be cumbersome to manipulate. Since most users
    only need to specify the kernel parameters at initialization time, we
    provide a helper function that specializes the general kernel.

    Example
    -------

    To initialize a SGHMC kernel one needs to specify a schedule function, which
    returns a step size at each sampling step, and a gradient estimator
    function. Here for a constant step size, and `data_size` data samples:

    .. code::

        grad_estimator = blackjax.sgmcmc.gradients.grad_estimator(logprior_fn, loglikelihood_fn, data_size)

    We can now initialize the sghmc kernel and the state. Like HMC, SGHMC needs the user to specify a number of integration steps.

    .. code::

        sghmc = blackjax.sghmc(grad_estimator, num_integration_steps)

    Assuming we have an iterator `batches` that yields batches of data we can
    perform one step:

    .. code::

        step_size = 1e-3
        minibatch = next(batches)
        new_position = sghmc.step(rng_key, position, minibatch, step_size)

    Kernels are not jit-compiled by default so you will need to do it manually:

    .. code::

       step = jax.jit(sghmc.step)
       new_position, info = step(rng_key, position, minibatch, step_size)

    Parameters
    ----------
    grad_estimator
       A function that takes a position, a batch of data and returns an estimation
       of the gradient of the log-density at this position.

    Returns
    -------
    A ``SamplingAlgorithm``.

    """

    kernel = build_kernel(alpha, beta)

    def init_fn(position: ArrayLikeTree, rng_key=None):
        del rng_key
        return init(position)

    def step_fn(
        rng_key: PRNGKey,
        state: ArrayLikeTree,
        minibatch: ArrayLikeTree,
        step_size: float,
        temperature: float = 1,
    ) -> ArrayTree:
        return kernel(
            rng_key,
            state,
            grad_estimator,
            minibatch,
            step_size,
            num_integration_steps,
            temperature,
        )

    return SamplingAlgorithm(init_fn, step_fn)  # type: ignore[arg-type]
