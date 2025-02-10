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
"""Public API for the Stochastic gradient Nosé-Hoover Thermostat kernel."""
from typing import Callable, NamedTuple, Union

import blackjax.sgmcmc.diffusions as diffusions
from blackjax.base import SamplingAlgorithm
from blackjax.types import ArrayLikeTree, ArrayTree, PRNGKey
from blackjax.util import generate_gaussian_noise

__all__ = ["SGNHTState", "init", "build_kernel", "as_top_level_api"]


class SGNHTState(NamedTuple):
    r"""State of the SGNHT algorithm.

    Parameters
    ----------
    position
        Current position in the sample space.
    momentum
        Current momentum in the sample space.
    xi
        Scalar thermostat controlling kinetic energy.

    """
    position: ArrayTree
    momentum: ArrayTree
    xi: float


def init(position: ArrayLikeTree, rng_key: PRNGKey, xi: float) -> SGNHTState:
    momentum = generate_gaussian_noise(rng_key, position)
    return SGNHTState(position, momentum, xi)


def build_kernel(alpha: float = 0.01, beta: float = 0) -> Callable:
    """Stochastic gradient Nosé-Hoover Thermostat (SGNHT) algorithm."""
    integrator = diffusions.sgnht(alpha, beta)

    def kernel(
        rng_key: PRNGKey,
        state: SGNHTState,
        grad_estimator: Callable,
        minibatch: ArrayLikeTree,
        step_size: float,
        temperature: float = 1.0,
    ) -> ArrayTree:
        position, momentum, xi = state
        logdensity_grad = grad_estimator(position, minibatch)
        position, momentum, xi = integrator(
            rng_key, position, momentum, xi, logdensity_grad, step_size, temperature
        )
        return SGNHTState(position, momentum, xi)

    return kernel


def as_top_level_api(
    grad_estimator: Callable,
    alpha: float = 0.01,
    beta: float = 0.0,
) -> SamplingAlgorithm:
    """Implements the (basic) user interface for the SGNHT kernel.

    The general sgnht kernel (:meth:`blackjax.sgmcmc.sgnht.build_kernel`, alias
    `blackjax.sgnht.build_kernel`) can be cumbersome to manipulate. Since most users
    only need to specify the kernel parameters at initialization time, we
    provide a helper function that specializes the general kernel.

    Example
    -------

    To initialize a SGNHT kernel one needs to specify a schedule function, which
    returns a step size at each sampling step, and a gradient estimator
    function. Here for a constant step size, and `data_size` data samples:

    .. code::

        grad_estimator = blackjax.sgmcmc.gradients.grad_estimator(logprior_fn, loglikelihood_fn, data_size)

    We can now initialize the sgnht kernel and the state.

    .. code::

        sgnht = blackjax.sgnht(grad_estimator)
        state = sgnht.init(rng_key, position)

    Assuming we have an iterator `batches` that yields batches of data we can
    perform one step:

    .. code::

        step_size = 1e-3
        minibatch = next(batches)
        new_state = sgnht.step(rng_key, state, minibatch, step_size)

    Kernels are not jit-compiled by default so you will need to do it manually:

    .. code::

       step = jax.jit(sgnht.step)
       new_state = step(rng_key, state, minibatch, step_size)

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

    def init_fn(
        position: ArrayLikeTree,
        rng_key: PRNGKey,
        init_xi: Union[None, float] = None,
    ):
        return init(position, rng_key, init_xi or alpha)

    def step_fn(
        rng_key: PRNGKey,
        state: SGNHTState,
        minibatch: ArrayLikeTree,
        step_size: float,
        temperature: float = 1,
    ) -> SGNHTState:
        return kernel(rng_key, state, grad_estimator, minibatch, step_size, temperature)

    return SamplingAlgorithm(init_fn, step_fn)  # type: ignore[arg-type]
