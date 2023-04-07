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
from typing import Callable, NamedTuple

import blackjax.sgmcmc.diffusions as diffusions
from blackjax.types import PRNGKey, PyTree
from blackjax.util import generate_gaussian_noise

__all__ = ["SGNHTState", "init" ,"kernel"]


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
    position: PyTree
    momentum: PyTree
    xi: float


def init(rng_key: PRNGKey, position: PyTree, alpha: float = 0.01):
    momentum = generate_gaussian_noise(rng_key, position)
    return SGNHTState(position, momentum, alpha)


def kernel(alpha: float = 0.01, beta: float = 0) -> Callable:
    """Stochastic gradient Nosé-Hoover Thermostat (SGNHT) algorithm."""
    integrator = diffusions.sgnht(alpha, beta)

    def one_step(
        rng_key: PRNGKey,
        state: SGNHTState,
        grad_estimator: Callable,
        minibatch: PyTree,
        step_size: float,
        temperature: float = 1.0,
    ) -> PyTree:
        position, momentum, xi = state
        logdensity_grad = grad_estimator(position, minibatch)
        position, momentum, xi = integrator(
            rng_key, position, momentum, xi, logdensity_grad, step_size, temperature
        )
        return SGNHTState(position, momentum, xi)

    return one_step