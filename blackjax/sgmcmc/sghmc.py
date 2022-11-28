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
import jax.numpy as jnp

import blackjax.sgmcmc.diffusions as diffusions
from blackjax.types import PRNGKey, PyTree
from blackjax.util import generate_gaussian_noise

__all__ = ["kernel"]


def kernel(alpha: float = 0.01, beta: float = 0) -> Callable:
    """Stochastic gradient Hamiltonian Monte Carlo (SgHMC) algorithm."""
    integrator = diffusions.sghmc(alpha, beta)

    def one_step(
        rng_key: PRNGKey,
        position: PyTree,
        grad_estimator: Callable,
        minibatch: PyTree,
        step_size: float,
        num_integration_steps: int,
    ) -> PyTree:
        def body_fn(state, rng_key):
            position, momentum = state
            logprob_grad = grad_estimator(position, minibatch)
            position, momentum = integrator(
                rng_key, position, momentum, logprob_grad, step_size
            )
            return ((position, momentum), position)

        momentum = generate_gaussian_noise(rng_key, position, 0, jnp.sqrt(step_size))
        keys = jax.random.split(rng_key, num_integration_steps)
        (position, momentum), _ = jax.lax.scan(body_fn, (position, momentum), keys)

        return position

    return one_step
