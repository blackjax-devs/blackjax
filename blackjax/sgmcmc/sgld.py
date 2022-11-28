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
"""Public API for the Stochastic gradient Langevin Dynamics kernel."""
from typing import Callable

import blackjax.sgmcmc.diffusions as diffusions
from blackjax.types import PRNGKey, PyTree

__all__ = ["kernel"]


def kernel() -> Callable:
    """Stochastic gradient Langevin Dynamics (SgLD) algorithm."""
    integrator = diffusions.overdamped_langevin()

    def one_step(
        rng_key: PRNGKey,
        position: PyTree,
        grad_estimator: Callable,
        minibatch: PyTree,
        step_size: float,
    ):

        logprob_grad = grad_estimator(position, minibatch)
        new_position = integrator(rng_key, position, logprob_grad, step_size)

        return new_position

    return one_step
