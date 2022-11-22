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
"""Public API for the Independent Rosenbluth-Metropolis-Hastings kernels."""
from typing import Callable, Tuple

from blackjax.mcmc.rmh import RMHInfo, RMHState, rmh
from blackjax.types import PRNGKey, PyTree

__all__ = ["kernel"]


def kernel(proposal_distribution: Callable) -> Callable:
    """
    Build an Independent Random Walk Rosenbluth-Metropolis-Hastings kernel. This implies
    that the proposal distribution does not depend on the particle being mutated.
    Reference: Algorithm 2 from https://arxiv.org/pdf/2008.02455.pdf

    Parameters
    ----------
    proposal_distribution
        A function that, given a PRNGKey, is able to produce a sample in the same
        domain of the target distribution.

    Returns
    -------
    A kernel that takes a rng_key and a Pytree that contains the current state
    of the chain and that returns a new state of the chain along with
    information about the transition.

    """

    def one_step(
        rng_key: PRNGKey, state: RMHState, logprob_fn: Callable
    ) -> Tuple[RMHState, RMHInfo]:
        def proposal_generator(rng_key: PRNGKey, position: PyTree):
            return proposal_distribution(rng_key)

        kernel = rmh(logprob_fn, proposal_generator)
        return kernel(rng_key, state)

    return one_step
