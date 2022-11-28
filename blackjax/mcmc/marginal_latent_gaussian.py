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
"""Public API for marginal latent Gaussian sampling."""
from typing import NamedTuple

import jax
import jax.numpy as jnp
import jax.scipy.linalg as linalg

from blackjax.types import Array, PRNGKey

__all__ = ["MarginalState", "MarginalInfo", "init_and_kernel"]


class MarginalState(NamedTuple):
    """State of the RMH chain.

    x
        Current position of the chain.
    log_p_x
        Current value of the log-likelihood of the model
    grad_x
        Current value of the gradient of the log-likelihood of the model

    Other Attributes:
    -----------------
    U_x, U_grad_x: Array
        Auxiliary attributes

    """

    position: Array
    logprob: float
    logprob_grad: Array

    U_x: Array
    U_grad_x: Array


class MarginalInfo(NamedTuple):
    """Additional information on the RMH chain.

    This additional information can be used for debugging or computing
    diagnostics.

    acceptance_rate
        The acceptance probability of the transition, linked to the energy
        difference between the original and the proposed states.
    is_accepted
        Whether the proposed position was accepted or the original position
        was returned.
    proposal
        The state proposed by the proposal.

    """

    acceptance_rate: float
    is_accepted: bool
    proposal: MarginalState


def init_and_kernel(logprob_fn, covariance, mean=None):
    """Build the marginal version of the auxiliary gradient-based sampler

    Returns
    -------
    A kernel that takes a rng_key and a Pytree that contains the current state
    of the chain and that returns a new state of the chain along with
    information about the transition.
    An init function.

    """
    U, Gamma, U_t = jnp.linalg.svd(covariance, hermitian=True)

    if mean is not None:
        shift = linalg.solve(covariance, mean, assume_a="pos")
        val_and_grad = jax.value_and_grad(lambda x: logprob_fn(x) + jnp.dot(x, shift))
    else:
        val_and_grad = jax.value_and_grad(logprob_fn)

    def step(key: PRNGKey, state: MarginalState, delta):
        y_key, u_key = jax.random.split(key, 2)

        position, logprob, logprob_grad, U_x, U_grad_x = state

        # Update Gamma(delta)
        # TODO: Ideally, we could have a dichotomy, where we only update Gamma(delta) if delta changes,
        #       but this is hardly the most expensive part of the algorithm (the multiplication by U below is).
        Gamma_1 = Gamma * delta / (delta + 2 * Gamma)
        Gamma_3 = (delta + 2 * Gamma) / (delta + 4 * Gamma)
        Gamma_2 = Gamma_1 / Gamma_3

        # Propose a new y
        temp = Gamma_1 * (U_x / (0.5 * delta) + U_grad_x)
        temp = temp + jnp.sqrt(Gamma_2) * jax.random.normal(y_key, position.shape)
        y = U @ temp

        # Bookkeeping
        log_p_y, grad_y = val_and_grad(y)
        U_y = U_t @ y
        U_grad_y = U_t @ grad_y

        # Acceptance step
        temp_x = Gamma_1 * (U_x / (0.5 * delta) + 0.5 * U_grad_x)
        temp_y = Gamma_1 * (U_y / (0.5 * delta) + 0.5 * U_grad_y)

        hxy = jnp.dot(U_x - temp_y, Gamma_3 * U_grad_y)
        hyx = jnp.dot(U_y - temp_x, Gamma_3 * U_grad_x)

        alpha = jnp.minimum(1, jnp.exp(log_p_y - logprob + hxy - hyx))
        accept = jax.random.uniform(u_key) < alpha

        proposed_state = MarginalState(y, log_p_y, grad_y, U_y, U_grad_y)
        state = jax.lax.cond(accept, lambda _: proposed_state, lambda _: state, None)
        info = MarginalInfo(alpha, accept, proposed_state)
        return state, info

    def init(position):
        logprob, logprob_grad = val_and_grad(position)
        return MarginalState(
            position, logprob, logprob_grad, U_t @ position, U_t @ logprob_grad
        )

    return init, step
