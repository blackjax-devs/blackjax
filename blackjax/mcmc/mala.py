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
"""Public API for Metropolis Adjusted Langevin kernels."""
import operator
from typing import Callable, NamedTuple, Tuple

import jax
import jax.numpy as jnp

import blackjax.mcmc.diffusions as diffusions
from blackjax.types import PRNGKey, PyTree

__all__ = ["MALAState", "MALAInfo", "init", "kernel"]


class MALAState(NamedTuple):
    """State of the MALA algorithm.

    The MALA algorithm takes one position of the chain and returns another
    position. In order to make computations more efficient, we also store
    the current log-probability density as well as the current gradient of the
    log-probability density.

    """

    position: PyTree
    logdensity: float
    logdensity_grad: PyTree


class MALAInfo(NamedTuple):
    """Additional information on the MALA transition.

    This additional information can be used for debugging or computing
    diagnostics.

    acceptance_rate
        The acceptance rate of the transition.
    is_accepted
        Whether the proposed position was accepted or the original position
        was returned.

    """

    acceptance_rate: float
    is_accepted: bool


def init(position: PyTree, logdensity_fn: Callable) -> MALAState:
    grad_fn = jax.value_and_grad(logdensity_fn)
    logdensity, logdensity_grad = grad_fn(position)
    return MALAState(position, logdensity, logdensity_grad)


def kernel():
    """Build a MALA kernel.

    Returns
    -------
    A kernel that takes a rng_key and a Pytree that contains the current state
    of the chain and that returns a new state of the chain along with
    information about the transition.

    """

    def transition_probability(state, new_state, step_size):
        """Transition probability to go from `state` to `new_state`"""
        theta = jax.tree_util.tree_map(
            lambda new_x, x, g: new_x - x - step_size * g,
            new_state.position,
            state.position,
            state.logdensity_grad,
        )
        theta_dot = jax.tree_util.tree_reduce(
            operator.add, jax.tree_util.tree_map(lambda x: jnp.sum(x * x), theta)
        )
        return -0.25 * (1.0 / step_size) * theta_dot

    def one_step(
        rng_key: PRNGKey, state: MALAState, logdensity_fn: Callable, step_size: float
    ) -> Tuple[MALAState, MALAInfo]:
        """Generate a new sample with the MALA kernel."""
        grad_fn = jax.value_and_grad(logdensity_fn)
        integrator = diffusions.overdamped_langevin(grad_fn)

        key_integrator, key_rmh = jax.random.split(rng_key)

        new_state = integrator(key_integrator, state, step_size)

        delta = (
            new_state.logdensity
            - state.logdensity
            + transition_probability(new_state, state, step_size)
            - transition_probability(state, new_state, step_size)
        )
        delta = jnp.where(jnp.isnan(delta), -jnp.inf, delta)
        p_accept = jnp.clip(jnp.exp(delta), a_max=1)

        do_accept = jax.random.bernoulli(key_rmh, p_accept)

        new_state = MALAState(*new_state)
        info = MALAInfo(p_accept, do_accept)

        return jax.lax.cond(
            do_accept,
            lambda _: (new_state, info),
            lambda _: (state, info),
            operand=None,
        )

    return one_step
