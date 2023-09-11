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
from typing import Callable, NamedTuple, Optional

import jax
import jax.numpy as jnp
import jax.scipy.linalg as linalg

from blackjax.base import SamplingAlgorithm
from blackjax.types import Array, PRNGKey

__all__ = ["MarginalState", "MarginalInfo", "init_and_kernel", "mgrad_gaussian"]


# [TODO](https://github.com/blackjax-devs/blackjax/issues/237)
class MarginalState(NamedTuple):
    """State of the RMH chain.

    x
        Current position of the chain.
    log_p_x
        Current value of the log-likelihood of the model
    grad_x
        Current value of the gradient of the log-likelihood of the model
    U_x
        Auxiliary attributes
    U_grad_x
        Gradient of the auxiliary attributes

    """

    position: Array
    logdensity: float
    logdensity_grad: Array

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


def init_and_kernel(logdensity_fn, covariance, mean=None):
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
        val_and_grad = jax.value_and_grad(
            lambda x: logdensity_fn(x) + jnp.dot(x, shift)
        )
    else:
        val_and_grad = jax.value_and_grad(logdensity_fn)

    def step(key: PRNGKey, state: MarginalState, delta):
        y_key, u_key = jax.random.split(key, 2)

        position, logdensity, logdensity_grad, U_x, U_grad_x = state

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

        alpha = jnp.minimum(1, jnp.exp(log_p_y - logdensity + hxy - hyx))
        accept = jax.random.uniform(u_key) < alpha

        proposed_state = MarginalState(y, log_p_y, grad_y, U_y, U_grad_y)
        state = jax.lax.cond(accept, lambda _: proposed_state, lambda _: state, None)
        info = MarginalInfo(alpha, accept, proposed_state)
        return state, info

    def init(position):
        logdensity, logdensity_grad = val_and_grad(position)
        return MarginalState(
            position, logdensity, logdensity_grad, U_t @ position, U_t @ logdensity_grad
        )

    return init, step


class mgrad_gaussian:
    """Implements the marginal sampler for latent Gaussian model of :cite:p:`titsias2018auxiliary`.

    It uses a first order approximation to the log_likelihood of a model with Gaussian prior.
    Interestingly, the only parameter that needs calibrating is the "step size" delta, which can be done very efficiently.
    Calibrating it to have an acceptance rate of roughly 50% is a good starting point.

    Examples
    --------
    A new marginal latent Gaussian MCMC kernel for a model q(x) ∝ exp(f(x)) N(x; m, C) can be initialized and
    used for a given "step size" delta with the following code:

    .. code::

        mgrad_gaussian = blackjax.mgrad_gaussian(f, C, use_inverse=False, mean=m)
        state = mgrad_gaussian.init(zeros)  # Starting at the mean of the prior
        new_state, info = mgrad_gaussian.step(rng_key, state, delta)

    We can JIT-compile the step function for better performance

    .. code::

        step = jax.jit(mgrad_gaussian.step)
        new_state, info = step(rng_key, state, delta)

    Parameters
    ----------
    logdensity_fn
        The logarithm of the likelihood function for the latent Gaussian model.
    covariance
        The covariance of the prior Gaussian density.
    mean: optional
        Mean of the prior Gaussian density. Default is zero.

    Returns
    -------
    A ``SamplingAlgorithm``.

    """

    def __new__(  # type: ignore[misc]
        cls,
        logdensity_fn: Callable,
        covariance: Array,
        mean: Optional[Array] = None,
    ) -> SamplingAlgorithm:
        init, kernel = init_and_kernel(logdensity_fn, covariance, mean)

        def init_fn(position: Array):
            return init(position)

        def step_fn(rng_key: PRNGKey, state, delta: float):
            return kernel(
                rng_key,
                state,
                delta,
            )

        return SamplingAlgorithm(init_fn, step_fn)  # type: ignore[arg-type]
