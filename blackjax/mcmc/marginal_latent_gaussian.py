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
from blackjax.mcmc.proposal import static_binomial_sampling
from blackjax.types import Array, PRNGKey

__all__ = [
    "MarginalState",
    "MarginalInfo",
    "init",
    "build_kernel",
    "as_top_level_api",
]


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


class CovarianceSVD(NamedTuple):
    """Singular Value Decomposition of the covariance matrix.

    U
        Unitary array of the covariance matrix.
    Gamma
        Singular values of the covariance matrix.
    U_t
        Transpose of the unitary array of the covariance matrix.

    """

    U: Array
    Gamma: Array
    U_t: Array


def svd_from_covariance(covariance: Array) -> CovarianceSVD:
    """Compute the singular value decomposition of the covariance matrix.

    Parameters
    ----------
    covariance
        The covariance matrix.

    Returns
    -------
    A ``CovarianceSVD`` object.

    """
    U, Gamma, U_t = jnp.linalg.svd(covariance, hermitian=True)
    return CovarianceSVD(U, Gamma, U_t)


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


def generate_mean_shifted_logprob(logdensity_fn, mean, covariance):
    """Generate a log-density function that is shifted by a constant

    Parameters
    ----------
    logdensity_fn
        The original log-density function
    mean
        The mean of the prior Gaussian density
    covariance
        The covariance of the prior Gaussian density.

    Returns
    -------
    A log-density function that is shifted by a constant

    """
    shift = linalg.solve(covariance, mean, assume_a="pos")

    def shifted_logdensity_fn(x):
        return logdensity_fn(x) + jnp.dot(x, shift)

    return shifted_logdensity_fn


def init(position, logdensity_fn, U_t):
    """Initialize the marginal version of the auxiliary gradient-based sampler.

    Parameters
    ----------
    position
        The initial position of the chain.
    logdensity_fn
        The logarithm of the likelihood function for the latent Gaussian model.
    U_t
        The unitary array of the covariance matrix.
    """
    logdensity, logdensity_grad = jax.value_and_grad(logdensity_fn)(position)
    return MarginalState(
        position, logdensity, logdensity_grad, U_t @ position, U_t @ logdensity_grad
    )


def build_kernel(cov_svd: CovarianceSVD):
    """Build the marginal version of the auxiliary gradient-based sampler.

    Parameters
    ----------
    cov_svd
        The singular value decomposition of the covariance matrix.

    Returns
    -------
    A kernel that takes a rng_key and a Pytree that contains the current state
    of the chain and that returns a new state of the chain along with
    information about the transition.
    """
    U, Gamma, U_t = cov_svd

    def kernel(key: PRNGKey, state: MarginalState, logdensity_fn, delta):
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
        log_p_y, grad_y = jax.value_and_grad(logdensity_fn)(y)
        U_y = U_t @ y
        U_grad_y = U_t @ grad_y

        # Acceptance step
        temp_x = Gamma_1 * (U_x / (0.5 * delta) + 0.5 * U_grad_x)
        temp_y = Gamma_1 * (U_y / (0.5 * delta) + 0.5 * U_grad_y)

        hxy = jnp.dot(U_x - temp_y, Gamma_3 * U_grad_y)
        hyx = jnp.dot(U_y - temp_x, Gamma_3 * U_grad_x)

        log_p_accept = log_p_y - logdensity + hxy - hyx
        proposed_state = MarginalState(y, log_p_y, grad_y, U_y, U_grad_y)
        accepted_state, info = static_binomial_sampling(
            u_key, log_p_accept, state, proposed_state
        )
        do_accept, p_accept, _ = info
        info = MarginalInfo(p_accept, do_accept, proposed_state)
        return accepted_state, info

    return kernel


def as_top_level_api(
    logdensity_fn: Callable,
    covariance: Optional[Array] = None,
    mean: Optional[Array] = None,
    cov_svd: Optional[CovarianceSVD] = None,
    step_size: float = 1.0,
) -> SamplingAlgorithm:
    """Implements the marginal sampler for latent Gaussian model of :cite:p:`titsias2018auxiliary`.

    It uses a first order approximation to the log_likelihood of a model with Gaussian prior.
    Interestingly, the only parameter that needs calibrating is the "step size" delta,
    which can be done very efficiently.
    Calibrating it to have an acceptance rate of roughly 50% is a good starting point.

    Examples
    --------
    A new marginal latent Gaussian MCMC kernel for a model q(x) ∝ exp(f(x)) N(x; m, C)
    can be initialized and used for a given "step size" delta with the following code:

    .. code::

        mgrad_gaussian = blackjax.mgrad_gaussian(f, C, mean=m, step_size=delta)
        state = mgrad_gaussian.init(zeros)  # Starting at the mean of the prior
        new_state, info = mgrad_gaussian.step(rng_key, state)

    We can JIT-compile the step function for better performance

    .. code::

        step = jax.jit(mgrad_gaussian.step)
        new_state, info = step(rng_key, state)

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

    if cov_svd is None:
        if covariance is None:
            raise ValueError("Either covariance or cov_svd must be provided.")
        cov_svd = svd_from_covariance(covariance)

    U, Gamma, U_t = cov_svd

    if mean is not None:
        logdensity_fn = generate_mean_shifted_logprob(logdensity_fn, mean, covariance)

    kernel = build_kernel(cov_svd)

    def init_fn(position: Array, rng_key=None):
        del rng_key
        return init(position, logdensity_fn, U_t)

    def step_fn(rng_key: PRNGKey, state):
        return kernel(
            rng_key,
            state,
            logdensity_fn,
            step_size,
        )

    return SamplingAlgorithm(init_fn, step_fn)
