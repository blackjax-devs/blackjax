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
"""Public API for the Elliptical Slice sampling Kernel"""
from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp

from blackjax.base import SamplingAlgorithm
from blackjax.types import Array, ArrayLikeTree, ArrayTree, PRNGKey
from blackjax.util import generate_gaussian_noise

__all__ = [
    "EllipSliceState",
    "EllipSliceInfo",
    "init",
    "build_kernel",
    "as_top_level_api",
]


class EllipSliceState(NamedTuple):
    """State of the Elliptical Slice sampling algorithm.

    position
        Current position of the chain.
    logdensity
        Current value of the logdensity (evaluated at current position).

    """

    position: ArrayTree
    logdensity: ArrayTree


class EllipSliceInfo(NamedTuple):
    r"""Additional information on the Elliptical Slice sampling chain.

    This additional information can be used for debugging or computing
    diagnostics.

    momentum
        The latent momentum variable returned at the end of the transition.
    theta
        A value between [-2\pi, 2\pi] identifying points in the ellipsis drawn
        from the positon and mommentum variables. This value indicates the theta
        value of the accepted proposal.
    subiter
        Number of sub iterations needed to accept a proposal. The more subiterations
        needed the less efficient the algorithm will be, and the more dependent the
        new value is likely to be to the previous value.

    """

    momentum: ArrayTree
    theta: float
    subiter: int


def init(position: ArrayLikeTree, logdensity_fn: Callable):
    logdensity = logdensity_fn(position)
    return EllipSliceState(position, logdensity)


def build_kernel(cov_matrix: Array, mean: Array):
    """Build an Elliptical Slice sampling kernel :cite:p:`murray2010elliptical`.

    Parameters
    ----------
    cov_matrix
        The value of the covariance matrix of the gaussian prior distribution from
        the posterior we wish to sample.

    Returns
    -------
    A kernel that takes a rng_key and a Pytree that contains the current state
    of the chain and that returns a new state of the chain along with
    information about the transition.

    """
    ndim = jnp.ndim(cov_matrix)  # type: ignore[arg-type]

    if ndim == 1:  # diagonal covariance matrix
        cov_matrix_sqrt = jnp.sqrt(cov_matrix)

    elif ndim == 2:
        cov_matrix_sqrt = jax.lax.linalg.cholesky(cov_matrix)

    else:
        raise ValueError(
            "The mass matrix has the wrong number of dimensions:"
            f" expected 1 or 2, got {jnp.ndim(cov_matrix)}."  # type: ignore[arg-type]
        )

    def momentum_generator(rng_key, position):
        return generate_gaussian_noise(rng_key, position, mean, cov_matrix_sqrt)

    def kernel(
        rng_key: PRNGKey,
        state: EllipSliceState,
        logdensity_fn: Callable,
    ) -> tuple[EllipSliceState, EllipSliceInfo]:
        proposal_generator = elliptical_proposal(
            logdensity_fn, momentum_generator, mean
        )
        return proposal_generator(rng_key, state)

    return kernel


def as_top_level_api(
    loglikelihood_fn: Callable,
    *,
    mean: Array,
    cov: Array,
) -> SamplingAlgorithm:
    """Implements the (basic) user interface for the Elliptical Slice sampling kernel.

    Examples
    --------

    A new Elliptical Slice sampling kernel can be initialized and used with the following code:

    .. code::

        ellip_slice = blackjax.elliptical_slice(loglikelihood_fn, cov_matrix)
        state = ellip_slice.init(position)
        new_state, info = ellip_slice.step(rng_key, state)

    We can JIT-compile the step function for better performance

    .. code::

        step = jax.jit(ellip_slice.step)
        new_state, info = step(rng_key, state)

    Parameters
    ----------
    loglikelihood_fn
        Only the log likelihood function from the posterior distributon we wish to sample.
    cov_matrix
        The value of the covariance matrix of the gaussian prior distribution from the posterior we wish to sample.

    Returns
    -------
    A ``SamplingAlgorithm``.
    """
    kernel = build_kernel(cov, mean)

    def init_fn(position: ArrayLikeTree, rng_key=None):
        del rng_key
        return init(position, loglikelihood_fn)

    def step_fn(rng_key: PRNGKey, state):
        return kernel(
            rng_key,
            state,
            loglikelihood_fn,
        )

    return SamplingAlgorithm(init_fn, step_fn)


def elliptical_proposal(
    logdensity_fn: Callable,
    momentum_generator: Callable,
    mean: Array,
) -> Callable:
    """Build an Ellitpical slice sampling kernel.

    The algorithm samples a latent parameter, traces an ellipse connecting the
    initial position and the latent parameter and does slice sampling on this
    ellipse to output a new sample from the posterior distribution.

    Parameters
    ----------
    logdensity_fn
        A function that returns the log-likelihood at a given position.
    momentum_generator
        A function that generates a new latent momentum variable.

    Returns
    -------
    A kernel that takes a rng_key and a Pytree that contains the current state
    of the chain and that returns a new state of the chain along with
    information about the transition.

    """

    def generate(
        rng_key: PRNGKey, state: EllipSliceState
    ) -> tuple[EllipSliceState, EllipSliceInfo]:
        position, logdensity = state
        key_slice, key_momentum, key_uniform, key_theta = jax.random.split(rng_key, 4)
        # step 1: sample momentum
        momentum = momentum_generator(key_momentum, position)
        # step 2: get slice (y)
        logy = logdensity + jnp.log(jax.random.uniform(key_uniform))
        # step 3: get theta (ellipsis move), set inital interval
        theta = 2 * jnp.pi * jax.random.uniform(key_theta)
        theta_min = theta - 2 * jnp.pi
        theta_max = theta
        # step 4: proposal
        p, m = ellipsis(position, momentum, theta, mean)
        # step 5: acceptance
        logdensity = logdensity_fn(p)

        def slice_fn(vals):
            """Perform slice sampling around the ellipsis.

            Checks if the proposed position's likelihood is larger than the slice
            variable. Returns the position if True, shrinks the bracket for sampling
            `theta` and samples a new proposal if False.

            As the bracket `[theta_min, theta_max]` shrinks, the proposal gets closer
            to the original position, which has likelihood larger than the slice variable.
            It is guaranteed to stop in a finite number of iterations as long as the
            likelihood is continuous with respect to the parameter being sampled.

            """
            _, subiter, theta, theta_min, theta_max, *_ = vals
            thetak = jax.random.fold_in(key_slice, subiter)
            theta = jax.random.uniform(thetak, minval=theta_min, maxval=theta_max)
            p, m = ellipsis(position, momentum, theta, mean)
            logdensity = logdensity_fn(p)
            theta_min = jnp.where(theta < 0, theta, theta_min)
            theta_max = jnp.where(theta > 0, theta, theta_max)
            subiter += 1
            return logdensity, subiter, theta, theta_min, theta_max, p, m

        logdensity, subiter, theta, *_, position, momentum = jax.lax.while_loop(
            lambda vals: vals[0] <= logy,
            slice_fn,
            (logdensity, 1, theta, theta_min, theta_max, p, m),
        )
        return (
            EllipSliceState(position, logdensity),
            EllipSliceInfo(momentum, theta, subiter),
        )

    return generate


def ellipsis(position, momentum, theta, mean):
    """Generate proposal from the ellipsis.

    Given a scalar theta indicating a point on the circumference of the ellipsis
    and the shared mean vector for both position and momentum variables,
    generate proposed position and momentum to later accept or reject
    depending on the slice variable.

    """
    position, unravel_fn = jax.flatten_util.ravel_pytree(position)
    momentum, _ = jax.flatten_util.ravel_pytree(momentum)
    position_centered = position - mean
    momentum_centered = momentum - mean
    return (
        unravel_fn(
            position_centered * jnp.cos(theta)
            + momentum_centered * jnp.sin(theta)
            + mean
        ),
        unravel_fn(
            momentum_centered * jnp.cos(theta)
            - position_centered * jnp.sin(theta)
            + mean
        ),
    )
