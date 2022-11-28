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
from typing import Callable, NamedTuple, Tuple

import jax
import jax.numpy as jnp

from blackjax.types import Array, PRNGKey, PyTree
from blackjax.util import generate_gaussian_noise

__all__ = ["EllipSliceState", "EllipSliceInfo", "init", "kernel"]


class EllipSliceState(NamedTuple):
    """State of the Elliptical Slice sampling algorithm.

    position
        Current position of the chain.
    loglikelihood
        Current value of the log likelihood only.

    """

    position: PyTree
    loglikelihood: PyTree


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

    momentum: PyTree
    theta: float
    subiter: int


def init(position: PyTree, loglikelihood_fn: Callable):
    loglikelihood = loglikelihood_fn(position)
    return EllipSliceState(position, loglikelihood)


def kernel(cov_matrix: Array, mean: Array):
    """Build an Elliptical Slice sampling kernel [1]_.

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

    References
    ----------
    .. [1]: Murray, Iain, Ryan Adams, and David MacKay. "Elliptical slice sampling."
            Proceedings of the thirteenth international conference on artificial intelligence
            and statistics. JMLR Workshop and Conference Proceedings, 2010.

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

    def one_step(
        rng_key: PRNGKey,
        state: EllipSliceState,
        loglikelihood_fn: Callable,
    ) -> Tuple[EllipSliceState, EllipSliceInfo]:
        proposal_generator = elliptical_proposal(
            loglikelihood_fn, momentum_generator, mean
        )
        return proposal_generator(rng_key, state)

    return one_step


def elliptical_proposal(
    loglikelihood_fn: Callable,
    momentum_generator: Callable,
    mean: Array,
) -> Callable:
    """Build an Ellitpical slice sampling kernel.

    The algorithm samples a latent parameter, traces an ellipse connecting the
    initial position and the latent parameter and does slice sampling on this
    ellipse to output a new sample from the posterior distribution.

    Parameters
    ----------
    loglikelihood_fn
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
    ) -> Tuple[EllipSliceState, EllipSliceInfo]:
        position, loglikelihood = state
        key_momentum, key_uniform, key_theta = jax.random.split(rng_key, 3)
        # step 1: sample momentum
        momentum = momentum_generator(key_momentum, position)
        # step 2: get slice (y)
        logy = loglikelihood + jnp.log(jax.random.uniform(key_uniform))
        # step 3: get theta (ellipsis move), set inital interval
        theta = 2 * jnp.pi * jax.random.uniform(key_theta)
        theta_min = theta - 2 * jnp.pi
        theta_max = theta
        # step 4: proposal
        p, m = ellipsis(position, momentum, theta, mean)
        # step 5: acceptance
        loglikelihood = loglikelihood_fn(p)

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
            rng, _, subiter, theta, theta_min, theta_max, *_ = vals
            rng, thetak = jax.random.split(rng)
            theta = jax.random.uniform(thetak, minval=theta_min, maxval=theta_max)
            p, m = ellipsis(position, momentum, theta, mean)
            loglikelihood = loglikelihood_fn(p)
            theta_min = jnp.where(theta < 0, theta, theta_min)
            theta_max = jnp.where(theta > 0, theta, theta_max)
            subiter += 1
            return rng, loglikelihood, subiter, theta, theta_min, theta_max, p, m

        _, loglikelihood, subiter, theta, *_, position, momentum = jax.lax.while_loop(
            lambda vals: vals[1] <= logy,
            slice_fn,
            (rng_key, loglikelihood, 1, theta, theta_min, theta_max, p, m),
        )
        return (
            EllipSliceState(position, loglikelihood),
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
