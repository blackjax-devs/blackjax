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
"""Public API for Barker's proposal with a Gaussian base kernel."""
from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp
from jax.scipy import stats
from jax.tree_util import tree_flatten

from blackjax.base import SamplingAlgorithm
from blackjax.types import ArrayLikeTree, ArrayTree, PRNGKey

__all__ = ["BarkerState", "BarkerInfo", "init", "build_kernel", "barker_proposal"]


class BarkerState(NamedTuple):
    """State of the Barker's proposal algorithm.

    The Barker algorithm takes one position of the chain and returns another
    position. In order to make computations more efficient, we also store
    the current log-probability density as well as the current gradient of the
    log-probability density.

    """

    position: ArrayTree
    logdensity: float
    logdensity_grad: ArrayTree


class BarkerInfo(NamedTuple):
    """Additional information on the Barker's proposal kernel transition.

    This additional information can be used for debugging or computing
    diagnostics.

    proposal
        The proposal that was sampled.
    acceptance_rate
        The acceptance rate of the transition.
    is_accepted
        Whether the proposed position was accepted or the original position
        was returned.

    """

    acceptance_rate: float
    is_accepted: bool
    proposal: BarkerState


def init(position: ArrayLikeTree, logdensity_fn: Callable) -> BarkerState:
    grad_fn = jax.value_and_grad(logdensity_fn)
    logdensity, logdensity_grad = grad_fn(position)
    return BarkerState(position, logdensity, logdensity_grad)


def build_kernel():
    """Build a Barker's proposal kernel.

    Returns
    -------
    A kernel that takes a rng_key and a Pytree that contains the current state
    of the chain and that returns a new state of the chain along with
    information about the transition.

    """

    def _compute_acceptance_probability(
        logdensity: float,
        logdensity_proposal: float,
        logdensity_grad: ArrayTree,
        logdensity_grad_proposal: ArrayTree,
        position: ArrayTree,
        position_proposal: ArrayTree,
        scale: float,
    ) -> float:
        """Compute the acceptance probability of the Barker's proposal kernel."""

        logdensity_grad, _ = tree_flatten(logdensity_grad)
        logdensity_grad_proposal, _ = tree_flatten(logdensity_grad_proposal)
        position, _ = tree_flatten(position)
        position_proposal, _ = tree_flatten(position_proposal)

        def ratio_proposal_nd(y, x, log_y, log_x):
            num = -_log1pexp(-log_y * (x - y))
            den = -_log1pexp(-log_x * (y - x))

            return jnp.sum(num - den)

        ratios_proposals = map(
            ratio_proposal_nd,
            position_proposal,
            position,
            logdensity_grad_proposal,
            logdensity_grad,
        )
        ratio_proposal = sum(ratios_proposals)
        log_p_accept = logdensity_proposal - logdensity + ratio_proposal
        p_accept = jnp.exp(log_p_accept)
        return jnp.minimum(1.0, p_accept)

    def kernel(
        rng_key: PRNGKey, state: BarkerState, logdensity_fn: Callable, step_size: float
    ) -> tuple[BarkerState, BarkerInfo]:
        """Generate a new sample with the MALA kernel."""
        grad_fn = jax.value_and_grad(logdensity_fn)

        key_sample, key_rmh = jax.random.split(rng_key)

        proposed_pos = _barker_sample(
            key_sample, state.position, state.logdensity_grad, step_size
        )
        proposed_logdensity, proposed_logdensity_grad = grad_fn(proposed_pos)

        p_accept = _compute_acceptance_probability(
            state.logdensity,
            proposed_logdensity,
            state.logdensity_grad,
            proposed_logdensity_grad,
            state.position,
            proposed_pos,
            step_size,
        )

        proposed_state = BarkerState(
            proposed_pos, proposed_logdensity, proposed_logdensity_grad
        )
        accept = jax.random.uniform(key_rmh) < p_accept

        state = jax.lax.cond(accept, lambda: proposed_state, lambda: state)
        info = BarkerInfo(p_accept, accept, proposed_state)
        return state, info

    return kernel


class barker_proposal:
    """Implements the (basic) user interface for the Barker's proposal kernel with a Gaussian base kernel.

    The general Barker kernel builder (:meth:`blackjax.mcmc.barker.build_kernel`, alias `blackjax.barker.build_kernel`) can be
    cumbersome to manipulate. Since most users only need to specify the kernel
    parameters at initialization time, we provide a helper function that
    specializes the general kernel.

    We also add the general kernel and state generator as an attribute to this class so
    users only need to pass `blackjax.barker` to SMC, adaptation, etc. algorithms.

    Examples
    --------

    A new Barker kernel can be initialized and used with the following code:

    .. code::

        barker = blackjax.barker(logdensity_fn, step_size)
        state = barker.init(position)
        new_state, info = barker.step(rng_key, state)

    Kernels are not jit-compiled by default so you will need to do it manually:

    .. code::

       step = jax.jit(barker.step)
       new_state, info = step(rng_key, state)

    Should you need to you can always use the base kernel directly:

    .. code::

       kernel = blackjax.barker.build_kernel(logdensity_fn)
       state = blackjax.barker.init(position, logdensity_fn)
       state, info = kernel(rng_key, state, logdensity_fn, step_size)

    Parameters
    ----------
    logdensity_fn
        The log-density function we wish to draw samples from.
    step_size
        The value to use for the step size in the symplectic integrator.

    Returns
    -------
    A ``SamplingAlgorithm``.

    """

    init = staticmethod(init)
    build_kernel = staticmethod(build_kernel)

    def __new__(  # type: ignore[misc]
        cls,
        logdensity_fn: Callable,
        step_size: float,
    ) -> SamplingAlgorithm:
        kernel = cls.build_kernel()

        def init_fn(position: ArrayLikeTree):
            return cls.init(position, logdensity_fn)

        def step_fn(rng_key: PRNGKey, state):
            return kernel(rng_key, state, logdensity_fn, step_size)

        return SamplingAlgorithm(init_fn, step_fn)


def _barker_sample_nd(key, mean, a, scale):
    """
    Sample from a multivariate Barker's proposal distribution. In 1D, this has the following probability density function:

    .. math::
        p(x; \\mu, a, \\sigma) = 2 \frac{N(x; \\mu, \\sigma^2)}{1 + \\exp(-a (x - \\mu)}

    where :math:`N(x; \\mu, \\sigma^2)` is the normal distribution with mean :math:`\\mu` and standard deviation :math:`\\sigma`.
    The multivariate Barker's proposal distribution is the product of one-dimensional Barker's proposal distributions.


    Parameters
    ----------
    key
        A PRNG key.
    mean
        The mean of the normal distribution, an Array. This corresponds to :math:`\\mu` in the equation above.
    a
        The parameter :math:`a` in the equation above, an Array. This is a skewness parameter.
    scale
        The standard deviation of the normal distribution, a scalar. This corresponds to :math:`\\sigma` in the equation above.
        It encodes the step size of the proposal.

    Returns
    -------
    A sample from the Barker's multidimensional proposal distribution.

    """

    key1, key2 = jax.random.split(key)
    z = scale * jax.random.normal(key1, shape=mean.shape)

    # Sample b=1 with probability p and 0 with probability 1 - p where
    # p = 1 / (1 + exp(-a * (z - mean)))
    log_p = -_log1pexp(-a * z)
    b = jax.random.bernoulli(key2, p=jnp.exp(log_p), shape=mean.shape)

    # return mean + z if b == 1 else mean - z
    return mean + b * z - (1 - b) * z


def _barker_sample(key, mean, a, scale):
    r"""
    Sample from a multivariate Barker's proposal distribution for PyTrees.

    Parameters
    ----------
    key
        A PRNG key.
    mean
        The mean of the normal distribution, a PyTree. This corresponds to :math:`\mu` in the equation above.
    a
        The parameter :math:`a` in the equation above, the same PyTree as `mean`. This is a skewness parameter.
    scale
        The standard deviation of the normal distribution, a scalar. This corresponds to :math:`\sigma` in the equation above.
        It encodes the step size of the proposal.

    """

    from jax.tree_util import tree_flatten, tree_map, tree_unflatten

    flat_mean, tree_def = tree_flatten(mean)
    flat_a, _ = tree_flatten(a)
    n_keys = len(flat_mean)

    keys = jax.random.split(key, n_keys)
    keys = [k for k in keys]
    sample = tree_map(
        lambda k, m, a: _barker_sample_nd(k, m, a, scale), keys, flat_mean, flat_a
    )
    # check that the pytrees have the same structure

    sample = tree_unflatten(tree_def, sample)

    return sample


def _log1pexp(a):
    return jnp.log1p(jnp.exp(a))


def _barker_logpdf(x, mean, a, scale):
    logpdf = jnp.log(2) + stats.norm.logpdf(x, mean, scale) - _log1pexp(-a * (x - mean))
    return logpdf


def _barker_pdf(x, mean, a, scale):
    return jnp.exp(_barker_logpdf(x, mean, a, scale))
