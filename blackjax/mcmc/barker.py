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
from jax.flatten_util import ravel_pytree
from jax.scipy import stats

import blackjax.mcmc.metrics as metrics
from blackjax.base import SamplingAlgorithm
from blackjax.mcmc.metrics import Metric
from blackjax.mcmc.proposal import static_binomial_sampling
from blackjax.types import ArrayLikeTree, ArrayTree, Numeric, PRNGKey
from blackjax.util import generate_gaussian_noise

__all__ = ["BarkerState", "BarkerInfo", "init", "build_kernel", "as_top_level_api"]


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
        state: BarkerState, proposal: BarkerState, metric: Metric
    ) -> Numeric:
        """Compute the acceptance probability of the Barker's proposal kernel."""

        x = state.position
        y = proposal.position
        log_x = state.logdensity_grad
        log_y = proposal.logdensity_grad

        y_minus_x = jax.tree_util.tree_map(lambda a, b: a - b, y, x)
        x_minus_y = jax.tree_util.tree_map(lambda a: -a, y_minus_x)
        z_tilde_x_to_y = metric.scale(x, y_minus_x, inv=True, trans=True)
        z_tilde_y_to_x = metric.scale(y, x_minus_y, inv=True, trans=True)

        c_x_to_y = metric.scale(x, log_x, inv=False, trans=True)
        c_y_to_x = metric.scale(y, log_y, inv=False, trans=True)

        z_tilde_x_to_y_flat, _ = ravel_pytree(z_tilde_x_to_y)
        z_tilde_y_to_x_flat, _ = ravel_pytree(z_tilde_y_to_x)

        c_x_to_y_flat, _ = ravel_pytree(c_x_to_y)
        c_y_to_x_flat, _ = ravel_pytree(c_y_to_x)

        num = metric.kinetic_energy(x_minus_y, y) - _log1pexp(
            -z_tilde_y_to_x_flat * c_y_to_x_flat
        )
        denom = metric.kinetic_energy(y_minus_x, x) - _log1pexp(
            -z_tilde_x_to_y_flat * c_x_to_y_flat
        )

        ratio_proposal = jnp.sum(num - denom)

        return proposal.logdensity - state.logdensity + ratio_proposal

    def kernel(
        rng_key: PRNGKey,
        state: BarkerState,
        logdensity_fn: Callable,
        step_size: float,
        inverse_mass_matrix: metrics.MetricTypes | None = None,
    ) -> tuple[BarkerState, BarkerInfo]:
        """Generate a new sample with the Barker kernel."""
        if inverse_mass_matrix is None:
            p, _ = ravel_pytree(state.position)
            (m,) = p.shape
            inverse_mass_matrix = jnp.ones((m,))
        metric = metrics.default_metric(inverse_mass_matrix)
        grad_fn = jax.value_and_grad(logdensity_fn)
        key_sample, key_rmh = jax.random.split(rng_key)

        proposed_pos = _barker_sample(
            key_sample,
            state.position,
            state.logdensity_grad,
            step_size,
            metric,
        )

        proposed_logdensity, proposed_logdensity_grad = grad_fn(proposed_pos)
        proposed_state = BarkerState(
            proposed_pos, proposed_logdensity, proposed_logdensity_grad
        )

        log_p_accept = _compute_acceptance_probability(state, proposed_state, metric)
        accepted_state, info = static_binomial_sampling(
            key_rmh, log_p_accept, state, proposed_state
        )
        do_accept, p_accept, _ = info
        return accepted_state, BarkerInfo(p_accept, do_accept, proposed_state)

    return kernel


def as_top_level_api(
    logdensity_fn: Callable,
    step_size: float,
    inverse_mass_matrix: metrics.MetricTypes | None = None,
) -> SamplingAlgorithm:
    """Implements the (basic) user interface for the Barker's proposal :cite:p:`Livingstone2022Barker` kernel with a
    Gaussian base kernel.

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
        The value of the step_size correspnoding to the global scale of the proposal distribution.
    inverse_mass_matrix
        The inverse mass matrix to use for pre-conditioning (see Appendix G of :cite:p:`Livingstone2022Barker`).

    Returns
    -------
    A ``SamplingAlgorithm``.

    """

    kernel = build_kernel()

    def init_fn(position: ArrayLikeTree, rng_key=None):
        del rng_key
        return init(position, logdensity_fn)

    def step_fn(rng_key: PRNGKey, state):
        return kernel(rng_key, state, logdensity_fn, step_size, inverse_mass_matrix)

    return SamplingAlgorithm(init_fn, step_fn)


def _generate_bernoulli(
    rng_key: PRNGKey, position: ArrayLikeTree, p: ArrayLikeTree
) -> ArrayTree:
    pos, unravel_fn = ravel_pytree(position)
    p_flat, _ = ravel_pytree(p)
    sample = jax.random.bernoulli(rng_key, p=p_flat, shape=pos.shape)
    return unravel_fn(sample)


def _barker_sample(key, mean, a, scale, metric):
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
        The global scale, a scalar. This corresponds to :math:`\\sigma` in the equation above.
        It encodes the step size of the proposal.
    metric
        A `metrics.MetricTypes` object encoding the mass matrix information.
    """

    key1, key2 = jax.random.split(key)

    z = generate_gaussian_noise(key1, mean, sigma=scale)
    c = metric.scale(mean, a, inv=False, trans=True)

    # Sample b=1 with probability p and 0 with probability 1 - p where
    # p = 1 / (1 + exp(-a * (z - mean)))
    log_p = jax.tree_util.tree_map(lambda x, y: -_log1pexp(-x * y), c, z)
    p = jax.tree_util.tree_map(lambda x: jnp.exp(x), log_p)
    b = _generate_bernoulli(key2, mean, p=p)

    bz = jax.tree_util.tree_map(lambda x, y: x * y - (1 - x) * y, b, z)

    return jax.tree_util.tree_map(
        lambda a, b: a + b, mean, metric.scale(mean, bz, inv=False, trans=False)
    )


def _log1pexp(a):
    return jnp.log1p(jnp.exp(a))


def _barker_logpdf(x, mean, a, scale):
    logpdf = jnp.log(2) + stats.norm.logpdf(x, mean, scale) - _log1pexp(-a * (x - mean))
    return logpdf


def _barker_pdf(x, mean, a, scale):
    return jnp.exp(_barker_logpdf(x, mean, a, scale))
