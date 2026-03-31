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
from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp
import jax.scipy as jsp
from optax import GradientTransformation, OptState

from blackjax.base import VIAlgorithm
from blackjax.types import ArrayLikeTree, ArrayTree, PRNGKey
from blackjax.vi._gaussian_vi import _elbo_step

__all__ = [
    "MFVIState",
    "MFVIInfo",
    "sample",
    "generate_meanfield_logdensity",
    "step",
    "as_top_level_api",
]


class MFVIState(NamedTuple):
    mu: ArrayTree
    rho: ArrayTree
    opt_state: OptState


class MFVIInfo(NamedTuple):
    elbo: float


def init(
    position: ArrayLikeTree,
    optimizer: GradientTransformation,
    *optimizer_args,
    **optimizer_kwargs,
) -> MFVIState:
    """Initialize the mean-field VI state.

    Parameters
    ----------
    position
        Reference PyTree used to infer shapes; mean is initialized to zero,
        log-scale ``rho`` to ``-2``.
    optimizer
        Optax optimizer whose state is initialized here.

    Returns
    -------
    Initial MFVIState with zero mean, rho=-2, and optimizer state.
    """
    mu = jax.tree.map(jnp.zeros_like, position)
    rho = jax.tree.map(lambda x: -2.0 * jnp.ones_like(x), position)
    opt_state = optimizer.init((mu, rho))
    return MFVIState(mu, rho, opt_state)


def step(
    rng_key: PRNGKey,
    state: MFVIState,
    logdensity_fn: Callable,
    optimizer: GradientTransformation,
    num_samples: int = 5,
    stl_estimator: bool = True,
) -> tuple[MFVIState, MFVIInfo]:
    """Approximate the target density using the mean-field approximation.

    Parameters
    ----------
    rng_key
        Key for JAX's pseudo-random number generator.
    state
        Current state of the mean-field approximation.
    logdensity_fn
        Function that represents the target log-density to approximate.
    optimizer
        Optax ``GradientTransformation`` to be used for optimization.
    num_samples
        The number of samples that are taken from the approximation
        at each step to compute the Kullback-Leibler divergence between
        the approximation and the target log-density.
    stl_estimator
        Whether to use the stick-the-landing (STL) gradient estimator
        :cite:p:`roeder2017sticking`. The STL estimator has lower gradient
        variance by removing the score function term from the gradient.
        :cite:p:`agrawal2020advances` recommend keeping it enabled.

    Returns
    -------
    Updated MFVIState and MFVIInfo containing the ELBO value.
    """

    parameters = (state.mu, state.rho)

    def sample_fn(rng_key, parameters, num_samples):
        return _sample(rng_key, parameters[0], parameters[1], num_samples)

    def logq_fn(parameters):
        return generate_meanfield_logdensity(parameters[0], parameters[1])

    new_parameters, new_opt_state, elbo = _elbo_step(
        rng_key,
        parameters,
        state.opt_state,
        logdensity_fn,
        optimizer,
        sample_fn,
        logq_fn,
        num_samples,
        stl_estimator,
    )
    new_state = MFVIState(new_parameters[0], new_parameters[1], new_opt_state)
    return new_state, MFVIInfo(elbo)


def sample(rng_key: PRNGKey, state: MFVIState, num_samples: int = 1):
    """Sample from the mean-field approximation.

    Parameters
    ----------
    rng_key
        Key for JAX's pseudo-random number generator.
    state
        Current MFVIState containing the variational parameters.
    num_samples
        Number of samples to draw.

    Returns
    -------
    A PyTree of samples with leading dimension ``num_samples``.
    """
    return _sample(rng_key, state.mu, state.rho, num_samples)


def as_top_level_api(
    logdensity_fn: Callable,
    optimizer: GradientTransformation,
    num_samples: int = 100,
):
    """High-level implementation of Mean-Field Variational Inference.

    Parameters
    ----------
    logdensity_fn
        A function that represents the log-density function associated with
        the distribution we want to sample from.
    optimizer
        Optax optimizer to use to optimize the ELBO.
    num_samples
        Number of samples to take at each step to optimize the ELBO.

    Returns
    -------
    A ``VIAlgorithm``.

    """

    def init_fn(position: ArrayLikeTree):
        return init(position, optimizer)

    def step_fn(rng_key: PRNGKey, state: MFVIState) -> tuple[MFVIState, MFVIInfo]:
        return step(rng_key, state, logdensity_fn, optimizer, num_samples)

    def sample_fn(rng_key: PRNGKey, state: MFVIState, num_samples: int):
        return sample(rng_key, state, num_samples)

    return VIAlgorithm(init_fn, step_fn, sample_fn)


def _sample(rng_key, mu, rho, num_samples):
    sigma = jax.tree.map(jnp.exp, rho)
    mu_flatten, unravel_fn = jax.flatten_util.ravel_pytree(mu)
    sigma_flat, _ = jax.flatten_util.ravel_pytree(sigma)
    flatten_sample = (
        jax.random.normal(rng_key, (num_samples,) + mu_flatten.shape) * sigma_flat
        + mu_flatten
    )
    return jax.vmap(unravel_fn)(flatten_sample)


def generate_meanfield_logdensity(mu, rho):
    sigma_param = jax.tree.map(jnp.exp, rho)

    def meanfield_logdensity(position):
        logq_pytree = jax.tree.map(jsp.stats.norm.logpdf, position, mu, sigma_param)
        logq = jax.tree.map(jnp.sum, logq_pytree)
        return jax.tree.reduce(jnp.add, logq)

    return meanfield_logdensity
