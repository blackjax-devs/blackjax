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
import jax.flatten_util
import jax.numpy as jnp
import jax.scipy as jsp
from optax import GradientTransformation, OptState

from blackjax.base import VIAlgorithm
from blackjax.types import ArrayLikeTree, ArrayTree, PRNGKey

__all__ = [
    "FRVIState",
    "FRVIInfo",
    "sample",
    "generate_fullrank_logdensity",
    "step",
    "as_top_level_api",
]


class FRVIState(NamedTuple):
    """State of the full-rank VI algorithm.

    mu:
        Mean of the Gaussian approximation.
    chol_params:
        Flattened Cholesky factor of the Gaussian approximation, used to parameterize
        the full-rank covariance matrix. A vector of length d(d+1)/2 for a
        d-dimensional Gaussian, containing d diagonal elements (in log space) followed
        by lower triangular elements in row-major order.
    opt_state:
        Optax optimizer state.

    """

    mu: ArrayTree
    chol_params: ArrayTree
    opt_state: OptState


class FRVIInfo(NamedTuple):
    """Extra information of the full-rank VI algorithm.

    elbo:
        ELBO of approximation wrt target distribution.

    """

    elbo: float


def init(
    position: ArrayLikeTree,
    optimizer: GradientTransformation,
    *optimizer_args,
    **optimizer_kwargs,
) -> FRVIState:
    """Initialize the full-rank VI state with zero mean and identity covariance."""
    mu = jax.tree.map(jnp.zeros_like, position)
    dim = jax.flatten_util.ravel_pytree(mu)[0].shape[0]
    chol_params = jnp.zeros(dim * (dim + 1) // 2)
    opt_state = optimizer.init((mu, chol_params))
    return FRVIState(mu, chol_params, opt_state)


def step(
    rng_key: PRNGKey,
    state: FRVIState,
    logdensity_fn: Callable,
    optimizer: GradientTransformation,
    num_samples: int = 5,
    stl_estimator: bool = True,
) -> tuple[FRVIState, FRVIInfo]:
    """Approximate the target density using the full-rank Gaussian approximation.

    Parameters
    ----------
    rng_key
        Key for JAX's pseudo-random number generator.
    init_state
        Initial state of the full-rank approximation.
    logdensity_fn
        Function that represents the target log-density to approximate.
    optimizer
        Optax `GradientTransformation` to be used for optimization.
    num_samples
        The number of samples that are taken from the approximation
        at each step to compute the Kullback-Leibler divergence between
        the approximation and the target log-density.
    stl_estimator
        Whether to use stick-the-landing (STL) gradient estimator :cite:p:`roeder2017sticking` for gradient estimation.
        The STL estimator has lower gradient variance by removing the score function term
        from the gradient. It is suggested by :cite:p:`agrawal2020advances` to always keep it in order for better results.

    """

    parameters = (state.mu, state.chol_params)

    def kl_divergence_fn(parameters):
        mu, chol_params = parameters
        z = _sample(rng_key, mu, chol_params, num_samples)
        if stl_estimator:
            parameters = jax.tree.map(jax.lax.stop_gradient, (mu, chol_params))
        logq = jax.vmap(generate_fullrank_logdensity(mu, chol_params))(z)
        logp = jax.vmap(logdensity_fn)(z)
        return (logq - logp).mean()

    elbo, elbo_grad = jax.value_and_grad(kl_divergence_fn)(parameters)
    updates, new_opt_state = optimizer.update(elbo_grad, state.opt_state, parameters)
    new_parameters = jax.tree.map(lambda p, u: p + u, parameters, updates)
    new_state = FRVIState(new_parameters[0], new_parameters[1], new_opt_state)
    return new_state, FRVIInfo(elbo)


def sample(rng_key: PRNGKey, state: FRVIState, num_samples: int = 1):
    """Sample from the full-rank approximation."""
    return _sample(rng_key, state.mu, state.chol_params, num_samples)


def as_top_level_api(
    logdensity_fn: Callable,
    optimizer: GradientTransformation,
    num_samples: int = 100,
):
    """High-level implementation of Full-Rank Variational Inference.

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

    def step_fn(rng_key: PRNGKey, state: FRVIState) -> tuple[FRVIState, FRVIInfo]:
        return step(rng_key, state, logdensity_fn, optimizer, num_samples)

    def sample_fn(rng_key: PRNGKey, state: FRVIState, num_samples: int):
        return sample(rng_key, state, num_samples)

    return VIAlgorithm(init_fn, step_fn, sample_fn)


def _unflatten_cholesky(chol_params, dim):
    """Construct the Cholesky factor from a flattened vector of Cholesky parameters.

    Transforms a flattened vector representation of the Cholesky factor (`chol_params`)
    into its proper lower triangular matrix form (`chol_factor`). It specifically
    reshapes the input vector `chol_params` into a lower triangular matrix with zeros
    above the diagonal and exponentiates the diagonal elements to ensure positivity.

    The Cholesky factor (L) is a lower triangular matrix with positive diagonal
    elements used to parameterize the full-rank covariance matrix of the Gaussian
    approximation as Sigma = LL^T.

    This parameterization allows for (1) efficient sampling and log density evaluation,
    and (2) ensuring the covariance matrix is symmetric and positive definite during
    (unconconstrained) optimization.

    Parameters
    ----------
    chol_params
        Flattened Cholesky factor of the full-rank covariance matrix.
    dim
        Dimensionality of the Gaussian distribution.

    Returns
    -------
    chol_factor
        Cholesky factor of the full-rank covariance matrix.

    """

    tril = jnp.zeros((dim, dim))
    tril = tril.at[jnp.tril_indices(dim, k=-1)].set(chol_params[dim:])
    diag = jnp.exp(chol_params[:dim])  # TODO: replace with softplus?
    chol_factor = tril + jnp.diag(diag)
    return chol_factor


def _sample(rng_key, mu, chol_params, num_samples):
    """Sample from the full-rank Gaussian approximation of the target distribution.

    Parameters
    ----------
    rng_key
        Key for JAX's pseudo-random number generator.
    mu
        Mean of the Gaussian approximation.
    chol_params
        Flattened Cholesky factor of the Gaussian approximation.
    num_samples
        Number of samples to draw.

    Returns
    -------
    Samples drawn from the full-rank Gaussian approximation.

    """

    mu_flatten, unravel_fn = jax.flatten_util.ravel_pytree(mu)
    dim = mu_flatten.size
    chol_factor = _unflatten_cholesky(chol_params, dim)
    eps = jax.random.normal(rng_key, (num_samples,) + (dim,))
    flatten_sample = eps @ chol_factor.T + mu_flatten
    return jax.vmap(unravel_fn)(flatten_sample)


def generate_fullrank_logdensity(mu, chol_params):
    """Generate the log-density function of a full-rank Gaussian distribution.

    Parameters
    ----------
    mu
        Mean of the Gaussian distribution.
    chol_params
        Flattened Cholesky factor of the Gaussian distribution.

    Returns
    -------
    A function that computes the log-density of the full-rank Gaussian distribution.

    """

    mu_flatten, _ = jax.flatten_util.ravel_pytree(mu)
    dim = mu_flatten.size
    chol_factor = _unflatten_cholesky(chol_params, dim)
    log_det = 2 * jnp.sum(jnp.log(jnp.diag(chol_factor)))
    const = -0.5 * dim * jnp.log(2 * jnp.pi)

    def fullrank_logdensity(position):
        position_flatten, _ = jax.flatten_util.ravel_pytree(position)
        centered_position = position_flatten - mu_flatten
        y = jsp.linalg.solve_triangular(chol_factor, centered_position, lower=True)
        mahalanobis_dist = jnp.sum(y**2)
        return const - 0.5 * (log_det + mahalanobis_dist)

    return fullrank_logdensity
