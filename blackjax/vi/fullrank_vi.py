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
from blackjax.types import Array, ArrayLikeTree, ArrayTree, PRNGKey
from blackjax.vi._gaussian_vi import KL, Objective, RenyiAlpha, _elbo_step

__all__ = [
    "KL",
    "RenyiAlpha",
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
    chol_params: Array
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
    """Initialize the full-rank VI state with zero mean and identity covariance.

    The Cholesky diagonal is initialized to ``exp(0) = 1``, giving an identity
    covariance as the starting approximation.
    """
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
    objective: Objective = KL(),
    stl_estimator: bool = True,
) -> tuple[FRVIState, FRVIInfo]:
    """Approximate the target density using the full-rank Gaussian approximation.

    Parameters
    ----------
    rng_key
        Key for JAX's pseudo-random number generator.
    state
        Current state of the full-rank approximation.
    logdensity_fn
        Function that represents the target log-density to approximate.
    optimizer
        Optax `GradientTransformation` to be used for optimization.
    num_samples
        The number of samples that are taken from the approximation
        at each step to compute the Kullback-Leibler divergence between
        the approximation and the target log-density.
    objective:
        The variational objective to minimize. `KL()` by default or
        `RenyiAlpha(alpha)`. For alpha = 1, Renyi reduces to KL.
    stl_estimator
        Whether to use the stick-the-landing (STL) gradient estimator
        :cite:p:`roeder2017sticking`. Reduces gradient variance by removing
        the score function term. Recommended in :cite:p:`agrawal2020advances`.

    Returns
    -------
    new_state
        Updated ``FRVIState``.
    info
        ``FRVIInfo`` containing the current ELBO value.

    """

    parameters = (state.mu, state.chol_params)

    def sample_fn(rng_key, parameters, num_samples):
        return _sample(rng_key, parameters[0], parameters[1], num_samples)

    def logq_fn(parameters):
        return generate_fullrank_logdensity(parameters[0], parameters[1])

    new_parameters, new_opt_state, elbo = _elbo_step(
        rng_key,
        parameters,
        state.opt_state,
        logdensity_fn,
        optimizer,
        sample_fn,
        logq_fn,
        num_samples,
        objective=objective,
        stl_estimator=stl_estimator,
    )
    new_state = FRVIState(new_parameters[0], new_parameters[1], new_opt_state)
    return new_state, FRVIInfo(elbo)


def sample(rng_key: PRNGKey, state: FRVIState, num_samples: int = 1):
    """Sample from the full-rank approximation.

    Parameters
    ----------
    rng_key
        Key for JAX's pseudo-random number generator.
    state
        Current ``FRVIState``.
    num_samples
        Number of samples to draw.

    Returns
    -------
    Samples from the full-rank Gaussian approximation, as a PyTree with a
    leading axis of size ``num_samples``.

    """
    return _sample(rng_key, state.mu, state.chol_params, num_samples)


def as_top_level_api(
    logdensity_fn: Callable,
    optimizer: GradientTransformation,
    num_samples: int = 100,
    objective: Objective = KL(),
    stl_estimator: bool = True,
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
    objective
        The variational objective to minimize. `KL()` by default or
        `RenyiAlpha(alpha)`. For alpha = 1, Renyi reduces to KL.
    stl_estimator
        Whether to use STL gradient estimator.
        Only supported when `objective` is `KL()` or `RenyiAlpha(alpha=1.0)`.

    Returns
    -------
    A ``VIAlgorithm``.

    """

    def init_fn(position: ArrayLikeTree):
        return init(position, optimizer)

    def step_fn(rng_key: PRNGKey, state: FRVIState) -> tuple[FRVIState, FRVIInfo]:
        return step(
            rng_key,
            state,
            logdensity_fn,
            optimizer,
            num_samples,
            objective=objective,
            stl_estimator=stl_estimator,
        )

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
    # exp ensures positivity of diagonal during unconstrained optimization;
    # its gradient is well-behaved near zero unlike softplus.
    diag = jnp.exp(chol_params[:dim])
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

    # We compute the log-density directly from the Cholesky factor rather than
    # using jax.scipy.stats.multivariate_normal.logpdf, which would require
    # forming Sigma = L L^T and re-decomposing it internally — wasting the
    # Cholesky factorization we already have.
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
