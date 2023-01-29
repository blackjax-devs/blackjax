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
from typing import Callable, NamedTuple, Tuple

import math
import jax
from jax import lax
import jax.numpy as jnp
import jax.scipy as jsp
from jax.flatten_util import ravel_pytree
from optax import GradientTransformation, OptState

from blackjax.types import PRNGKey, PyTree, Array

__all__ = ["FullrankVIState", "FullrankVIInfo", "sample", "generate_fullrank_logdensity", "step"]


def _real_vector_to_cholesky(X):
    m = len(X)
    n = int(.5 * (-1 + math.sqrt(1 + 8 * m)))
    tril = X[:-n]
    diag = jax.nn.softplus(X[-n:])
    mat = jnp.zeros((n, n))
    mat = mat.at[jnp.tril_indices(n, -1)].set(tril)
    mat = mat + jnp.expand_dims(diag, -1) * jnp.identity(n)
    return mat


class FullrankVIState(NamedTuple):
    mu: PyTree
    L: Array # Flattened scale tril
    opt_state: OptState


class FullrankVIInfo(NamedTuple):
    elbo: float


def init(
    position: PyTree,
    optimizer: GradientTransformation,
    *optimizer_args,
    **optimizer_kwargs
) -> FullrankVIState:
    """Initialize the fullrank VI state."""
    mu = jax.tree_map(jnp.zeros_like, position) # Is this a good init strategy?
    num_latent = len(ravel_pytree(mu)[0])
    L = jnp.concatenate(
            [
                jnp.zeros((num_latent * (num_latent + 1) // 2) - num_latent),
                jnp.ones((num_latent,))
            ]
        )
    opt_state = optimizer.init((mu, L))
    return FullrankVIState(mu, L, opt_state)


def step(
    rng_key: PRNGKey,
    state: FullrankVIState,
    logdensity_fn: Callable,
    optimizer: GradientTransformation,
    num_samples: int = 5,
    stl_estimator: bool = True,
) -> Tuple[FullrankVIState, FullrankVIInfo]:
    """Approximate the target density using the mean-field approximation.

    Parameters
    ----------
    rng_key
        Key for JAX's pseudo-random number generator.
    init_state
        Initial state of the mean-field approximation.
    logdensity_fn
        Function that represents the target log-density to approximate.
    optimizer
        Optax `GradientTransformation` to be used for optimization.
    num_samples
        The number of samples that are taken from the approximation
        at each step to compute the Kullback-Leibler divergence between
        the approximation and the target log-density.
    stl_estimator
        Whether to use stick-the-landing (STL) gradient estimator [1] for gradient estimation.
        The STL estimator has lower gradient variance by removing the score function term
        from the gradient. It is suggested by [2] to always keep it in order for better results.

    References
    ----------
    .. [1]: Roeder, G., Wu, Y., & Duvenaud, D. K. (2017).
        Sticking the landing: Simple, lower-variance gradient estimators for variational inference.
        Advances in Neural Information Processing Systems, 30.
    .. [2]: Agrawal, A., Sheldon, D. R., & Domke, J. (2020).
        Advances in black-box VI: Normalizing flows, importance weighting, and optimization.
        Advances in Neural Information Processing Systems, 33.
    """

    parameters = (state.mu, state.L)

    def kl_divergence_fn(parameters):
        mu, L = parameters
        z = _sample(rng_key, mu, L, num_samples)
        if stl_estimator:
            mu = jax.lax.stop_gradient(mu)
            L = jax.lax.stop_gradient(L)
        logq = jax.vmap(generate_fullrank_logdensity(mu, L))(z)
        logp = jax.vmap(logdensity_fn)(z)
        return (logq - logp).mean()

    elbo, elbo_grad = jax.value_and_grad(kl_divergence_fn)(parameters)
    updates, new_opt_state = optimizer.update(elbo_grad, state.opt_state, parameters)
    new_parameters = jax.tree_map(lambda p, u: p + u, parameters, updates)
    new_state = FullrankVIState(new_parameters[0], new_parameters[1], new_opt_state)
    return new_state, FullrankVIInfo(elbo)


def sample(rng_key: PRNGKey, state: FullrankVIState, num_samples: int = 1):
    """Sample from the fullrank VI approximation."""
    return _sample(rng_key, state.mu, state.L, num_samples)


def _sample(rng_key, mu, L, num_samples):
    mu_flatten, unravel_fn = jax.flatten_util.ravel_pytree(mu)
    L = _real_vector_to_cholesky(L)
    flatten_sample = (
        jax.random.normal(rng_key, (num_samples,) + mu_flatten.shape) @ L
        + mu_flatten
    )
    return jax.vmap(unravel_fn)(flatten_sample)


def generate_fullrank_logdensity(mu, L):
    mu_flatten, _ = jax.flatten_util.ravel_pytree(mu)
    L = _real_vector_to_cholesky(L)
    def fullrank_logdensity(position):
        position = ravel_pytree(position)[0]
        return jsp.stats.multivariate_normal.logpdf(position, mu_flatten, L @ L.T)
    return fullrank_logdensity
