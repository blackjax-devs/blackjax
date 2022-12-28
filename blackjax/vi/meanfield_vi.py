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

import jax
import jax.numpy as jnp
import jax.scipy as jsp
from optax import GradientTransformation, OptState

from blackjax.types import PRNGKey, PyTree

__all__ = ["MFVIState", "MFVIInfo", "sample", "generate_meanfield_logdensity", "step"]


class MFVIState(NamedTuple):
    mu: PyTree
    rho: PyTree
    opt_state: OptState


class MFVIInfo(NamedTuple):
    elbo: float


def init(
    position: PyTree,
    optimizer: GradientTransformation,
    *optimizer_args,
    **optimizer_kwargs
) -> MFVIState:
    """Initialize the mean-field VI state."""
    mu = jax.tree_map(jnp.zeros_like, position)
    rho = jax.tree_map(lambda x: -2.0 * jnp.ones_like(x), position)
    opt_state = optimizer.init((mu, rho))
    return MFVIState(mu, rho, opt_state)


def step(
    rng_key: PRNGKey,
    state: MFVIState,
    logdensity_fn: Callable,
    optimizer: GradientTransformation,
    num_samples: int = 5,
    stl_estimator: bool = True,
) -> Tuple[MFVIState, MFVIInfo]:
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

    parameters = (state.mu, state.rho)

    def kl_divergence_fn(parameters):
        mu, rho = parameters
        z = _sample(rng_key, mu, rho, num_samples)
        if stl_estimator:
            mu = jax.lax.stop_gradient(mu)
            rho = jax.lax.stop_gradient(rho)
        logq = jax.vmap(generate_meanfield_logdensity(mu, rho))(z)
        logp = jax.vmap(logdensity_fn)(z)
        return (logq - logp).mean()

    elbo, elbo_grad = jax.value_and_grad(kl_divergence_fn)(parameters)
    updates, new_opt_state = optimizer.update(elbo_grad, state.opt_state, parameters)
    new_parameters = jax.tree_map(lambda p, u: p + u, parameters, updates)
    new_state = MFVIState(new_parameters[0], new_parameters[1], new_opt_state)
    return new_state, MFVIInfo(elbo)


def sample(rng_key: PRNGKey, state: MFVIState, num_samples: int = 1):
    """Sample from the mean-field approximation."""
    return _sample(rng_key, state.mu, state.rho, num_samples)


def _sample(rng_key, mu, rho, num_samples):
    sigma = jax.tree_map(jnp.exp, rho)
    mu_flatten, unravel_fn = jax.flatten_util.ravel_pytree(mu)
    sigma_flat, _ = jax.flatten_util.ravel_pytree(sigma)
    flatten_sample = (
        jax.random.normal(rng_key, (num_samples,) + mu_flatten.shape) * sigma_flat
        + mu_flatten
    )
    return jax.vmap(unravel_fn)(flatten_sample)


def generate_meanfield_logdensity(mu, rho):
    sigma_param = jax.tree_map(jnp.exp, rho)

    def meanfield_logdensity(position):
        logq_pytree = jax.tree_map(jsp.stats.norm.logpdf, position, mu, sigma_param)
        logq = jax.tree_map(jnp.sum, logq_pytree)
        return jax.tree_util.tree_reduce(jnp.add, logq)

    return meanfield_logdensity
