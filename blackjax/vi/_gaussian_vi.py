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
"""Shared ELBO optimization step for Gaussian VI variants (MFVI, FRVI)."""
from typing import Callable

import jax
from optax import GradientTransformation, OptState


def _elbo_step(
    rng_key,
    parameters: tuple,
    opt_state: OptState,
    logdensity_fn: Callable,
    optimizer: GradientTransformation,
    sample_fn: Callable,
    logq_fn: Callable,
    num_samples: int,
    stl_estimator: bool,
) -> tuple[tuple, OptState, float]:
    """Single ELBO optimization step shared by Gaussian VI variants.

    Computes the KL divergence ``E_q[log q - log p]`` via Monte Carlo,
    differentiates with respect to ``parameters``, and applies one optimizer
    update.

    Parameters
    ----------
    rng_key
        Key for JAX's pseudo-random number generator.
    parameters
        Tuple of variational parameters ``(mu, covariance_params)``.
    opt_state
        Current optimizer state.
    logdensity_fn
        Target log-density (unnormalized).
    optimizer
        Optax ``GradientTransformation``.
    sample_fn
        ``(rng_key, parameters, num_samples) -> samples``; draws samples from
        the current approximation.
    logq_fn
        ``parameters -> (sample -> log_q_scalar)``; returns the log-density
        function of the current approximation given its parameters.
    num_samples
        Number of Monte Carlo samples used to estimate the ELBO.
    stl_estimator
        If ``True``, apply ``stop_gradient`` to the parameters used in
        ``logq_fn`` (stick-the-landing estimator). Gradients still flow
        through the samples drawn by ``sample_fn``.

    Returns
    -------
    new_parameters
        Updated variational parameters after one optimizer step.
    new_opt_state
        Updated optimizer state.
    elbo
        Current ELBO estimate (scalar).

    """

    def kl_divergence_fn(parameters):
        z = sample_fn(rng_key, parameters, num_samples)
        logq_parameters = (
            jax.lax.stop_gradient(parameters) if stl_estimator else parameters
        )
        logq = jax.vmap(logq_fn(logq_parameters))(z)
        logp = jax.vmap(logdensity_fn)(z)
        return (logq - logp).mean()

    elbo, elbo_grad = jax.value_and_grad(kl_divergence_fn)(parameters)
    updates, new_opt_state = optimizer.update(elbo_grad, opt_state, parameters)
    new_parameters = jax.tree.map(lambda p, u: p + u, parameters, updates)
    return new_parameters, new_opt_state, elbo
