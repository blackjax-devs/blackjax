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
"""Shared Gaussian VI optimization step for:
   * mean field variational inference (MFVI)
   * full rank variational inference (FRVI)"""
from dataclasses import dataclass
from typing import Callable

import jax
import jax.numpy as jnp
import jax.scipy as jsp
from optax import GradientTransformation, OptState


@dataclass(frozen=True)
class KL:
    """standard reverse-KL objective"""

    pass


@dataclass(frozen=True)
class RenyiAlpha:
    """Rényi alpha objective.

    Notes
    -----
    A smooth interpolation from the evidence lower-bound to the
    log (marginal) likelihood that is controlled by the value of alpha
    that parametrises the divergence.
    """

    alpha: float


Objective = KL | RenyiAlpha


def _objective_value_from_log_ratio(
    log_ratio: jax.Array,
    objective: Objective,
) -> jax.Array:
    """Returns a scalar loss to minimize from the given log-ratio array and
    supports two objective types.:

    * KL: returns mean of the log-ratio, corresponding to KL divergence loss
    * RenyiAlpha: returns negative Monte Carlo Rényi variational bound.
      For alpha = 1.0 it recovers the reverse-KL objective.
      For other alpha values, it computes:
      (logsumexp((alpha - 1) * log_ratio) - log(N)) / (alpha - 1)
      where N is the number of samples.

     Parameters
     ----------
     log_ratio: A JAX array of log-ratio values (log q - log p)
     objective: An instance of objective (KL or RenyiAlpha)

     Returns
     -------
     A scalar JAX array representing the loss value to be minimized.

    """
    if isinstance(objective, KL):
        return jnp.mean(log_ratio)

    if isinstance(objective, RenyiAlpha):
        alpha = objective.alpha

        # for alpha = 1.0 it recovers the reverse-KL objective.
        if alpha == 1.0:
            return jnp.mean(log_ratio)

        # negative Monte Carlo Renyi variational bound:
        # -L_hat_alpha = (1 / (alpha - 1)) * log mean(exp((alpha - 1) * (logq - logp)))
        scaled = (alpha - 1.0) * log_ratio
        return (jsp.special.logsumexp(scaled) - jnp.log(log_ratio.shape[0])) / (
            alpha - 1.0
        )

    raise TypeError(f"Unsupported objective type: {type(objective)!r}")


def _elbo_step(
    rng_key,
    parameters: tuple,
    opt_state: OptState,
    logdensity_fn: Callable,
    optimizer: GradientTransformation,
    sample_fn: Callable,
    logq_fn: Callable,
    num_samples: int,
    objective: Objective = KL(),
    stl_estimator: bool = True,
) -> tuple[tuple, OptState, float]:
    """Single Gaussian VI optimization step shared by MFVI and FRVI.

    Single step of variational optimisation (ELBO or Renyi bound)
    shared by Gaussian VI variants. Computes a variational loss
    (KL or Renyi) via Monte Carlo, differentiates with respect to
    ``parameters``, and applies one optimizer update.

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
    objective
        The variational objective (KL or Rényi). Defaults to KL.
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
    loss
        Current estimate of the variational loss (scalar).

    """

    if stl_estimator and isinstance(objective, RenyiAlpha) and objective.alpha != 1.0:
        raise ValueError(
            "stl_estimator is currently only supported with KL() or "
            "RenyiAlpha(alpha=1.0). Use stl_estimator=False for "
            "RenyiAlpha(alpha != 1.0)."
        )

    def objective_fn(parameters):
        z = sample_fn(rng_key, parameters, num_samples)
        logq_parameters = (
            jax.lax.stop_gradient(parameters) if stl_estimator else parameters
        )
        logq = jax.vmap(logq_fn(logq_parameters))(z)
        logp = jax.vmap(logdensity_fn)(z)
        log_ratio = logq - logp
        return _objective_value_from_log_ratio(log_ratio, objective)

    objective_value, objective_grad = jax.value_and_grad(objective_fn)(parameters)
    updates, new_opt_state = optimizer.update(objective_grad, opt_state, parameters)
    new_parameters = jax.tree.map(lambda p, u: p + u, parameters, updates)
    return new_parameters, new_opt_state, objective_value
