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
   * mean field variational inference
   * full rank variational inference"""
from dataclasses import dataclass
from typing import Callable, Union

import jax
import jax.numpy as jnp
import jax.scipy as jsp
from optax import GradientTransformation, OptState


@dataclass(frozen=True)
class KL:
    """Standard reverse-KL objective."""

    pass


@dataclass(frozen=True)
class RenyiAlpha:
    """Rényi alpha objective."""

    alpha: float


@dataclass(frozen=True)
class TailAdaptive:
    """Tail-adaptive f-divergence objective.

    Parameters
    ----------
    beta
        Tail-adaptation exponent. The paper recommends negative values to obtain
        mass-covering behaviour. Theory uses ``beta > -1`` while the empirical
        default in the paper is often ``beta = -1``.
    """

    beta: float = -1.0


Objective = Union[KL, RenyiAlpha, TailAdaptive]


def _tail_adaptive_weights_from_log_ratio(
    log_ratio: jax.Array,
    beta: float,
) -> jax.Array:
    """Compute normalized tail adaptive weights for variational inference.

    The weights are based on the empirical survival function of the importance
    weights `w = p(x) / q(x)`.  Large values of `w` indicate that the proposal
    `q` underrepresents the target `p`; such samples receive high weights when
    `β < 0`, encouraging a mass covering behaviour. Using the inverse tail
    probability guarantees finite moments even when `w` is heavy tailed,
    unlike fixed alpha divergence.

    Function works in log space: `log_ratio = log q(x) - log p(x) = -log w(x)`.
    The condition `w_j >= w_i` is equivalent to `log_ratio_j <= log_ratio_i`.
    So that the empirical survival probability for sample `i` is given by:

    F_hat(w_i) = (1/n) * sum_{j=1}^n I(w_j >= w_i) =
                 (1/n) * sum_{j=1}^n I(log_ratio_j <= log_ratio_i)

    Parameters
    ----------
    log_ratio : jax.Array
        One dimensional array of values `log q(x_i) - log p(x_i)` for a batch of
        samples `{x_i}` drawn from the current variational distribution `q`.

    beta : float
        Exponent controlling the tail of probability.  Negative values
        (typically `-1`) give larger weights to samples with high `w`
        (where `q` underestimates `p`), enforcing mass covering.

    Returns
    -------
    jax.Array
        Normalised tail adaptive weights, shape `(n,)`, summing to one.
        These weights are used to reweight the gradient estimate.
    """
    if log_ratio.ndim != 1:
        raise ValueError(
            "Tail-adaptive weighting expects a one-dimensional batch of "
            f"log-ratios got shape {log_ratio.shape}."
        )

    stopped_log_ratio = jax.lax.stop_gradient(log_ratio)
    empirical_survival = jnp.mean(
        (stopped_log_ratio[None, :] <= stopped_log_ratio[:, None]).astype(
            log_ratio.dtype
        ),
        axis=1,
    )
    raw_weights = jnp.power(empirical_survival, jnp.asarray(beta, log_ratio.dtype))
    return raw_weights / jnp.sum(raw_weights)


def _objective_value_from_log_ratio(
    log_ratio: jax.Array,
    objective: Objective,
) -> jax.Array:
    """Return a scalar loss to minimize."""
    if isinstance(objective, KL):
        return jnp.mean(log_ratio)

    if isinstance(objective, RenyiAlpha):
        alpha = objective.alpha

        # Continuous recovery of KL at alpha = 1.
        if alpha == 1.0:
            return jnp.mean(log_ratio)

        # Negative Monte Carlo Rényi variational bound:
        #   -L_hat_alpha
        # = (1 / (alpha - 1)) * log mean(exp((alpha - 1) * (logq - logp)))
        scaled = (alpha - 1.0) * log_ratio
        return (jsp.special.logsumexp(scaled) - jnp.log(log_ratio.shape[0])) / (
            alpha - 1.0
        )

    if isinstance(objective, TailAdaptive):
        weights = jax.lax.stop_gradient(
            _tail_adaptive_weights_from_log_ratio(log_ratio, objective.beta)
        )
        return jnp.sum(weights * log_ratio)

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

    Single ELBO optimization step shared by Gaussian VI variants.

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

    if stl_estimator and isinstance(objective, RenyiAlpha) and objective.alpha != 1.0:
        raise ValueError(
            "stl_estimator is currently only supported with KL() or "
            "RenyiAlpha(alpha=1.0). Use stl_estimator=False for "
            "RenyiAlpha(alpha != 1.0)."
        )

    if isinstance(objective, TailAdaptive) and not stl_estimator:
        raise ValueError(
            "TailAdaptive is implemented via the reparameterization-gradient "
            "and requires stl_estimator=True"
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
