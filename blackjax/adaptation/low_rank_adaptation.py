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
"""Adaptation of the low-rank-modified mass matrix for HMC-family samplers.

Implements Algorithm 1 of :cite:p:`seyboldt2026preconditioning`, following the
nutpie reference implementation.  The mass matrix has the form

.. math::

    M^{-1} = \\operatorname{diag}(\\sigma)
             \\bigl(I + U(\\Lambda - I)U^\\top\\bigr)
             \\operatorname{diag}(\\sigma)

and is adapted by minimising the sample Fisher divergence.  All HMC operations
cost :math:`O(dk)` where :math:`k` is the low rank.

Key algorithmic choices that match nutpie:

* **Population variance** (divide by *n*, not *n-1*) for diagonal scaling.
* **σ clipping** to ``[1e-20, 1e20]`` to avoid premature saturation.
* **Optimal translation** μ* = x̄ + σ²⊙ᾱ is computed and returned.
* **Regularisation**: projected covariance is ``P P^T / (n·γ) + I``
  (nutpie's convention; default γ=1 gives ``P P^T / n + I``).
* **SPD mean** via eigendecomposition of the gradient covariance (not
  Cholesky of the draw covariance).
* **Eigenvalue masking**: components with λ ∈ [1/cutoff, cutoff] are set
  to λ=1 rather than clipped (default cutoff=2, matching nutpie's ``c=2``).

The warmup schedule mirrors Stan's window adaptation: an initial fast phase,
a series of doubling slow windows (metric + step-size), and a final fast
phase.
"""
import inspect
from typing import Callable, NamedTuple

import jax
import jax.flatten_util as fu
import jax.numpy as jnp

import blackjax.mcmc as mcmc
from blackjax.adaptation.base import AdaptationResults, return_all_adapt_info
from blackjax.adaptation.step_size import (
    DualAveragingAdaptationState,
    dual_averaging_adaptation,
)
from blackjax.adaptation.window_adaptation import build_schedule
from blackjax.base import AdaptationAlgorithm
from blackjax.mcmc.metrics import gaussian_euclidean_low_rank
from blackjax.progress_bar import gen_scan_fn
from blackjax.types import Array, ArrayLikeTree, PRNGKey
from blackjax.util import pytree_size

__all__ = [
    "LowRankAdaptationState",
    "base",
    "low_rank_window_adaptation",
]


class LowRankAdaptationState(NamedTuple):
    """State for the low-rank mass matrix window adaptation.

    ss_state
        Internal state of the dual-averaging step-size adapter.
    sigma
        Current diagonal scaling, shape ``(d,)``.
    mu_star
        Current optimal translation ``x̄ + σ² ⊙ ᾱ``, shape ``(d,)``.
    U
        Current low-rank eigenvectors, shape ``(d, max_rank)``.
    lam
        Current eigenvalues, shape ``(max_rank,)``.
    step_size
        Current step size (updated every iteration).
    draws_buffer
        Circular buffer storing the last ``buffer_size`` chain positions,
        shape ``(buffer_size, d)``.
    grads_buffer
        Circular buffer storing the corresponding log-density gradients,
        shape ``(buffer_size, d)``.
    buffer_idx
        Number of samples written to the current buffer (resets at each slow
        window boundary).
    """

    ss_state: DualAveragingAdaptationState
    sigma: Array
    mu_star: Array
    U: Array
    lam: Array
    step_size: float
    draws_buffer: Array
    grads_buffer: Array
    buffer_idx: int


# ---------------------------------------------------------------------------
# Core batch algorithm
# ---------------------------------------------------------------------------


def _spd_mean(A: Array, B: Array) -> Array:
    """Symmetric positive-definite (AIRM) geometric mean of A and B.

    Computes :math:`A \\#_{1/2} B = B^{1/2}(B^{-1/2}AB^{-1/2})^{1/2}B^{1/2}`
    via the eigendecomposition of B (the gradient covariance), following the
    nutpie convention.  Both matrices must be SPD with shape ``(k, k)``.
    """
    # Eigendecompose B: B = V_b D_b V_b^T
    vals_b, vecs_b = jnp.linalg.eigh(B)
    vals_b = jnp.maximum(vals_b, 0.0)
    sqrt_b = jnp.sqrt(vals_b)
    inv_sqrt_b = jnp.where(vals_b > 0, 1.0 / jnp.maximum(sqrt_b, 1e-30), 0.0)

    # M = B^{-1/2} A B^{-1/2} in B's eigenbasis
    tmp = vecs_b.T @ A @ vecs_b  # (k, k)
    M = inv_sqrt_b[:, None] * tmp * inv_sqrt_b[None, :]

    vals_m, vecs_m = jnp.linalg.eigh(M)
    vals_m = jnp.maximum(vals_m, 0.0)
    sqrt_m = jnp.sqrt(vals_m)

    # A # B = B^{1/2} M^{1/2} B^{1/2}
    # = (V_b diag(sqrt_b) V_m) diag(sqrt_m) (V_b diag(sqrt_b) V_m)^T
    W = vecs_b @ (sqrt_b[:, None] * vecs_m)  # (k, k)
    return (W * sqrt_m[None, :]) @ W.T


def _compute_low_rank_metric(
    draws_buffer: Array,
    grads_buffer: Array,
    n: int,
    max_rank: int,
    gamma: float,
    cutoff: float,
) -> tuple[Array, Array, Array, Array]:
    """Compute the low-rank metric from a buffer of draws and gradients.

    Implements Algorithm 1 of :cite:p:`seyboldt2026preconditioning`, following
    the nutpie reference implementation.

    Parameters
    ----------
    draws_buffer
        Shape ``(B, d)``.  The first ``n`` rows are valid; remaining rows are
        zero-padded and masked out.
    grads_buffer
        Shape ``(B, d)``.  Log-density gradients corresponding to each draw.
    n
        Number of valid samples in the buffer (may be a traced integer).
    max_rank
        Maximum number of eigenvectors to retain.
    gamma
        Regularisation scale.  The projected covariance is divided by
        ``n * gamma`` before adding the identity, following nutpie's convention.
        Larger values → stronger regularisation toward the identity.
    cutoff
        Eigenvectors whose eigenvalue falls in ``[1/cutoff, cutoff]`` are
        masked (eigenvalue set to 1), as they provide no useful preconditioning.

    Returns
    -------
    sigma
        Shape ``(d,)``.  Diagonal scaling.
    mu_star
        Shape ``(d,)``.  Optimal translation ``x̄ + σ² ⊙ ᾱ``.
    U
        Shape ``(d, max_rank)``.  Low-rank eigenvectors (orthonormal columns).
    lam
        Shape ``(max_rank,)``.  Eigenvalues (1 for masked components).
    """
    B, d = draws_buffer.shape

    # Mask valid rows
    mask = (jnp.arange(B) < n).astype(draws_buffer.dtype)  # (B,)
    n_safe = jnp.maximum(n, 2).astype(draws_buffer.dtype)  # avoid div-by-zero

    # --- Step 1: diagonal scaling  σ = (Var[x] / Var[∇log p])^{1/4} ---
    mean_x = (mask[:, None] * draws_buffer).sum(0) / n_safe  # (d,)
    mean_g = (mask[:, None] * grads_buffer).sum(0) / n_safe  # (d,)

    diff_x = mask[:, None] * (draws_buffer - mean_x[None, :])  # (B, d)
    diff_g = mask[:, None] * (grads_buffer - mean_g[None, :])  # (B, d)

    # Population variance (n not n-1), matching nutpie
    var_x = (diff_x**2).sum(0) / n_safe  # (d,)
    var_g = (diff_g**2).sum(0) / n_safe  # (d,)

    sigma = jnp.power(jnp.clip(var_x / jnp.maximum(var_g, 1e-10), 0.0, None), 0.25)
    sigma = jnp.clip(sigma, 1e-20, 1e20)  # nutpie range

    # Optimal translation μ* = x̄ + σ² ⊙ ᾱ  (paper §3.2)
    mu_star = mean_x + sigma**2 * mean_g

    # --- Step 2: scale draws and gradients ---
    X = diff_x / sigma[None, :]  # (B, d)  scaled centered draws
    A = diff_g * sigma[None, :]  # (B, d)  scaled centered gradients

    # --- Step 3: principal subspaces via thin SVD ---
    _, _, Vt_x = jnp.linalg.svd(X, full_matrices=False)  # Vt_x: (min(B,d), d)
    _, _, Vt_a = jnp.linalg.svd(A, full_matrices=False)
    U_x = Vt_x[:max_rank].T  # (d, max_rank)
    U_a = Vt_a[:max_rank].T  # (d, max_rank)

    # --- Step 4: combined orthonormal basis Q ∈ R^{d × q}, q = min(d, 2k) ---
    combined = jnp.concatenate([U_x, U_a], axis=1)  # (d, 2*max_rank)
    Q, _ = jnp.linalg.qr(combined)  # Q: (d, min(d, 2*max_rank))
    q = Q.shape[1]  # actual subspace dimension

    # --- Step 5: project onto Q ---
    P_x = Q.T @ X.T  # (q, B)
    P_a = Q.T @ A.T  # (q, B)

    # --- Step 6: projected covariance matrices ---
    # nutpie: C = P P^T / (n * gamma) + I  (scale by 1/gamma, then add identity)
    scale = n_safe * gamma
    C_x = (P_x @ P_x.T) / scale + jnp.eye(q)
    C_a = (P_a @ P_a.T) / scale + jnp.eye(q)

    # --- Step 7: SPD geometric mean Σ = C_x # C_a ---
    Sigma = _spd_mean(C_x, C_a)

    # --- Step 8: eigendecompose Σ in the projected subspace ---
    vals, vecs = jnp.linalg.eigh(Sigma)  # vals ascending, (2k,)
    U_full = Q @ vecs  # (d, 2k) back to original space

    # --- Step 9: select top max_rank by |λ-1|; mask near-unity eigenvalues ---
    # nutpie keeps λ < 1/cutoff or λ > cutoff; others carry no preconditioning
    # benefit.  In JAX (fixed shapes) we retain the slots but set λ=1 to
    # effectively zero out those directions in the metric.
    # When q < max_rank (i.e. d < 2k), only q eigenvectors exist; pad the
    # remainder with zero columns (λ=1 → no effect on the metric).
    actual_rank = min(max_rank, q)  # static: both are Python ints at trace time
    distances = jnp.abs(vals - 1.0)
    order = jnp.argsort(-distances)[:actual_rank]  # (actual_rank,) indices
    U_out = U_full[:, order]  # (d, actual_rank)
    lam_raw = vals[order]
    is_informative = (lam_raw < 1.0 / cutoff) | (lam_raw > cutoff)
    lam_out = jnp.where(is_informative, lam_raw, 1.0)

    if actual_rank < max_rank:
        pad = max_rank - actual_rank
        U_out = jnp.concatenate([U_out, jnp.zeros((d, pad))], axis=1)
        lam_out = jnp.concatenate([lam_out, jnp.ones(pad)])

    return sigma, mu_star, U_out, lam_out


# ---------------------------------------------------------------------------
# Warmup primitives  (init / update / final)
# ---------------------------------------------------------------------------


def base(
    max_rank: int = 10,
    target_acceptance_rate: float = 0.80,
    gamma: float = 1.0,
    cutoff: float = 2.0,
) -> tuple[Callable, Callable, Callable]:
    """Warmup scheme using the low-rank mass matrix adaptation.

    Mirrors Stan's three-phase schedule but replaces Welford covariance
    estimation with the Fisher-divergence-minimising low-rank metric of
    :cite:p:`seyboldt2026preconditioning`, following nutpie's implementation.

    Parameters
    ----------
    max_rank
        Maximum number of eigenvectors retained in the low-rank correction.
    target_acceptance_rate
        Target acceptance rate for dual-averaging step-size adaptation.
    gamma
        Regularisation scale.  The projected covariance is divided by
        ``n * gamma`` before adding identity (nutpie convention).  Default
        ``1.0`` gives ``C = P P^T / n + I``.
    cutoff
        Eigenvectors with eigenvalue in ``[1/cutoff, cutoff]`` are masked
        (eigenvalue set to 1).  Default ``2.0`` matches nutpie's ``c=2``.

    Returns
    -------
    ``(init, update, final)``
        The three adaptation primitives expected by the window-adaptation loop.
    """
    da_init, da_update, da_final = dual_averaging_adaptation(target_acceptance_rate)

    def init(
        position: ArrayLikeTree,
        grad: ArrayLikeTree,
        initial_step_size: float,
        buffer_size: int,
    ) -> LowRankAdaptationState:
        d = pytree_size(position)
        sigma = jnp.ones(d)
        mu_star = jnp.zeros(d)
        U = jnp.zeros((d, max_rank))
        lam = jnp.ones(max_rank)
        ss_state = da_init(initial_step_size)
        draws_buffer = jnp.zeros((buffer_size, d))
        grads_buffer = jnp.zeros((buffer_size, d))
        return LowRankAdaptationState(
            ss_state,
            sigma,
            mu_star,
            U,
            lam,
            initial_step_size,
            draws_buffer,
            grads_buffer,
            0,
        )

    def fast_update(
        position: ArrayLikeTree,
        grad: ArrayLikeTree,
        acceptance_rate: float,
        state: LowRankAdaptationState,
    ) -> LowRankAdaptationState:
        """Fast window: only adapt step size."""
        del position, grad
        new_ss = da_update(state.ss_state, acceptance_rate)
        return LowRankAdaptationState(
            new_ss,
            state.sigma,
            state.mu_star,
            state.U,
            state.lam,
            jnp.exp(new_ss.log_step_size),
            state.draws_buffer,
            state.grads_buffer,
            state.buffer_idx,
        )

    def slow_update(
        position: ArrayLikeTree,
        grad: ArrayLikeTree,
        acceptance_rate: float,
        state: LowRankAdaptationState,
    ) -> LowRankAdaptationState:
        """Slow window: adapt step size and accumulate draws/grads in buffer."""
        pos_flat, _ = fu.ravel_pytree(position)
        grad_flat, _ = fu.ravel_pytree(grad)
        B = state.draws_buffer.shape[0]
        idx = state.buffer_idx % B  # wrap to avoid out-of-bounds during trace
        new_draws = jax.lax.dynamic_update_slice(
            state.draws_buffer, pos_flat[None, :], (idx, 0)
        )
        new_grads = jax.lax.dynamic_update_slice(
            state.grads_buffer, grad_flat[None, :], (idx, 0)
        )
        new_ss = da_update(state.ss_state, acceptance_rate)
        return LowRankAdaptationState(
            new_ss,
            state.sigma,
            state.mu_star,
            state.U,
            state.lam,
            jnp.exp(new_ss.log_step_size),
            new_draws,
            new_grads,
            state.buffer_idx + 1,
        )

    def slow_final(state: LowRankAdaptationState) -> LowRankAdaptationState:
        """End of slow window: recompute metric and reset buffer."""
        sigma, mu_star, U, lam = _compute_low_rank_metric(
            state.draws_buffer,
            state.grads_buffer,
            state.buffer_idx,
            max_rank,
            gamma,
            cutoff,
        )
        # Re-initialise dual averaging at the current averaged step size
        new_ss = da_init(da_final(state.ss_state))
        B, d = state.draws_buffer.shape
        return LowRankAdaptationState(
            new_ss,
            sigma,
            mu_star,
            U,
            lam,
            jnp.exp(new_ss.log_step_size),
            jnp.zeros((B, d)),
            jnp.zeros((B, d)),
            0,
        )

    def update(
        state: LowRankAdaptationState,
        adaptation_stage: tuple,
        position: ArrayLikeTree,
        grad: ArrayLikeTree,
        acceptance_rate: float,
    ) -> LowRankAdaptationState:
        stage, is_window_end = adaptation_stage

        new_state = jax.lax.switch(
            stage,
            (fast_update, slow_update),
            position,
            grad,
            acceptance_rate,
            state,
        )
        new_state = jax.lax.cond(
            is_window_end,
            slow_final,
            lambda s: s,
            new_state,
        )
        return new_state

    def final(
        state: LowRankAdaptationState,
    ) -> tuple[float, Array, Array, Array, Array]:
        step_size = jnp.exp(state.ss_state.log_step_size_avg)
        return step_size, state.sigma, state.mu_star, state.U, state.lam

    return init, update, final


# ---------------------------------------------------------------------------
# High-level API
# ---------------------------------------------------------------------------


def low_rank_window_adaptation(
    algorithm,
    logdensity_fn: Callable,
    max_rank: int = 10,
    initial_step_size: float = 1.0,
    target_acceptance_rate: float = 0.80,
    gamma: float = 1.0,
    cutoff: float = 2.0,
    progress_bar: bool = False,
    adaptation_info_fn: Callable = return_all_adapt_info,
    integrator=mcmc.integrators.velocity_verlet,
    **extra_parameters,
) -> AdaptationAlgorithm:
    """Adapt step size and a low-rank mass matrix for HMC-family samplers.

    Uses the three-phase Stan-style warmup schedule while replacing Welford
    covariance estimation with the Fisher-divergence-minimising low-rank
    metric of :cite:p:`seyboldt2026preconditioning`.

    The returned ``AdaptationAlgorithm`` has a single ``run`` method::

        (state, params), info = warmup.run(rng_key, position, num_steps=1000)
        nuts = blackjax.nuts(logdensity_fn, **params)

    Parameters
    ----------
    algorithm
        An HMC-family algorithm object (e.g. ``blackjax.nuts``).
    logdensity_fn
        Log-density of the target distribution.
    max_rank
        Maximum number of eigenvectors in the low-rank correction.
    initial_step_size
        Starting step size (adapted automatically).
    target_acceptance_rate
        Target acceptance rate for dual averaging.
    gamma
        Regularisation scale; projected covariance is divided by ``n * gamma``
        before adding identity (nutpie convention).
    cutoff
        Eigenvectors with eigenvalue in ``[1/cutoff, cutoff]`` are masked.
        Default ``2.0`` matches nutpie's ``c=2``.
    progress_bar
        Show a progress bar during warmup.
    adaptation_info_fn
        Controls what adaptation info is retained; see
        ``blackjax.adaptation.base``.
    integrator
        Integrator to pass to ``algorithm.build_kernel``.
    **extra_parameters
        Additional keyword arguments forwarded to the kernel at every step
        (e.g. ``num_integration_steps`` for HMC).

    Returns
    -------
    An ``AdaptationAlgorithm`` whose ``run`` method returns
    ``(AdaptationResults, info)``.  ``AdaptationResults.parameters`` contains
    ``step_size``, ``inverse_mass_matrix`` (a :func:`gaussian_euclidean_low_rank`
    ``Metric`` object), and any ``extra_parameters``.
    ``AdaptationResults.state`` is re-initialised at the optimal translation
    μ* = x̄ + σ²⊙ᾱ, so it can be passed directly as the starting state for
    production sampling.  The last chain state from warmup is available as
    ``warmup_info[-1].state``, and μ* as
    ``warmup_info[-1].adaptation_state.mu_star``.
    """
    if len(inspect.signature(algorithm.build_kernel).parameters) > 0:
        mcmc_kernel = algorithm.build_kernel(integrator)
    else:
        mcmc_kernel = algorithm.build_kernel()

    adapt_init, adapt_step, adapt_final = base(
        max_rank=max_rank,
        target_acceptance_rate=target_acceptance_rate,
        gamma=gamma,
        cutoff=cutoff,
    )

    def one_step(carry, xs):
        _, rng_key, adaptation_stage = xs
        state, adaptation_state = carry

        metric = gaussian_euclidean_low_rank(
            adaptation_state.sigma,
            adaptation_state.U,
            adaptation_state.lam,
        )
        new_state, info = mcmc_kernel(
            rng_key,
            state,
            logdensity_fn,
            adaptation_state.step_size,
            metric,
            **extra_parameters,
        )
        new_adaptation_state = adapt_step(
            adaptation_state,
            adaptation_stage,
            new_state.position,
            new_state.logdensity_grad,
            info.acceptance_rate,
        )
        return (
            (new_state, new_adaptation_state),
            adaptation_info_fn(new_state, info, new_adaptation_state),
        )

    def run(rng_key: PRNGKey, position: ArrayLikeTree, num_steps: int = 1000):
        init_state = algorithm.init(position, logdensity_fn)
        # Size the buffer to the expected largest slow window rather than the
        # full warmup length.  The modular indexing in slow_update means that
        # if a window exceeds buffer_size only the most recent buffer_size
        # draws are kept — matching nutpie's fixed-buffer behaviour and
        # avoiding O(num_steps × d) allocations for large d.
        typical_window = max(num_steps // 5, 128)
        buffer_size = min(typical_window * 2, max(num_steps, 1))
        init_adaptation_state = adapt_init(
            position,
            init_state.logdensity_grad,
            initial_step_size,
            buffer_size,
        )

        if progress_bar:
            print("Running low-rank window adaptation")
        scan_fn = gen_scan_fn(num_steps, progress_bar=progress_bar)
        keys = jax.random.split(rng_key, num_steps)
        schedule = build_schedule(num_steps)
        last_state, info = scan_fn(
            one_step,
            (init_state, init_adaptation_state),
            (jnp.arange(num_steps), keys, schedule),
        )
        _, last_warmup_state, *_ = last_state
        step_size, sigma, mu_star, U, lam = adapt_final(last_warmup_state)
        metric = gaussian_euclidean_low_rank(sigma, U, lam)
        parameters = {
            "step_size": step_size,
            "inverse_mass_matrix": metric,
            **extra_parameters,
        }
        # Re-initialise chain state at the optimal translation μ* = x̄ + σ²⊙ᾱ.
        # mu_star is flat (d,); unravel to the original position pytree structure
        # before passing to algorithm.init.
        _, unravel = fu.ravel_pytree(position)
        mu_star_state = algorithm.init(unravel(mu_star), logdensity_fn)
        return AdaptationResults(mu_star_state, parameters), info

    return AdaptationAlgorithm(run)
