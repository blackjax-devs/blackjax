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
* **Regularisation**: projected covariance is ``P P^T / γ + I`` (nutpie's
  convention: the *unnormalised* sum-of-outer-products is divided by ``γ``
  directly, with no ``n`` scaling; see ``nuts-rs``
  ``src/transform/adapt/low_rank.rs::estimate_mass_matrix``). Default
  ``γ=1e-5`` matches nutpie's ``LowRankSettings::default``. The
  regularisation therefore only matters when the projected subspace is
  rank-deficient (few draws relative to ``2·max_rank``); it fades away as
  the number of draws grows, consistent with Theorem 2.4 of
  :cite:p:`seyboldt2026preconditioning` (exact recovery once draws exceed
  ``d+1``).
* **SPD mean of the draw covariance and the *inverse* score covariance**:
  Theorem 2.3 / Eq. 9 of :cite:p:`seyboldt2026preconditioning` give the
  (regularised) optimal inverse mass matrix as
  ``M_γ⁻¹ = (cov(x)+γI) # (cov(∇log p)+γI)⁻¹`` — the AIRM geometric mean of
  the draw covariance with the *inverse* of the score/gradient covariance.
  Cross-validated against nutpie's own Rust ``spd_mean`` (``nuts-rs``
  ``src/transform/adapt/low_rank.rs``), whose own unit test confirms
  ``spd_mean(cov_draws, cov_grads) == cov_draws # cov_grads⁻¹``.
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
from blackjax.mcmc.metrics import LowRankInverseMassMatrix, gaussian_euclidean_low_rank
from blackjax.progress_bar import gen_scan_fn
from blackjax.types import Array, ArrayLikeTree, PRNGKey
from blackjax.util import pytree_size

__all__ = [
    "LowRankAdaptationState",
    "base",
    "build_growing_window_schedule",
    "window_adaptation_low_rank",
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
        Regularisation scale.  The projected covariance is divided by ``gamma``
        (not scaled by ``n``) before adding the identity, following nutpie's
        convention.  Smaller values → weaker regularisation (the identity term
        matters less relative to the data term); the influence of the
        regularisation fades as the number of draws grows.
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
    # nutpie: C = P P^T / gamma + I  (raw gamma, NOT scaled by n -- nuts-rs
    # estimate_mass_matrix divides the unnormalised sum-of-outer-products by
    # gamma directly; there is no /n anywhere in that pipeline).
    C_x = (P_x @ P_x.T) / gamma + jnp.eye(q)
    C_a = (P_a @ P_a.T) / gamma + jnp.eye(q)

    # --- Step 7: SPD geometric mean Σ = C_x # C_a^{-1} ---
    # Theorem 2.3 / Eq. 9 (arXiv:2603.18845): the regularized optimal inverse
    # mass matrix is M_gamma^{-1} = (cov(x)+gamma*I) # (cov(alpha)+gamma*I)^{-1}
    # -- the score/gradient covariance must be INVERTED before the geometric
    # mean. Cross-validated against nutpie's own `spd_mean` (nuts-rs
    # src/transform/adapt/low_rank.rs), whose own unit test confirms
    # spd_mean(cov_draws, cov_grads) == cov_draws # cov_grads^{-1}.
    Sigma = _spd_mean(C_x, jnp.linalg.inv(C_a))

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
# Schedule builders
# ---------------------------------------------------------------------------


def build_growing_window_schedule(
    num_steps: int,
    early_window: float = 0.3,
    step_size_window: float = 0.15,
    early_window_size: int = 10,
    window_size: int = 80,
    window_growth: float = 1.5,
) -> Array:
    """Proportional-to-tune, geometrically-growing-window warmup schedule.

    An alternate to :func:`~blackjax.adaptation.window_adaptation.build_schedule`
    (Stan's fixed-absolute, 2x-doubling schedule) that instead sizes windows
    *proportionally to* ``num_steps`` and grows them by ``window_growth``
    (1.5x) rather than doubling, matching nutpie's window-sizing and
    growth-factor choices (see ``nuts-rs`` ``src/adapt_strategy.rs``,
    ``EuclideanAdaptOptions::default``):

    * ``early_window=0.3``, ``step_size_window=0.15`` -- fractions of
      ``num_steps``, vs Stan/blackjax's fixed absolute defaults
      (``initial_buffer_size=75``, ``final_buffer_size=50``) that are only
      rescaled when they don't fit the budget.
    * ``window_growth=1.5`` -- vs Stan's 2x doubling
      (``mass_matrix_window_growth`` in nutpie's receipts).

    **Scope note.** This function (together with the ``gradient_based_init``
    option on :func:`base` / :func:`window_adaptation_low_rank`) implements
    the window-sizing and gradient-based-init components of nutpie's warmup.
    Recompute cadence and buffer semantics follow the *host* Stan-style
    machinery unchanged: nutpie's actual schedule is an *online*, per-draw
    decision (``adapt_strategy.rs``'s ``is_late`` look-ahead + a
    partial-forget circular buffer + up-to-every-draw metric recomputation,
    ``mass_matrix_update_freq=1``), whereas blackjax's warmup runs the
    entire schedule as a static array through a single ``jax.lax.scan``
    (fixed ahead of time, like Stan's own :func:`build_schedule`), so this
    function precomputes an equivalent *offline* schedule with the same
    growth/sizing character. Recompute cadence stays window-boundary-only
    (unchanged from :func:`build_schedule`'s semantics: ``slow_final`` still
    only fires at a window end) and buffer memory stays a hard reset
    (unchanged: ``slow_final`` still zeros the buffer) -- nutpie's
    continuous per-draw recomputation and partial-forget buffer are a
    "faithful port" follow-up; see the note in
    :func:`window_adaptation_low_rank`'s docstring.

    Unlike Stan's schedule, there is no purely step-size-only *initial*
    buffer: nutpie starts adapting the mass matrix from the very first draw
    (paper §3.2, "More frequent updates"), so the entire region up to the
    final step-size-only window is labelled "slow" (mass-matrix-adapting),
    split into windows of size ``early_window_size`` during the early phase
    and growing windows (starting at ``window_size``, x``window_growth``
    each switch) during the main phase.

    Parameters
    ----------
    num_steps
        Total number of warmup steps.
    early_window
        Fraction of ``num_steps`` devoted to the early phase (fixed small
        windows of size ``early_window_size``). Default ``0.3`` matches
        nutpie's ``early_window``.
    step_size_window
        Fraction of ``num_steps`` devoted to the final step-size-only
        phase (no mass-matrix updates). Default ``0.15`` matches nutpie's
        ``step_size_window``.
    early_window_size
        Fixed window size during the early phase. Default ``10`` matches
        nutpie's ``early_mass_matrix_switch_freq``.
    window_size
        Starting window size for the main (post-early) phase, before
        growth. Default ``80`` matches nutpie's ``mass_matrix_switch_freq``.
    window_growth
        Multiplicative growth factor applied to the window size after each
        switch in the main phase. Default ``1.5`` matches nutpie's
        ``mass_matrix_window_growth``.

    Returns
    -------
    A ``(num_steps, 2)`` array of ``(stage, is_window_end)`` pairs, in the
    same format as :func:`~blackjax.adaptation.window_adaptation.build_schedule`
    (stage ``0`` = fast/step-size-only, stage ``1`` = slow/mass-matrix-adapting).
    """
    if num_steps < 20:
        return jnp.array([(0, False)] * num_steps)

    final_buffer_size = max(int(round(step_size_window * num_steps)), 1)
    final_buffer_start = num_steps - final_buffer_size
    early_end = min(max(int(round(early_window * num_steps)), 1), final_buffer_start)

    schedule = []

    # Early phase: fixed-size windows of `early_window_size`, slow stage.
    pos = 0
    while pos < early_end:
        size = min(early_window_size, early_end - pos)
        schedule += [(1, False)] * (size - 1)
        schedule.append((1, True))
        pos += size

    # Main phase: windows starting at `window_size`, growing by
    # `window_growth` after each switch, slow stage.
    current_size = window_size
    while pos < final_buffer_start:
        remaining = final_buffer_start - pos
        size = min(current_size, remaining)
        schedule += [(1, False)] * (size - 1)
        schedule.append((1, True))
        pos += size
        current_size = max(current_size + 1, int(round(current_size * window_growth)))

    # Final phase: step-size-only, fast stage.
    schedule += [(0, False)] * (num_steps - pos - 1)
    schedule.append((0, False))

    return jnp.array(schedule)


# ---------------------------------------------------------------------------
# Warmup primitives  (init / update / final)
# ---------------------------------------------------------------------------


def base(
    max_rank: int = 10,
    target_acceptance_rate: float = 0.80,
    gamma: float = 1e-5,
    cutoff: float = 2.0,
    gradient_based_init: bool = False,
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
        Regularisation scale.  The projected covariance is divided by ``gamma``
        (nutpie convention -- no ``n`` scaling).  Default ``1e-5`` matches
        nutpie's ``LowRankSettings::default``.
    cutoff
        Eigenvectors with eigenvalue in ``[1/cutoff, cutoff]`` are masked
        (eigenvalue set to 1).  Default ``2.0`` matches nutpie's ``c=2``.
    gradient_based_init
        If ``True``, seed the diagonal scale from the initial gradient
        instead of the identity: nutpie's own ``init`` calls
        ``update_from_grad`` on the very first observed point (``nuts-rs``
        ``src/transform/adapt/low_rank.rs::init``), which the paper's §3.1
        motivates as ``M = diag(|alpha^(0)|)`` -- a regularised diagonal of
        the gradient outer-product, a common Hessian approximation at the
        starting point (cf. L-BFGS). Since blackjax's ``sigma**2`` is the
        *inverse*-mass-matrix diagonal, this sets
        ``sigma = 1/sqrt(clip(|grad|, 1e-20, 1e20))`` so that
        ``M^{-1}_diag = sigma**2 = 1/|grad|``, matching ``M = diag(|grad|)``
        -- **except per-coordinate where** ``|grad_i| < 1e-10``, **where
        sigma_i falls back to 1.0** (the identity) instead of propagating
        the ``1e-20`` clip floor into an astronomically loose ``sigma_i =
        1e10``. This defends the real edge case of initialising at (or very
        near) a stationary point of the target -- e.g. ``x=0`` on any
        centered/standardised density -- where the gradient is exactly (or
        near-)zero and an extreme initial scale causes near-certain
        divergence on the very first trajectory (see the fisher-2x2
        calibration study's root-caused finding). Only the diagonal scale
        changes; ``U``/``lam`` still start at no-correction (``U=0``,
        ``lam=1``), same as the default. Default ``False`` reproduces the
        original identity/zero initialisation exactly (see also
        :func:`build_growing_window_schedule`, which implements the
        companion window-sizing piece of nutpie's warmup).

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
        if gradient_based_init:
            grad_flat, _ = fu.ravel_pytree(grad)
            abs_grad = jnp.abs(grad_flat)
            # Per-coordinate fallback to the identity (sigma_i=1.0) below a
            # near-zero-gradient threshold, rather than propagating the
            # 1e-20 clip floor into sigma_i=1e10. A real user-facing edge:
            # x=0 initialisation on any centered/standardised target gives
            # an EXACTLY zero gradient, and nuts-rs's own
            # array_update_var_inv_std_grad has no formula-level defense for
            # this either (clamp(0, 1e-20, 1e20).recip() = 1e20 is finite,
            # so its `fill_invalid` branch -- reserved for non-finite results
            # -- never fires); nutpie avoids this in practice purely by
            # jittering the initial position elsewhere in its pipeline, not
            # via this formula. sigma=1e10 at every dimension mistunes the
            # metric so severely relative to initial_step_size that the
            # first trajectory diverges immediately, freezing the chain for
            # the whole first window and collapsing the subsequent estimate
            # -- root-caused via the Fisher 2x2 calibration study's
            # instrumented re-run (design doc "Calibration verdict" section).
            # Threshold 1e-10 is a defensible, disclosed choice (no nuts-rs
            # precedent to cite at this exact boundary).
            near_zero_grad_threshold = 1e-10
            safe_sigma = jnp.power(jnp.clip(abs_grad, 1e-20, 1e20), -0.5)
            sigma = jnp.where(abs_grad < near_zero_grad_threshold, 1.0, safe_sigma)
        else:
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


def window_adaptation_low_rank(
    algorithm,
    logdensity_fn: Callable,
    max_rank: int = 10,
    initial_step_size: float = 1.0,
    target_acceptance_rate: float = 0.80,
    gamma: float = 1e-5,
    cutoff: float = 2.0,
    progress_bar: bool = False,
    adaptation_info_fn: Callable = return_all_adapt_info,
    integrator=mcmc.integrators.velocity_verlet,
    gradient_based_init: bool = False,
    schedule_fn: Callable[[int], Array] = build_schedule,
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
        Regularisation scale; projected covariance is divided by ``gamma``
        before adding identity (nutpie convention -- no ``n`` scaling).
        Default ``1e-5`` matches nutpie's ``LowRankSettings::default``.
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
    gradient_based_init
        Seed the diagonal scale from the initial gradient instead of the
        identity, matching nutpie's own initialisation (see :func:`base`).
        Default ``False`` reproduces the original behaviour exactly.
    schedule_fn
        Schedule-generator function ``num_steps -> (num_steps, 2)`` array of
        ``(stage, is_window_end)`` pairs. Default is Stan's fixed-absolute,
        2x-doubling :func:`~blackjax.adaptation.window_adaptation.build_schedule`
        (unchanged default behaviour). Pass
        :func:`build_growing_window_schedule` for nutpie's proportional-to-tune,
        1.5x-growing-window schedule -- see that function's docstring for
        exactly what it does and does not capture relative to nutpie's own
        (online, per-draw) schedule.
    **extra_parameters
        Additional keyword arguments forwarded to the kernel at every step
        (e.g. ``num_integration_steps`` for HMC).

    Returns
    -------
    An ``AdaptationAlgorithm`` whose ``run`` method returns
    ``(AdaptationResults, info)``.  ``AdaptationResults.parameters`` contains
    ``step_size``, ``inverse_mass_matrix`` (a
    :class:`~blackjax.mcmc.metrics.LowRankInverseMassMatrix` NamedTuple holding
    the pure-array payload ``(sigma, U, lam)``), and any ``extra_parameters``.
    The kernel layer normalises this into a full
    :class:`~blackjax.mcmc.metrics.Metric` via
    :func:`~blackjax.mcmc.metrics.default_metric` at call time. Returning the
    pure-array form (rather than the closure-bearing ``Metric``) lets the
    warmup compose with ``jax.vmap`` over chains; see GH #916.

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
        gradient_based_init=gradient_based_init,
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
        schedule = schedule_fn(num_steps)
        last_state, info = scan_fn(
            one_step,
            (init_state, init_adaptation_state),
            (jnp.arange(num_steps), keys, schedule),
        )
        _, last_warmup_state, *_ = last_state
        step_size, sigma, mu_star, U, lam = adapt_final(last_warmup_state)
        # Return the inverse mass matrix as a pure-array NamedTuple so that the
        # warmup composes with `jax.vmap` over chains. The kernel layer expands
        # this into a full `Metric` via `default_metric` at call time. See #916.
        inverse_mass_matrix = LowRankInverseMassMatrix(sigma=sigma, U=U, lam=lam)
        parameters = {
            "step_size": step_size,
            "inverse_mass_matrix": inverse_mass_matrix,
            **extra_parameters,
        }
        # Re-initialise chain state at the optimal translation μ* = x̄ + σ²⊙ᾱ.
        # mu_star is flat (d,); unravel to the original position pytree structure
        # before passing to algorithm.init.
        _, unravel = fu.ravel_pytree(position)
        mu_star_state = algorithm.init(unravel(mu_star), logdensity_fn)
        return AdaptationResults(mu_star_state, parameters), info

    return AdaptationAlgorithm(run)
