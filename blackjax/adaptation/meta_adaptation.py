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
"""Meta-adaptation controller for the HMC-family warmup.

At each window boundary the controller computes two signals: (1) held-out
score-linearity R² — the curvature gate (funnel R²≈0.007 vs ≥0.54 for all
metric-fixable classes); (2) S_gap(k) = λ₁/λ_{k+1} of the Welford-whitened
residual — the magnitude predictor (Spearman 1.0 with measured rank-k payoff).
Escalate diagonal → rank-k iff R² ≥ _R_MIN AND S_gap ≥ _S_MIN AND stable
over two consecutive windows AND budget deadline clear. Growing-window schedule
(nutpie-style) is the default; AIRM-velocity early exit is advisory in v1
(the scan runs its full length; ``converged_at_step`` records where stopping
would have helped — the actual early-stop host is the named v1.1 upgrade).

**Dtype note**: the composed estimator ``_compute_low_rank_metric`` produces
numerically indefinite metrics under float32 (~98% of runs). Enable x64 via
``jax.config.update("jax_enable_x64", True)`` for production use and for the
V-phase acceptance runs; all optpath harnesses ran with x64 enabled.

See :mod:`blackjax.adaptation.metric_recipes` for the MetricCore protocol and
:mod:`blackjax.adaptation.staged_adaptation` for the host engine.
"""
from __future__ import annotations

from typing import Any, NamedTuple

import jax
import jax.flatten_util as fu
import jax.numpy as jnp

from blackjax.adaptation.metric_estimators import _compute_low_rank_metric
from blackjax.adaptation.metric_recipes import MetricCore
from blackjax.mcmc.metrics import LowRankInverseMassMatrix
from blackjax.types import Array, ArrayLikeTree

__all__ = [
    "MetaAdaptationCoreState",
    "MetaAdaptationVerdict",
    "build_meta_adaptation_core",
    "extract_meta_verdict",
]

# ---------------------------------------------------------------------------
# Module constants — V-phase calibration anchors (not user knobs).
# ---------------------------------------------------------------------------

_R_MIN: float = 0.5
"""R² gate: curvature targets score ≈0.007–0.09; metric-fixable ≥0.54 (optpath-B)."""

_S_MIN: float = 2.0
"""S_gap magnitude gate: stoch_vol (marginal, S_gap≈1.5) must not escalate."""

_S_GAP_STABILITY_TOL: float = 0.3
"""Max relative S_gap change between consecutive windows before acting.
Warmup-phase S_gap is transient-inflated; two stable reads prevent acting early."""

_MIN_TRAIN_D_RATIO: int = 8
"""Full-affine fit: n_half ≥ 8d required; stoch_vol d=503: R²=-109 at n/d≈2."""

_MIN_TRAIN_K_RATIO: int = 8
"""Projected fit: n_half ≥ 8·(max_rank+1) required. d-independent threshold."""

_AIRM_VELOCITY_TOL: float = 0.05
"""Frobenius-norm lam-change threshold for AIRM early-exit (advisory, v1)."""

_STEP_SIZE_READAPT_BUFFER: int = 50
"""Steps reserved for step-size re-adaptation after escalation."""

_ASSUMED_AVG_LEAPFROGS_PER_STEP: int = 20
"""Conservative grad-per-step divisor: max_grad_budget → num_steps.
Most NUTS runs average 4–15 leapfrogs per step; 20 is conservative."""

_TRANSIENT_MIXING_THRESHOLD: float = 1.0
"""Split-half normalised mean-diff threshold for slow-mixing class detection."""

_MAX_RANK_CAP: int = 50
"""Static rank cap for buffer allocation; per-window rank ≤ min(cap, n//2)."""

# R² mode integers (stored in carry as int8).
_R2_DEFERRED: int = 0
_R2_PROJECTED: int = 1
_R2_FULL_AFFINE: int = 2


# ---------------------------------------------------------------------------
# State types
# ---------------------------------------------------------------------------


class MetaAdaptationCoreState(NamedTuple):
    """Scan-carry state for the meta-adaptation MetricCore.

    Buffer fields mirror ``LowRankMetricCoreState`` so the state is
    interchangeable in the staged_adaptation engine.  The ``inverse_mass_matrix``
    is always a :class:`~blackjax.mcmc.metrics.LowRankInverseMassMatrix`; before
    escalation, U=0 and lam=1 (bit-equivalent to the diagonal metric).
    """

    # Buffer (mirrors LowRankMetricCoreState layout)
    inverse_mass_matrix: LowRankInverseMassMatrix
    mu_star: Array  # optimal translation, (d,); zero before escalation
    draws_buffer: Array  # (buffer_size, d)
    grads_buffer: Array  # (buffer_size, d)
    buffer_idx: Array  # int32 scalar; reset to 0 after each window (v1 reset policy)
    background_split: Array  # always 0 in v1
    recompute_counter: Array  # always 0 in v1
    # Controller carry
    has_escalated: Array  # bool scalar; monotone True-once
    escalation_rank: Array  # int32; the rank k chosen at escalation (0 before)
    s_gap_prev: Array  # float32; S_gap from window before last (NaN initially)
    s_gap_curr: Array  # float32; S_gap from most recent window (NaN initially)
    r2_latest: Array  # float32; most recent R² (NaN if deferred or not yet computed)
    r2_mode: Array  # int32; 0=deferred, 1=projected, 2=full_affine (last window)
    budget_used: Array  # int32; warmup steps elapsed (step-clock proxy)
    converged_at_step: Array  # int32; step when AIRM velocity first fired (<0 = not yet)
    prev_lam: Array  # (max_rank,); lam from previous window for AIRM velocity
    airm_vel_prev: Array  # float32; AIRM velocity proxy from window before last
    airm_vel_curr: Array  # float32; AIRM velocity proxy from most recent window
    is_slow_mixing: Array  # bool; True = slow-mixing (RESET preferred by v1.1 switch)


class MetaAdaptationVerdict(NamedTuple):
    """Verdict emitted by :func:`extract_meta_verdict` after the warmup scan.

    All budget numbers are in warmup steps (step-clock proxy) unless the
    info stream is provided for true gradient counts.

    ``budget_returned_steps`` is ADVISORY in v1: the scan runs its full length;
    a stopping host (lax.while) is the named v1.1 upgrade.
    """

    route: str  # "diagonal" | "low_rank" | "reparam_suggested"
    metric: LowRankInverseMassMatrix
    effective_rank: int  # k at escalation time (0 for diagonal route)
    confidence: str  # "high" | "low"
    exit_reason: str  # "warmup_complete" | "airm_velocity_converged" | "warmup_budget_exhausted"
    budget_used_steps: int
    budget_returned_steps: int  # advisory (see docstring)
    budget_used_grads: int  # -1 if info stream not provided
    r2_final: float
    s_gap_final: float
    transient_mixing_class: str  # "slow" | "fast"
    buffer_policy: str  # always "reset" in v1
    flags: dict  # reparam_hint, marginal_s_gap, wall_cost_discount, high_d_r2_mode, mode_coverage


# ---------------------------------------------------------------------------
# Signal computation — pure JAX functions, module-private
# ---------------------------------------------------------------------------


def _compute_whitened_spectrum(
    draws_buffer: Array,
    sigma: Array,
    n: Array,
    max_rank: int,
) -> tuple[Array, Array]:
    """Top ``max_rank`` eigenvalues and eigenvectors of the diagonal-whitened
    sample covariance R = D^{-1/2}ΣD^{-1/2} (D = diag(σ²)), computed via the
    thin SVD of the centred whitened draw matrix.

    Returns ``(eigenvalues, U_k)`` with shapes ``(max_rank,)`` and
    ``(d, max_rank)``; zero-padded when fewer than max_rank components are
    available.  Invalid rows (beyond index n) are zeroed via a mask.
    """
    B, d = draws_buffer.shape
    n_safe = jnp.maximum(n.astype(draws_buffer.dtype), 1.0)
    mask = (jnp.arange(B) < n).astype(draws_buffer.dtype)
    sigma_safe = jnp.maximum(sigma, 1e-20)
    mean_x = (mask[:, None] * draws_buffer).sum(0) / n_safe
    w = mask[:, None] * (draws_buffer - mean_x[None, :]) / sigma_safe[None, :]
    _, s, Vt = jnp.linalg.svd(w, full_matrices=False)
    eigs_all = (s**2) / n_safe
    actual = min(max_rank, min(B, d))  # static
    if actual < max_rank:
        pad = max_rank - actual
        eigenvalues = jnp.concatenate(
            [eigs_all[:actual], jnp.zeros(pad, dtype=eigs_all.dtype)]
        )
        U_k = jnp.concatenate(
            [Vt[:actual].T, jnp.zeros((d, pad), dtype=Vt.dtype)], axis=1
        )
    else:
        eigenvalues = eigs_all[:max_rank]
        U_k = Vt[:max_rank].T
    return eigenvalues, U_k


def _choose_rank(
    eigenvalues: Array,
    n: Array,
    max_rank: int,
    cutoff: float = 2.0,
) -> Array:
    """Count informative eigenvalues (outside [1/cutoff, cutoff]), capped by
    n//2 (estimation-support limit) and max_rank.
    """
    is_informative = (eigenvalues > cutoff) | (eigenvalues < 1.0 / cutoff)
    count = is_informative.sum().astype(jnp.int32)
    support_cap = (n // 2).astype(jnp.int32)
    return jnp.minimum(count, jnp.minimum(support_cap, jnp.int32(max_rank)))


def _compute_s_gap(eigenvalues: Array, k: Array) -> Array:
    """S_gap at rank cut k: λ₁/λ_{k+1}.  Returns 1.0 when k=0."""
    max_rank = eigenvalues.shape[0]
    k_clipped = jnp.clip(k.astype(jnp.int32), 0, max_rank - 1)
    lambda_1 = jnp.maximum(eigenvalues[0], 1e-10)
    lambda_k1 = jax.lax.dynamic_index_in_dim(eigenvalues, k_clipped, keepdims=False)
    lambda_k1_safe = jnp.maximum(lambda_k1, 1e-10)
    return jnp.where(
        k.astype(jnp.int32) == 0, jnp.ones_like(lambda_1), lambda_1 / lambda_k1_safe
    )


def _compute_r2_score_linearity(
    draws_buffer: Array,
    grads_buffer: Array,
    sigma: Array,
    n: Array,
    U_k: Array,
    max_rank: int,
) -> tuple[Array, Array]:
    """Held-out score-linearity R² with three-tier fallback.

    Fit s_w ≈ A·w + b train/test (first/last n//2 valid rows).  Three modes:
    full-affine (n ≥ 2·8·d), projected on max_rank+1 candidate directions
    (n ≥ 2·8·(max_rank+1)), deferred/NaN otherwise.  Returns (r2, mode_int)
    where mode_int is a traced JAX int32: _R2_DEFERRED=0, _R2_PROJECTED=1,
    _R2_FULL_AFFINE=2.  Mode is observed from the actual branch taken — not
    inferred post-hoc.
    """
    B, d = draws_buffer.shape
    n_f = n.astype(jnp.float32)
    n_safe = jnp.maximum(n_f, 2.0)
    mask = (jnp.arange(B) < n).astype(draws_buffer.dtype)
    sigma_safe = jnp.maximum(sigma, 1e-20)
    mean_x = (mask[:, None] * draws_buffer).sum(0) / n_safe
    mean_g = (mask[:, None] * grads_buffer).sum(0) / n_safe
    w = mask[:, None] * (draws_buffer - mean_x[None, :]) / sigma_safe[None, :]
    s_w = mask[:, None] * (grads_buffer - mean_g[None, :]) * sigma_safe[None, :]
    n_train_int = n // 2
    train_mask = mask * (jnp.arange(B) < n_train_int).astype(mask.dtype)
    test_mask = mask * (jnp.arange(B) >= n_train_int).astype(mask.dtype)

    def _r2_from_features(feats: Array) -> Array:
        n_feats = feats.shape[1]
        FtF = (train_mask[:, None] * feats).T @ (train_mask[:, None] * feats)
        FtS = (train_mask[:, None] * feats).T @ (train_mask[:, None] * s_w)
        A = jnp.linalg.lstsq(
            FtF + 1e-8 * jnp.eye(n_feats, dtype=FtF.dtype), FtS, rcond=None
        )[0]
        s_pred = (test_mask[:, None] * feats) @ A
        s_test = test_mask[:, None] * s_w
        n_test_safe = jnp.maximum(test_mask.sum().astype(jnp.float32), 2.0)
        s_mean = s_test.sum(0) / n_test_safe
        tss = ((s_test - test_mask[:, None] * s_mean[None, :]) ** 2).sum(0)
        rss = ((s_test - s_pred) ** 2).sum(0)
        return jnp.median(1.0 - rss / jnp.maximum(tss, 1e-10))

    def _full_affine() -> tuple[Array, Array]:
        feats = jnp.concatenate([w, jnp.ones((B, 1), dtype=w.dtype)], axis=1)
        return _r2_from_features(feats), jnp.int32(_R2_FULL_AFFINE)

    def _projected() -> tuple[Array, Array]:
        feats = jnp.concatenate([w @ U_k, jnp.ones((B, 1), dtype=w.dtype)], axis=1)
        return _r2_from_features(feats), jnp.int32(_R2_PROJECTED)

    def _deferred() -> tuple[Array, Array]:
        # dtype must match sibling branches (s_w.dtype); hardcoding float32 crashes
        # under x64 where sibling branches return float64.
        return jnp.asarray(float("nan"), dtype=s_w.dtype), jnp.int32(_R2_DEFERRED)

    min_n_full = float(2 * _MIN_TRAIN_D_RATIO * d)  # each half needs ≥ 8d samples
    min_n_proj = float(2 * _MIN_TRAIN_K_RATIO * (max_rank + 1))  # each half ≥ 8(k+1)

    return jax.lax.cond(
        n_f >= min_n_full,
        _full_affine,
        lambda: jax.lax.cond(n_f >= min_n_proj, _projected, _deferred),
    )


def _compute_transient_mixing_signal(
    draws_buffer: Array,
    sigma: Array,
    n: Array,
) -> Array:
    """Split-half mean-difference proxy for the warmup transient-mixing class.

    Returns traced bool: True = slow-mixing (RESET preferred by v1.1 switch).
    V1 always uses RESET; this signal is reported for the v1.1 anchor.
    """
    B, _ = draws_buffer.shape
    n_f = n.astype(draws_buffer.dtype)
    n_safe = jnp.maximum(n_f, 2.0)
    mask = (jnp.arange(B) < n).astype(draws_buffer.dtype)
    sigma_safe = jnp.maximum(sigma, 1e-20)
    mean_x = (mask[:, None] * draws_buffer).sum(0) / n_safe
    w = mask[:, None] * (draws_buffer - mean_x[None, :]) / sigma_safe[None, :]
    n_train_int = n // 2
    m1 = mask * (jnp.arange(B) < n_train_int).astype(mask.dtype)
    m2 = mask * (jnp.arange(B) >= n_train_int).astype(mask.dtype)
    n1 = jnp.maximum(m1.sum().astype(jnp.float32), 1.0)
    n2 = jnp.maximum(m2.sum().astype(jnp.float32), 1.0)
    mu1 = (m1[:, None] * w).sum(0) / n1
    mu2 = (m2[:, None] * w).sum(0) / n2
    std = jnp.maximum(((mask[:, None] * w**2).sum(0) / n_safe) ** 0.5, 1e-10)
    return jnp.max(jnp.abs(mu1 - mu2) / std) > _TRANSIENT_MIXING_THRESHOLD


# ---------------------------------------------------------------------------
# Core builder
# ---------------------------------------------------------------------------


def build_meta_adaptation_core(
    max_grad_budget: int,
    *,
    max_rank: int | None = None,
    gamma: float = 1e-5,
    cutoff: float = 2.0,
) -> MetricCore:
    """Build the meta-adaptation :class:`~blackjax.adaptation.metric_recipes.MetricCore`.

    Parameters
    ----------
    max_grad_budget
        Maximum total gradient budget (leapfrog evaluations).  Converted to
        warmup steps via ``_ASSUMED_AVG_LEAPFROGS_PER_STEP`` at Python time.
    max_rank
        Maximum low-rank rank; ``None`` uses :data:`_MAX_RANK_CAP`.
    gamma, cutoff
        Fisher-estimator parameters; defaults match ``fisher_low_rank`` recipe.

    Returns
    -------
    MetricCore
        Embeddable init/update/final bundle.
    """
    _max_rank: int = _MAX_RANK_CAP if max_rank is None else max_rank
    max_budget_steps: int = max(max_grad_budget // _ASSUMED_AVG_LEAPFROGS_PER_STEP, 1)

    def init(n_dims: int) -> MetaAdaptationCoreState:
        buf = min(max(max_budget_steps // 5, 128), max_budget_steps)
        buf = max(buf, 2 * (_max_rank + 1) * _MIN_TRAIN_K_RATIO)
        buf = min(buf, max_budget_steps)
        actual_rank = min(_max_rank, max(n_dims // 2, 1), _MAX_RANK_CAP)
        return MetaAdaptationCoreState(
            inverse_mass_matrix=LowRankInverseMassMatrix(
                sigma=jnp.ones(n_dims),
                U=jnp.zeros((n_dims, actual_rank)),
                lam=jnp.ones(actual_rank),
            ),
            mu_star=jnp.zeros(n_dims),
            draws_buffer=jnp.zeros((buf, n_dims)),
            grads_buffer=jnp.zeros((buf, n_dims)),
            buffer_idx=jnp.zeros((), dtype=jnp.int32),
            background_split=jnp.zeros((), dtype=jnp.int32),
            recompute_counter=jnp.zeros((), dtype=jnp.int32),
            has_escalated=jnp.zeros((), dtype=jnp.bool_),
            escalation_rank=jnp.zeros((), dtype=jnp.int32),
            s_gap_prev=jnp.array(float("nan"), dtype=jnp.float32),
            s_gap_curr=jnp.array(float("nan"), dtype=jnp.float32),
            r2_latest=jnp.array(float("nan"), dtype=jnp.float32),
            r2_mode=jnp.array(_R2_DEFERRED, dtype=jnp.int32),
            budget_used=jnp.zeros((), dtype=jnp.int32),
            converged_at_step=jnp.array(-1, dtype=jnp.int32),  # -1 = not yet converged
            prev_lam=jnp.ones(actual_rank, dtype=jnp.float32),
            airm_vel_prev=jnp.array(float("inf"), dtype=jnp.float32),
            airm_vel_curr=jnp.array(float("inf"), dtype=jnp.float32),
            is_slow_mixing=jnp.zeros((), dtype=jnp.bool_),
        )

    def update(
        state: MetaAdaptationCoreState,
        position: ArrayLikeTree,
        grad: ArrayLikeTree | None = None,
    ) -> MetaAdaptationCoreState:
        pos_flat, _ = fu.ravel_pytree(position)
        grad_flat, _ = fu.ravel_pytree(grad)
        B = state.draws_buffer.shape[0]
        idx = state.buffer_idx % B
        # Cast the column offset to idx.dtype — under x64 the literal 0 is int64
        # but idx is int32, causing "index arguments must be integers of the same
        # type" at trace time.
        col0 = jnp.zeros((), dtype=idx.dtype)
        return state._replace(
            draws_buffer=jax.lax.dynamic_update_slice(
                state.draws_buffer, pos_flat[None, :], (idx, col0)
            ),
            grads_buffer=jax.lax.dynamic_update_slice(
                state.grads_buffer, grad_flat[None, :], (idx, col0)
            ),
            buffer_idx=state.buffer_idx + 1,
            budget_used=state.budget_used + 1,
        )

    def final(state: MetaAdaptationCoreState) -> MetaAdaptationCoreState:
        """Window-boundary controller: compute signals → escalation decision →
        choose IMM → hard-reset buffer (v1 reset policy always).
        """
        B, d = state.draws_buffer.shape
        n = state.buffer_idx
        actual_rank = state.inverse_mass_matrix.U.shape[1]  # static

        # Fisher-LR metric from buffer (sigma_lr used for all downstream whitening).
        sigma_lr, mu_star_new, U_lr, lam_lr = _compute_low_rank_metric(
            state.draws_buffer, state.grads_buffer, n, actual_rank, gamma, cutoff
        )
        lr_imm = LowRankInverseMassMatrix(sigma=sigma_lr, U=U_lr, lam=lam_lr)
        # Diagonal: same sigma_lr, U=0, lam=1 (LowRankIMM(U=0,lam=1) ≡ diagonal metric).
        diag_imm = LowRankInverseMassMatrix(
            sigma=sigma_lr,
            U=jnp.zeros((d, actual_rank), dtype=sigma_lr.dtype),
            lam=jnp.ones(actual_rank, dtype=sigma_lr.dtype),
        )

        # Whitened-residual spectrum.
        eigenvalues, U_k = _compute_whitened_spectrum(
            state.draws_buffer, sigma_lr, n, actual_rank
        )
        k_new = _choose_rank(eigenvalues, n, actual_rank, cutoff)
        s_gap_new = _compute_s_gap(eigenvalues, k_new)

        # Score-linearity R² (returns (r2, mode_int) — mode is observed, not inferred).
        r2_new, mode_new = _compute_r2_score_linearity(
            state.draws_buffer, state.grads_buffer, sigma_lr, n, U_k, actual_rank
        )

        # Transient-mixing class (reported in verdict; v1 always uses RESET).
        is_slow = _compute_transient_mixing_signal(state.draws_buffer, sigma_lr, n)

        # ---- Escalation decision (all three gates must pass) ----
        r2_gate = (
            r2_new >= _R_MIN
        )  # False when r2_new is NaN (JAX comparison semantics)
        s_gap_prev_valid = ~jnp.isnan(state.s_gap_curr)
        relative_change = jnp.abs(s_gap_new - state.s_gap_curr) / jnp.maximum(
            s_gap_new, 1e-10
        )
        s_gap_gate = (
            (s_gap_new >= _S_MIN)
            & s_gap_prev_valid
            & (relative_change < _S_GAP_STABILITY_TOL)
        )
        budget_remaining = jnp.int32(max_budget_steps) - state.budget_used.astype(
            jnp.int32
        )
        deadline_ok = budget_remaining >= 2 * k_new.astype(jnp.int32) + jnp.int32(
            _STEP_SIZE_READAPT_BUFFER
        )

        escalate_now = ~state.has_escalated & r2_gate & s_gap_gate & deadline_ok
        new_has_escalated = state.has_escalated | escalate_now
        new_escalation_rank = jnp.where(escalate_now, k_new, state.escalation_rank)

        chosen_imm = jax.lax.cond(new_has_escalated, lambda: lr_imm, lambda: diag_imm)
        chosen_mu = jax.lax.cond(
            new_has_escalated, lambda: mu_star_new, lambda: jnp.zeros_like(mu_star_new)
        )

        # ---- AIRM velocity proxy + advisory convergence step ----
        lam_diff = jnp.linalg.norm(lam_lr - state.prev_lam)
        new_airm_vel_prev = state.airm_vel_curr
        new_airm_vel_curr = jnp.where(new_has_escalated, lam_diff, state.airm_vel_curr)
        # Set converged_at_step once (monotone, like has_escalated) when AIRM criterion fires.
        airm_converged_now = (
            new_has_escalated
            & (new_airm_vel_curr < _AIRM_VELOCITY_TOL)
            & (new_airm_vel_prev < _AIRM_VELOCITY_TOL)
        )
        new_converged_at = jnp.where(
            (state.converged_at_step < 0) & airm_converged_now,
            state.budget_used,
            state.converged_at_step,
        )

        return MetaAdaptationCoreState(
            inverse_mass_matrix=chosen_imm,
            mu_star=chosen_mu,
            draws_buffer=jnp.zeros_like(state.draws_buffer),
            grads_buffer=jnp.zeros_like(state.grads_buffer),
            buffer_idx=jnp.zeros_like(state.buffer_idx),
            background_split=jnp.zeros_like(state.background_split),
            recompute_counter=jnp.zeros_like(state.recompute_counter),
            has_escalated=new_has_escalated,
            escalation_rank=new_escalation_rank,
            s_gap_prev=state.s_gap_curr,
            s_gap_curr=s_gap_new,
            r2_latest=r2_new,
            r2_mode=mode_new,
            budget_used=state.budget_used,
            converged_at_step=new_converged_at,
            prev_lam=lam_lr,
            airm_vel_prev=new_airm_vel_prev,
            airm_vel_curr=new_airm_vel_curr,
            is_slow_mixing=is_slow,
        )

    return MetricCore(init=init, update=update, final=final)


# ---------------------------------------------------------------------------
# Verdict extraction — Python-side, runs after the warmup scan
# ---------------------------------------------------------------------------


def extract_meta_verdict(
    final_state: MetaAdaptationCoreState,
    max_grad_budget: int,
    num_warmup_steps: int,
    adaptation_info: Any = None,
) -> MetaAdaptationVerdict:
    """Build a :class:`MetaAdaptationVerdict` from the final core state.

    Call after ``warmup.run()`` completes.  Pass ``adaptation_info`` (the
    second return of ``warmup.run()``) for true gradient counts.

    ``budget_returned_steps`` is ADVISORY: the scan runs its full length in v1;
    this field shows where a stopping host would have saved steps.
    """
    import numpy as np

    has_esc = bool(np.asarray(final_state.has_escalated))
    k = int(np.asarray(final_state.escalation_rank))
    budget_used = int(np.asarray(final_state.budget_used))
    s_gap = float(np.asarray(final_state.s_gap_curr))
    r2 = float(np.asarray(final_state.r2_latest))
    mode_int = int(np.asarray(final_state.r2_mode))
    airm_v_prev = float(np.asarray(final_state.airm_vel_prev))
    airm_v_curr = float(np.asarray(final_state.airm_vel_curr))
    converged_at = int(np.asarray(final_state.converged_at_step))
    is_slow = bool(np.asarray(final_state.is_slow_mixing))

    r2_nan = np.isnan(r2)

    # Route
    r2_blocked = (not r2_nan) and (r2 < _R_MIN)
    if not has_esc and r2_blocked:
        route = "reparam_suggested"
    elif has_esc:
        route = "low_rank"
    else:
        route = "diagonal"

    # Confidence
    s_gap_valid = not np.isnan(s_gap)
    confidence = (
        "high"
        if (has_esc and not r2_nan and r2 >= _R_MIN and s_gap_valid and s_gap >= _S_MIN)
        else "low"
    )

    # Exit reason and advisory budget return
    airm_converged = (airm_v_prev < _AIRM_VELOCITY_TOL) and (
        airm_v_curr < _AIRM_VELOCITY_TOL
    )
    if airm_converged and has_esc:
        exit_reason = "airm_velocity_converged"
    elif budget_used >= num_warmup_steps:
        exit_reason = "warmup_budget_exhausted"
    else:
        exit_reason = "warmup_complete"

    # Advisory budget return: steps from first AIRM convergence to end of warmup.
    # converged_at_step = -1 means never converged (scan ran full length).
    budget_returned = (
        max(num_warmup_steps - converged_at, 0) if converged_at >= 0 else 0
    )

    # True gradient count from info stream
    budget_used_grads = -1
    if adaptation_info is not None:
        try:
            budget_used_grads = int(
                np.asarray(adaptation_info.num_integration_steps).sum()
            )
        except AttributeError:
            pass

    # Flags — mode_int is observed from the carry, not inferred.
    high_d_r2_mode = {
        _R2_DEFERRED: "deferred",
        _R2_PROJECTED: "projected",
        _R2_FULL_AFFINE: "full_affine",
    }.get(mode_int, "deferred")
    marginal_s_gap = (not has_esc) and s_gap_valid and (_S_MIN <= s_gap < 2.0 * _S_MIN)
    flags = {
        "reparam_hint": route == "reparam_suggested",
        "marginal_s_gap": marginal_s_gap,
        "wall_cost_discount": k > 0,
        "high_d_r2_mode": high_d_r2_mode,
        "mode_coverage": "single_chain_uncertified",
    }

    return MetaAdaptationVerdict(
        route=route,
        metric=final_state.inverse_mass_matrix,
        effective_rank=k,
        confidence=confidence,
        exit_reason=exit_reason,
        budget_used_steps=budget_used,
        budget_returned_steps=budget_returned,
        budget_used_grads=budget_used_grads,
        r2_final=r2,
        s_gap_final=s_gap,
        transient_mixing_class="slow" if is_slow else "fast",
        buffer_policy="reset",
        flags=flags,
    )
