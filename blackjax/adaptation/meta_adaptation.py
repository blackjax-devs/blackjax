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

The meta-adaptation scheme takes any sampler in the (inverse_mass_matrix,
step_size) family — HMC, NUTS, GHMC, MCLMC — and produces an optimal warmup
path (window-length × rank) without user-supplied rank or schedule knobs.
Only one parameter is exposed: ``max_grad_budget``.

**The scheme (one paragraph).** Start diagonal (Fisher estimator) always; at
each window boundary compute two signals from the accumulated draws and
gradients: (1) held-out score-linearity R² — a binary gate that says whether
the residual anisotropy after diagonal preconditioning is *linear correlation*
(a rank-k metric can fix it) or *curvature* (reparameterize; no fixed metric
helps); and (2) the diagonal-whitened residual spectrum gap S_gap(k) = λ₁/λ_{k+1}
— the magnitude predictor that orders the rank-k payoff among linear-residual
targets (Spearman 1.0 with measured data). Escalate from diagonal to rank-k iff
BOTH the R² gate passes AND the S_gap is large AND stable over two consecutive
windows AND the remaining budget allows a rank-k fit plus step-size re-adaptation.
The rank k is the number of informative directions (eigenvalues outside the
uninformative band), capped by the estimation-support limit n//2. The growing-window
schedule (nutpie-style) is the default; AIRM-velocity early exit returns unused
budget when the metric has converged. On curvature targets (funnel, banana) the
controller never escalates and emits ``reparam_suggested``.

**Recovers-classical.** On isotropic-after-diagonal targets (S_gap ≈ 1) the
controller never escalates and the emitted schedule is the diagonal growing-window
path. On concentrated-low-rank targets it escalates to small k on growing windows.
Stan-doubling diagonal is available via ``schedule="stan"`` and is exactly what
the auto-controller emits in the isotropic limiting case.

**Out of v1.** Dense rank beyond k ≈ d/2; multi-chain certification; MEADS/ChEES
hosting; ess/wallsec objective; accumulating buffer policy (the switch is a v1.1
upgrade governed by the transient-mixing class the verdict already reports);
online grad-clock deadline (v1 uses a conservative step-clock proxy).

**Budget semantics (v1).** ``budget_used`` in the carry is tracked in *warmup
steps*. The public ``max_grad_budget`` (true gradient count) is converted to
``num_steps`` at Python time via ``_ASSUMED_AVG_LEAPFROGS_PER_STEP``. The
verdict's ``budget_used_grads`` is computed post-scan by summing
``info.num_integration_steps`` in :func:`extract_meta_verdict`. The step-clock
deadline is conservative (under-counts grads per step), so the controller will
escalate with more remaining budget than the headline number suggests.

See Also
--------
blackjax.adaptation.metric_recipes : MetricCore protocol and REGISTRY.
blackjax.adaptation.staged_adaptation : Host engine that plugs in this core.
blackjax.adaptation.metric_estimators : Signal computation substrates.
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
# Module constants — V-phase calibration anchors.
# These are NOT user knobs. Adjust only after new benchmark evidence;
# do not expose through the public API.
# ---------------------------------------------------------------------------

_R_MIN: float = 0.5
"""Score-linearity R² gate threshold.

Measured geometry chasm: curvature targets (funnel, banana) score R² ≈ 0.007–0.09;
linear-residual targets (radon, ill_cond, german, stoch_vol) score R² ≥ 0.54.
The threshold 0.5 sits in the empty region. V-phase calibration after extended
benchmark runs across more geometry classes may tighten this.
"""

_S_MIN: float = 2.0
"""S_gap magnitude gate threshold for escalation.

S_gap(k) = λ₁/λ_{k+1} of the diagonal-whitened residual. Targets with
S_gap ≈ 1.4–1.6 (stoch_vol-like, diffuse AR(1)) are marginal and net-negative
at rank-10; the controller stays diagonal for them. s_min = 2.0 sits above the
measured marginal band. The named experiment F-stochvol-rank will probe whether
a deeper rank rescues such targets; the verdict already flags marginal cases.
"""

_S_GAP_STABILITY_TOL: float = 0.3
"""Stability band for consecutive S_gap reads before acting.

A window-boundary read is "stable" when:
    |s_gap_curr - s_gap_prev| / max(s_gap_curr, ε) < _S_GAP_STABILITY_TOL.
Warmup-phase S_gap is transient-inflated (decreasing monotonically toward the
stationary value). Requiring two reads within the band prevents acting on the
inflated early window read.
"""

_MIN_TRAIN_D_RATIO: int = 8
"""Minimum n-to-dimension ratio for a reliable full-affine R² fit.

Full-affine score fit (s_w ≈ Aw + b, d predictors) is under-powered when
n < 8·d (stoch_vol d=503: R² = −109 at n=1000 with n/d≈2, R²=0.54 at n=4000
with n/d≈8). Below this threshold the controller uses the projected fit.
"""

_MIN_TRAIN_K_RATIO: int = 8
"""Minimum n per predictor for the projected R² fit.

The projected fit (max_rank+1 predictors: max_rank candidate eigenvectors plus
intercept) needs n ≥ _MIN_TRAIN_K_RATIO × (max_rank+1). Below this, R² is
deferred (returned as NaN).
"""

_AIRM_VELOCITY_TOL: float = 0.05
"""Frobenius-subspace velocity threshold for AIRM early exit.

Two consecutive window boundaries with metric-improvement velocity (Frobenius
norm of the lam change) below this threshold trigger early exit and return
unused budget. V-phase calibration will determine whether a tighter threshold
improves budget efficiency.
"""

_STEP_SIZE_READAPT_BUFFER: int = 50
"""Minimum warmup steps reserved for step-size re-adaptation after escalation.

After switching to a rank-k metric, dual-averaging needs approximately this
many steps to re-converge.
"""

_ASSUMED_AVG_LEAPFROGS_PER_STEP: int = 20
"""Conservative average NUTS leapfrog count per warmup step.

Used to convert ``max_grad_budget`` (gradient count) to ``num_steps`` (warmup
step count) at Python time:
    num_steps = max_grad_budget // _ASSUMED_AVG_LEAPFROGS_PER_STEP.

This is conservative (most NUTS runs average 4–15 leapfrogs per step).
The true grad-clock deadline (counting actual ``num_integration_steps``) is
the named v1.1 upgrade.
"""

_TRANSIENT_MIXING_THRESHOLD: float = 1.0
"""Split-half mean-difference threshold for transient-mixing class detection.

Values above this threshold indicate slow mixing (first and second halves of
the window buffer have different means after whitening), suggesting the RESET
buffer policy. V1 always uses RESET regardless; the detected class is reported
in the verdict to anchor the v1.1 policy switch.
"""

_MAX_RANK_CAP: int = 50
"""Static maximum rank cap for buffer shape allocation.

The buffer must be allocated with a static shape at Python time. This cap
accommodates full-rank-anisotropy targets up to d=100. The per-window rank
chosen by _choose_rank is always ≤ min(max_rank, n//2).
"""


# ---------------------------------------------------------------------------
# State types
# ---------------------------------------------------------------------------


class MetaAdaptationCoreState(NamedTuple):
    """Scan-carry state for the meta-adaptation MetricCore.

    Extends the LowRankMetricCoreState buffer layout with controller carry
    fields. The ``inverse_mass_matrix`` is always a
    :class:`~blackjax.mcmc.metrics.LowRankInverseMassMatrix`; when the
    controller has not yet escalated, U=0 and lam=1, making it bit-equivalent
    to the diagonal metric.

    Parameters
    ----------
    inverse_mass_matrix
        Current IMM emitted to the MCMC kernel.
    mu_star
        Optimal translation, shape ``(d,)``. Zero before escalation.
    draws_buffer
        Circular draw buffer, shape ``(buffer_size, d)``.
    grads_buffer
        Circular gradient buffer, shape ``(buffer_size, d)``.
    buffer_idx
        Number of draws written (reset to 0 after each window in v1).
    background_split
        Always 0 in v1 (reset policy placeholder).
    recompute_counter
        Always 0 in v1 (reset policy placeholder).
    has_escalated
        Bool scalar. Monotone: once True, stays True.
    escalation_rank
        Int scalar. The rank k chosen at escalation time (0 before).
    s_gap_prev
        S_gap from the window before last (NaN before the second window).
    s_gap_curr
        S_gap from the most recent window (NaN before the first window).
    r2_latest
        Most recent R² (NaN if deferred or not yet computed).
    budget_used
        Warmup steps used (step-clock proxy for gradient budget).
    prev_lam
        Lam vector from the previous window, shape ``(max_rank,)``.
        Used for AIRM-velocity early-exit computation.
    airm_vel_prev
        AIRM velocity proxy from the window before last.
    airm_vel_curr
        AIRM velocity proxy from the most recent window.
    is_slow_mixing
        Bool scalar. True = slow-mixing (RESET preferred by the v1.1 switch).
        V1 always uses RESET regardless.
    """

    # Buffer fields (same layout as LowRankMetricCoreState)
    inverse_mass_matrix: LowRankInverseMassMatrix
    mu_star: Array
    draws_buffer: Array
    grads_buffer: Array
    buffer_idx: Array
    background_split: Array
    recompute_counter: Array
    # Controller carry
    has_escalated: Array
    escalation_rank: Array
    s_gap_prev: Array
    s_gap_curr: Array
    r2_latest: Array
    budget_used: Array
    prev_lam: Array
    airm_vel_prev: Array
    airm_vel_curr: Array
    is_slow_mixing: Array


class MetaAdaptationVerdict(NamedTuple):
    """Verdict emitted by the meta-adaptation controller after warmup.

    Populated by :func:`extract_meta_verdict` from the final
    :class:`MetaAdaptationCoreState` after the warmup scan completes.

    Parameters
    ----------
    route
        Routing decision: ``"diagonal"``, ``"low_rank"``, or
        ``"reparam_suggested"`` (R² gate blocked escalation).
    metric
        Final :class:`~blackjax.mcmc.metrics.LowRankInverseMassMatrix`.
    effective_rank
        Rank k at escalation time (0 for diagonal route).
    confidence
        ``"high"`` if both gates passed and S_gap was stable;
        ``"low"`` otherwise (budget ran out, R² deferred, S_gap marginal).
    exit_reason
        One of ``"warmup_complete"``, ``"airm_velocity_converged"``,
        or ``"warmup_budget_exhausted"``.
    budget_used_steps
        Warmup steps used (step-clock proxy).
    budget_returned_steps
        Warmup steps returned (0 if budget was exhausted).
    budget_used_grads
        True gradient count summed from the info stream; -1 if the info
        stream was not provided to :func:`extract_meta_verdict`.
    r2_final
        Last R² reading (NaN if always deferred).
    s_gap_final
        Last S_gap reading (NaN if not yet computed).
    transient_mixing_class
        ``"slow"`` or ``"fast"`` (from split-half mixing signal).
    buffer_policy
        Always ``"reset"`` in v1.
    flags
        Dictionary with keys:

        ``reparam_hint`` (bool): True if ``route == "reparam_suggested"``.

        ``marginal_s_gap`` (bool): True if S_gap was above _S_MIN but below
        2·_S_MIN at the final window (the stayed-diagonal decision was close).

        ``wall_cost_discount`` (bool): True if effective_rank > 0 — the
        ess/grad improvement overstates the wall-time win by O(dk) per step.

        ``high_d_r2_mode`` (str): ``"full_affine"``, ``"projected"``, or
        ``"deferred"`` — which R² fit was likely used in the final window.

        ``mode_coverage`` (str): always ``"single_chain_uncertified"`` in v1.
    """

    route: str
    metric: LowRankInverseMassMatrix
    effective_rank: int
    confidence: str
    exit_reason: str
    budget_used_steps: int
    budget_returned_steps: int
    budget_used_grads: int
    r2_final: float
    s_gap_final: float
    transient_mixing_class: str
    buffer_policy: str
    flags: dict


# ---------------------------------------------------------------------------
# Signal computation — pure JAX functions, module-private
# ---------------------------------------------------------------------------


def _compute_whitened_spectrum(
    draws_buffer: Array,
    sigma: Array,
    n: Array,
    max_rank: int,
) -> tuple[Array, Array]:
    """Eigenvalues and top eigenvectors of the diagonal-whitened sample covariance.

    Computes R = D^{-1/2}ΣD^{-1/2} where D = diag(σ²), via the thin SVD of
    the centred, whitened draws matrix. Returns the top ``max_rank`` eigenvalues
    (descending) and their eigenvectors.

    These serve two roles in the controller: the eigenvalue gap S_gap(k) = λ₁/λ_{k+1}
    measures the rank-k payoff, and the eigenvectors U_k provide the projected
    basis for the R² curvature gate.

    Parameters
    ----------
    draws_buffer
        Shape ``(B, d)``.  First ``n`` rows are valid; rest are zero-padded.
    sigma
        Shape ``(d,)``.  Diagonal scale (positive).
    n
        Number of valid rows (traced JAX integer).
    max_rank
        Static maximum rank to retain.

    Returns
    -------
    eigenvalues : Array
        Shape ``(max_rank,)``. Descending eigenvalues; zero-padded when fewer
        than max_rank are available.
    U_k : Array
        Shape ``(d, max_rank)``. Top eigenvectors; zero-padded.
    """
    B, d = draws_buffer.shape  # static Python ints
    n_safe = jnp.maximum(n.astype(draws_buffer.dtype), 1.0)
    mask = (jnp.arange(B) < n).astype(draws_buffer.dtype)

    sigma_safe = jnp.maximum(sigma, 1e-20)
    mean_x = (mask[:, None] * draws_buffer).sum(0) / n_safe
    w = mask[:, None] * (draws_buffer - mean_x[None, :]) / sigma_safe[None, :]

    # SVD of w/√n: singular values squared = eigenvalues of R
    _, s, Vt = jnp.linalg.svd(w, full_matrices=False)  # s:(min(B,d),), Vt:(min(B,d),d)
    eigs_all = (s**2) / n_safe

    # Pad or trim to static shape (max_rank,)
    actual = min(max_rank, min(B, d))  # static Python int
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
    """Choose the escalation rank from the whitened-residual spectrum.

    The rank k is the number of informative eigenvalues (those outside the
    uninformative band [1/cutoff, cutoff]), capped by the estimation-support
    limit n//2 and by ``max_rank``. This gives k≈2 for a concentrated spike
    and k growing toward d/2 for a declining spectrum.

    Parameters
    ----------
    eigenvalues
        Shape ``(max_rank,)``. Descending whitened-residual eigenvalues.
    n
        Number of valid draws (traced JAX integer).
    max_rank
        Static maximum rank.
    cutoff
        Eigenvalues in ``[1/cutoff, cutoff]`` are uninformative.

    Returns
    -------
    Array
        Traced integer scalar k ∈ [0, max_rank].
    """
    is_informative = (eigenvalues > cutoff) | (eigenvalues < 1.0 / cutoff)
    count = is_informative.sum().astype(jnp.int32)
    support_cap = (n // 2).astype(jnp.int32)
    return jnp.minimum(count, jnp.minimum(support_cap, jnp.int32(max_rank)))


def _compute_s_gap(eigenvalues: Array, k: Array) -> Array:
    """S_gap at the chosen rank cut: λ₁/λ_{k+1}.

    Measures the gap between the largest eigenvalue and the first eigenvalue
    after the rank-k cut of the diagonal-whitened residual. S_gap ≈ 1 means
    rank-k adds little; large S_gap predicts large rank-k payoff (Spearman 1.0
    with measured payoff across geometry classes).

    Parameters
    ----------
    eigenvalues
        Shape ``(max_rank,)``. Descending; zero-padded beyond actual spectrum.
    k
        Traced integer scalar. Rank cut position.

    Returns
    -------
    Array
        Scalar S_gap = λ₁/λ_{k+1}. Returns 1.0 when k=0.
    """
    max_rank = eigenvalues.shape[0]  # static
    k_clipped = jnp.clip(k.astype(jnp.int32), 0, max_rank - 1)
    lambda_1 = jnp.maximum(eigenvalues[0], 1e-10)
    lambda_k1 = jax.lax.dynamic_index_in_dim(eigenvalues, k_clipped, keepdims=False)
    lambda_k1_safe = jnp.maximum(lambda_k1, 1e-10)
    return jnp.where(k.astype(jnp.int32) == 0, jnp.ones_like(lambda_1), lambda_1 / lambda_k1_safe)


def _compute_r2_score_linearity(
    draws_buffer: Array,
    grads_buffer: Array,
    sigma: Array,
    n: Array,
    U_k: Array,
    max_rank: int,
) -> tuple[Array, str]:
    """Held-out score-linearity R² with projected fallback for high-d targets.

    Tests whether the whitened score s_w is approximately linear in the
    whitened position w. High R² indicates linear-correlation geometry (a rank-k
    metric can fix it); low R² indicates curvature (reparameterize).

    Three fit modes determined at trace time by static thresholds on n:

    - **Full-affine** (n ≥ 8d): regress all d whitened position components.
    - **Projected** (n ≥ 8·(max_rank+1)): regress only the max_rank candidate
      eigenvectors U_k from the whitened spectrum — d-independent predictor count.
      Tests linearity precisely in the candidate escalation directions.
    - **Deferred** (n too small): return NaN. The controller treats NaN as a
      blocking gate (does not escalate), so this is conservative.

    The R² is computed as median held-out R² over all d output coordinates.
    Train/test split: first n//2 samples train, last n//2 test.

    Parameters
    ----------
    draws_buffer, grads_buffer
        Shape ``(B, d)``. First ``n`` rows are valid.
    sigma
        Shape ``(d,)``. Diagonal scale.
    n
        Number of valid rows (traced JAX integer).
    U_k
        Shape ``(d, max_rank)``. Top eigenvectors from the whitened spectrum.
    max_rank
        Static maximum rank (projected feature count = max_rank + 1).

    Returns
    -------
    r2 : Array
        Scalar median held-out R². NaN if deferred.
    mode_label : str
        Static string labelling the largest feasible fit mode for this buffer:
        ``"full_affine"``, ``"projected"``, or ``"deferred"``. This reflects
        what the computation does when n is large; early windows with small n
        may use a lower-tier mode inside the JAX conditional.
    """
    B, d = draws_buffer.shape  # static Python ints
    n_f = n.astype(jnp.float32)
    n_safe = jnp.maximum(n_f, 2.0)
    mask = (jnp.arange(B) < n).astype(draws_buffer.dtype)
    sigma_safe = jnp.maximum(sigma, 1e-20)

    # Whiten and centre draws and scores
    mean_x = (mask[:, None] * draws_buffer).sum(0) / n_safe
    mean_g = (mask[:, None] * grads_buffer).sum(0) / n_safe
    w = mask[:, None] * (draws_buffer - mean_x[None, :]) / sigma_safe[None, :]
    s_w = mask[:, None] * (grads_buffer - mean_g[None, :]) * sigma_safe[None, :]

    # Dynamic n-split: first n//2 samples → train, remaining n//2 → test.
    # Both halves have ≥ n//2 valid samples when n ≥ min_n_full (or min_n_proj).
    n_train_int = n // 2  # traced int
    train_mask = mask * (jnp.arange(B) < n_train_int).astype(mask.dtype)
    test_mask = mask * (jnp.arange(B) >= n_train_int).astype(mask.dtype)

    def _r2_from_features(feats: Array) -> Array:
        """OLS on train split; median held-out R² on test split."""
        n_feats = feats.shape[1]  # static
        feats_train = train_mask[:, None] * feats
        feats_test = test_mask[:, None] * feats
        s_train = train_mask[:, None] * s_w
        s_test = test_mask[:, None] * s_w

        # Regularised normal equations: (FᵀF + εI)β = FᵀS
        FtF = feats_train.T @ feats_train  # (n_feats, n_feats) static shape
        FtS = feats_train.T @ s_train  # (n_feats, d) static shape
        A = jnp.linalg.lstsq(
            FtF + 1e-8 * jnp.eye(n_feats, dtype=FtF.dtype), FtS, rcond=None
        )[0]  # (n_feats, d)

        s_pred_test = feats_test @ A  # (B, d)
        n_test_safe = jnp.maximum(test_mask.sum().astype(jnp.float32), 2.0)
        s_test_mean = s_test.sum(0) / n_test_safe
        tss = ((s_test - test_mask[:, None] * s_test_mean[None, :]) ** 2).sum(0)
        rss = ((s_test - s_pred_test) ** 2).sum(0)
        r2_per = 1.0 - rss / jnp.maximum(tss, 1e-10)
        return jnp.median(r2_per)

    def _r2_full_affine() -> Array:
        feats = jnp.concatenate([w, jnp.ones((B, 1), dtype=w.dtype)], axis=1)
        return _r2_from_features(feats)

    def _r2_projected() -> Array:
        proj = w @ U_k  # (B, max_rank)
        feats = jnp.concatenate([proj, jnp.ones((B, 1), dtype=proj.dtype)], axis=1)
        return _r2_from_features(feats)

    # Thresholds based on total n (both halves must have ≥ threshold/2 samples;
    # using n ≥ 2 × ratio × predictors so each half gets ratio × predictors).
    min_n_full = 2 * _MIN_TRAIN_D_RATIO * d  # need n_half ≥ 8d
    min_n_proj = 2 * _MIN_TRAIN_K_RATIO * (max_rank + 1)  # need n_half ≥ 8(k+1)

    r2 = jax.lax.cond(
        n_f >= float(min_n_full),
        _r2_full_affine,
        lambda: jax.lax.cond(
            n_f >= float(min_n_proj),
            _r2_projected,
            lambda: jnp.array(float("nan"), dtype=jnp.float32),
        ),
    )

    # Static mode label for the verdict: reflects what the computation does when
    # the buffer is fully filled (n = B). Early windows may use a lower-tier mode.
    if min_n_full <= B:
        mode_label = "full_affine"
    elif min_n_proj <= B:
        mode_label = "projected"
    else:
        mode_label = "deferred"

    return r2, mode_label


def _compute_transient_mixing_signal(
    draws_buffer: Array,
    sigma: Array,
    n: Array,
) -> Array:
    """Split-half mean-difference proxy for the warmup transient-mixing class.

    Estimates whether the chain is mixing slowly (buffer halves have different
    means) or quickly (halves agree). Slow mixing suggests RESET buffer policy.

    Returns a traced bool scalar: True = slow-mixing (RESET preferred).
    V1 policy is always RESET regardless; this signal is reported in the verdict
    to anchor the v1.1 policy switch.

    Parameters
    ----------
    draws_buffer
        Shape ``(B, d)``. First ``n`` rows are valid.
    sigma
        Shape ``(d,)``. Diagonal scale for whitening.
    n
        Number of valid rows (traced JAX integer).
    """
    B, _ = draws_buffer.shape
    n_f = n.astype(draws_buffer.dtype)
    n_safe = jnp.maximum(n_f, 2.0)
    mask = (jnp.arange(B) < n).astype(draws_buffer.dtype)
    sigma_safe = jnp.maximum(sigma, 1e-20)

    mean_x = (mask[:, None] * draws_buffer).sum(0) / n_safe
    w = mask[:, None] * (draws_buffer - mean_x[None, :]) / sigma_safe[None, :]

    n_train_int = n // 2
    mask_first = mask * (jnp.arange(B) < n_train_int).astype(mask.dtype)
    mask_second = mask * (jnp.arange(B) >= n_train_int).astype(mask.dtype)
    n_first = jnp.maximum(mask_first.sum().astype(jnp.float32), 1.0)
    n_second = jnp.maximum(mask_second.sum().astype(jnp.float32), 1.0)

    mean_first = (mask_first[:, None] * w).sum(0) / n_first
    mean_second = (mask_second[:, None] * w).sum(0) / n_second
    std_all = jnp.maximum(
        ((mask[:, None] * w**2).sum(0) / n_safe) ** 0.5, 1e-10
    )

    split_half_stat = jnp.max(jnp.abs(mean_first - mean_second) / std_all)
    return split_half_stat > _TRANSIENT_MIXING_THRESHOLD


# ---------------------------------------------------------------------------
# Core builder — the public factory
# ---------------------------------------------------------------------------


def build_meta_adaptation_core(
    max_grad_budget: int,
    *,
    max_rank: int | None = None,
    gamma: float = 1e-5,
    cutoff: float = 2.0,
) -> MetricCore:
    """Build a meta-adaptation :class:`~blackjax.adaptation.metric_recipes.MetricCore`.

    The returned core implements the init/update/final protocol compatible with
    :func:`~blackjax.adaptation.staged_adaptation.staged_adaptation`. Pass it
    as ``metric=`` or let :func:`staged_adaptation` build it automatically when
    ``metric="auto"`` and ``max_grad_budget`` are supplied.

    Parameters
    ----------
    max_grad_budget
        Maximum total gradient budget (leapfrog evaluations).  Converted to
        warmup steps via ``_ASSUMED_AVG_LEAPFROGS_PER_STEP``.
    max_rank
        Maximum low-rank correction rank. ``None`` (default) uses
        :data:`_MAX_RANK_CAP`. Set to a smaller value for memory-limited scenarios.
    gamma
        Fisher-estimator regularisation scale. Default 1e-5 matches
        ``fisher_low_rank`` recipe.
    cutoff
        Eigenvalue informativeness cutoff (also used for rank selection).
        Default 2.0.

    Returns
    -------
    MetricCore
        Embeddable init/update/final bundle.
    """
    _max_rank: int = _MAX_RANK_CAP if max_rank is None else max_rank
    max_budget_steps: int = max(max_grad_budget // _ASSUMED_AVG_LEAPFROGS_PER_STEP, 1)

    def init(n_dims: int) -> MetaAdaptationCoreState:
        # Buffer large enough to hold one large window (20 % of budget), capped by budget.
        buf = min(max(max_budget_steps // 5, 128), max_budget_steps)
        buf = max(buf, 2 * (_max_rank + 1) * _MIN_TRAIN_K_RATIO)  # ensure R² feasibility
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
            budget_used=jnp.zeros((), dtype=jnp.int32),
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
        """Accumulate one draw/gradient pair into the buffer."""
        pos_flat, _ = fu.ravel_pytree(position)
        grad_flat, _ = fu.ravel_pytree(grad)
        B = state.draws_buffer.shape[0]
        idx = state.buffer_idx % B  # circular wrap; v1 resets so wrap doesn't fire
        new_draws = jax.lax.dynamic_update_slice(
            state.draws_buffer, pos_flat[None, :], (idx, 0)
        )
        new_grads = jax.lax.dynamic_update_slice(
            state.grads_buffer, grad_flat[None, :], (idx, 0)
        )
        return state._replace(
            draws_buffer=new_draws,
            grads_buffer=new_grads,
            buffer_idx=state.buffer_idx + 1,
            budget_used=state.budget_used + 1,
        )

    def final(state: MetaAdaptationCoreState) -> MetaAdaptationCoreState:
        """Window-boundary controller.

        Computes signals (spectrum, S_gap, R², mixing class), applies the
        escalation decision, chooses the IMM to emit, then hard-resets the
        buffer (v1 reset policy).
        """
        B, d = state.draws_buffer.shape  # static Python ints
        n = state.buffer_idx
        actual_rank = state.inverse_mass_matrix.U.shape[1]  # static Python int

        # Compute Fisher-LR metric from buffer (one call; yields sigma AND U,lam).
        sigma_lr, mu_star_new, U_lr, lam_lr = _compute_low_rank_metric(
            state.draws_buffer, state.grads_buffer, n, actual_rank, gamma, cutoff
        )
        lr_imm = LowRankInverseMassMatrix(sigma=sigma_lr, U=U_lr, lam=lam_lr)
        # Diagonal: same sigma_lr but U=0, lam=1 (LowRankIMM with U=0,lam=1
        # IS the diagonal metric — the equivalence is verified in the timing arm).
        diag_imm = LowRankInverseMassMatrix(
            sigma=sigma_lr,
            U=jnp.zeros((d, actual_rank), dtype=sigma_lr.dtype),
            lam=jnp.ones(actual_rank, dtype=sigma_lr.dtype),
        )

        # Whitened-residual spectrum (using sigma_lr as the diagonal scale).
        eigenvalues, U_k = _compute_whitened_spectrum(
            state.draws_buffer, sigma_lr, n, actual_rank
        )

        # Rank choice: number of informative eigenvalues, capped by n//2.
        k_new = _choose_rank(eigenvalues, n, actual_rank, cutoff)

        # S_gap at the chosen rank cut: λ₁/λ_{k+1}.
        s_gap_new = _compute_s_gap(eigenvalues, k_new)

        # Score-linearity R² (projected fallback for high-d targets).
        r2_new, _mode_label = _compute_r2_score_linearity(
            state.draws_buffer, state.grads_buffer, sigma_lr, n, U_k, actual_rank
        )

        # Transient-mixing class (for verdict; v1 policy is always RESET).
        is_slow = _compute_transient_mixing_signal(state.draws_buffer, sigma_lr, n)

        # ---- Escalation decision (all three conditions must hold) ----
        # Gate 1: R² ≥ r_min (NaN → gate fails, controller stays diagonal)
        r2_gate = r2_new >= _R_MIN  # False when r2_new is NaN (comparison semantics)

        # Gate 2: S_gap magnitude AND stability over two consecutive windows.
        s_gap_prev_valid = ~jnp.isnan(state.s_gap_curr)
        relative_change = jnp.abs(s_gap_new - state.s_gap_curr) / jnp.maximum(
            s_gap_new, 1e-10
        )
        s_gap_stable = s_gap_prev_valid & (relative_change < _S_GAP_STABILITY_TOL)
        s_gap_gate = (s_gap_new >= _S_MIN) & s_gap_stable

        # Gate 3: Remaining step budget ≥ 2k (support) + step-size re-adaptation tail.
        budget_remaining = jnp.int32(max_budget_steps) - state.budget_used.astype(jnp.int32)
        deadline_ok = budget_remaining >= (
            2 * k_new.astype(jnp.int32) + jnp.int32(_STEP_SIZE_READAPT_BUFFER)
        )

        # Monotone escalation: once escalated, stays escalated.
        escalate_now = ~state.has_escalated & r2_gate & s_gap_gate & deadline_ok
        new_has_escalated = state.has_escalated | escalate_now
        new_escalation_rank = jnp.where(escalate_now, k_new, state.escalation_rank)

        # Choose IMM to emit for the next window.
        chosen_imm = jax.lax.cond(new_has_escalated, lambda: lr_imm, lambda: diag_imm)
        chosen_mu_star = jax.lax.cond(
            new_has_escalated, lambda: mu_star_new, lambda: jnp.zeros_like(mu_star_new)
        )

        # ---- AIRM velocity proxy (Frobenius norm of lam change) ----
        # After escalation, lam should converge; two consecutive small velocities
        # trigger early exit (detected in extract_meta_verdict).
        lam_diff = jnp.linalg.norm(lam_lr - state.prev_lam)
        new_airm_vel_prev = state.airm_vel_curr
        new_airm_vel_curr = jnp.where(new_has_escalated, lam_diff, state.airm_vel_curr)

        # ---- Hard-reset buffer (v1: reset policy always) ----
        return MetaAdaptationCoreState(
            inverse_mass_matrix=chosen_imm,
            mu_star=chosen_mu_star,
            draws_buffer=jnp.zeros_like(state.draws_buffer),
            grads_buffer=jnp.zeros_like(state.grads_buffer),
            buffer_idx=jnp.zeros_like(state.buffer_idx),
            background_split=jnp.zeros_like(state.background_split),
            recompute_counter=jnp.zeros_like(state.recompute_counter),
            has_escalated=new_has_escalated,
            escalation_rank=new_escalation_rank,
            s_gap_prev=state.s_gap_curr,  # shift history
            s_gap_curr=s_gap_new,
            r2_latest=r2_new,
            budget_used=state.budget_used,
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
    """Extract a :class:`MetaAdaptationVerdict` from the final core state.

    Call this after ``warmup.run()`` completes to get the controller's
    structured verdict. Pass ``adaptation_info`` (the second return of
    ``warmup.run()``) to get true gradient counts; omit it for step-proxy
    counts only.

    Parameters
    ----------
    final_state
        The ``imm_state`` field of the last :class:`StagedAdaptationState`
        from the warmup scan.
    max_grad_budget
        The ``max_grad_budget`` passed to ``staged_adaptation``.
    num_warmup_steps
        The ``num_steps`` used in ``warmup.run()``.
    adaptation_info
        Optional adaptation info from ``warmup.run()``. If it contains a
        ``num_integration_steps`` field, true gradient counts are computed.

    Returns
    -------
    MetaAdaptationVerdict
    """
    import numpy as np

    has_esc = bool(np.asarray(final_state.has_escalated))
    k = int(np.asarray(final_state.escalation_rank))
    budget_used = int(np.asarray(final_state.budget_used))
    s_gap = float(np.asarray(final_state.s_gap_curr))
    r2 = float(np.asarray(final_state.r2_latest))
    airm_v_prev = float(np.asarray(final_state.airm_vel_prev))
    airm_v_curr = float(np.asarray(final_state.airm_vel_curr))
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
    gates_passed = has_esc and (not r2_nan) and r2 >= _R_MIN and s_gap_valid and s_gap >= _S_MIN
    confidence = "high" if gates_passed else "low"

    # Exit reason
    airm_converged = (airm_v_prev < _AIRM_VELOCITY_TOL) and (airm_v_curr < _AIRM_VELOCITY_TOL)
    if airm_converged and has_esc:
        exit_reason = "airm_velocity_converged"
    elif budget_used >= num_warmup_steps:
        exit_reason = "warmup_budget_exhausted"
    else:
        exit_reason = "warmup_complete"

    # Budget accounting
    budget_returned = max(num_warmup_steps - budget_used, 0)

    # True gradient count from info stream
    budget_used_grads = -1
    if adaptation_info is not None:
        try:
            nis = np.asarray(adaptation_info.num_integration_steps)
            budget_used_grads = int(nis.sum())
        except AttributeError:
            pass

    # Marginal S_gap: stayed diagonal but was close to threshold
    marginal_s_gap = (not has_esc) and s_gap_valid and (_S_MIN <= s_gap < 2.0 * _S_MIN)

    # High-d R² mode: infer from NaN status (NaN → deferred or no window yet)
    if r2_nan:
        high_d_r2_mode = "deferred"
    else:
        # The mode actually used depends on n vs static thresholds; report "projected"
        # as the conservative label (full-affine is rare in practice for high-d targets).
        high_d_r2_mode = "projected"

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
