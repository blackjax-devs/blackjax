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
"""Single-chain signal computation for the meta-adaptation controller.

Functions
---------
:func:`_compute_whitened_spectrum` — top eigenvalues + eigenvectors of the
    diagonal-whitened sample covariance.
:func:`_choose_rank` — count informative eigenvalues.
:func:`_compute_s_gap` — S_gap(k) = λ₁/λ_{k+1} magnitude predictor.
:func:`_compute_r2_score_linearity` — held-out score-linearity R².
:func:`_compute_transient_mixing_signal` — split-half mean-diff mixing proxy.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from blackjax.adaptation.meta._calibration import (
    _MIN_TRAIN_D_RATIO,
    _MIN_TRAIN_K_RATIO,
    _R2_DEFERRED,
    _R2_FULL_AFFINE,
    _R2_PROJECTED,
    _TRANSIENT_MIXING_THRESHOLD,
)
from blackjax.types import Array


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

    def _r2_from_features(feats: Array, resp: Array) -> Array:
        """Held-out R² of ``resp`` predicted from ``feats`` (train/test split)."""
        n_feats = feats.shape[1]
        FtF = (train_mask[:, None] * feats).T @ (train_mask[:, None] * feats)
        FtS = (train_mask[:, None] * feats).T @ (train_mask[:, None] * resp)
        A = jnp.linalg.lstsq(
            FtF + 1e-8 * jnp.eye(n_feats, dtype=FtF.dtype), FtS, rcond=None
        )[0]
        s_pred = (test_mask[:, None] * feats) @ A
        s_test = test_mask[:, None] * resp
        n_test_safe = jnp.maximum(test_mask.sum().astype(jnp.float32), 2.0)
        s_mean = s_test.sum(0) / n_test_safe
        tss = ((s_test - test_mask[:, None] * s_mean[None, :]) ** 2).sum(0)
        rss = ((s_test - s_pred) ** 2).sum(0)
        return jnp.median(1.0 - rss / jnp.maximum(tss, 1e-10))

    def _full_affine() -> tuple[Array, Array]:
        feats = jnp.concatenate([w, jnp.ones((B, 1), dtype=w.dtype)], axis=1)
        return _r2_from_features(feats, s_w), jnp.int32(_R2_FULL_AFFINE)

    def _projected() -> tuple[Array, Array]:
        # Project BOTH position and score onto U_k.  Rank-k acts only in
        # span(U_k), so linearity of the score RESTRICTED to that subspace is
        # the escalation-safety question; nonlinearity orthogonal to U_k is
        # invisible to rank-k and should not gate escalation.
        # Bug fixed: old code regressed d-dim score on k+1 features → median
        # R² ≈ 0 at d≫k (radon d=390 always emitted reparam_suggested).
        w_proj = w @ U_k  # (B, actual_rank): position projected onto U_k
        s_w_proj = s_w @ U_k  # (B, actual_rank): score projected onto U_k
        feats = jnp.concatenate([w_proj, jnp.ones((B, 1), dtype=w.dtype)], axis=1)
        return _r2_from_features(feats, s_w_proj), jnp.int32(_R2_PROJECTED)

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
