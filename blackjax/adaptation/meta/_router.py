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
"""Routing functions for the multi-chain meta-adaptation controller.

Functions
---------
:func:`_compute_projected_gain_r2_mc` — slope-heterogeneity GAIN for the
    projected router tier (M-chain path).
:func:`_build_pc_centered_time_major_pool` — per-chain-centered draws/grads
    in time-major layout with the valid-row mask.
:func:`_geometric_mean_deploy_scale` — geometric-mean metric variance for the
    slow direction.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from blackjax.adaptation.meta._calibration import _GAIN_READABILITY_FLOOR
from blackjax.types import Array


def _geometric_mean_deploy_scale(
    chain_means: Array,
    pooled_grads: Array,
    step_mask_all: Array,
    grand_mean: Array,
    e: Array,
    n_pool: Array,
    M: int,
) -> Array:
    """Geometric-mean metric variance for the slow direction.

    σ̂²_deploy = √( (B/n) · 1/(eᵀF̂e) )

    where B/n is the between-chain variance of chain-means in direction e and
    1/(eᵀF̂e) is the pooled-Fisher curvature scale in direction e.  The
    geometric mean cancels the init-dispersion factor f_disp that makes each
    term individually biased: B/n over-estimates by f_disp, Fisher curvature
    under-estimates by f_disp; the product is f_disp-independent.

    Parameters
    ----------
    chain_means
        Shape ``(M, d)`` — per-chain means.
    pooled_grads
        Shape ``(M*B, d)`` — all chains' gradient buffers flattened.
    step_mask_all
        Shape ``(M*B,)`` — 1 for valid rows, 0 otherwise.
    grand_mean
        Shape ``(d,)`` — grand mean of chain means.
    e
        Shape ``(d,)`` — unit-vector slow direction in original (unwhitened) space.
    n_pool
        Dynamic int32 — total valid gradient count (M * n_per_chain).
    M
        Static int — number of chains.

    Returns
    -------
    float32 scalar — σ̂²_deploy (positive).
    """
    # B/n: between-chain variance of chain-means projected onto e
    mu_proj = (chain_means - grand_mean[None, :]) @ e  # (M,)
    B_over_n = (mu_proj**2).sum() / jnp.float32(M - 1)

    # Fisher curvature: eᵀF̂_pooled e
    n_pool_f = n_pool.astype(pooled_grads.dtype)
    n_pool_safe = jnp.maximum(n_pool_f, jnp.float32(1.0))
    g_proj = pooled_grads @ e  # (M*B,)
    fisher_curv_e = (step_mask_all * g_proj**2).sum() / n_pool_safe

    # Geometric mean: sqrt(B/n * 1/fisher_curv_e)
    sigma_sq_deploy = jnp.sqrt(
        jnp.maximum(B_over_n, jnp.float32(1e-20))
        / jnp.maximum(fisher_curv_e, jnp.float32(1e-20))
    )
    return sigma_sq_deploy.astype(jnp.float32)


def _compute_projected_gain_r2_mc(
    pc_draws_tm: Array,
    pc_grads_tm: Array,
    sigma_w_diag: Array,
    n: Array,
    M: int,
    U_k: Array,
) -> tuple[Array, Array]:
    """Slope-heterogeneity GAIN for the projected router tier (M-chain path).

    Both the per-chain fit and the pooled-shared fit are evaluated on held-out
    data (train = first half of each chain, test = second half).  The GAIN

    .. code-block:: text

        GAIN = R2_perchain - R2_shared

    is positive only when genuine curvature heterogeneity exists across chain
    regions.  A Gaussian null produces GAIN <= 0 (per-chain fits overfit),
    so the threshold 0.3 gives near-zero false-reparam rate.

    **Abstain rule**: if R2_perchain < :data:`~blackjax.adaptation.meta._calibration._GAIN_READABILITY_FLOOR`
    (garbage fits due to starvation / transience), return (NaN, NaN).  The
    caller maps NaN to diagonal + confidence=low (not reparam).

    Parameters
    ----------
    pc_draws_tm
        Per-chain-centered time-major pooled draws, shape ``(B*M, d)``.
        Row ``t*M + m`` = (step t, chain m).  Valid rows: first ``n*M``.
    pc_grads_tm
        Same layout as ``pc_draws_tm`` for score vectors.
    sigma_w_diag
        Within-chain whitening std, shape ``(d,)``; ``sqrt(W_diag)``.
    n
        Per-chain valid draw count (dynamic JAX int32).
    M
        Chain count (static Python int; baked into reshape).
    U_k
        Top-k whitened eigenvectors, shape ``(d, k)`` where k = actual_rank.

    Returns
    -------
    tuple[Array, Array]
        ``(gain, r2_perchain)`` both float32.  ``NaN`` on abstain (unreadable
        fits).
    """
    B_M = pc_draws_tm.shape[0]  # static: B*M
    B = B_M // M  # per-chain buffer size (static)
    k = U_k.shape[1]  # projected rank (static)

    sigma_safe = jnp.maximum(sigma_w_diag, jnp.float32(1e-20))

    # Reshape to step-major-chain: (B, M, d); chain m at [:, m, :].
    draws_3d = pc_draws_tm.reshape(B, M, -1)  # (B, M, d)
    grads_3d = pc_grads_tm.reshape(B, M, -1)  # (B, M, d)

    # Whiten and project to k directions.
    w_proj_3d = (draws_3d / sigma_safe[None, None, :]) @ U_k  # (B, M, k)
    s_proj_3d = (grads_3d * sigma_safe[None, None, :]) @ U_k  # (B, M, k)

    # Per-chain train/test masks (same split boundary for all chains).
    n_half = n // 2
    step_idx = jnp.arange(B, dtype=jnp.int32)
    train_mask = (step_idx < n_half).astype(pc_draws_tm.dtype)  # (B,)
    test_mask = ((step_idx >= n_half) & (step_idx < n)).astype(
        pc_draws_tm.dtype
    )  # (B,)
    n_test_safe = jnp.maximum(test_mask.sum().astype(jnp.float32), jnp.float32(2.0))

    # ---- Shared fit: pool all chains' train data ----
    # Layout: (B, M, k) → chain-major (M, B, k) → reshape (M*B, k).
    # Shared train mask: broadcast step mask over M chains.
    w_pool = w_proj_3d.transpose(1, 0, 2).reshape(M * B, k)  # (M*B, k)
    s_pool = s_proj_3d.transpose(1, 0, 2).reshape(M * B, k)  # (M*B, k)
    # train_mask for pool: repeat each step's mask M times (chain-major order)
    train_pool = jnp.tile(train_mask, M)  # (M*B,) — step t repeated for M chains
    feats_sh = jnp.concatenate(
        [w_pool, jnp.ones((M * B, 1), dtype=w_pool.dtype)], axis=1
    )
    tm = train_pool[:, None]
    FtF_sh = (tm * feats_sh).T @ (tm * feats_sh)  # (k+1, k+1)
    FtS_sh = (tm * feats_sh).T @ (tm * s_pool)  # (k+1, k)
    A_sh = jnp.linalg.lstsq(
        FtF_sh + jnp.float32(1e-8) * jnp.eye(k + 1, dtype=FtF_sh.dtype),
        FtS_sh,
        rcond=None,
    )[
        0
    ]  # (k+1, k)

    # ---- Per-chain r2_shared and r2_perchain via vmap ----
    # Transpose to chain-major so vmap sees chain as batch dim.
    w_proj_MC = w_proj_3d.transpose(1, 0, 2)  # (M, B, k)
    s_proj_MC = s_proj_3d.transpose(1, 0, 2)  # (M, B, k)

    def _chain_r2s(w_m: Array, s_m: Array) -> tuple[Array, Array]:
        """Per-chain R2_shared and R2_perchain for one chain (vmapped)."""
        feats_m = jnp.concatenate(
            [w_m, jnp.ones((B, 1), dtype=w_m.dtype)], axis=1
        )  # (B, k+1)
        tm_col = test_mask[:, None]

        # Shared model predictions on this chain's test data
        pred_sh = (tm_col * feats_m) @ A_sh  # (B, k)
        s_test = tm_col * s_m
        s_mean = s_test.sum(0) / n_test_safe
        tss = ((s_test - tm_col * s_mean[None, :]) ** 2).sum(0)
        rss_sh = ((s_test - pred_sh) ** 2).sum(0)
        r2_sh = jnp.median(
            jnp.float32(1.0) - rss_sh / jnp.maximum(tss, jnp.float32(1e-10))
        )

        # Per-chain OLS on this chain's train, evaluated on its test
        tr_col = train_mask[:, None]
        FtF_m = (tr_col * feats_m).T @ (tr_col * feats_m)
        FtS_m = (tr_col * feats_m).T @ (tr_col * s_m)
        A_m = jnp.linalg.lstsq(
            FtF_m + jnp.float32(1e-8) * jnp.eye(k + 1, dtype=FtF_m.dtype),
            FtS_m,
            rcond=None,
        )[0]
        pred_pc = (tm_col * feats_m) @ A_m
        rss_pc = ((s_test - pred_pc) ** 2).sum(0)
        r2_pc = jnp.median(
            jnp.float32(1.0) - rss_pc / jnp.maximum(tss, jnp.float32(1e-10))
        )
        return r2_sh, r2_pc

    r2_sh_all, r2_pc_all = jax.vmap(_chain_r2s)(w_proj_MC, s_proj_MC)  # (M,) each
    r2_shared = jnp.median(r2_sh_all)
    r2_perchain = jnp.median(r2_pc_all)

    gain = r2_perchain - r2_shared

    # Abstain: unreadable fits → return NaN to signal "no evidence"
    abstain = (r2_perchain < jnp.float32(_GAIN_READABILITY_FLOOR)) | ~jnp.isfinite(
        r2_perchain
    )
    nan32 = jnp.array(float("nan"), dtype=jnp.float32)
    gain_out = jnp.where(abstain, nan32, gain.astype(jnp.float32))
    r2_pc_out = jnp.where(abstain, nan32, r2_perchain.astype(jnp.float32))
    return gain_out, r2_pc_out


def _build_pc_centered_time_major_pool(
    draws_buffer_mc: Array,
    grads_buffer_mc: Array,
    chain_means: Array,
    n: Array,
    M: int,
) -> tuple[Array, Array, Array]:
    """Per-chain-centered draws/grads in time-major layout with the valid-row mask.

    Subtracts each chain's own mean from positions and gradients, then
    reshapes from chain-major ``(M, B, d)`` to time-major ``(B*M, d)`` where
    the first ``n*M`` rows are valid (one real draw per chain per time step).

    This simultaneously fixes two v2 defects in the pooled R² path:
    1. **Padding bug**: chain-major layout puts zero-padded rows in the
       middle of the buffer (rows ``m*B + n .. m*B + B-1``), so the valid-row
       mask ``arange < n_pool`` incorrectly includes padding from early chains
       and drops later chains.  Time-major layout makes all valid rows
       contiguous at the start.
    2. **Between-chain inflation**: grand-centered pooled data conflates local
       curvature with transient chain-offset variance, producing wildly wrong
       R² on real multi-chain draws.  Per-chain centering removes the offset.

    Parameters
    ----------
    draws_buffer_mc
        Shape ``(M, B, d)``.
    grads_buffer_mc
        Shape ``(M, B, d)``.
    chain_means
        Shape ``(M, d)`` — per-chain position means (computed from valid rows).
    n
        Dynamic valid-row fill count, int32.
    M
        Static chain count.

    Returns
    -------
    pc_draws_tm : shape ``(B*M, d)`` — per-chain-centered draws, time-major.
    pc_grads_tm : shape ``(B*M, d)`` — per-chain-centered grads, time-major.
    step_mask_tm : shape ``(B*M,)`` — 1 for valid rows (first n*M), 0 otherwise.
    """
    _M, B, d = draws_buffer_mc.shape

    # Gradient means per chain (computed from valid rows)
    step_mask = (jnp.arange(B) < n).astype(draws_buffer_mc.dtype)  # (B,)
    n_f = jnp.maximum(n.astype(draws_buffer_mc.dtype), jnp.float32(1.0))

    def _chain_grad_mean(g_m: Array) -> Array:
        return (step_mask[:, None] * g_m).sum(0) / n_f

    grad_means = jax.vmap(_chain_grad_mean)(grads_buffer_mc)  # (M, d)

    # Per-chain centering
    pc_draws = draws_buffer_mc - chain_means[:, None, :]  # (M, B, d)
    pc_grads = grads_buffer_mc - grad_means[:, None, :]  # (M, B, d)

    # Time-major reshape: (M, B, d) → swap → (B, M, d) → reshape → (B*M, d)
    # Valid rows: first n*M (all M chains at time step t, for t = 0..n-1)
    pc_draws_tm = pc_draws.swapaxes(0, 1).reshape(B * M, d)
    pc_grads_tm = pc_grads.swapaxes(0, 1).reshape(B * M, d)
    # step_mask[t] = 1 for t < n; repeat M times → M*B mask, valid = first n*M
    step_mask_tm = jnp.repeat(step_mask, M)  # (B*M,)

    return pc_draws_tm, pc_grads_tm, step_mask_tm
