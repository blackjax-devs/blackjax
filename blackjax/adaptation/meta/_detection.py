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
"""Multi-chain detection statistics for the meta-adaptation controller.

Functions
---------
:func:`_compute_within_chain_stats` — per-chain means and pooled within-chain variance.
:func:`_between_chain_detection` — between-chain detection via M×M Gram.
:func:`_loo_detection_passes` — leave-one-out robustness check.
:func:`_unimodality_gap_stat` — gap-statistic check on projected chain-means.
:func:`_compute_pooled_within_spectrum` — pooled within-chain residual spectrum.
:func:`_compute_mode_consistency_flag` — per-direction mode-consistency flag.
:func:`_compute_contraction_stat` — per-chain split-half drift t-statistic.
:func:`_compute_chain_consistency_psi` — cross-chain consistency Ψ.
:func:`_compute_lag1_autocorr_top_dir` — pooled lag-1 autocorrelation.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from blackjax.adaptation.meta._calibration import _mc_unimodality_threshold
from blackjax.types import Array


def _compute_within_chain_stats(
    draws_buffer_mc: Array,
    n: Array,
) -> tuple[Array, Array]:
    """Per-chain means and pooled within-chain diagonal variance.

    Parameters
    ----------
    draws_buffer_mc
        Per-chain draw buffers, shape ``(M, buf_size, d)``.
    n
        Dynamic fill count (same for all chains), int32 scalar.

    Returns
    -------
    chain_means
        Shape ``(M, d)`` — per-chain mean over valid draws.
    W_diag
        Shape ``(d,)`` — average within-chain diagonal variance across M chains.
    """
    M_s, B, d = draws_buffer_mc.shape  # all static
    n_f = n.astype(draws_buffer_mc.dtype)
    n_safe = jnp.maximum(n_f, 1.0)
    step_mask = (jnp.arange(B) < n).astype(draws_buffer_mc.dtype)  # (B,)

    # Per-chain means: (M, d)
    chain_means = (step_mask[None, :, None] * draws_buffer_mc).sum(
        1
    ) / n_safe  # broadcast (M, B, d) × (B,)

    # Per-chain within-chain variance, vmapped over M: (M, d)
    def _chain_var(draws_m: Array, mean_m: Array) -> Array:
        centered = step_mask[:, None] * (draws_m - mean_m[None, :])
        return (centered**2).sum(0) / jnp.maximum(n_safe - 1.0, 1.0)

    per_chain_vars = jax.vmap(_chain_var)(draws_buffer_mc, chain_means)  # (M, d)
    W_diag = per_chain_vars.mean(0)  # (d,) pooled within-chain variance
    return chain_means, W_diag


def _between_chain_detection(
    chain_means: Array,
    W_diag: Array,
    n: Array,
    M: int,
    d: int,
) -> tuple[Array, Array, Array]:
    """Between-chain detection statistic via the M×M Gram.

    Computes T = (n/(M−1))·ZᵀZ (rank ≤ M−1) via the M×M Gram ZZᵀ.
    T's eigenvalues are the per-direction Gelman–Rubin B/W ratios (R̂²).
    Null distribution has top eigenvalue → (1+√(d/(M−1)))² (the detection edge).

    Parameters
    ----------
    chain_means
        Shape ``(M, d)`` — per-chain mean.
    W_diag
        Shape ``(d,)`` — within-chain diagonal variance (whitening basis).
    n
        Dynamic fill count, int32 scalar (within-chain number of draws).
    M, d
        Static integers.

    Returns
    -------
    T_eigenvalues
        Shape ``(M,)`` — eigenvalues of T in descending order (last M−rank ≈ 0).
    V_top
        Shape ``(d, M−1)`` — top M−1 directions of T in whitened space (columns).
    f1
        float32 scalar — collinearity score: fraction of total between-chain
        scatter variance in the leading direction (→1 for genuine slow dir,
        ≈1/(M−1) for isotropic scatter).
    """
    n_f = n.astype(chain_means.dtype)
    grand_mean = chain_means.mean(0)  # (d,)
    W_safe = jnp.maximum(W_diag, jnp.float32(1e-20))
    sigma_w = jnp.sqrt(W_safe)  # (d,)

    # Whitened chain-mean deviations: z_m = (mu_m − mu_bar) / sigma_w, (M, d)
    Z = (chain_means - grand_mean[None, :]) / sigma_w[None, :]

    # M×M Gram: cheaper than d×d when M ≪ d
    gram = Z @ Z.T  # (M, M)

    # Symmetric eigendecomp; eigh returns ascending order
    eigvals_gram, eigvecs_gram = jnp.linalg.eigh(gram)  # (M,), (M, M)
    # Descending order
    eigvals_gram = jnp.flip(eigvals_gram)
    eigvecs_gram = jnp.flip(eigvecs_gram, axis=1)

    # T eigenvalues: scale by c = n/(M−1)
    c = n_f / jnp.float32(M - 1)
    T_eigenvalues = eigvals_gram * c  # (M,)

    # Collinearity score f₁ = λ_max(gram) / trace(gram)
    trace_gram = jnp.maximum(jnp.trace(gram), jnp.float32(1e-20))
    f1 = (eigvals_gram[0] / trace_gram).astype(jnp.float32)

    # Top M−1 directions in d-space via Z^T U / sqrt(gram_eigval)
    # (eigenvectors of Z^T Z = right singular vectors of Z)
    eps = jnp.float32(1e-10)
    top_m1 = min(M - 1, d)  # static
    s_safe = jnp.sqrt(jnp.maximum(eigvals_gram[:top_m1], eps))  # (M−1,)
    V_top = Z.T @ eigvecs_gram[:, :top_m1] / s_safe[None, :]  # (d, M−1)

    return T_eigenvalues, V_top, f1


def _loo_detection_passes(
    chain_means: Array,
    W_diag: Array,
    n: Array,
    M: int,
    d: int,
    edge_loo: float,
) -> Array:
    """Leave-one-out check: detection must survive dropping any single chain.

    For each m' = 0..M−1, re-centres the remaining M−1 chains, computes the
    top T eigenvalue with (M−2) dof, and checks it clears ``edge_loo``.
    All M checks must pass (conjunction).

    Parameters
    ----------
    chain_means
        Shape ``(M, d)``.
    W_diag
        Shape ``(d,)`` — within-chain diagonal variance.
    n
        Dynamic fill count, int32.
    M, d
        Static.
    edge_loo
        Detection edge for M−2 dof: ``(1+√(d/(M−2)))²`` (Python float).

    Returns
    -------
    bool scalar (JAX) — True if all M leave-one-out checks pass.
    """
    n_f = n.astype(chain_means.dtype)
    c_loo = n_f / jnp.float32(M - 2)
    W_safe = jnp.maximum(W_diag, jnp.float32(1e-20))
    sigma_w = jnp.sqrt(W_safe)
    edge_f32 = jnp.float32(edge_loo)

    all_pass = jnp.ones((), dtype=jnp.bool_)
    for m_drop in range(M):
        # Remaining M−1 chain means (static Python-time indexing)
        rows = [chain_means[m] for m in range(M) if m != m_drop]
        Z_loo = jnp.stack(rows)  # (M−1, d)
        mu_loo = Z_loo.mean(0)  # (d,)
        Z_loo_c = (Z_loo - mu_loo[None, :]) / sigma_w[None, :]  # whitened + centred
        gram_loo = Z_loo_c @ Z_loo_c.T  # (M−1, M−1)
        max_eigval_loo = jnp.linalg.eigvalsh(gram_loo)[-1]  # largest (ascending)
        T_max_loo = max_eigval_loo * c_loo
        all_pass = all_pass & (T_max_loo > edge_f32)

    return all_pass


def _unimodality_gap_stat(
    chain_means: Array,
    top_direction: Array,
    M: int,
) -> tuple[Array, Array]:
    """Gap-statistic check on projected chain-means.

    Projects the M chain-means onto ``top_direction`` and measures whether
    the projections cluster into one group (unimodal) or two or more groups
    (mode-split).  Uses max_gap / mean_gap as the discreteness signal.

    Parameters
    ----------
    chain_means
        Shape ``(M, d)``.
    top_direction
        Shape ``(d,)`` — unit-vector slow direction (unwhitened).
    M
        Static int — number of chains.

    Returns
    -------
    is_unimodal
        bool scalar — True iff gap ratio < :func:`~blackjax.adaptation.meta._calibration._mc_unimodality_threshold`.
    gap_ratio
        float32 scalar — max_gap / mean_gap (diagnostic; large ⇒ mode-split).
    """
    threshold = _mc_unimodality_threshold(M)  # Python float, static
    projections = chain_means @ top_direction  # (M,)
    sorted_proj = jnp.sort(projections)  # ascending
    gaps = sorted_proj[1:] - sorted_proj[:-1]  # (M−1,) inter-point gaps
    mean_gap = gaps.mean()
    max_gap = gaps.max()
    gap_ratio = max_gap / jnp.maximum(mean_gap, jnp.float32(1e-10))
    is_unimodal = gap_ratio < jnp.float32(threshold)
    return is_unimodal, gap_ratio.astype(jnp.float32)


def _compute_pooled_within_spectrum(
    draws_buffer_mc: Array,
    chain_means: Array,
    W_diag: Array,
    n: Array,
    M: int,
    max_rank: int,
) -> tuple[Array, Array]:
    """Top eigenvalue and top eigenvector of the pooled within-chain residual
    correlation matrix R_W.

    Pools M chains' per-chain-centered, diag-whitened residuals into an
    M(n-1)-dof estimate of the within-chain correlation.  Computed via thin SVD
    of the stacked (M*B, d) masked residual matrix — never a d×d eigendecomp.

    Structural facts (exact, not asymptotic):
    - diag(R_W) = 1 exactly: per-chain centering removes the between-chain
      component so W_diag is exactly R_W's diagonal.
    - R_W is f_disp-free: chain means are removed before pooling so init
      overdispersion never enters the estimate.
    - R_W is mode-blind: per-chain centering within each mode gives white
      residuals if the modes have similar within-mode geometry; multi-modal
      chains escalate on the within-mode correlation structure only.

    Parameters
    ----------
    draws_buffer_mc
        Shape ``(M, B, d)``.
    chain_means
        Shape ``(M, d)``; per-chain means over valid draws.
    W_diag
        Shape ``(d,)``; pooled within-chain diagonal variance.
    n
        Dynamic fill count, int32 scalar.
    M
        Static chain count.
    max_rank
        Maximum rank for the SVD (determines how many right singular vectors
        are returned; only the top one is used for the W-branch gate).

    Returns
    -------
    lam1
        float32 scalar — top eigenvalue of R_W.
    top_eigvec
        Shape ``(d,)`` float32 — top right singular vector (principal direction
        in the whitened space).
    """
    _M, B, d = draws_buffer_mc.shape
    W_safe = jnp.maximum(W_diag, jnp.float32(1e-20))
    sigma_w = jnp.sqrt(W_safe)

    # Per-chain centering + whitening: Y[m,t] = (X[m,t] - mu_m) / sigma_w
    centered = draws_buffer_mc - chain_means[:, None, :]  # (M, B, d)
    whitened = centered / sigma_w[None, None, :]  # (M, B, d)

    # Apply valid-row mask within each chain; flatten chain-major → (M*B, d)
    step_mask = (jnp.arange(B) < n).astype(whitened.dtype)  # (B,)
    Y_pool = (step_mask[None, :, None] * whitened).reshape(M * B, d)  # (M*B, d)

    # Dof: N = M*(n-1).  Scale so S = Y_scaled^T Y_scaled ≈ R_W = (1/N) Y_pool^T Y_pool
    N_dof = jnp.maximum(
        n.astype(jnp.int32) * jnp.int32(M) - jnp.int32(M), jnp.int32(1)
    )  # M*(n-1)
    N_f = N_dof.astype(Y_pool.dtype)
    Y_scaled = Y_pool / jnp.sqrt(jnp.maximum(N_f, jnp.float32(1.0)))

    # Finite guard before SVD (NaN/Inf rows from near-zero W_diag)
    Y_safe = jnp.where(jnp.isfinite(Y_scaled), Y_scaled, jnp.zeros_like(Y_scaled))

    # Thin SVD: singular values s satisfy lam_i(R_W) = s_i^2
    _, s, Vt = jnp.linalg.svd(Y_safe, full_matrices=False)

    lam1 = (s[0] ** 2).astype(jnp.float32)
    top_eigvec = Vt[0].astype(jnp.float32)  # (d,) — top direction in whitened space

    return lam1, top_eigvec


def _compute_mode_consistency_flag(
    pc_draws_tm: Array,
    pc_grads_tm: Array,
    grads_buffer_mc: Array,
    chain_means: Array,
    grand_mean: Array,
    V_top: Array,
    sigma_w_diag: Array,
    T_eigenvalues: Array,
    edge_full,
    n: Array,
    M: int,
) -> Array:
    """Per-direction mode-consistency flag for T-branch multimodality detection.

    For each admitted T-spike direction ``e_j`` (where
    ``T_eigenvalues[j] > edge_full``), computes

    .. code-block:: text

        mode_flag(j) = (R²_local(e_j) − R²_global(e_j) > 0.3) AND (R²_local(e_j) ≥ 0.5)

    **R²_local**: per-chain OLS of (per-chain-centered score, per-chain-centered
    position) projected to ``e_j``, then median over chains.  Captures local
    score linearity within each chain's region.

    **R²_global**: pooled OLS of (grand-centered score, grand-centered position)
    projected to ``e_j``.  For a unimodal target the global score is globally
    linear → R²_global ≈ R²_local.  For a mode-split the scores have different
    intercepts per mode → the global linear fit collapses → R²_global << R²_local.

    **Critical distinction**: R²_global must use the RAW (not per-chain-centered)
    gradients, grand-mean-centered.  If per-chain-centered gradients were used for
    R²_global, a unimodal overdispersed target (chains at different positions on
    the same Gaussian) would falsely flag: the per-chain-centered score doesn't
    correlate with the grand-centered position (the offset contributes to x but
    not to pc_s), mimicking a mode-split.  Using grand-centered raw gradients
    restores the global linear relationship for unimodal targets.

    Returns True if ANY admitted direction flags.  W-branch is unaffected (never
    gated on this — the W-branch is structurally mode-blind via per-chain centering).

    Parameters
    ----------
    pc_draws_tm
        Per-chain-centered time-major draws, shape ``(B*M, d)``.  Row ``t*M+m``.
    pc_grads_tm
        Per-chain-centered time-major grads, shape ``(B*M, d)``.  Used for R²_local.
    grads_buffer_mc
        Raw (non-centered) grads, shape ``(M, B, d)``.  Used for R²_global (grand-
        centered).  For a unimodal Gaussian the raw grads ARE globally linear in
        position; per-chain-centered grads lose this between-chain signal.
    chain_means
        Shape ``(M, d)`` — per-chain draw means.
    grand_mean
        Shape ``(d,)`` — mean of chain means.
    V_top
        Shape ``(d, k)`` — top T-eigenvectors in W-whitened space.
    sigma_w_diag
        Shape ``(d,)`` — within-chain whitening std.
    T_eigenvalues
        Shape ``(k,)`` — T-branch eigenvalues (to mask admitted dirs).
    edge_full
        Scalar float32 — T magnitude threshold.
    n
        Per-chain valid draw count (dynamic int32).
    M
        Static chain count.

    Returns
    -------
    bool scalar — True if any T-spike direction exhibits mode-consistency signature.
    """
    BM, d = pc_draws_tm.shape
    B = BM // M  # static

    # Grand-mean-centered raw grads for R²_global (shape M, B, d → B*M, d time-major)
    step_mask_col = (jnp.arange(B) < n).astype(grads_buffer_mc.dtype)  # (B,)
    n_f = jnp.maximum(n.astype(grads_buffer_mc.dtype), jnp.float32(1.0))
    # grand mean of raw grads (over all valid draws across all chains)
    total_valid = n_f * jnp.float32(M)
    grad_sum = (step_mask_col[None, :, None] * grads_buffer_mc).sum(axis=(0, 1))  # (d,)
    grand_mean_grad = grad_sum / jnp.maximum(total_valid, jnp.float32(1.0))  # (d,)
    # Grand-mean-centered raw grads, time-major layout
    gc_grads_mc = grads_buffer_mc - grand_mean_grad[None, None, :]  # (M, B, d)
    # swapaxes(0,1) → (B, M, d) → reshape → (B*M, d)
    gc_grads_tm = gc_grads_mc.swapaxes(0, 1).reshape(BM, d)  # (B*M, d)

    # Grand-centered position offset in time-major layout
    gc_offset = chain_means - grand_mean[None, :]  # (M, d)
    gc_offset_tm = jnp.tile(gc_offset, (B, 1))  # (B*M, d)

    # Per-row valid mask: row t*M+m is valid when step t < n
    row_idx = jnp.arange(BM, dtype=jnp.int32)
    t_idx = row_idx // M  # (B*M,) time step per row
    valid_mask = (t_idx < n).astype(pc_draws_tm.dtype)  # (B*M,)

    # Step-wise valid mask for per-chain reshape
    step_valid = step_mask_col.astype(pc_draws_tm.dtype)  # (B,)

    def _r2_for_direction(j: int) -> Array:
        """Mode-consistency flag (bool Array) for direction j."""
        # Unwhiten V_top[:, j] → original space unit vector
        e_unnorm = sigma_w_diag * V_top[:, j]
        e_j = e_unnorm / jnp.maximum(jnp.linalg.norm(e_unnorm), jnp.float32(1e-10))

        # ---- R²_global: OLS on grand-centered raw grads vs grand-centered positions ----
        x_gc_proj = (pc_draws_tm + gc_offset_tm) @ e_j  # (B*M,) grand-centered position
        s_gc_proj = gc_grads_tm @ e_j  # (B*M,) grand-centered raw score

        n_total = jnp.maximum(valid_mask.sum(), jnp.float32(2.0))
        x_gc_mean = (valid_mask * x_gc_proj).sum() / n_total
        s_gc_mean = (valid_mask * s_gc_proj).sum() / n_total
        x_gc_c = x_gc_proj - x_gc_mean
        s_gc_c = s_gc_proj - s_gc_mean
        Sxx_g = (valid_mask * x_gc_c**2).sum()
        Sxs_g = (valid_mask * x_gc_c * s_gc_c).sum()
        beta_g = Sxs_g / jnp.maximum(Sxx_g, jnp.float32(1e-20))
        rss_g = (valid_mask * (s_gc_c - beta_g * x_gc_c) ** 2).sum()
        ss_tot_g = (valid_mask * s_gc_c**2).sum()
        r2_global = jnp.clip(
            jnp.float32(1.0) - rss_g / jnp.maximum(ss_tot_g, jnp.float32(1e-20)),
            jnp.float32(-10.0),
            jnp.float32(1.0),
        )

        # ---- R²_local: per-chain OLS on per-chain-centered (x, s), median ----
        x_pc_proj = pc_draws_tm @ e_j  # (B*M,) per-chain-centered position
        s_pc_proj = pc_grads_tm @ e_j  # (B*M,) per-chain-centered score

        # Reshape time-major (B*M,) → (B, M): column m = chain m's values
        x_pc_2d = x_pc_proj.reshape(B, M)  # (B, M)
        s_2d = s_pc_proj.reshape(B, M)  # (B, M)

        def _chain_r2_1d(x_col: Array, s_col: Array) -> Array:
            n_c = jnp.maximum(step_valid.sum(), jnp.float32(2.0))
            x_m = (step_valid * x_col).sum() / n_c
            s_m = (step_valid * s_col).sum() / n_c
            x_c = x_col - x_m
            s_c = s_col - s_m
            Sxx = (step_valid * x_c**2).sum()
            Sxs = (step_valid * x_c * s_c).sum()
            beta = Sxs / jnp.maximum(Sxx, jnp.float32(1e-20))
            rss = (step_valid * (s_c - beta * x_c) ** 2).sum()
            ss_tot = (step_valid * s_c**2).sum()
            r2 = jnp.float32(1.0) - rss / jnp.maximum(ss_tot, jnp.float32(1e-20))
            return jnp.clip(r2, jnp.float32(-10.0), jnp.float32(1.0))

        r2_per_chain = jax.vmap(_chain_r2_1d, in_axes=(1, 1))(x_pc_2d, s_2d)  # (M,)
        r2_local = jnp.median(r2_per_chain)

        # Admitted direction only (mask out by eigenvalue threshold)
        admitted = T_eigenvalues[j] > edge_full
        return (
            admitted
            & (r2_local - r2_global > jnp.float32(0.3))
            & (r2_local >= jnp.float32(0.5))
        )

    # Compute flag for each static direction index; OR together
    k = V_top.shape[1]  # static Python int
    flags = jnp.stack([_r2_for_direction(j) for j in range(k)])  # (k,) bool
    return flags.any()


def _compute_contraction_stat(
    draws_buffer: Array,
    chain_means: Array,
    grand_mean: Array,
    n: Array,
    M: int,
) -> Array:
    """Per-chain split-half drift t-statistic for the T-branch three-way rule.

    For each chain m, computes the whitened drift along the chain's own offset
    direction

    .. code-block:: text

        c_m = ⟨mean(late half) − mean(early half), ô_m⟩ / SE_m

    where ``ô_m = (chain_mean_m − grand_mean) / ‖·‖`` and ``SE_m`` is the
    pooled SE of the two half-means along ``ô_m``.  A one-sided t across the
    M chains is returned.

    Interpretation (at M=8, critical value -2.365 at α=2.5%):

    - ``t_contr < -2.365``: chains are converging toward the grand mean →
      unimodal-safe, escalate if the other T gates pass.
    - ``t_contr ≈ 0``: chains are frozen / not drifting (mode-split or
      genuinely slow direction); combine with mode-consistency flag.
    - ``t_contr >> 0``: chains are diverging (unusual, flag as inconclusive).

    Parameters
    ----------
    draws_buffer
        Shape ``(M, B, d)`` — raw (not centered) per-chain draws.
    chain_means
        Shape ``(M, d)`` — per-chain draw means.
    grand_mean
        Shape ``(d,)`` — mean of chain means.
    n
        Per-chain valid draw count (dynamic int32).
    M
        Static chain count.

    Returns
    -------
    float32 scalar — one-sided t statistic across M chains.
    """
    _M, B, d = draws_buffer.shape  # all static

    # Per-chain offset directions ô_m = (chain_means_m - grand_mean) / ||·||
    offsets = chain_means - grand_mean[None, :]  # (M, d)
    norms = jnp.linalg.norm(offsets, axis=1, keepdims=True)  # (M, 1)
    o_hat = offsets / jnp.maximum(norms, jnp.float32(1e-10))  # (M, d)

    # Project draws onto ô_m: (M, B, d) * (M, 1, d) → sum over d → (M, B)
    proj = (draws_buffer * o_hat[:, None, :]).sum(axis=-1)  # (M, B)

    # Build valid/early/late masks
    step_idx = jnp.arange(B, dtype=jnp.int32)
    n_half = n // 2
    early_mask = (step_idx < n_half).astype(proj.dtype)  # (B,)
    late_mask = ((step_idx >= n_half) & (step_idx < n)).astype(proj.dtype)
    valid_mask = (step_idx < n).astype(proj.dtype)

    n_half_f = jnp.maximum(n_half.astype(jnp.float32), jnp.float32(1.0))
    n_late_f = jnp.maximum((n - n_half).astype(jnp.float32), jnp.float32(1.0))
    n_total_f = jnp.maximum(n.astype(jnp.float32), jnp.float32(1.0))

    # Per-chain half-means: broadcast mask (B,) over chains (M, B)
    early_proj = (proj * early_mask[None, :]).sum(axis=1) / n_half_f  # (M,)
    late_proj = (proj * late_mask[None, :]).sum(axis=1) / n_late_f  # (M,)

    # Per-chain within-window std for SE estimation
    proj_mean = (proj * valid_mask[None, :]).sum(axis=1) / n_total_f  # (M,)
    proj_var = (valid_mask[None, :] * (proj - proj_mean[:, None]) ** 2).sum(
        axis=1
    ) / jnp.maximum(n_total_f - jnp.float32(1.0), jnp.float32(1.0))
    se = jnp.sqrt(
        jnp.maximum(proj_var, jnp.float32(1e-10)) * jnp.float32(2.0) / n_half_f
    )  # (M,) — SE of mean difference between two half-means

    # Standardized drift per chain
    c_std = (late_proj - early_proj) / jnp.maximum(se, jnp.float32(1e-10))  # (M,)

    # One-sided t across M chains
    c_mean = c_std.mean()
    c_se = jnp.std(c_std, ddof=1) / jnp.sqrt(jnp.float32(M))
    t_contr = c_mean / jnp.maximum(c_se, jnp.float32(1e-10))

    return t_contr.astype(jnp.float32)


def _compute_chain_consistency_psi(
    draws_buffer_mc: Array,
    chain_means: Array,
    W_diag: Array,
    n: Array,
    M: int,
) -> Array:
    """Cross-chain consistency Ψ of the off-diagonal residual correlation.

    Splits the M chains into fixed halves A (first M//2) and B (remaining),
    computes the pooled within-chain residual correlation for each half using
    the shared W_diag whitener, then computes the cosine similarity of their
    off-diagonal parts in Frobenius matrix space.

    Null property: chains are independent, so E[<C_A, C_B>_F] = ||E C||²_F
    is the off-diagonal signal energy; the cross-noise term is exactly zero
    regardless of within-chain autocorrelation law.  This makes Ψ τ-blind:
    it carries genuine target structure but is insensitive to per-chain
    mixing rate.

    Parameters
    ----------
    draws_buffer_mc
        Shape ``(M, B, d)``.
    chain_means
        Shape ``(M, d)``.
    W_diag
        Shape ``(d,)`` — pooled within-chain diagonal variance.
    n
        Dynamic fill count, int32 scalar.
    M
        Static chain count.

    Returns
    -------
    float32 scalar — Ψ (ranges nominally in [−1, 1]; genuine deep-spread
    ≈ 0.91–0.98; iid-null q999 ≤ 0.095 across all null types tested;
    threshold floor = :data:`~blackjax.adaptation.meta._calibration._W_BRANCH_PSI_FLOOR` = 0.15).
    """
    _M, B, d = draws_buffer_mc.shape
    M_A = M // 2
    M_B = M - M_A

    W_safe = jnp.maximum(W_diag, jnp.float32(1e-20))
    sigma_w = jnp.sqrt(W_safe)

    # Per-chain centering + whitening
    centered = draws_buffer_mc - chain_means[:, None, :]  # (M, B, d)
    whitened = centered / sigma_w[None, None, :]  # (M, B, d)

    step_mask = (jnp.arange(B) < n).astype(whitened.dtype)  # (B,)

    # Half-A chains: flatten chain-major (M_A*B, d) with valid-row mask
    Y_A = (step_mask[None, :, None] * whitened[:M_A]).reshape(M_A * B, d)
    # Half-B chains: flatten chain-major (M_B*B, d)
    Y_B = (step_mask[None, :, None] * whitened[M_A:]).reshape(M_B * B, d)

    # Finite guards
    Y_A = jnp.where(jnp.isfinite(Y_A), Y_A, jnp.zeros_like(Y_A))
    Y_B = jnp.where(jnp.isfinite(Y_B), Y_B, jnp.zeros_like(Y_B))

    # <R_A, R_B>_F = (1/(N_A*N_B)) * ||Y_A Y_B^T||_F^2 = (1/(N_A*N_B)) * ||cross_gram||_F^2
    # (uses the identity trace(A^T A B^T B) = ||A B^T||_F^2 for A=Y_A, B=Y_B)
    cross_gram = Y_A @ Y_B.T  # (M_A*B, M_B*B)
    inner_R_AB = jnp.sum(cross_gram**2)  # = N_A*N_B * <R_A,R_B>_F

    # Diagonal correction: sum_i (R_A)_{ii}(R_B)_{ii} = (1/(N_A*N_B)) * dot(d_A, d_B)
    # where d_A[i] = ||Y_A[:,i]||^2 = N_A * (R_A)_{ii}
    d_A = jnp.sum(Y_A**2, axis=0)  # (d,) = N_A * diag(R_A)
    d_B = jnp.sum(Y_B**2, axis=0)  # (d,) = N_B * diag(R_B)
    diag_corr = jnp.dot(d_A, d_B)  # = N_A*N_B * sum_i (R_A)_{ii}(R_B)_{ii}

    # Frobenius inner product of off-diagonal parts
    inner_C_AB = inner_R_AB - diag_corr  # = N_A*N_B * <C_A, C_B>_F

    # Norms: ||C_X||_F^2 = ||R_X||_F^2 - sum_i (R_X)_{ii}^2
    # ||R_A||_F^2 = (1/N_A^2) * trace((Y_A^T Y_A)^2) = (1/N_A^2) * ||Y_A Y_A^T||_F^2
    auto_gram_A = Y_A @ Y_A.T  # (M_A*B, M_A*B)
    auto_gram_B = Y_B @ Y_B.T  # (M_B*B, M_B*B)
    inner_R_AA = jnp.sum(auto_gram_A**2)  # = N_A^2 * ||R_A||_F^2
    inner_R_BB = jnp.sum(auto_gram_B**2)  # = N_B^2 * ||R_B||_F^2
    diag_AA = jnp.dot(d_A, d_A)  # = N_A^2 * sum_i (R_A)_{ii}^2
    diag_BB = jnp.dot(d_B, d_B)  # = N_B^2 * sum_i (R_B)_{ii}^2
    inner_C_AA = inner_R_AA - diag_AA  # = N_A^2 * ||C_A||_F^2
    inner_C_BB = inner_R_BB - diag_BB  # = N_B^2 * ||C_B||_F^2

    # Ψ = <C_A,C_B>_F / (||C_A||_F * ||C_B||_F)
    # Numerator: inner_C_AB / (N_A * N_B)
    # Denominator: sqrt(inner_C_AA / N_A^2) * sqrt(inner_C_BB / N_B^2)
    #            = sqrt(inner_C_AA * inner_C_BB) / (N_A * N_B)
    # Simplifies to: inner_C_AB / sqrt(max(inner_C_AA * inner_C_BB, eps))
    psi = inner_C_AB / jnp.maximum(
        jnp.sqrt(jnp.maximum(inner_C_AA * inner_C_BB, jnp.float32(1e-30))),
        jnp.float32(1e-20),
    )
    return psi.astype(jnp.float32)


def _compute_lag1_autocorr_top_dir(
    draws_buffer_mc: Array,
    chain_means: Array,
    W_diag: Array,
    top_eigvec: Array,
    n: Array,
    M: int,
) -> Array:
    """Pooled lag-1 autocorrelation of projections onto the top W-branch direction.

    The oscillation screen: a genuine under-resolved slow direction is
    diffusive (r₁ > 0), while integrator resonance alternates (r₁ < 0).
    Pooling M chains' lag-1 estimates reduces variance relative to single-chain.

    Gate: ``r1_top > _W_BRANCH_R1_TOL`` (= −0.2).  The gate is a LOWER bound
    only — no requirement of large positive r₁, or every isotropic-AC null
    direction would pass (at ρ=0.8, AR direction r₁ ≈ +0.8 — the screen
    must not penalize high-AC genuine slow directions).

    Parameters
    ----------
    draws_buffer_mc
        Shape ``(M, B, d)``.
    chain_means
        Shape ``(M, d)``.
    W_diag
        Shape ``(d,)`` — pooled within-chain diagonal variance.
    top_eigvec
        Shape ``(d,)`` — top right singular vector in whitened space (from
        :func:`_compute_pooled_within_spectrum`).
    n
        Dynamic fill count, int32 scalar.
    M
        Static chain count.

    Returns
    -------
    float32 scalar — pooled lag-1 autocorrelation (mean across M chains).
    """
    _M, B, d = draws_buffer_mc.shape
    W_safe = jnp.maximum(W_diag, jnp.float32(1e-20))
    sigma_w = jnp.sqrt(W_safe)

    # Per-chain centering + whitening, then project onto top_eigvec (whitened space)
    centered = draws_buffer_mc - chain_means[:, None, :]  # (M, B, d)
    whitened = centered / sigma_w[None, None, :]  # (M, B, d)
    proj = whitened @ top_eigvec  # (M, B)

    step_mask = (jnp.arange(B) < n).astype(proj.dtype)  # (B,)
    n_f = jnp.maximum(n.astype(proj.dtype), jnp.float32(2.0))

    def _chain_lag1(proj_m: Array) -> Array:
        """Lag-1 autocorrelation for one chain's projection time series."""
        mu = (step_mask * proj_m).sum() / n_f
        p_c = step_mask * (proj_m - mu)  # centered
        var = (p_c**2).sum() / jnp.maximum(n_f - jnp.float32(1.0), jnp.float32(1.0))
        lag1_cov = (
            p_c[:-1] * p_c[1:] * step_mask[:-1] * step_mask[1:]
        ).sum() / jnp.maximum(n_f - jnp.float32(2.0), jnp.float32(1.0))
        return lag1_cov / jnp.maximum(var, jnp.float32(1e-20))

    per_chain_r1 = jax.vmap(_chain_lag1)(proj)  # (M,)
    return per_chain_r1.mean().astype(jnp.float32)
