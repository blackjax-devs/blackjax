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
"""Core builders for the meta-adaptation controller.

This file is the primary entry point for understanding the controller decision
logic.  Read :mod:`~blackjax.adaptation.meta._calibration` for the calibration
surface (all thresholds and swappable gate functions).

Functions
---------
:func:`build_meta_adaptation_core` — single-chain MetricCore builder.
:func:`build_multi_chain_meta_core` — multi-chain MetricCore builder.
"""
from __future__ import annotations

import jax
import jax.flatten_util as fu
import jax.numpy as jnp

from blackjax.adaptation.meta._calibration import (
    _AIRM_VELOCITY_TOL,
    _ASSUMED_AVG_LEAPFROGS_PER_STEP,
    _DETECTION_BRANCH_BETWEEN_MEANS,
    _DETECTION_BRANCH_BOTH,
    _DETECTION_BRANCH_NONE,
    _DETECTION_BRANCH_POOLED_WITHIN,
    _GAIN_THRESHOLD,
    _MAX_RANK_CAP,
    _MC_COLLINEARITY_TOL,
    _MC_MIN_CHAINS,
    _MC_UNIMODALITY_CONFIRM_WINDOWS,
    _MIN_TRAIN_K_RATIO,
    _MULTI_CHAIN_DEFAULT_N_CHAINS,
    _R2_DEFERRED,
    _R2_PROJECTED,
    _R_MIN,
    _S_GAP_STABILITY_TOL,
    _S_MIN,
    _STEP_SIZE_READAPT_BUFFER,
    _W_BRANCH_R1_TOL,
    _mc_detection_edge,
    _w_branch_null_edge,
    _w_branch_psi_threshold,
)
from blackjax.adaptation.meta._detection import (
    _between_chain_detection,
    _compute_chain_consistency_psi,
    _compute_contraction_stat,
    _compute_lag1_autocorr_top_dir,
    _compute_mode_consistency_flag,
    _compute_pooled_within_spectrum,
    _compute_within_chain_stats,
    _loo_detection_passes,
    _unimodality_gap_stat,
)
from blackjax.adaptation.meta._router import (
    _build_pc_centered_time_major_pool,
    _compute_projected_gain_r2_mc,
    _geometric_mean_deploy_scale,
)
from blackjax.adaptation.meta._signals import (
    _choose_rank,
    _compute_r2_score_linearity,
    _compute_s_gap,
    _compute_transient_mixing_signal,
    _compute_whitened_spectrum,
)
from blackjax.adaptation.meta._state import (
    MetaAdaptationCoreState,
    MultiChainMetaAdaptationCoreState,
)
from blackjax.adaptation.metric_estimators import _compute_low_rank_metric
from blackjax.adaptation.metric_recipes import MetricCore
from blackjax.mcmc.metrics import LowRankInverseMassMatrix
from blackjax.types import Array, ArrayLikeTree


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
        Maximum low-rank rank; ``None`` uses :data:`~blackjax.adaptation.meta._calibration._MAX_RANK_CAP`.
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
        # Buffer sized to half the budget so the growing-window schedule's largest
        # window (≈ max_budget_steps * 0.4–0.5) does not overflow the buffer.
        # With the cap n=min(buffer_idx, B) in final(), overflow is still safe
        # (RESET policy keeps the most-recent B draws = the more-stationary tail).
        buf = min(max(max_budget_steps // 2, 256), max_budget_steps)
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
        n = jnp.minimum(state.buffer_idx, jnp.int32(B))  # cap at B (Fix 4)
        actual_rank = state.inverse_mass_matrix.U.shape[1]  # static

        # Welford diagonal sigma: sample std from the buffer.  Used for:
        # (a) stay-diagonal metric (measured ≥0.62× welford vs 0.11× fisher-diag
        #     on classes where escalation is correctly refused), (b) S_gap whitening
        #     basis (where _S_MIN=2.0 and the anchors radon≈8.5/stoch_vol≈1.6 are
        #     calibrated).
        n_f = n.astype(state.draws_buffer.dtype)
        n_safe = jnp.maximum(n_f, 1.0)
        mask_w = (jnp.arange(B) < n).astype(state.draws_buffer.dtype)
        mean_x = (mask_w[:, None] * state.draws_buffer).sum(0) / n_safe
        var_x = (mask_w[:, None] * (state.draws_buffer - mean_x[None, :]) ** 2).sum(
            0
        ) / jnp.maximum(n_safe - 1.0, 1.0)
        sigma_welford = jnp.sqrt(jnp.maximum(var_x, 1e-10))

        # Fisher-LR metric (sigma + low-rank U, lam) — used only for the escalated
        # path.  Its sigma is informative for the escalated metric but measured worse
        # on the stay-diagonal classes (funnel 0.11×, stoch_vol 0.54× vs welford).
        sigma_lr, mu_star_new, U_lr, lam_lr = _compute_low_rank_metric(
            state.draws_buffer, state.grads_buffer, n, actual_rank, gamma, cutoff
        )

        # Stay-diagonal metric: Welford sigma, U=0, lam=1.
        # Recovers-classical anchor: this IS the nutpie welford-diagonal scheme that
        # the controller converges to on isotropic/curvature targets.
        diag_imm = LowRankInverseMassMatrix(
            sigma=sigma_welford,
            U=jnp.zeros((d, actual_rank), dtype=sigma_welford.dtype),
            lam=jnp.ones(actual_rank, dtype=sigma_welford.dtype),
        )
        # Escalated metric: full Fisher-LR (measured payoffs are relative to welford
        # baseline, so the two-phase design diag(welford) → rank-k(fisher) is correct).
        lr_imm = LowRankInverseMassMatrix(sigma=sigma_lr, U=U_lr, lam=lam_lr)

        # Whitened-residual spectrum on the WELFORD basis (where _S_MIN and anchors
        # were calibrated; the Fisher whitening inflates S_gap vs the calibrated scale).
        eigenvalues, U_k = _compute_whitened_spectrum(
            state.draws_buffer, sigma_welford, n, actual_rank
        )
        k_new = _choose_rank(eigenvalues, n, actual_rank, cutoff)
        s_gap_new = _compute_s_gap(eigenvalues, k_new)

        # Score-linearity R² on the Welford-whitened space.
        # (mode is observed from the taken branch — not inferred post-hoc)
        r2_new, mode_new = _compute_r2_score_linearity(
            state.draws_buffer, state.grads_buffer, sigma_welford, n, U_k, actual_rank
        )
        # FOLD-IN 6: clip R² to [-10, 1]; values below -10 → NaN (deferred path).
        r2_new = jnp.where(
            r2_new < jnp.float32(-10.0),
            jnp.array(float("nan"), dtype=r2_new.dtype),
            jnp.clip(r2_new, max=jnp.float32(1.0)),
        )

        # Transient-mixing class (reported in verdict; v1 always uses RESET).
        is_slow = _compute_transient_mixing_signal(state.draws_buffer, sigma_welford, n)

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
        # Cast prev_lam to lam_lr dtype for the norm so the subtraction is
        # numerically sound; then cast back to float32 before storing.
        lam_diff = jnp.linalg.norm(lam_lr - state.prev_lam.astype(lam_lr.dtype))
        # Controller-signal scalars are always stored as float32: they are gate
        # comparisons and AIRM velocities that don't benefit from float64 precision.
        # Explicit casts here are required so that both branches of the
        # jax.lax.cond in staged_adaptation produce identical types under x64
        # (the false branch returns the init-time float32 carry unchanged).
        lam_diff_f32 = lam_diff.astype(jnp.float32)
        new_airm_vel_prev = state.airm_vel_curr  # already float32
        new_airm_vel_curr = jnp.where(
            new_has_escalated, lam_diff_f32, state.airm_vel_curr
        )
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
            s_gap_prev=state.s_gap_curr,  # float32, propagated from prior window
            s_gap_curr=s_gap_new.astype(jnp.float32),
            r2_latest=r2_new.astype(jnp.float32),
            r2_mode=mode_new,
            budget_used=state.budget_used,
            converged_at_step=new_converged_at,
            prev_lam=lam_lr.astype(jnp.float32),
            airm_vel_prev=new_airm_vel_prev,
            airm_vel_curr=new_airm_vel_curr,
            is_slow_mixing=is_slow,
        )

    return MetricCore(init=init, update=update, final=final)


def build_multi_chain_meta_core(
    max_grad_budget: int,
    n_chains: int = _MULTI_CHAIN_DEFAULT_N_CHAINS,
    *,
    max_rank: int | None = None,
    gamma: float = 1e-5,
    cutoff: float = 2.0,
) -> MetricCore:
    """Build the multi-chain meta-adaptation :class:`~blackjax.adaptation.metric_recipes.MetricCore`.

    Runs M independent chains sharing one adapted metric; the escalation
    decision uses pooled M-chain information instead of a single-chain
    stability check.  The pooled between-chain signal makes the escalation
    decision robust to seed variation for near-edge posterior structures.

    The multi-chain gate (replaces the single-chain S_gap-stability check).
    Five conditions must all hold to escalate:

    1. **Magnitude.** Top eigenvalue of the between-chain T matrix exceeds the
       detection edge ``(1 + √(d/(M−1)))²`` (M−1 dof, grand-mean constraint).
    2. **Collinearity.** Fraction of total between-chain scatter in the top
       singular direction f₁ ≥ :data:`~blackjax.adaptation.meta._calibration._MC_COLLINEARITY_TOL`.
       Genuine slow directions produce near-rank-1 concentration (f₁→1);
       isotropic spurious scatter gives f₁ ≈ 1/(M−1).
    3. **Leave-one-out.** Detection must survive dropping any single chain,
       preventing a single outlier chain from driving the verdict.  Leave-two-out
       (dropping any pair) is subsumed by the collinearity + unimodality conjunction
       for the aligned-pair threat model and is deferred to v2.1.
    4. **Support floor.** At least one spike is admitted (k ≥ 1).
    5. **Unimodality guard.** Gap-statistic on the projected chain-means must
       not flag mode-split; mode-separated chains are deferred to the ensemble
       (Paper-3 scope) and reported via ``deferred_to_ensemble=True`` in the
       verdict.

    Plus R² curvature gate and budget deadline (same as single-chain).

    Budget re-allocation: ``max_grad_budget`` is the TOTAL gradient budget,
    shared across all M chains.  Providing ``n_chains=M`` overdispersed
    starting positions to ``run()`` causes each chain to run for
    ``total // M`` leapfrog evaluations — the total cost equals the
    single-chain budget, not M× it.

    For ``n_chains=1`` use :func:`build_meta_adaptation_core` directly to
    obtain exact single-chain (v1) behaviour; the ``staged_adaptation`` engine
    routes to it automatically when ``n_chains=1``.

    Parameters
    ----------
    max_grad_budget
        Maximum total gradient budget (leapfrog evaluations) across all M chains.
    n_chains
        Number of independent chains.  Must be ≥ 2.  Defaults to
        :data:`~blackjax.adaptation.meta._calibration._MULTI_CHAIN_DEFAULT_N_CHAINS` (8).
    max_rank, gamma, cutoff
        Same as :func:`build_meta_adaptation_core`.

    Returns
    -------
    MetricCore
        Embeddable init/update/final bundle.  ``update`` expects ``position``
        of shape ``(n_chains, d)`` and ``grad`` of shape ``(n_chains, d)``.
    """
    if n_chains < 2:
        raise ValueError(
            f"build_multi_chain_meta_core: n_chains must be >= 2, got {n_chains}. "
            "For single-chain use, call build_meta_adaptation_core instead."
        )
    if n_chains < _MC_MIN_CHAINS:
        import warnings

        warnings.warn(
            f"build_multi_chain_meta_core: n_chains={n_chains} < {_MC_MIN_CHAINS} "
            f"(the recommended minimum).  At M<{_MC_MIN_CHAINS} the collinearity "
            "null-margin is unsafe (iid null f₁ can exceed the 0.70 threshold) and "
            "the unimodality gap-ratio for a perfect 2-cluster split falls below "
            "the detection threshold.  Use n_chains >= 6 (default 8) for reliable "
            "gate behaviour.",
            stacklevel=2,
        )
    _max_rank: int = _MAX_RANK_CAP if max_rank is None else max_rank
    # Per-chain step budget: total budget divided across M chains
    max_budget_steps_total: int = max(
        max_grad_budget // _ASSUMED_AVG_LEAPFROGS_PER_STEP, 1
    )
    max_budget_steps_per_chain: int = max(max_budget_steps_total // n_chains, 1)

    def init(n_dims: int) -> MultiChainMetaAdaptationCoreState:
        # buf_size: sized to hold the largest single-chain window.  With the
        # budget split across n_chains, each chain runs max_budget_steps_per_chain
        # steps, so buf_size is half that (growing-window largest window ≈ 40–50%).
        buf = min(max(max_budget_steps_per_chain // 2, 256), max_budget_steps_per_chain)
        buf = max(buf, 2 * (_max_rank + 1) * _MIN_TRAIN_K_RATIO)
        buf = min(buf, max_budget_steps_per_chain)
        actual_rank = min(_max_rank, max(n_dims // 2, 1), _MAX_RANK_CAP)
        return MultiChainMetaAdaptationCoreState(
            inverse_mass_matrix=LowRankInverseMassMatrix(
                sigma=jnp.ones(n_dims),
                U=jnp.zeros((n_dims, actual_rank)),
                lam=jnp.ones(actual_rank),
            ),
            mu_star=jnp.zeros(n_dims),
            draws_buffer=jnp.zeros((n_chains, buf, n_dims)),
            grads_buffer=jnp.zeros((n_chains, buf, n_dims)),
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
            converged_at_step=jnp.array(-1, dtype=jnp.int32),
            prev_lam=jnp.ones(actual_rank, dtype=jnp.float32),
            airm_vel_prev=jnp.array(float("inf"), dtype=jnp.float32),
            airm_vel_curr=jnp.array(float("inf"), dtype=jnp.float32),
            is_slow_mixing=jnp.zeros((), dtype=jnp.bool_),
            chain_collinearity=jnp.array(float("nan"), dtype=jnp.float32),
            unimodality_passed=jnp.ones((), dtype=jnp.bool_),  # True until first window
            deferred_to_ensemble=jnp.zeros((), dtype=jnp.bool_),
            # v2.1 new fields — NaN / 0 until first window
            within_lam1=jnp.array(float("nan"), dtype=jnp.float32),
            chain_consistency_psi=jnp.array(float("nan"), dtype=jnp.float32),
            r1_top=jnp.array(float("nan"), dtype=jnp.float32),
            detection_branch=jnp.array(_DETECTION_BRANCH_NONE, dtype=jnp.int32),
            unimodality_flag_count=jnp.zeros((), dtype=jnp.int32),
        )

    def update(
        state: MultiChainMetaAdaptationCoreState,
        positions: Array,
        grads: Array,
    ) -> MultiChainMetaAdaptationCoreState:
        """Accumulate one step of M chains into the per-chain buffers.

        Parameters
        ----------
        positions
            Shape ``(n_chains, d)`` — flat 2-D array, one row per chain.
        grads
            Shape ``(n_chains, d)`` — flat 2-D array, one row per chain.
        """
        # positions and grads are already flat (n_chains, d) arrays.
        # Ravel each chain's row in case a pytree is passed (no-op for 1-D rows).
        B = state.draws_buffer.shape[1]  # buf_size, static
        idx = state.buffer_idx % B
        col0 = jnp.zeros((), dtype=idx.dtype)

        def _write_chain(draws_m: Array, grads_m: Array, pos_m: Array, grad_m: Array):
            pos_flat, _ = fu.ravel_pytree(pos_m)
            grad_flat, _ = fu.ravel_pytree(grad_m)
            new_d = jax.lax.dynamic_update_slice(
                draws_m, pos_flat[None, :], (idx, col0)
            )
            new_g = jax.lax.dynamic_update_slice(
                grads_m, grad_flat[None, :], (idx, col0)
            )
            return new_d, new_g

        new_draws_buffer, new_grads_buffer = jax.vmap(_write_chain)(
            state.draws_buffer, state.grads_buffer, positions, grads
        )

        return state._replace(
            draws_buffer=new_draws_buffer,
            grads_buffer=new_grads_buffer,
            buffer_idx=state.buffer_idx + 1,
            budget_used=state.budget_used + n_chains,
        )

    def final(
        state: MultiChainMetaAdaptationCoreState,
    ) -> MultiChainMetaAdaptationCoreState:
        """Window-boundary controller: W-branch ∪ T-branch detection (v2.1).

        v2.1 two-branch union:
        - **W-branch (primary, deep spread):** pools per-chain-centered,
          within-chain-whitened residuals into a M(n−1)-dof spectrum; escalates
          when λ₁ clears the MP null edge AND the cross-chain consistency Ψ
          confirms AND the oscillation screen passes AND R² ≥ 0.5.
        - **T-branch (pure spike):** v2's between-means detector kept for the
          fully-unmixed regime.  Unimodality guard recalibrated to q99 (4.54
          at M=8) with 2-window confirmation; latch is non-monotone.

        Router fix: per-chain-centered draws/grads in time-major layout (first
        n*M rows valid), replacing the v2 chain-major reshape that caused
        r2 ∈ {−1.96×10¹⁷, +1.000} on real multi-chain draws.
        """
        M_stat, B, d = state.draws_buffer.shape  # all static
        n = jnp.minimum(state.buffer_idx, jnp.int32(B))
        actual_rank = state.inverse_mass_matrix.U.shape[1]  # static

        # ---- Within-chain statistics ----
        chain_means, W_diag = _compute_within_chain_stats(state.draws_buffer, n)
        grand_mean = chain_means.mean(0)  # (d,)
        W_safe = jnp.maximum(W_diag, jnp.float32(1e-20))
        sigma_w_diag = jnp.sqrt(W_safe)

        # ---- T-branch: between-chain detection via M×M Gram ----
        dof = M_stat - 1  # rank of T (grand-mean constraint)
        edge_full = _mc_detection_edge(d, dof)  # (1+√(d/(M−1)))²
        edge_loo = _mc_detection_edge(d, max(dof - 1, 1))

        T_eigenvalues, V_top, f1 = _between_chain_detection(
            chain_means, W_diag, n, M_stat, d
        )
        T_top = T_eigenvalues[0]

        k_raw = (T_eigenvalues > jnp.float32(edge_full)).sum().astype(jnp.int32)
        k_new = jnp.minimum(
            jnp.minimum(k_raw, jnp.int32(max(dof - 1, 1))), jnp.int32(actual_rank)
        )

        t_magnitude = T_top > jnp.float32(edge_full)
        t_collinearity = f1 >= jnp.float32(_MC_COLLINEARITY_TOL)
        t_loo = _loo_detection_passes(chain_means, W_diag, n, M_stat, d, edge_loo)
        t_support = k_new >= jnp.int32(1)

        # ---- T-branch Gate 5: unimodality (gap-stat, corroborator after BLOCKER-3) ----
        # Unwhiten top T direction: V_top[:, 0] is in whitened space → unwhiten by sigma_w
        e_unnorm = sigma_w_diag * V_top[:, 0]  # (d,) in original space
        e_norm = jnp.linalg.norm(e_unnorm)
        e_dir = e_unnorm / jnp.maximum(e_norm, jnp.float32(1e-10))
        is_unimodal, _gap_ratio = _unimodality_gap_stat(chain_means, e_dir, M_stat)
        # NOTE: new_flag_count is computed AFTER BLOCKER-3 (mode-consistency + contraction)
        # because the primary multimodality signal is any_mode_flag (mode-consistency),
        # not the gap-stat alone.  The flag counter uses multimodality_signal (either signal)
        # to count consecutive windows for the 2-window confirmation gate.

        # T pre-unimodality conjunction (four non-unimodality gates)
        t_pre_uni = t_magnitude & t_collinearity & t_loo & t_support

        # ---- Per-chain-centered pooled buffers in time-major layout ----
        # Fixes the v2 padding bug (chain-major zeros contaminated R² / Fisher paths)
        # and removes between-chain transient inflation from the curvature gate.
        pc_draws_tm, pc_grads_tm, step_mask_tm = _build_pc_centered_time_major_pool(
            state.draws_buffer, state.grads_buffer, chain_means, n, M_stat
        )
        n_pool = n.astype(jnp.int32) * jnp.int32(M_stat)
        # step_mask_all for geometric_mean_deploy_scale (time-major, repeat)
        step_mask = (jnp.arange(B) < n).astype(state.draws_buffer.dtype)  # (B,)
        step_mask_all = jnp.repeat(step_mask, M_stat)  # (B*M,)

        # Finite guard before SVD/eigh (sanitise NaN/Inf from near-zero W_diag dims)
        pc_draws_safe = jnp.where(
            jnp.isfinite(pc_draws_tm), pc_draws_tm, jnp.zeros_like(pc_draws_tm)
        )
        pc_grads_safe = jnp.where(
            jnp.isfinite(pc_grads_tm), pc_grads_tm, jnp.zeros_like(pc_grads_tm)
        )

        # ---- Fisher-LR metric on per-chain-centered pooled buffers ----
        # Used by the W-branch metric and by the T-branch sigma baseline.
        sigma_lr, mu_star_new, U_lr, lam_lr = _compute_low_rank_metric(
            pc_draws_safe, pc_grads_safe, n_pool, actual_rank, gamma, cutoff
        )

        # ---- R² curvature gate (per-chain-centered, time-major) ----
        # Per-chain centering removes between-chain variance, so the shared-slope
        # fit measures local metric-fixability (curvature), not global linearity.
        _, U_k_pooled = _compute_whitened_spectrum(
            pc_draws_safe, sigma_w_diag, n_pool, actual_rank
        )
        r2_new, mode_new = _compute_r2_score_linearity(
            pc_draws_safe, pc_grads_safe, sigma_w_diag, n_pool, U_k_pooled, actual_rank
        )
        # FOLD-IN 6: clip R² to [-10, 1] before any gate or carry.
        # Fit values below -10 are garbage (starved / degenerate data) and should
        # trigger the deferred path (NaN), not pollute the carry with extreme negatives.
        r2_new = jnp.where(
            r2_new < jnp.float32(-10.0),
            jnp.array(float("nan"), dtype=r2_new.dtype),
            jnp.clip(r2_new, max=jnp.float32(1.0)),
        )

        # W-branch r² gate uses the raw (pre-GAIN-override) R² so that a genuine
        # linear Gaussian with GAIN ≈ 0 in the projected tier is not blocked.
        # The W-branch question is "is the metric fixable?" — answered by the
        # original per-chain-centered projected fit, not by the slope-heterogeneity test.
        r2_gate_w = r2_new >= jnp.float32(_R_MIN)  # for W-branch only

        # BLOCKER-2: projected-tier GAIN+abstain router for T-branch routing.
        # When _R2_PROJECTED was taken (n_pool < 16d), the raw r2_new is
        # meaningless at k<<d (can be ≤ 0 for valid curvature targets → false
        # reparam_suggested in verdict; stoch_vol r2=-0.117, radon r2=-0.981).
        # Override: compute slope-heterogeneity GAIN = R²_perchain − R²_shared.
        #   abstain (NaN) → r2_routing = NaN → no T escalation / no reparam
        #   GAIN > 0.3 AND R²_pc ≥ 0.5 → r2_routing = R²_pc → T can escalate
        #   GAIN ≤ 0.3 (no evidence) → r2_routing = NaN → diagonal, no T action
        # W-branch is NOT gated on r2_routing (uses r2_gate_w from the raw R²).
        def _gain_r2_override() -> Array:
            gain_proj, r2_pc_proj = _compute_projected_gain_r2_mc(
                pc_draws_safe, pc_grads_safe, sigma_w_diag, n, M_stat, U_k_pooled
            )
            reparam_signal = (
                jnp.isfinite(gain_proj)
                & (gain_proj > jnp.float32(_GAIN_THRESHOLD))
                & (r2_pc_proj >= jnp.float32(_R_MIN))
            )
            return jnp.where(
                reparam_signal,
                r2_pc_proj,
                jnp.array(float("nan"), dtype=r2_new.dtype),
            )

        r2_routing = jax.lax.cond(
            mode_new == jnp.int32(_R2_PROJECTED),
            _gain_r2_override,
            lambda: r2_new,
        )
        r2_gate = r2_routing >= jnp.float32(
            _R_MIN
        )  # for T-branch + verdict; False when NaN

        # ---- W-branch: pooled within-chain whiteness detector ----
        lam1_w, top_eigvec_w = _compute_pooled_within_spectrum(
            state.draws_buffer, chain_means, W_diag, n, M_stat, actual_rank
        )
        w_edge = _w_branch_null_edge(M_stat, n, d)
        w_magnitude = lam1_w > w_edge

        psi_w = _compute_chain_consistency_psi(
            state.draws_buffer, chain_means, W_diag, n, M_stat
        )
        # FOLD-IN 4: adaptive Ψ threshold max(3*q99_null(M,n,d), 0.15).
        # The flat 0.15 causes 16-17% null leak at d=10; the calibrated table
        # corrects this without changing the floor for large d (sparse N).
        w_psi_thresh = _w_branch_psi_threshold(M_stat, n, d)
        w_psi_gate = psi_w > w_psi_thresh

        r1_w = _compute_lag1_autocorr_top_dir(
            state.draws_buffer, chain_means, W_diag, top_eigvec_w, n, M_stat
        )
        w_r1_gate = r1_w > jnp.float32(_W_BRANCH_R1_TOL)

        # ---- Budget deadline ----
        budget_remaining = jnp.int32(max_budget_steps_per_chain) - (
            state.budget_used.astype(jnp.int32) // jnp.int32(n_chains)
        )
        deadline_ok = budget_remaining >= 2 * k_new.astype(jnp.int32) + jnp.int32(
            _STEP_SIZE_READAPT_BUFFER
        )

        # ---- W-branch conjunction ----
        # No unimodality gate: per-chain centering makes W structurally mode-blind.
        # No support gate: W detects continuous spread, not discrete spike count.
        # Uses r2_gate_w (raw R², pre-GAIN-override) so that a genuine linear
        # Gaussian with low projected GAIN is not incorrectly blocked here.
        escalate_W = (
            ~state.has_escalated
            & w_magnitude
            & w_psi_gate
            & w_r1_gate
            & r2_gate_w
            & deadline_ok
        )

        # ---- BLOCKER-3: mode-consistency flag + contraction stat (T-branch guard) ----
        # Mode-consistency (BLOCKER-3): per direction e_j, flag iff
        # (R²_local(e_j) − R²_global(e_j) > 0.3) AND (R²_local(e_j) ≥ 0.5).
        # Replaces the gap-stat as the primary multimodality signal — k-mode-agnostic.
        any_mode_flag = _compute_mode_consistency_flag(
            pc_draws_safe,
            pc_grads_safe,
            state.grads_buffer,  # raw (M, B, d) for grand-centered R²_global
            chain_means,
            grand_mean,
            V_top,
            sigma_w_diag,
            T_eigenvalues,
            jnp.float32(edge_full),
            n,
            M_stat,
        )
        # Contraction stat (BLOCKER-3): per-chain split-half drift t.
        # t < -2.365 at M=8 → chains are converging → unimodal-safe, T can escalate.
        t_contr = _compute_contraction_stat(
            state.draws_buffer, chain_means, grand_mean, n, M_stat
        )
        _T_CONTR_CRIT = jnp.float32(-2.365)  # one-sided at α=2.5%, M=8 dof=7
        is_converging = t_contr < _T_CONTR_CRIT

        # ---- T-branch three-way (BLOCKER-3 fix, replaces binary gap-stat gate) ----
        # (i) Converging → chains drifting toward grand mean → unimodal-safe → escalate
        #     if pre-uni gates pass (override gap-stat).
        # (ii) Any mode_flag AND NOT converging → defer (genuine or uncertain multimodality).
        # (iii) Inconclusive (not converging, no mode flag) → existing gap-stat gate (is_unimodal).
        t_unimodality_override = (
            is_converging  # (i): converging overrides gap-stat block
        )
        t_unimodality_default = (
            is_unimodal & ~any_mode_flag
        )  # (iii): gap-stat + no flag
        t_unimodality = t_unimodality_override | t_unimodality_default

        mc_detection_T = t_pre_uni & t_unimodality
        escalate_T = ~state.has_escalated & r2_gate & mc_detection_T & deadline_ok

        # ---- Union escalation (computed before deferred to enable the gate) ----
        escalate_now = escalate_W | escalate_T
        new_has_escalated = state.has_escalated | escalate_now

        # ---- Scoped latch: deferred_to_ensemble (non-monotone, v2.1 rule) ----
        # Deferred is set when T-branch sees a spike (t_magnitude + t_support) AND
        # the mode-consistency flag fires (any_mode_flag), AND this has been true
        # for _MC_UNIMODALITY_CONFIRM_WINDOWS=2 consecutive windows, AND no T-escalation
        # fires this window.
        #
        # t_collinearity (f1 ≥ 0.7) is NOT in the defer gate: many-mode targets with
        # scattered eigenvalue mass (e.g. gmm, f1≈0.27) lose the P3 handoff if we gate
        # behind rank-1 collinearity.  Mode-consistency (any_mode_flag) already implies
        # an admitted T-spike; collinearity is irrelevant to a scattered-mode split.
        #
        # ~escalate_T (branch-scoped, not ~new_has_escalated): a W-escalation is legal
        # coexistence with defer; only a T-escalation (which requires t_unimodality=True)
        # is mutually exclusive with defer (confirmed_split resets flag_count → deferred=False).
        #
        # Non-monotone: recomputed each window; if mode flags clear, defer resets to False.
        # Impossible combo (route=low_rank ∧ deferred ∧ detection_branch=between_means)
        # remains impossible: T-escalation requires t_unimodality=True → any_mode_flag
        # drives flag_count to 0 via the False branch → confirmed_split=False. ✓
        multimodality_signal = any_mode_flag | ~is_unimodal
        new_flag_count = jnp.where(
            multimodality_signal,
            state.unimodality_flag_count + jnp.int32(1),
            jnp.int32(0),
        )
        # Confirmed split requires _MC_UNIMODALITY_CONFIRM_WINDOWS (=2) consecutive windows
        unimodality_confirmed_split = new_flag_count >= jnp.int32(
            _MC_UNIMODALITY_CONFIRM_WINDOWS
        )

        new_deferred = (
            t_magnitude  # T-spike above null edge
            & t_loo  # leave-one-out cross-chain validation
            & t_support  # at least one eigenvalue above edge
            & multimodality_signal  # any_mode_flag (primary) | ~is_unimodal (gap-stat corroborator)
            & unimodality_confirmed_split
            & r2_gate
            & ~escalate_T  # branch-scoped: W-escalation coexists; T-escalation precludes
        )
        new_escalation_rank = jnp.where(escalate_now, k_new, state.escalation_rank)

        # Detection branch code (carry from last firing window if no fire this window)
        branch_when_fires = jnp.where(
            escalate_W & escalate_T,
            jnp.int32(_DETECTION_BRANCH_BOTH),
            jnp.where(
                escalate_W,
                jnp.int32(_DETECTION_BRANCH_POOLED_WITHIN),
                jnp.int32(_DETECTION_BRANCH_BETWEEN_MEANS),
            ),
        )
        new_detection_branch = jnp.where(
            escalate_now, branch_when_fires, state.detection_branch
        )

        # ---- Metric selection ----
        # W fires: full Fisher-LR on per-chain-centered pooled buffers.
        # T fires (alone): Fisher-LR sigma + rank-1 geometric-mean slow-dir correction.
        # Neither fires post-escalation: carry the branch-appropriate LR metric.

        # T-branch escalated metric (v2 rank-1 update, unchanged)
        sigma_sq_deploy = _geometric_mean_deploy_scale(
            chain_means, pc_grads_safe, step_mask_all, grand_mean, e_dir, n_pool, M_stat
        )
        sigma_lr_e_sq = jnp.maximum(
            ((sigma_lr**2) * (e_dir**2)).sum(), jnp.float32(1e-20)
        )
        lam_slow = sigma_sq_deploy / sigma_lr_e_sq
        U_slow = e_dir[:, None]  # (d, 1)
        lam_slow_vec = jnp.concatenate(
            [lam_slow[None], jnp.ones(actual_rank - 1, dtype=lam_slow.dtype)]
        )
        t_lr_imm = LowRankInverseMassMatrix(
            sigma=sigma_lr,
            U=jnp.concatenate([U_slow, U_lr[:, 1:]], axis=1),
            lam=lam_slow_vec,
        )

        # W-branch metric: full Fisher-LR (all directions, no rank-1 truncation)
        w_lr_imm = LowRankInverseMassMatrix(sigma=sigma_lr, U=U_lr, lam=lam_lr)

        # Stay-diagonal metric (pre-escalation)
        diag_imm = LowRankInverseMassMatrix(
            sigma=sigma_w_diag,
            U=jnp.zeros((d, actual_rank), dtype=sigma_w_diag.dtype),
            lam=jnp.ones(actual_rank, dtype=sigma_w_diag.dtype),
        )

        # Select escalated metric: W fires (now or previously) → W metric; else T metric.
        prev_was_w = (
            new_detection_branch == jnp.int32(_DETECTION_BRANCH_POOLED_WITHIN)
        ) | (new_detection_branch == jnp.int32(_DETECTION_BRANCH_BOTH))
        escalated_imm = jax.lax.cond(prev_was_w, lambda: w_lr_imm, lambda: t_lr_imm)
        chosen_imm = jax.lax.cond(
            new_has_escalated, lambda: escalated_imm, lambda: diag_imm
        )
        chosen_mu = jax.lax.cond(
            new_has_escalated, lambda: mu_star_new, lambda: jnp.zeros_like(mu_star_new)
        )

        # ---- AIRM velocity proxy (on per-chain-centered pooled lam) ----
        lam_diff = jnp.linalg.norm(lam_lr - state.prev_lam.astype(lam_lr.dtype))
        lam_diff_f32 = lam_diff.astype(jnp.float32)
        new_airm_vel_prev = state.airm_vel_curr
        new_airm_vel_curr = jnp.where(
            new_has_escalated, lam_diff_f32, state.airm_vel_curr
        )
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

        return MultiChainMetaAdaptationCoreState(
            inverse_mass_matrix=chosen_imm,
            mu_star=chosen_mu,
            draws_buffer=jnp.zeros_like(state.draws_buffer),
            grads_buffer=jnp.zeros_like(state.grads_buffer),
            buffer_idx=jnp.zeros_like(state.buffer_idx),
            background_split=jnp.zeros_like(state.background_split),
            recompute_counter=jnp.zeros_like(state.recompute_counter),
            has_escalated=new_has_escalated,
            escalation_rank=new_escalation_rank,
            s_gap_prev=state.s_gap_curr,  # NaN chain; diagnostic compat
            s_gap_curr=jnp.array(float("nan"), dtype=jnp.float32),
            r2_latest=r2_routing.astype(jnp.float32),  # GAIN-corrected for verdict
            r2_mode=mode_new,
            budget_used=state.budget_used,
            converged_at_step=new_converged_at,
            prev_lam=lam_lr.astype(jnp.float32),
            airm_vel_prev=new_airm_vel_prev,
            airm_vel_curr=new_airm_vel_curr,
            is_slow_mixing=jnp.zeros((), dtype=jnp.bool_),
            chain_collinearity=f1,
            unimodality_passed=is_unimodal,
            deferred_to_ensemble=new_deferred,
            # v2.1 new fields
            within_lam1=lam1_w.astype(jnp.float32),
            chain_consistency_psi=psi_w.astype(jnp.float32),
            r1_top=r1_w.astype(jnp.float32),
            detection_branch=new_detection_branch,
            unimodality_flag_count=new_flag_count,
        )

    return MetricCore(init=init, update=update, final=final)
