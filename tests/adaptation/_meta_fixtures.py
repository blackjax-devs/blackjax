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
"""Shared test geometry helpers for the meta-adaptation test suite.

All buffer-generation helpers in one place so the test split files don't
duplicate them.
"""
import jax
import jax.numpy as jnp
import numpy as np

from blackjax.adaptation.meta._state import (
    MetaAdaptationCoreState,
    MultiChainMetaAdaptationCoreState,
)

_RNG_SEED = 424242


def _make_isotropic_buffer(n_dims: int, n: int, seed: int = _RNG_SEED):
    """Standard-normal draws with linear score (score = -position for N(0,I))."""
    key = jax.random.key(seed)
    k1, _ = jax.random.split(key)
    draws = jax.random.normal(k1, (n, n_dims))
    grads = -draws
    return draws, grads


def _make_curvature_buffer(n_dims: int, n: int, seed: int = _RNG_SEED):
    """Draws with random independent gradients: R²≈0 (curvature regime proxy).

    Real curvature targets (funnel, banana) have score-linearity R²≈0.007–0.09
    (optpath-B.md): the score is a highly non-linear function of position.
    Using independent random gradients is the minimal model for this:
    R² is exactly 0 by construction.
    """
    key = jax.random.key(seed)
    k1, k2 = jax.random.split(key)
    draws = jax.random.normal(k1, (n, n_dims))
    grads = jax.random.normal(k2, (n, n_dims))  # independent of draws → R²≈0
    return draws, grads


def _make_correlated_buffer(
    n_dims: int,
    n: int,
    rank: int = 2,
    lam_spike: float = 25.0,
    seed: int = _RNG_SEED,
):
    """Correlated Gaussian with rank-k spike: R²=1.0, S_gap >> _S_MIN.

    Σ = I + U*(lam_spike - 1)*Uᵀ, score = -Σ⁻¹ x (linear in position).
    After Welford-diagonal whitening (D = diag(Σ)), the whitened residual has
    a rank-k spike with eigenvalue ≈ lam_spike / max(diag(Σ)), giving S_gap
    well above _S_MIN for large lam_spike.
    """
    key = jax.random.key(seed)
    k1, k2 = jax.random.split(key)
    raw = jax.random.normal(k2, (n_dims, rank))
    U, _ = jnp.linalg.qr(raw)
    U = U[:, :rank]

    z = jax.random.normal(k1, (n, n_dims))
    z_orth = z - (z @ U) @ U.T
    draws = z_orth + jnp.sqrt(lam_spike) * (z @ U) @ U.T
    grads = -(draws - (1.0 - 1.0 / lam_spike) * (draws @ U) @ U.T)
    return draws, grads


def _make_marginal_sgap_curvature_buffer(
    n_dims: int,
    n: int,
    seed: int = _RNG_SEED,
):
    """Rank-1 spike with S_gap ∈ [_S_MIN, 2·_S_MIN) = [2.0, 4.0) + random grads.

    Uses lam_spike=4.5 (random non-axis-aligned direction, d=20, n=500).
    Measured with seed=42: top Welford-whitened eigenvalue ≈ 3.5, k_new=1,
    S_gap ≈ 2.94 ∈ [2, 4).  lam_spike=4.5 gives margins of ≥0.3 above _S_MIN
    across the seed sweep [1,7,13,17,21,37,42,55,63,100], making the fixture robust
    to seed variation (all tested seeds remain in band).

    Random grads (R²≈0) block escalation via the R² gate, leaving the S_gap in
    the MARGINAL band as the only reason for staying diagonal — the correct fixture
    for the 'stays-diag-marginal' decision row.

    The rank-1 (non-axis-aligned) spike is load-bearing: an axis-aligned spike is
    perfectly cancelled by Welford diagonal whitening (D^{-1/2} Sigma D^{-1/2} = I),
    giving S_gap=1.  A random direction leaks residual anisotropy into the whitened
    space, producing a measurable spike with a non-trivial k_new=1 cut.
    """
    key = jax.random.key(seed)
    k1, k2, k3 = jax.random.split(key, 3)
    # Rank-1 random direction (non-axis-aligned -> Welford whitening is imperfect)
    raw = jax.random.normal(k3, (n_dims, 1))
    U, _ = jnp.linalg.qr(raw)
    U = U[:, :1]

    z = jax.random.normal(k1, (n, n_dims))
    z_orth = z - (z @ U) @ U.T
    # lam_spike=4.5 gives S_gap ∈ [2.05, 2.94] across probed seeds; well clear of
    # both boundaries [_S_MIN=2.0, 2*_S_MIN=4.0).
    lam_spike = 4.5
    draws = z_orth + jnp.sqrt(lam_spike) * (z @ U) @ U.T
    grads = jax.random.normal(k2, (n, n_dims))  # independent of draws -> R2 approx 0
    return draws, grads


def _make_high_sgap_curvature_buffer(
    n_dims: int,
    n: int,
    rank: int = 2,
    lam_spike: float = 20.0,
    seed: int = _RNG_SEED,
):
    """Correlated draws (high S_gap) + random independent grads (R²≈0).

    Fixture for the funnel-refusal test: the S_gap gate passes (S_gap≥_S_MIN)
    but the R² gate blocks (R²≈0).  Proves the R² gate is the sole discriminator
    for non-linear-score targets.
    """
    key = jax.random.key(seed)
    k1, k2, k3 = jax.random.split(key, 3)
    raw = jax.random.normal(k3, (n_dims, rank))
    U, _ = jnp.linalg.qr(raw)
    U = U[:, :rank]

    z = jax.random.normal(k1, (n, n_dims))
    z_orth = z - (z @ U) @ U.T
    draws = z_orth + jnp.sqrt(lam_spike) * (z @ U) @ U.T
    grads = jax.random.normal(k2, (n, n_dims))  # independent of draws → R²≈0
    return draws, grads


def _fill_state_from_buffer(
    state: MetaAdaptationCoreState,
    draws: jax.Array,
    grads: jax.Array,
) -> MetaAdaptationCoreState:
    """Copy draws/grads into the state buffer (bypasses update() for direct testing)."""
    B, d = state.draws_buffer.shape
    n = draws.shape[0]
    n_fill = min(n, B)
    return state._replace(
        draws_buffer=jnp.concatenate(
            [draws[:n_fill], jnp.zeros((B - n_fill, d), dtype=draws.dtype)], axis=0
        ),
        grads_buffer=jnp.concatenate(
            [grads[:n_fill], jnp.zeros((B - n_fill, d), dtype=grads.dtype)], axis=0
        ),
        buffer_idx=jnp.array(n_fill, dtype=jnp.int32),
    )


def _fill_mc_state_from_buffers(
    state: MultiChainMetaAdaptationCoreState,
    draws_mc: jax.Array,
    grads_mc: jax.Array,
) -> MultiChainMetaAdaptationCoreState:
    """Copy (n_chains, n, d) draws/grads into a MultiChainMetaAdaptationCoreState.

    Analogous to :func:`_fill_state_from_buffer` for single-chain tests.
    """
    M, B, d = state.draws_buffer.shape
    n_fill = min(draws_mc.shape[1], B)
    new_draws = jnp.concatenate(
        [
            draws_mc[:, :n_fill, :],
            jnp.zeros((M, B - n_fill, d), dtype=draws_mc.dtype),
        ],
        axis=1,
    )
    new_grads = jnp.concatenate(
        [
            grads_mc[:, :n_fill, :],
            jnp.zeros((M, B - n_fill, d), dtype=grads_mc.dtype),
        ],
        axis=1,
    )
    return state._replace(
        draws_buffer=new_draws,
        grads_buffer=new_grads,
        buffer_idx=jnp.array(n_fill, dtype=jnp.int32),
    )


def _make_mc_correlated_buffers(
    n_dims: int,
    n: int,
    n_chains: int,
    rank: int = 2,
    lam_spike: float = 25.0,
    seed: int = _RNG_SEED,
) -> tuple[jax.Array, jax.Array]:
    """Correlated (spike) data replicated across all M chains.

    All chains receive draws from the SAME distribution, so their leading
    subspaces are aligned.  Used in projector-agreement-accepts tests.
    Returns ``(draws_mc, grads_mc)`` with shapes ``(n_chains, n, n_dims)``.
    """
    draws_1, grads_1 = _make_correlated_buffer(
        n_dims, n, rank=rank, lam_spike=lam_spike, seed=seed
    )
    draws_mc = jnp.stack([jnp.array(draws_1, dtype=jnp.float32)] * n_chains)
    grads_mc = jnp.stack([jnp.array(grads_1, dtype=jnp.float32)] * n_chains)
    return draws_mc, grads_mc


def _make_mc_misaligned_buffers(
    n_dims: int,
    n: int,
    n_chains: int,
    rank: int = 1,
    lam_spike: float = 25.0,
    seed: int = _RNG_SEED,
) -> tuple[jax.Array, jax.Array]:
    """Correlated spike data with a DIFFERENT spike direction per chain.

    Each chain has a rank-1 spike along an independent random direction, so
    within each chain the whitened spectrum shows a spike, but the leading
    subspaces are mutually near-orthogonal across chains.  Used in
    projector-agreement-rejects tests (agreement → 0) and in the oscillatory
    null test.
    Returns ``(draws_mc, grads_mc)`` with shapes ``(n_chains, n, n_dims)``.
    """
    key = jax.random.key(seed)
    draws_all = []
    grads_all = []
    for m in range(n_chains):
        k_m, key = jax.random.split(key)
        k_dir, k_data = jax.random.split(k_m)
        # Random unit vector (different per chain)
        raw = jax.random.normal(k_dir, (n_dims, rank))
        U_m, _ = jnp.linalg.qr(raw)
        U_m = U_m[:, :rank]
        z = jax.random.normal(k_data, (n, n_dims))
        z_orth = z - (z @ U_m) @ U_m.T
        draws_m = z_orth + jnp.sqrt(lam_spike) * (z @ U_m) @ U_m.T
        # Linear score for this chain's distribution
        grads_m = -(draws_m - (1.0 - 1.0 / lam_spike) * (draws_m @ U_m) @ U_m.T)
        draws_all.append(jnp.array(draws_m, dtype=jnp.float32))
        grads_all.append(jnp.array(grads_m, dtype=jnp.float32))
    return jnp.stack(draws_all), jnp.stack(grads_all)


def _make_overdispersed_slow_chains(
    n_dims: int,
    n: int,
    n_chains: int,
    slow_offset_scale: float = 5.0,
    within_chain_noise: float = 0.1,
    slow_var: float = 25.0,
    seed: int = _RNG_SEED,
) -> tuple[jax.Array, jax.Array]:
    """M chains overdispersed along one slow direction with true target gradients.

    Each chain m is stuck near ``offset_m * e_slow`` where offsets are evenly
    spaced over ``[-slow_offset_scale, +slow_offset_scale]``.  Within-chain
    variance is small (``within_chain_noise``).  Gradients are the TRUE score of
    the target ``N(0, Σ)`` where ``Σ = I + (slow_var−1) * e_slow @ e_slow^T``,
    so R² is high.

    The between-chain scatter of means is large and rank-1 along ``e_slow``, so
    the between-chain detection matrix T fires and the collinearity gate passes.
    The projected chain-means are uniformly spaced (unimodal) so the unimodality
    gate passes.

    Returns ``(draws_mc, grads_mc)`` with shapes ``(n_chains, n, n_dims)``.
    """
    key = jax.random.key(seed)
    k_dir, k_data = jax.random.split(key)
    raw = jax.random.normal(k_dir, (n_dims,))
    e_slow = raw / jnp.linalg.norm(raw)

    # Precision correction for the slow direction:
    # Σ^{-1} = I + (1/slow_var − 1) * e_slow @ e_slow^T
    prec_corr = 1.0 / slow_var - 1.0  # negative

    offsets = np.linspace(-slow_offset_scale, slow_offset_scale, n_chains)

    draws_all = []
    grads_all = []
    for m in range(n_chains):
        k_m = jax.random.fold_in(k_data, m)
        mu_m = float(offsets[m]) * e_slow
        noise = jax.random.normal(k_m, (n, n_dims)) * within_chain_noise
        draws_m = noise + mu_m[None, :]
        # True score: −Σ^{-1} x = −(x + prec_corr * (x @ e_slow) * e_slow)
        x_proj = draws_m @ e_slow  # (n,)
        grads_m = -(draws_m + prec_corr * x_proj[:, None] * e_slow[None, :])
        draws_all.append(jnp.array(draws_m, dtype=jnp.float32))
        grads_all.append(jnp.array(grads_m, dtype=jnp.float32))
    return jnp.stack(draws_all), jnp.stack(grads_all)


def _make_mode_split_chains(
    n_dims: int,
    n: int,
    n_chains: int,
    mode_separation: float = 8.0,
    within_chain_noise: float = 0.1,
    slow_var: float = 25.0,
    seed: int = _RNG_SEED,
) -> tuple[jax.Array, jax.Array]:
    """Half of M chains near mode A, half near mode B along the same axis.

    Uses the true linear score for the unimodal target ``N(0, Σ)`` with
    ``Σ = I + (slow_var−1) * e_mode @ e_mode^T``, so the magnitude, collinearity,
    leave-one-out, and R² gates all pass.  But the projected chain-means are
    bimodal (half at ``−mode_separation/2``, half at ``+mode_separation/2``),
    so the gap-stat unimodality guard fires and blocks escalation.

    Requires ``n_chains ≥ 4`` with even ``n_chains`` for the split to have
    ``≥ 2`` chains per cluster, ensuring LOO coverage.

    Returns ``(draws_mc, grads_mc)`` with shapes ``(n_chains, n, n_dims)``.
    """
    assert n_chains % 2 == 0, "n_chains must be even for mode-split fixture"
    key = jax.random.key(seed)
    k_dir, k_data = jax.random.split(key)
    raw = jax.random.normal(k_dir, (n_dims,))
    e_mode = raw / jnp.linalg.norm(raw)

    prec_corr = 1.0 / slow_var - 1.0

    half = n_chains // 2
    draws_all = []
    grads_all = []
    for m in range(n_chains):
        k_m = jax.random.fold_in(k_data, m)
        # First half near −a, second half near +a
        offset = -mode_separation / 2.0 if m < half else mode_separation / 2.0
        mu_m = offset * e_mode
        noise = jax.random.normal(k_m, (n, n_dims)) * within_chain_noise
        draws_m = noise + mu_m[None, :]
        # True score for the unimodal target: −Σ^{-1} x (linear in x)
        x_proj = draws_m @ e_mode
        grads_m = -(draws_m + prec_corr * x_proj[:, None] * e_mode[None, :])
        draws_all.append(jnp.array(draws_m, dtype=jnp.float32))
        grads_all.append(jnp.array(grads_m, dtype=jnp.float32))
    return jnp.stack(draws_all), jnp.stack(grads_all)


def _make_underdispersed_chains(
    n_dims: int,
    n: int,
    n_chains: int,
    within_chain_noise: float = 0.1,
    seed: int = _RNG_SEED,
) -> tuple[jax.Array, jax.Array]:
    """All M chains starting near the origin (under-dispersed starts).

    Chain means are all approximately zero; the between-chain scatter is
    structurally near zero.  The detector cannot see the slow direction
    because all chains sampled the same basin — but this is one-sided safe
    (never a dangerous over-escalation, just conservative under-detection).

    Returns ``(draws_mc, grads_mc)`` with shapes ``(n_chains, n, n_dims)``.
    """
    draws_all = []
    grads_all = []
    for m in range(n_chains):
        k_m = jax.random.fold_in(jax.random.key(seed), m)
        noise = jax.random.normal(k_m, (n, n_dims)) * within_chain_noise
        grads_m = -noise / (within_chain_noise**2)  # score for N(0, noise²·I)
        draws_all.append(jnp.array(noise, dtype=jnp.float32))
        grads_all.append(jnp.array(grads_m, dtype=jnp.float32))
    return jnp.stack(draws_all), jnp.stack(grads_all)


# ---------------------------------------------------------------------------
# W-branch and detection test helpers (used by test_meta_detection.py)
# ---------------------------------------------------------------------------


def _make_mc_deep_spread(M, n, d, lam_within=20.0, seed=_RNG_SEED):
    """M chains each with the same within-chain slow direction, all centered at 0.

    Generates the ill_cond-like scenario: W-branch should detect anisotropy
    (lam1 >> w_edge) but T-branch stays silent (f1 ≈ 0 since chain means ≈ 0).
    Returns arrays of shape (M, n, d).
    """
    key = jax.random.key(seed)
    k_dir, k_chains = jax.random.split(key)
    u_raw = jax.random.normal(k_dir, (d,))
    u = u_raw / jnp.linalg.norm(u_raw)
    chain_keys = jax.random.split(k_chains, M)
    draws_list, grads_list = [], []
    for m in range(M):
        z = jax.random.normal(chain_keys[m], (n, d))
        z_orth = z - (z @ u[:, None]) @ u[None, :]
        x = z_orth + jnp.sqrt(jnp.float32(lam_within)) * (z @ u[:, None]) @ u[None, :]
        g = -(x - (1.0 - 1.0 / lam_within) * (x @ u[:, None]) @ u[None, :])
        draws_list.append(x)
        grads_list.append(g)
    return jnp.stack(draws_list), jnp.stack(grads_list)


def _make_mc_isotropic(M, n, d, seed=_RNG_SEED):
    """M chains, each isotropic N(0, I): W-branch should NOT fire."""
    key = jax.random.key(seed)
    chain_keys = jax.random.split(key, M)
    draws_list, grads_list = [], []
    for m in range(M):
        x = jax.random.normal(chain_keys[m], (n, d))
        draws_list.append(x)
        grads_list.append(-x)
    return jnp.stack(draws_list), jnp.stack(grads_list)


def _make_mc_split_means(M, n, d, split_scale=10.0, seed=_RNG_SEED):
    """Half-and-half bimodal chains: first M//2 at -split_scale, rest at +split_scale.

    Triggers gap-stat flagging (gap_ratio >> threshold) when projected onto e0.
    """
    key = jax.random.key(seed)
    chain_keys = jax.random.split(key, M)
    draws_list, grads_list = [], []
    for m in range(M):
        mean = split_scale * (1.0 if m < M // 2 else -1.0)
        center = jnp.zeros(d).at[0].set(mean)
        x = jax.random.normal(chain_keys[m], (n, d)) * 0.5 + center
        g = -x
        draws_list.append(x)
        grads_list.append(g)
    return jnp.stack(draws_list), jnp.stack(grads_list)


def _make_mc_even_spread(M, n, d, spread_scale=5.0, seed=_RNG_SEED):
    """M chains with evenly-spaced means (unimodal): gap-stat should NOT flag."""
    key = jax.random.key(seed)
    chain_keys = jax.random.split(key, M)
    means = jnp.linspace(-spread_scale, spread_scale, M)
    draws_list, grads_list = [], []
    for m in range(M):
        center = jnp.zeros(d).at[0].set(means[m])
        x = jax.random.normal(chain_keys[m], (n, d)) * 0.3 + center
        g = -x
        draws_list.append(x)
        grads_list.append(g)
    return jnp.stack(draws_list), jnp.stack(grads_list)


def _make_mc_ar_null(M, n, d, rho=0.8, seed=_RNG_SEED):
    """M independent AR(1) chains: magnitude fires but Ψ refuses (F4).

    Each chain: x_t = rho * x_{t-1} + sqrt(1-rho^2) * eps_t, eps_t ~ N(0,I).
    The within-chain pooled spectrum is inflated (lam1 >> edge due to
    autocorrelation).  But cross-chain consistency Ψ ≈ 0: each chain's AR
    noise is independent, so the off-diagonal correlation matrices C_A and C_B
    (between chain-halves) are near-zero for isotropic AR.  This proves Ψ is
    load-bearing — without it the W-branch would fire false positives on any
    well-mixed chain with slow mixing.

    Use d >= 26 where the adaptive Ψ threshold exceeds the flat 0.15 floor.
    """
    key = jax.random.key(seed)
    chain_keys = jax.random.split(key, M)
    noise_scale = float((1.0 - rho**2) ** 0.5)
    draws_list, grads_list = [], []
    for m in range(M):
        eps = jax.random.normal(chain_keys[m], (n, d)).astype(jnp.float32)
        # Build AR(1) via Python-level loop (fixture construction only)
        rows = [jnp.zeros(d, dtype=jnp.float32)]
        for t in range(1, n):
            rows.append(rho * rows[-1] + noise_scale * eps[t])
        x = jnp.stack(rows)  # (n, d)
        draws_list.append(x)
        grads_list.append(-x)  # N(0,I) stationary score
    return jnp.stack(draws_list), jnp.stack(grads_list)


def _make_mc_multi_spread(M, n, d, n_dirs=5, lam_within=10.0, seed=_RNG_SEED):
    """M chains each with n_dirs COMPARABLE slow directions (f1 ≈ 1/n_dirs).

    Unlike _make_mc_deep_spread (rank-1 spike, f1 ≈ 0.65, NOT the W-branch
    target), this represents the GENUINE W-branch target: continuous spread
    across multiple directions with no single dominant eigenvalue.

    Construction: lam_within applied equally to all n_dirs directions → all
    slow-direction eigenvalues equal → f1 = 1/n_dirs by construction.

    Use this fixture for W-branch detection tests; _make_mc_deep_spread is
    retained for T-branch and single-chain compatibility tests.
    """
    key = jax.random.key(seed)
    k_dir, k_chains = jax.random.split(key)
    raw = jax.random.normal(k_dir, (d, n_dirs))
    U, _ = jnp.linalg.qr(raw)
    U = U[:, :n_dirs]  # (d, n_dirs) orthonormal slow directions

    chain_keys = jax.random.split(k_chains, M)
    draws_list, grads_list = [], []
    for m in range(M):
        z = jax.random.normal(chain_keys[m], (n, d))
        z_proj = z @ U  # (n, n_dirs)
        z_orth = z - z_proj @ U.T  # (n, d)
        x = z_orth + jnp.sqrt(jnp.float32(lam_within)) * z_proj @ U.T
        # Score: -(I + (lam-1)*U*U^T)^{-1} x = -(x - (1-1/lam)*U*U^T*x)
        g = -(x - (1.0 - 1.0 / lam_within) * (x @ U) @ U.T)
        draws_list.append(x)
        grads_list.append(g)
    return jnp.stack(draws_list), jnp.stack(grads_list)


def _make_mc_coexistence(M, n, d, lam_within=20.0, split_scale=5.0, seed=_RNG_SEED):
    """W fires (within-chain anisotropy) AND any_mode_flag fires (modal split grads).

    Used for the scoped-latch coexistence test: a combined fixture where both the
    W-branch and the T-branch defer gate can be active simultaneously.

    Construction:
    - Within-chain slow direction u_within along dims 1..d-1 (orthogonal to e_0).
    - Split means: first M//2 chains at +split_scale*e_0, rest at -split_scale*e_0.
    - Mode-specific gradients: g = -Sigma^{-1}@(x - center_m), which gives
      GAIN = R²_local - R²_global ≈ 1 >> 0.3 so any_mode_flag fires.

    Why modal grads matter: g=-x (global N(0,I) gradient) gives R²_local≈R²_global≈1
    → GAIN≈0 → any_mode_flag=False.  Mode-specific g=-prec@(x-center_m) decouples
    per-chain from global fit, making R²_global low along the split direction.
    """
    key = jax.random.key(seed)
    k_dir, k_chains = jax.random.split(key)

    # Slow direction orthogonal to e_0 (split direction) so the two effects are separable
    u_raw = jax.random.normal(k_dir, (d,)).at[0].set(0.0)
    u = u_raw / jnp.linalg.norm(u_raw)  # unit vector in dims 1..d-1

    chain_keys = jax.random.split(k_chains, M)
    draws_list, grads_list = [], []
    for m in range(M):
        sign = 1.0 if m < M // 2 else -1.0
        center = jnp.zeros(d).at[0].set(sign * split_scale)
        z = jax.random.normal(chain_keys[m], (n, d))
        # Scale along u_within → within-chain lam_within spike (W fires)
        proj_u = (z @ u)[:, None] * u[None, :]
        z_aniso = z - proj_u + jnp.sqrt(jnp.float32(lam_within)) * proj_u
        x = z_aniso + center
        # Mode-specific gradient: score of N(center_m, Sigma) where Sigma = I + (lam-1)u uT
        # score = -(x - center) + (1 - 1/lam) * ((x-center) @ u) * u = -prec @ (x - center)
        z_m = x - center
        proj_zm = (z_m @ u)[:, None] * u[None, :]
        g = -(z_m - (1.0 - 1.0 / lam_within) * proj_zm)
        draws_list.append(x)
        grads_list.append(g)
    return jnp.stack(draws_list), jnp.stack(grads_list)


def _fill_mc_state(
    state: MultiChainMetaAdaptationCoreState,
    draws_mc: jax.Array,
    grads_mc: jax.Array,
) -> MultiChainMetaAdaptationCoreState:
    """Copy (M, n, d) draws/grads into the MultiChain state buffer.

    Alias for :func:`_fill_mc_state_from_buffers` used in W-branch tests.
    """
    M, B, d = state.draws_buffer.shape
    n = draws_mc.shape[1]
    n_fill = min(n, B)
    draws_buf = jnp.concatenate(
        [draws_mc[:, :n_fill, :], jnp.zeros((M, B - n_fill, d), dtype=draws_mc.dtype)],
        axis=1,
    )
    grads_buf = jnp.concatenate(
        [grads_mc[:, :n_fill, :], jnp.zeros((M, B - n_fill, d), dtype=grads_mc.dtype)],
        axis=1,
    )
    return state._replace(
        draws_buffer=draws_buf,
        grads_buffer=grads_buf,
        buffer_idx=jnp.array(n_fill, dtype=jnp.int32),
    )
