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
"""Pure metric estimator functions for the low-rank metric adaptation layer.

Each function is a **pure transformation**: explicit arrays in → metric
representation out.  No buffer state, no scheduling logic, no side effects.
All functions are JAX-traceable and safe to use inside ``jax.lax.scan`` /
``jax.vmap`` provided that ``max_rank`` (where applicable) is a static
Python integer (required to determine output shapes).

**Source lineage** (as of 2026-07-11, main @ 532631c1):

+--------------------------------------+--------------------------------------------+
| Estimator                            | Extracted from                             |
+======================================+============================================+
| :func:`fisher_score_low_rank`        | ``low_rank_adaptation._compute_low_rank``  |
|                                      | ``_metric`` + ``_spd_mean``                |
+--------------------------------------+--------------------------------------------+
| :func:`draws_singular_value_low_rank`| ``mclmc_lrd_adaptation``                   |
|                                      | ``._extract_lrd_from_samples``             |
+--------------------------------------+--------------------------------------------+
| :func:`sample_covariance_eigh_low``  | ``meads_adaptation``                       |
| ``_rank`                             | ``._lrd_from_accumulated_covariance``      |
+--------------------------------------+--------------------------------------------+
| :func:`welford_diagonal`             | ``mass_matrix.welford_algorithm``          |
| :func:`welford_dense`                | (thin scan-wrappers; algorithm untouched)  |
+--------------------------------------+--------------------------------------------+
| :func:`fisher_score_diagonal`        | branch b197f1e2                            |
|                                      | ``mass_matrix._fisher_diagonal_inverse``   |
|                                      | ``_mass`` (dormant; wiring stays dormant)  |
+--------------------------------------+--------------------------------------------+
| :func:`sample_variance_diagonal`     | ``mclmc_adaptation.py:342`` and            |
|                                      | ``adjusted_mclmc_adaptation.py:375``       |
|                                      | (verbatim duplicate; one function here)    |
+--------------------------------------+--------------------------------------------+

**Composition note:** estimators, data-feeding policy, and schedule are
co-adapted package components.  The functions here are the *estimator*
component only — callers supply the buffer view.  Gating logic (support gates,
fraction-window checks, 2·d thresholds) belongs in the caller; it is
explicitly *not* implemented here (docstrings note the relevant gates for each
estimator).

**Registration:** these functions are module-public (importable from
``blackjax.adaptation.metric_estimators``) but are NOT exported at the
``blackjax`` top-level.  Top-level export and consumer re-wiring are
deliberate follow-up work.
"""

from typing import Literal

import jax
import jax.numpy as jnp

from blackjax.adaptation.mass_matrix import welford_algorithm
from blackjax.mcmc.metrics import LowRankInverseMassMatrix
from blackjax.types import Array

# ---------------------------------------------------------------------------
# SPD utilities (moved from low_rank_adaptation; it imports from here)
# ---------------------------------------------------------------------------


def _relative_pd_floor(vals: Array) -> Array:
    """Machine-epsilon floor, SCALED to ``vals``' own magnitude.

    An absolute floor (e.g. a bare ``jnp.finfo(dtype).eps``) is wrong here:
    this module's SPD matrices routinely span many orders of magnitude --
    e.g. ``C_a = P_a P_a^T / gamma + I`` scales like ``O(n / gamma)``
    (``~1e10`` at ``n=50_000`` draws, default ``gamma=1e-5``), so
    ``inv(C_a)``'s eigenvalues legitimately live around ``~1e-10`` -- an
    absolute ``eps`` (``~1.2e-7`` for float32) would incorrectly clamp that
    perfectly-conditioned small eigenvalue UP by several orders of
    magnitude, corrupting the result (caught by
    ``LowRankDiagonalConsistencyTest``: a bare absolute floor turned the 1D
    case's correct ``lam=1.0`` into ``lam=34.6``). Flooring relative to the
    largest eigenvalue IN THE SAME SPECTRUM correctly leaves a
    uniformly-small-but-well-conditioned spectrum untouched while still
    catching a genuinely near-zero-relative-to-its-own-scale eigenvalue
    (the actual rounding-noise failure mode the PD guard targets).

    **Moved from** ``blackjax.adaptation.low_rank_adaptation``;
    ``low_rank_adaptation`` now imports from here.
    """
    scale = jnp.maximum(jnp.max(jnp.abs(vals)), jnp.finfo(vals.dtype).tiny)
    return jnp.finfo(vals.dtype).eps * scale


def _spd_mean(A: Array, B: Array) -> Array:
    """Symmetric positive-definite (AIRM) geometric mean of A and B.

    Computes :math:`A \\#_{1/2} B = B^{1/2}(B^{-1/2}AB^{-1/2})^{1/2}B^{1/2}`
    via the eigendecomposition of B (the gradient covariance), following the
    nutpie convention.  Both matrices must be SPD with shape ``(k, k)``.

    **PD guard** (round-9 schedule-port audit, GAP-2): both intermediate
    eigenspectra are floored at :func:`_relative_pd_floor` rather than
    ``0.0``. nuts-rs's own unit test (``low_rank.rs::test_estimate_mass_matrix``)
    *asserts* this pipeline returns strictly-positive eigenvalues -- it is
    PD by construction in exact arithmetic, since ``A``/``B`` are each
    ``P P^T / gamma + I``. The float32 audit found that rounding in this
    eigendecomposition-heavy pipeline (condition number up to ``~1/gamma``)
    can nonetheless produce small negative eigenvalues that silently make
    the whole geometric mean indefinite; a *scale-relative* floor (see
    :func:`_relative_pd_floor`'s docstring for why an absolute one is wrong
    here) fixes this without perturbing legitimately-informative
    eigenvalues, matching nuts-rs's own PD-by-construction invariant.

    **Moved from** ``blackjax.adaptation.low_rank_adaptation``;
    ``low_rank_adaptation`` now imports from here.
    """
    # Eigendecompose B: B = V_b D_b V_b^T
    vals_b, vecs_b = jnp.linalg.eigh(B)
    vals_b = jnp.maximum(vals_b, _relative_pd_floor(vals_b))
    sqrt_b = jnp.sqrt(vals_b)
    inv_sqrt_b = 1.0 / sqrt_b  # vals_b > 0 (floored), so this never divides by 0

    # M = B^{-1/2} A B^{-1/2} in B's eigenbasis
    tmp = vecs_b.T @ A @ vecs_b  # (k, k)
    M = inv_sqrt_b[:, None] * tmp * inv_sqrt_b[None, :]

    vals_m, vecs_m = jnp.linalg.eigh(M)
    vals_m = jnp.maximum(vals_m, _relative_pd_floor(vals_m))
    sqrt_m = jnp.sqrt(vals_m)

    # A # B = B^{1/2} M^{1/2} B^{1/2}
    # = (V_b diag(sqrt_b) V_m) diag(sqrt_m) (V_b diag(sqrt_b) V_m)^T
    W = vecs_b @ (sqrt_b[:, None] * vecs_m)  # (k, k)
    return (W * sqrt_m[None, :]) @ W.T


__all__ = [
    "eigenvalue_informativeness",
    "select_top_eigenvalues_by_informativeness",
    "fisher_score_low_rank",
    "draws_singular_value_low_rank",
    "sample_covariance_eigh_low_rank",
    "welford_diagonal",
    "welford_dense",
    "fisher_score_diagonal_from_moments",
    "fisher_score_diagonal",
    "sample_variance_diagonal",
]


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------


def eigenvalue_informativeness(eigenvalues: Array) -> Array:
    r"""Informativeness of each eigenvalue: how far it deviates from isotropic.

    Computes :math:`|\lambda - 1|` for each entry of ``eigenvalues``.
    Directions with large informativeness deviate the most from the identity
    metric and benefit the most from low-rank preconditioning.

    This idiom is used identically in:

    * :func:`fisher_score_low_rank` (step 9 — top-k selection in projected
      subspace; see ``low_rank_adaptation._compute_low_rank_metric`` :468).
    * :func:`draws_singular_value_low_rank` (top-k SVD eigenvalue selection;
      see ``mclmc_lrd_adaptation._extract_lrd_from_samples`` :271).
    * :func:`sample_covariance_eigh_low_rank` (top-k eigh eigenvalue selection;
      see ``meads_adaptation._lrd_from_accumulated_covariance`` :309).

    Parameters
    ----------
    eigenvalues
        Shape ``(q,)``.  Eigenvalues of a correlation (or preconditioned)
        matrix.  An isotropic Gaussian has all eigenvalues equal to 1 and
        informativeness 0.

    Returns
    -------
    Array, shape ``(q,)``
        :math:`|\lambda_i - 1|` for each :math:`\lambda_i`.
    """
    return jnp.abs(eigenvalues - 1.0)


def select_top_eigenvalues_by_informativeness(
    eigenvalues: Array,
    eigenvectors: Array,
    max_rank: int,
    *,
    tail_handling: Literal["mask_pad", "raw"] = "mask_pad",
    cutoff: float = 2.0,
) -> tuple[Array, Array]:
    r"""Select the top-``max_rank`` eigenpairs by :func:`eigenvalue_informativeness`.

    Two production consumers use this pattern with **divergent tail handling**:

    ``tail_handling="mask_pad"`` (default)
        Used by :func:`fisher_score_low_rank`.  Eigenvalues inside the
        uninformative band :math:`[1/\text{cutoff},\ \text{cutoff}]` are
        **masked to 1** (they carry no preconditioning benefit), making the
        corresponding direction a no-op in the metric.  When the input
        subspace has fewer than ``max_rank`` eigenvectors (``q < max_rank``,
        which arises when ``d < 2 · max_rank``), the output is **zero-padded**
        to the requested shape — zero columns in ``U`` and ``lam=1`` entries
        — so the return shape is always ``(d, max_rank)`` / ``(max_rank,)``.
        Matches ``low_rank_adaptation._compute_low_rank_metric`` steps 8–9
        (:468–484).

    ``tail_handling="raw"``
        Used by :func:`draws_singular_value_low_rank` and
        :func:`sample_covariance_eigh_low_rank`.  The top-k pairs are
        returned **as-is**: no informativeness masking, no padding.  The
        caller must ensure ``max_rank <= q``; behaviour is undefined (JAX
        will silently return ``q`` columns) when ``max_rank > q``.  Matches
        ``mclmc_lrd_adaptation._extract_lrd_from_samples`` :271–275 and
        ``meads_adaptation._lrd_from_accumulated_covariance`` :309–312.

        **Tie-break divergence:** the two modes differ not only in masking
        but also in sort stability for exact ``|λ−1|`` ties:
        ``"mask_pad"`` uses ``argsort(-scores)`` (ascending original-index
        among ties; Fisher consumer convention); ``"raw"`` uses
        ``argsort(scores)[::-1]`` (descending original-index; both raw-source
        conventions).  On real ``eigh``/``svd`` spectra bit-exact ties do
        not occur — even truly degenerate inputs return ulp-broken eigenvalues
        — so the difference is behaviorally inert on continuous data; the raw
        path preserves byte-fidelity to its sources nonetheless.

    Parameters
    ----------
    eigenvalues
        Shape ``(q,)``.  Must be the eigenvalues associated with the columns
        of ``eigenvectors``.
    eigenvectors
        Shape ``(d, q)``.  Columns are the eigenvectors.
    max_rank
        Number of eigenpairs to return.  Must be a **static Python integer**
        (determines the output shape).
    tail_handling
        See above.  Default ``"mask_pad"`` matches the Fisher-score consumer.
    cutoff
        Only used when ``tail_handling="mask_pad"``.  Eigenvalues in
        :math:`[1/\text{cutoff},\ \text{cutoff}]` are masked to 1.
        Default ``2.0`` matches nutpie's ``c=2``.

    Returns
    -------
    U_out : Array, shape ``(d, max_rank)`` (mask_pad) or ``(d, ≤max_rank)`` (raw)
        Selected (and possibly masked/padded) eigenvectors.
    lam_out : Array, shape ``(max_rank,)`` or ``(≤max_rank,)``
        Selected (and possibly masked) eigenvalues.
    """
    if tail_handling not in ("mask_pad", "raw"):
        raise ValueError(
            f"tail_handling must be 'mask_pad' or 'raw', got {tail_handling!r}"
        )

    q = eigenvalues.shape[0]
    scores = eigenvalue_informativeness(eigenvalues)

    if tail_handling == "mask_pad":
        # argsort(-scores): ascending original-index tie-break — matches the
        # fisher consumer (low_rank_adaptation._compute_low_rank_metric :468).
        order = jnp.argsort(-scores)
        actual_rank = min(max_rank, q)  # static: both are Python ints at trace time
        top_order = order[:actual_rank]
        U_out = eigenvectors[:, top_order]  # (d, actual_rank)
        lam_raw = eigenvalues[top_order]  # (actual_rank,)

        # Mask eigenvalues inside the uninformative band to 1.0.
        # nutpie: keep λ < 1/cutoff or λ > cutoff; others set to 1.
        is_informative = (lam_raw < 1.0 / cutoff) | (lam_raw > cutoff)
        lam_out = jnp.where(is_informative, lam_raw, 1.0)

        # Zero-pad to max_rank when the input subspace is smaller.
        if actual_rank < max_rank:
            d = eigenvectors.shape[0]
            pad = max_rank - actual_rank
            U_out = jnp.concatenate([U_out, jnp.zeros((d, pad))], axis=1)
            lam_out = jnp.concatenate([lam_out, jnp.ones(pad)])

        return U_out, lam_out

    else:  # "raw"
        # argsort(scores)[::-1]: descending original-index tie-break — matches
        # BOTH raw consumers (mclmc_lrd_adaptation._extract_lrd_from_samples :271
        # and meads_adaptation._lrd_from_accumulated_covariance :309).
        # This differs from mask_pad's argsort(-scores) tie-break in degenerate
        # spectra; on real eigh/SVD spectra bit-exact |λ−1| ties never occur
        # (ulp-broken even for truly degenerate inputs), so the difference is
        # behaviorally inert on continuous data — but the raw path stays
        # byte-faithful to its sources.
        order = jnp.argsort(scores)[::-1]
        top_order = order[:max_rank]
        return eigenvectors[:, top_order], eigenvalues[top_order]


# ---------------------------------------------------------------------------
# Low-rank estimators
# ---------------------------------------------------------------------------


def fisher_score_low_rank(
    draws: Array,
    grads: Array,
    max_rank: int,
    *,
    gamma: float = 1e-5,
    cutoff: float = 2.0,
) -> LowRankInverseMassMatrix:
    r"""Fisher-divergence-minimising low-rank inverse mass matrix.

    Implements Steps 1–9 of Algorithm 1 of
    :cite:p:`seyboldt2026preconditioning`, following the nutpie reference
    implementation (``nuts-rs`` ``src/transform/adapt/low_rank.rs``
    ``estimate_mass_matrix``).

    The inverse mass matrix has the form

    .. math::

        M^{-1} = \operatorname{diag}(\sigma)
                 \bigl(I + U(\Lambda - I)U^\top\bigr)
                 \operatorname{diag}(\sigma)

    where :math:`\sigma`, :math:`U`, :math:`\Lambda` minimise the sample
    Fisher divergence from :math:`\{(x_i, \nabla \log p(x_i))\}`.

    **Extracted from:** ``blackjax.adaptation.low_rank_adaptation``
    ``._compute_low_rank_metric`` + ``._spd_mean`` (main @ 532631c1).

    **Key asymmetry vs** :func:`draws_singular_value_low_rank`: this estimator
    uses BOTH draws and score gradients and applies γ-regularisation plus
    cutoff-masking on the eigenvalues.  :func:`draws_singular_value_low_rank`
    uses draws only and applies neither — the docstring notes this divergence
    explicitly.

    **Diagonal scale** ``σ = (Var[x] / Var[∇ log p])^{1/4}`` is the
    per-coordinate optimal scale (paper §3.1); clipped to ``[1e-20, 1e20]``
    (nutpie range).

    **AIRM geometric mean** ``Σ = C_x # C_a^{-1}`` (AIRM = affine-invariant
    Riemannian metric): Theorem 2.3 / Eq. 9 of
    :cite:p:`seyboldt2026preconditioning` — the regularised optimal inverse
    mass matrix is the geometric mean of the draw covariance with the
    *inverse* score covariance.

    **γ-regularisation** ``C = P P^T / γ + I`` (nutpie convention: the
    *unnormalised* sum-of-outer-products divided by ``γ`` directly, no ``n``
    scaling). Influence fades as ``n`` grows (Theorem 2.4).

    **Cutoff masking**: eigenvalues in ``[1/cutoff, cutoff]`` are set to 1
    (no preconditioning benefit); default ``cutoff=2`` matches nutpie.

    **dtype promotion** (round-9 schedule-port audit): promotes internally to
    ``float64`` when ``jax_enable_x64`` is active, regardless of the chain's
    working dtype.  Returns in the caller's original dtype.

    **G-layer note:** the 2·d support gate (``n ≥ 2·d`` before the LR
    estimate is trusted) and any fraction-window guard are G-layer concerns
    and are NOT implemented here.

    Parameters
    ----------
    draws
        Shape ``(n, d)``.  Chain positions (all rows must be valid samples —
        no zero-padding).
    grads
        Shape ``(n, d)``.  Log-density gradients at the corresponding draws.
    max_rank
        Maximum number of eigenvectors in the low-rank correction.  Must be
        a **static Python integer** (determines output shape).
    gamma
        Regularisation scale (nutpie convention).  Default ``1e-5`` matches
        nutpie's ``LowRankSettings::default``.
    cutoff
        Eigenvalue cutoff for informativeness masking.  Default ``2.0``
        matches nutpie's ``c=2``.

    Returns
    -------
    LowRankInverseMassMatrix
        ``(sigma, U, lam)`` with shapes ``(d,)``, ``(d, max_rank)``,
        ``(max_rank,)``.

    Notes
    -----
    The optimal translation ``μ* = x̄ + σ² ⊙ ᾱ`` (paper §3.2) is an
    adaptation-layer output, not part of the metric.  Compute it separately
    from the per-draw means if needed by the warmup wiring (a warmup-wiring
    concern, deliberately outside this estimator).
    """
    orig_dtype = draws.dtype
    compute_dtype = jnp.float64 if jax.config.jax_enable_x64 else orig_dtype
    draws = draws.astype(compute_dtype)
    grads = grads.astype(compute_dtype)

    n, d = draws.shape

    # --- Step 1: diagonal scaling  σ = (Var[x] / Var[∇log p])^{1/4} ---
    # draws.shape[0] is a static Python int for the pure estimator interface.
    mean_x = draws.mean(0)  # (d,)
    mean_g = grads.mean(0)  # (d,)

    diff_x = draws - mean_x[None, :]  # (n, d)
    diff_g = grads - mean_g[None, :]  # (n, d)

    # Population variance (n not n-1), matching nutpie.
    var_x = (diff_x**2).sum(0) / n  # (d,)
    var_g = (diff_g**2).sum(0) / n  # (d,)

    sigma = jnp.power(jnp.clip(var_x / jnp.maximum(var_g, 1e-10), 0.0, None), 0.25)
    sigma = jnp.clip(sigma, 1e-20, 1e20)  # nutpie range

    # --- Step 2: scale draws and gradients ---
    X = diff_x / sigma[None, :]  # (n, d)  scaled centered draws
    A = diff_g * sigma[None, :]  # (n, d)  scaled centered gradients

    # --- Step 3: principal subspaces via thin SVD ---
    _, _, Vt_x = jnp.linalg.svd(X, full_matrices=False)  # Vt_x: (min(n,d), d)
    _, _, Vt_a = jnp.linalg.svd(A, full_matrices=False)
    U_x = Vt_x[:max_rank].T  # (d, max_rank)
    U_a = Vt_a[:max_rank].T  # (d, max_rank)

    # --- Step 4: combined orthonormal basis Q ∈ ℝ^{d × q} ---
    combined = jnp.concatenate([U_x, U_a], axis=1)  # (d, 2*max_rank)
    Q, _ = jnp.linalg.qr(combined)  # Q: (d, min(d, 2*max_rank))
    q = Q.shape[1]  # actual subspace dimension

    # --- Step 5: project onto Q ---
    P_x = Q.T @ X.T  # (q, n)
    P_a = Q.T @ A.T  # (q, n)

    # --- Step 6: projected covariance matrices ---
    # nutpie: C = P P^T / gamma + I  (raw gamma, no /n).
    C_x = (P_x @ P_x.T) / gamma + jnp.eye(q, dtype=compute_dtype)
    C_a = (P_a @ P_a.T) / gamma + jnp.eye(q, dtype=compute_dtype)

    # --- Step 7: SPD geometric mean Σ = C_x # C_a^{-1} ---
    # Theorem 2.3 / Eq. 9 of arXiv:2603.18845.
    Sigma = _spd_mean(C_x, jnp.linalg.inv(C_a))

    # --- Step 8: eigendecompose Σ in the projected subspace ---
    vals, vecs = jnp.linalg.eigh(Sigma)  # ascending, (q,)
    vals = jnp.maximum(vals, _relative_pd_floor(vals))
    U_full = Q @ vecs  # (d, q) back to original space

    # --- Step 9: select top max_rank by |λ-1|; mask near-unity eigenvalues ---
    U_out, lam_out = select_top_eigenvalues_by_informativeness(
        vals, U_full, max_rank, tail_handling="mask_pad", cutoff=cutoff
    )

    return LowRankInverseMassMatrix(
        sigma=sigma.astype(orig_dtype),
        U=U_out.astype(orig_dtype),
        lam=lam_out.astype(orig_dtype),
    )


def draws_singular_value_low_rank(
    draws: Array,
    max_rank: int,
) -> LowRankInverseMassMatrix:
    r"""Draws-only SVD low-rank inverse mass matrix (MCLMC Scheme A estimator).

    Estimates the low-rank inverse mass matrix from the SVD of centred,
    standardised draws.  No gradient information is required.

    **Extracted from:** ``blackjax.adaptation.mclmc_lrd_adaptation``
    ``._extract_lrd_from_samples`` (main @ 532631c1).

    **Key asymmetry vs** :func:`fisher_score_low_rank`:

    * Draws only — no score gradients.
    * **No regularisation** (no γ): the covariance is estimated from raw
      outer products without a ``P P^T / γ + I`` regularisation term.
    * **No masking**: eigenvalues are returned as-is (``tail_handling="raw"``
      in :func:`select_top_eigenvalues_by_informativeness`); non-informative
      eigenvalues are NOT masked to 1.  This is a deliberate design choice
      in the MCLMC-LRD pilot estimator: the raw eigenspectrum is what drives
      the effective condition number diagnostics downstream.

    This asymmetry is preserved faithfully here; see the module docstring's
    source-lineage table for context.

    **G-layer note:** the Geyer-ESS support gate
    (``k_used = min(k, ⌊n_eff/2⌋)``, ``mclmc_lrd_adaptation.py:636``) and
    the ``n ≥ 2·d`` threshold are G-layer concerns.  Ensure ``max_rank ≤
    min(n, d)`` before calling (behaviour is undefined otherwise; JAX will
    return fewer than ``max_rank`` columns when ``max_rank > min(n, d)``).

    Parameters
    ----------
    draws
        Shape ``(n, d)``.  All rows must be valid samples (no zero-padding).
    max_rank
        Number of eigenpairs to return.  Must be a static Python integer and
        satisfy ``max_rank ≤ min(n, d)``.

    Returns
    -------
    LowRankInverseMassMatrix
        ``(sigma, U, lam)`` where ``sigma`` is the per-coordinate standard
        deviation and ``lam`` are the raw SVD eigenvalues of the sample
        correlation matrix (not masked to 1 for near-unity values).

    Notes
    -----
    The full eigenspectrum needed for :func:`~blackjax.adaptation.mclmc_lrd_adaptation._kappa_eff_pilot`
    diagnostics is NOT returned here — that function operates on the full
    sorted spectrum (``lam_all_sorted``).  If you need the full spectrum,
    call ``_extract_lrd_from_samples`` directly (G-layer concern, not part
    of the pure estimator).
    """
    mean = jnp.mean(draws, axis=0)  # (d,)
    sigma = jnp.std(draws, axis=0)  # (d,)
    sigma = jnp.where(sigma == 0.0, 1.0, sigma)  # avoid div-by-zero

    standardised = (draws - mean[None, :]) / sigma[None, :]  # (n, d)
    n = draws.shape[0]

    # Thin SVD; eigenvalues of the sample correlation matrix: lam_i = s_i^2 / n.
    _, S, Vt = jnp.linalg.svd(standardised, full_matrices=False)
    V = Vt.T  # (d, min(n, d))
    lam = (S**2) / n  # (min(n, d),)

    # Raw top-k selection by |λ-1|: no masking, no padding.
    U_k, lam_k = select_top_eigenvalues_by_informativeness(
        lam, V, max_rank, tail_handling="raw"
    )

    return LowRankInverseMassMatrix(sigma=sigma, U=U_k, lam=lam_k)


def sample_covariance_eigh_low_rank(
    m2: Array,
    count: Array | int,
    max_rank: int,
) -> LowRankInverseMassMatrix:
    r"""Chan/Welford-accumulated covariance → low-rank metric via ``eigh``.

    Extracts a low-rank inverse mass matrix from a Chan-parallel-accumulated
    sum of squared deviations matrix (MEADS / MCLMC-LRD Scheme-B estimator).

    **Extracted from:** ``blackjax.adaptation.meads_adaptation``
    ``._lrd_from_accumulated_covariance`` (main @ 532631c1, landed via #954).

    The input ``m2`` is the accumulated Chan-parallel M2 matrix (shape
    ``(d, d)``):

    .. math::

        M_2 = \sum_{i=1}^n (x_i - \bar{x}_n)(x_i - \bar{x}_n)^T

    which gives the Bessel-corrected sample covariance as
    ``C = M_2 / (n - 1)``.  The correlation matrix is then
    ``R = D^{-1/2} C D^{-1/2}`` where ``D = diag(C)``, and ``eigh`` of ``R``
    gives the eigenbasis.

    **Tail handling:** raw (no cutoff masking, no zero-padding) — matching the
    MEADS implementation.  The G-layer (``meads_adaptation``'s
    ``low_rank_window_fraction`` / 2·d gate) ensures the estimate is only used
    when the accumulated support suffices.

    **G-layer note:** the fraction-window gate (only accumulate draws in the
    second half of the adaptation window, ``low_rank_window_fraction=0.5``) and
    the 2·d support threshold are G-layer concerns; they gate *when* to call
    this estimator, not what it computes.  Callers must ensure ``count``
    reflects enough effective support before calling.

    Parameters
    ----------
    m2
        Shape ``(d, d)``.  Accumulated Chan-Welford sum of squared deviations
        (NOT divided by ``count``; this function applies the Bessel correction
        ``/ max(count - 1, 1)`` internally).
    count
        Total number of samples accumulated into ``m2``.  May be a traced
        JAX integer (safe inside ``jax.lax.scan``).
    max_rank
        Number of eigenpairs to return.  Must be a static Python integer and
        satisfy ``max_rank ≤ d``.

    Returns
    -------
    LowRankInverseMassMatrix
        ``(sigma, U, lam)`` with shapes ``(d,)``, ``(d, max_rank)``,
        ``(max_rank,)``.
    """
    # Bessel-corrected covariance from the accumulated M2.
    covariance = m2 / jnp.maximum(count - 1.0, 1.0)  # (d, d)
    variance = jnp.diag(covariance)  # (d,)
    sigma = jnp.sqrt(jnp.maximum(variance, 0.0))  # (d,)
    sigma = jnp.where(sigma <= 0.0, 1.0, sigma)  # avoid div-by-zero

    inv_sigma = 1.0 / sigma  # (d,)
    correlation = covariance * inv_sigma[:, None] * inv_sigma[None, :]  # (d, d)

    # eigh gives ascending eigenvalues of the (symmetric) correlation matrix.
    lam_all, V = jnp.linalg.eigh(correlation)  # (d,), (d, d)

    # Raw top-k selection by |λ-1|: no masking, no padding.
    U, lam = select_top_eigenvalues_by_informativeness(
        lam_all, V, max_rank, tail_handling="raw"
    )

    return LowRankInverseMassMatrix(sigma=sigma, U=U, lam=lam)


# ---------------------------------------------------------------------------
# Diagonal estimators
# ---------------------------------------------------------------------------


def welford_diagonal(draws: Array) -> Array:
    r"""Diagonal (per-coordinate) sample variance via Welford's algorithm.

    Thin scan-wrapper over :func:`blackjax.adaptation.mass_matrix.welford_algorithm`
    with ``is_diagonal_matrix=True``.  Computes the Bessel-corrected sample
    variance :math:`s^2_i = \frac{1}{n-1}\sum_{j=1}^n (x_{ji} - \bar{x}_i)^2`
    for each coordinate ``i``.

    **Why a wrapper instead of direct ``jnp.var``:** having a single
    estimator import surface ensures future callers can swap to streaming
    Welford (e.g., for online adaptation) without changing call sites.
    The algorithm itself is NOT moved or duplicated — only the call site
    is unified here.

    **Source:** ``blackjax.adaptation.mass_matrix.welford_algorithm``
    (``is_diagonal_matrix=True``).

    Parameters
    ----------
    draws
        Shape ``(n, d)``.

    Returns
    -------
    Array, shape ``(d,)``
        Bessel-corrected per-coordinate sample variance (= diagonal of the
        sample covariance matrix).
    """
    n, d = draws.shape
    wc_init, wc_update, wc_final = welford_algorithm(is_diagonal_matrix=True)

    def scan_fn(state, draw):
        return wc_update(state, draw), None

    final_state, _ = jax.lax.scan(scan_fn, wc_init(d), draws)
    covariance, _, _ = wc_final(final_state)
    return covariance


def welford_dense(draws: Array) -> Array:
    r"""Dense sample covariance via Welford's algorithm.

    Thin scan-wrapper over :func:`blackjax.adaptation.mass_matrix.welford_algorithm`
    with ``is_diagonal_matrix=False``.  Computes the Bessel-corrected sample
    covariance matrix.

    **Source:** ``blackjax.adaptation.mass_matrix.welford_algorithm``
    (``is_diagonal_matrix=False``).

    Parameters
    ----------
    draws
        Shape ``(n, d)``.

    Returns
    -------
    Array, shape ``(d, d)``
        Bessel-corrected sample covariance matrix.
    """
    n, d = draws.shape
    wc_init, wc_update, wc_final = welford_algorithm(is_diagonal_matrix=False)

    def scan_fn(state, draw):
        return wc_update(state, draw), None

    final_state, _ = jax.lax.scan(scan_fn, wc_init(d), draws)
    covariance, _, _ = wc_final(final_state)
    return covariance


def fisher_score_diagonal_from_moments(
    variance: Array,
    gradient_variance: Array,
) -> Array:
    r"""Core math of the Fisher-diagonal estimator from pre-computed variances.

    Computes :math:`\sigma^2 = \sqrt{\mathrm{Var}[x] / \mathrm{Var}[\nabla
    \log p]}` per coordinate — the same formula as :func:`fisher_score_diagonal`
    but operating on **already-computed** per-coordinate variances rather than
    raw draw arrays.

    This entry point is intended for callers that accumulate moments online
    (e.g. via :class:`~blackjax.adaptation.metric_buffers._FisherMomentBlock`)
    and want to avoid materialising the full draw array.  The caller is
    responsible for supplying Bessel-corrected (or otherwise normalised)
    variances; the ratio is invariant to a shared ``n`` vs ``n-1`` factor so
    either convention is acceptable for the diagonal estimator.

    **Near-zero gradient protection** (identical to :func:`fisher_score_low_rank`
    and :func:`fisher_score_diagonal`): ``gradient_variance`` is floored at
    ``1e-10`` before division, and the result is clipped to nutpie's
    ``[1e-20, 1e20]`` range before squaring.

    **Planned extension note:** this entry point is intentionally separate so
    that future updates to the estimator (e.g. adding draw-grad cross moments)
    can extend the *from_moments* signature without changing the *raw-draws*
    wrapper :func:`fisher_score_diagonal`.

    Parameters
    ----------
    variance
        Shape ``(d,)``.  Per-coordinate position variance
        :math:`\mathrm{Var}[x]`.  Must be non-negative; typically
        Bessel-corrected.
    gradient_variance
        Shape ``(d,)``.  Per-coordinate log-density-gradient variance
        :math:`\mathrm{Var}[\nabla \log p]`.  Must be non-negative; floored
        at ``1e-10`` internally.

    Returns
    -------
    Array, shape ``(d,)``
        Diagonal inverse mass matrix :math:`\sigma^2 =
        \sqrt{\mathrm{Var}[x] / \max(\mathrm{Var}[\nabla \log p],\, 10^{-10})}`,
        clipped to ``[1e-20, 1e20]``.
    """
    sigma = jnp.power(
        jnp.clip(variance / jnp.maximum(gradient_variance, 1e-10), 0.0, None), 0.25
    )
    sigma = jnp.clip(sigma, 1e-20, 1e20)  # nutpie range
    return sigma**2


def fisher_score_diagonal(
    draws: Array,
    grads: Array,
) -> Array:
    r"""Fisher-divergence-minimising diagonal inverse mass matrix.

    Computes :math:`\sigma^2 = \sqrt{\mathrm{Var}[x] / \mathrm{Var}[\nabla
    \log p]}`, the per-coordinate diagonal estimator of
    :cite:p:`seyboldt2026preconditioning`.

    This is the **diagonal-only** analogue of :func:`fisher_score_low_rank`:
    that function's diagonal scale :math:`\sigma =
    (\mathrm{Var}[x] / \mathrm{Var}[\nabla \log p])^{1/4}` gives the
    corresponding inverse mass matrix as
    :math:`\sigma^2 = \sqrt{\mathrm{Var}[x] / \mathrm{Var}[\nabla \log p]}`
    when the low-rank correction ``(U, lam)`` is dropped.

    **Extracted from:** branch ``b197f1e2`` (``feat/window-adaptation-fisher-diag``,
    2026-07-04), ``blackjax.adaptation.mass_matrix._fisher_diagonal_inverse_mass``.

    **Near-zero gradient protection** (same as :func:`fisher_score_low_rank`):
    ``Var[∇ log p]`` is floored at ``1e-10`` before division, and the result
    is clipped to nutpie's ``[1e-20, 1e20]`` range before squaring.

    **Note on variance convention:** ``Var[x]`` and ``Var[∇ log p]`` are
    computed with the same normalisation (Bessel-corrected, ``n-1`` divisor,
    via :func:`welford_diagonal`); the ratio is invariant to a shared
    ``n`` vs ``n-1`` factor (branch ``b197f1e2`` commit message, verified).

    **Implementation:** thin wrapper over
    :func:`fisher_score_diagonal_from_moments` — computes Bessel-corrected
    per-coordinate variances via two :func:`welford_diagonal` scans, then
    delegates all arithmetic to the from-moments entry point.

    Parameters
    ----------
    draws
        Shape ``(n, d)``.  Chain positions.
    grads
        Shape ``(n, d)``.  Log-density gradients at the corresponding draws.

    Returns
    -------
    Array, shape ``(d,)``
        Diagonal inverse mass matrix :math:`\sigma^2`.
    """
    var_x = welford_diagonal(draws)  # (d,)  Bessel-corrected
    var_g = welford_diagonal(grads)  # (d,)
    return fisher_score_diagonal_from_moments(var_x, var_g)


def sample_variance_diagonal(draws: Array) -> Array:
    r"""Coordinate-wise population variance from raw draws.

    Computes :math:`\hat{\sigma}^2_i = \mathbb{E}[x_i^2] - \mathbb{E}[x_i]^2`
    (population variance, no Bessel correction) for each coordinate ``i``.

    **Extracted from (verbatim duplicate):**

    * ``blackjax.adaptation.mclmc_adaptation``, ``L_step_size_adaptation``
      :341–342: ``variances = x_squared_average - jnp.square(x_average)``
      where ``x_average = E[x]`` and ``x_squared_average = E[x^2]`` are
      streaming step-size-weighted averages.

    * ``blackjax.adaptation.adjusted_mclmc_adaptation``,
      ``adjusted_mclmc_find_L_and_step_size`` :374–375: identical formula.

    Both inline occurrences use the result as ``inverse_mass_matrix = variances``,
    i.e., the population variance IS the diagonal IMM.  The streaming-average
    formulation (weighted by step size via ``incremental_value_update``) is an
    adaptation-layer scheduling concern; the pure estimator here takes the batch
    of accumulated draws directly.

    **Population (not Bessel-corrected)** to match the MCLMC streaming form:
    the weighted incremental average
    ``avg ← avg + w·(x − avg) / Σw`` converges to :math:`\mathbb{E}[x]` under
    uniform weights, not the unbiased :math:`n/(n-1)` form.

    Parameters
    ----------
    draws
        Shape ``(n, d)``.  The draws accumulated over the estimation window.

    Returns
    -------
    Array, shape ``(d,)``
        Per-coordinate population variance
        :math:`\mathbb{E}[x^2] - \mathbb{E}[x]^2`.
    """
    x_average = jnp.mean(draws, axis=0)  # (d,)
    x_squared_average = jnp.mean(draws**2, axis=0)  # (d,)
    return x_squared_average - jnp.square(x_average)
