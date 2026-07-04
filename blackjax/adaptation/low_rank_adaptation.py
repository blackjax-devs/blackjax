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

**Buffer policy and recompute cadence** (opt-in, default unchanged). The
schedule above hard-resets the draw/gradient buffer to empty at every window
switch and only recomputes the metric at a window's end. nutpie instead keeps
an *accumulating*, partial-forget buffer: at a switch it pops only the draws
that were already "background" (i.e. from the window before last), so the
buffer retains the just-completed window's draws in addition to whatever
accumulates in the next one -- and it recomputes the metric up to every draw
(``mass_matrix_update_freq=1`` in ``nuts-rs``), not just at window ends
(``nuts-rs`` ``src/transform/adapt/low_rank.rs::switch`` /
``src/adapt_strategy.rs``). Passing ``buffer_policy="accumulating"`` to
:func:`base` / :func:`window_adaptation_low_rank` enables this; the default
``"reset"`` reproduces the original hard-reset behaviour exactly.
"""
import inspect
from typing import Callable, NamedTuple

import jax
import jax.flatten_util as fu
import jax.numpy as jnp
import numpy as np

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
        Number of currently-valid samples in the buffer (the first
        ``buffer_idx`` rows). Under ``buffer_policy="reset"`` this resets to
        0 at each slow window boundary; under ``"accumulating"`` it only
        shrinks by ``background_split`` at a switch (nutpie's partial-forget
        pop), so it persists across window boundaries.
    background_split
        Number of the buffer's leading (oldest) rows considered "background"
        -- to be dropped at the *next* switch, matching nuts-rs's
        ``LowRankMassMatrixStrategy::background_split`` (``switch()`` pops
        this many draws from the front, then resets it to the post-pop
        buffer length). Always ``0`` and inert under ``buffer_policy="reset"``.
    recompute_counter
        Number of slow-stage steps since the metric was last recomputed;
        gates the ``recompute_every`` cadence under ``buffer_policy=
        "accumulating"``. Always inert under ``buffer_policy="reset"``
        (recompute there is tied solely to ``is_window_end``).
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
    background_split: int
    recompute_counter: int


# ---------------------------------------------------------------------------
# Core batch algorithm
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


def _shift_buffer_left(buf: Array, shift: Array) -> Array:
    """Drop the first ``shift`` rows of ``buf``, shifting the remainder to
    the front and zero-filling the newly-vacated tail.

    Implements nutpie's partial-forget buffer pop under JAX's static-shape
    constraint (``nuts-rs`` ``src/transform/adapt/low_rank.rs::switch``:
    ``for _ in 0..background_split { draws.pop_front() }``): pad the buffer
    with its own capacity in zeros, then take a single dynamic-length-free
    ``dynamic_slice`` starting at ``shift`` (a traced integer is fine here,
    only the slice *size* -- the static ``capacity`` -- needs to be a
    concrete Python int).
    """
    capacity = buf.shape[0]
    shift = jnp.clip(shift, 0, capacity)
    padded = jnp.concatenate([buf, jnp.zeros_like(buf)], axis=0)
    return jax.lax.dynamic_slice_in_dim(padded, shift, capacity, axis=0)


def _accumulating_buffer_capacity(schedule: Array) -> int:
    """Static buffer capacity required by ``buffer_policy="accumulating"``,
    derived from a concrete ``(stage, is_window_end)`` schedule array.

    Under the partial-forget switch, the buffer holds at most the
    just-completed window's draws (kept as the new background) plus however
    many draws have accumulated in the in-progress next window at *its* own
    switch (up to that window's own size) -- so the tight worst case across
    the whole schedule is ``max(window[i] + window[i-1])`` over consecutive
    window-size pairs. Computed once, outside any trace (the schedule is a
    concrete array by construction -- ``num_steps`` must already be a static
    Python int for ``jax.random.split``/the scan length elsewhere in
    :func:`window_adaptation_low_rank`'s ``run``), so ordinary numpy suffices.
    """
    is_end = np.asarray(schedule[:, 1]).astype(bool)
    window_end_idx = np.flatnonzero(is_end)
    if window_end_idx.size == 0:
        return 1
    window_sizes = np.diff(np.concatenate([[-1], window_end_idx]))
    if window_sizes.size == 1:
        return int(window_sizes[0])
    pair_sums = window_sizes[1:] + window_sizes[:-1]
    return int(max(window_sizes[0], pair_sums.max()))


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

    Notes
    -----
    **Dtype promotion** (round-9 schedule-port audit, GAP-1). nuts-rs runs
    this whole estimator in ``f64`` unconditionally; blackjax chains
    typically run in ``float32`` (JAX's default), and the audit found that
    running THIS pipeline specifically (inverting + AIRM-geometric-meaning
    matrices whose condition number can reach ``~1/gamma``, e.g. ``1e5`` at
    the default ``gamma=1e-5``) in float32 produces small negative
    eigenvalues ~98% of the time on the models tested, silently making the
    returned metric indefinite. If the caller has enabled JAX's ``x64``
    mode (``jax.config.update("jax_enable_x64", True)``), this function
    promotes its inputs to ``float64`` internally regardless of the
    incoming (possibly ``float32``) chain dtype, then casts the result back
    -- decoupling the metric estimate's numerical precision from the
    sampler's own working dtype, at zero cost to any other part of the
    chain. If ``x64`` is not enabled, a cast to ``float64`` would silently
    be truncated back to ``float32`` by JAX (there is no per-call way to
    opt into ``float64`` without the global flag), so this function instead
    proceeds in the input's native dtype and relies on the PD guard in
    :func:`_spd_mean` (and the floor below) to keep the returned metric
    positive-definite regardless. **Enabling x64 is strongly recommended**
    for ``buffer_policy="accumulating"``/nutpie-schedule low-rank warmup,
    matching nuts-rs's own dtype and the shipped sampling-book example.
    """
    orig_dtype = draws_buffer.dtype
    compute_dtype = jnp.float64 if jax.config.jax_enable_x64 else orig_dtype
    draws_buffer = draws_buffer.astype(compute_dtype)
    grads_buffer = grads_buffer.astype(compute_dtype)

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
    # PD guard (round-9 audit, GAP-2): Sigma is PD by construction in exact
    # arithmetic (both C_x, C_a are `P P^T / gamma + I`, hence PD, and
    # _spd_mean itself is now floored -- see its docstring), but float32
    # rounding through this eigendecomposition-heavy pipeline can still tip
    # a near-zero eigenvalue negative. Flooring here is the same
    # scale-relative guard as _spd_mean's (see _relative_pd_floor -- an
    # ABSOLUTE eps floor is wrong for this pipeline's wide dynamic range),
    # applied to the metric's OWN final eigenvalues (belt-and-suspenders,
    # matching nuts-rs's own `vals.all(|x| x > 0.)` assertion in
    # `test_estimate_mass_matrix`).
    vals = jnp.maximum(vals, _relative_pd_floor(vals))
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

    # Cast back to the caller's original dtype -- the metric's INTERNAL
    # computation may have been promoted to float64 above, but the returned
    # state fields (sigma/mu_star/U/lam, folded into LowRankAdaptationState)
    # must stay in the chain's own working dtype so the rest of the warmup
    # loop's pytree structure is unaffected.
    return (
        sigma.astype(orig_dtype),
        mu_star.astype(orig_dtype),
        U_out.astype(orig_dtype),
        lam_out.astype(orig_dtype),
    )


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
    buffer_policy: str = "reset",
    recompute_every: int = 1,
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
    buffer_policy
        ``"reset"`` (default) hard-resets the draw/gradient buffer to empty
        at every window switch, matching the original Stan-schedule
        behaviour exactly -- zero default-behavior change.
        ``"accumulating"`` instead ports nutpie's partial-forget buffer
        (``nuts-rs`` ``src/transform/adapt/low_rank.rs::switch``): at a
        window switch, only the draws that were already "background" (the
        window before last) are dropped, so the buffer keeps the
        just-completed window's draws in addition to the next window's, and
        the metric is recomputed both at every switch (unconditionally,
        nutpie's ``force_update``) and periodically in between per
        ``recompute_every`` (nutpie's ``mass_matrix_update_freq``). Composes
        with any ``schedule_fn`` -- the buffer policy only changes what
        happens *at* a window boundary the schedule already defines, not
        when those boundaries occur.
    recompute_every
        Only used when ``buffer_policy="accumulating"``. Number of
        slow-stage steps between metric recomputes *between* window
        switches (switches themselves always force a recompute,
        independent of this cadence). Default ``1`` recomputes on every
        slow-stage step, matching nutpie's default
        ``mass_matrix_update_freq=1`` (the fully faithful port). Raising
        this trades fidelity for compute: an SVD-based recompute every
        single step can be costly in JAX for large ``d``/buffer size; see
        the PR description for measured timings before deviating from the
        default. Ignored under ``buffer_policy="reset"`` (recompute there is
        tied solely to ``is_window_end``, as before).

    Returns
    -------
    ``(init, update, final)``
        The three adaptation primitives expected by the window-adaptation loop.
    """
    if buffer_policy not in ("reset", "accumulating"):
        raise ValueError(
            f"buffer_policy must be 'reset' or 'accumulating', got {buffer_policy!r}"
        )
    if recompute_every < 1:
        raise ValueError(f"recompute_every must be >= 1, got {recompute_every!r}")
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
            0,
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
            state.background_split,
            state.recompute_counter,
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
        # Only accumulate under "accumulating" -- keeps the field genuinely
        # inert (always 0) rather than merely unread under "reset", so state
        # inspection under the default policy is unambiguous.
        new_recompute_counter = (
            state.recompute_counter + 1
            if buffer_policy == "accumulating"
            else state.recompute_counter
        )
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
            state.background_split,
            new_recompute_counter,
        )

    def slow_final(state: LowRankAdaptationState) -> LowRankAdaptationState:
        """End of slow window under ``buffer_policy="reset"``: recompute
        metric and hard-reset the buffer to empty. Unchanged from the
        original (pre-``buffer_policy``) behaviour -- ``background_split``/
        ``recompute_counter`` are inert under this policy, zeroed here to
        match the buffer's own reset for cleanliness."""
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
            0,
            0,
        )

    def slow_switch(state: LowRankAdaptationState) -> LowRankAdaptationState:
        """End of slow window under ``buffer_policy="accumulating"``: nutpie's
        partial-forget ``switch()`` -- drop only the ``background_split``
        oldest rows (shift the rest to the front), then the *entire*
        surviving buffer (the just-completed window) becomes the new
        background, matching ``nuts-rs``'s
        ``self.background_split = self.draws.len()`` (set *after* the pop).
        The recompute is unconditional here (nutpie's ``force_update=true``
        on every switch), guarded only by the same ``n>=3`` minimum ``adapt()``
        floor nuts-rs itself uses (``current_count() < 3 => return false``)
        to avoid fitting a metric from a near-empty buffer."""
        shift = state.background_split
        new_draws = _shift_buffer_left(state.draws_buffer, shift)
        new_grads = _shift_buffer_left(state.grads_buffer, shift)
        new_n_valid = state.buffer_idx - shift

        def _recompute():
            return _compute_low_rank_metric(
                new_draws, new_grads, new_n_valid, max_rank, gamma, cutoff
            )

        def _keep():
            return state.sigma, state.mu_star, state.U, state.lam

        sigma, mu_star, U, lam = jax.lax.cond(new_n_valid >= 3, _recompute, _keep)
        # Re-initialise dual averaging at the current averaged step size --
        # unchanged Stan-schedule convention, orthogonal to the buffer
        # question (nuts-rs restarts step size only once, on the *first*
        # successful mass-matrix change; porting that is out of this task's
        # scope, see the module docstring / PR description).
        new_ss = da_init(da_final(state.ss_state))
        return LowRankAdaptationState(
            new_ss,
            sigma,
            mu_star,
            U,
            lam,
            jnp.exp(new_ss.log_step_size),
            new_draws,
            new_grads,
            new_n_valid,
            new_n_valid,
            0,
        )

    def slow_recompute_only(state: LowRankAdaptationState) -> LowRankAdaptationState:
        """Continuous (mid-window) metric recompute under
        ``buffer_policy="accumulating"``: reuses whatever is currently in
        the buffer (the retained previous window plus the partial current
        window), matching nuts-rs's ``adapt()`` firing on (up to) every draw
        independent of the window-end ``switch()`` event
        (``mass_matrix_update_freq=1``, gated here by ``recompute_every``).
        Buffer contents, ``background_split``, and the step-size
        dual-averaging state are left untouched -- only the mass-matrix
        factors change."""
        n = state.buffer_idx

        def _recompute():
            return _compute_low_rank_metric(
                state.draws_buffer, state.grads_buffer, n, max_rank, gamma, cutoff
            )

        def _keep():
            return state.sigma, state.mu_star, state.U, state.lam

        sigma, mu_star, U, lam = jax.lax.cond(n >= 3, _recompute, _keep)
        return LowRankAdaptationState(
            state.ss_state,
            sigma,
            mu_star,
            U,
            lam,
            state.step_size,
            state.draws_buffer,
            state.grads_buffer,
            state.buffer_idx,
            state.background_split,
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
        if buffer_policy == "accumulating":

            def _maybe_periodic_recompute(s: LowRankAdaptationState):
                due = jnp.logical_and(
                    stage == 1, s.recompute_counter % recompute_every == 0
                )
                return jax.lax.cond(due, slow_recompute_only, lambda x: x, s)

            new_state = jax.lax.cond(
                is_window_end,
                slow_switch,
                _maybe_periodic_recompute,
                new_state,
            )
        else:
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
    buffer_policy: str = "reset",
    recompute_every: int = 1,
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
    buffer_policy
        ``"reset"`` (default, unchanged behaviour) or ``"accumulating"``
        (nutpie's partial-forget buffer) -- see :func:`base` for the exact
        semantics. Composes with any ``schedule_fn``.
    recompute_every
        Only used when ``buffer_policy="accumulating"``; see :func:`base`.
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
        buffer_policy=buffer_policy,
        recompute_every=recompute_every,
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
        # `schedule` must be computed before sizing the buffer under
        # "accumulating" (its capacity is schedule-derived); `num_steps` is
        # already required to be a concrete Python int here regardless (it
        # sizes `jax.random.split` and the scan length below), so `schedule`
        # is a concrete array too -- safe to inspect with plain numpy.
        schedule = schedule_fn(num_steps)
        if buffer_policy == "accumulating":
            # Capacity = the accumulating policy's own worst-case buffer
            # content (previous + current window), not the "reset" policy's
            # heuristic below -- see _accumulating_buffer_capacity.
            buffer_size = max(_accumulating_buffer_capacity(schedule), 1)
        else:
            # Size the buffer to the expected largest slow window rather than
            # the full warmup length.  The modular indexing in slow_update
            # means that if a window exceeds buffer_size only the most
            # recent buffer_size draws are kept — matching nutpie's
            # fixed-buffer behaviour and avoiding O(num_steps × d)
            # allocations for large d.
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
