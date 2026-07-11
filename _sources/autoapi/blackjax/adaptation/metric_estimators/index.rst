blackjax.adaptation.metric_estimators
=====================================

.. py:module:: blackjax.adaptation.metric_estimators

.. autoapi-nested-parse::

   E-layer: pure metric estimator functions for the low-rank unification RFC.

   Each function is a **pure transformation**: explicit arrays in → metric
   representation out.  No buffer state, no scheduling logic, no side effects.
   All functions are JAX-traceable and safe to use inside ``jax.lax.scan`` /
   ``jax.vmap`` provided that ``max_rank`` (where applicable) is a static
   Python integer (required to determine output shapes).

   **Source lineage** (R2a census, 2026-07-11, main @ 532631c1):

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

   **Composition law (RFC §0 L2):** estimators, data-feeding policy, and schedule
   are co-adapted package components.  The functions here are the *estimator*
   component only — callers supply the buffer view.  Gating logic (support gates,
   fraction-window checks, 2·d thresholds) belongs in the G-layer; it is
   explicitly *not* implemented here (docstrings note the relevant gates for each
   estimator).

   **Registration (R2a phase):** these functions are module-public (importable
   from ``blackjax.adaptation.metric_estimators``) but are NOT exported at the
   ``blackjax`` top-level.  Top-level registration and consumer re-wiring happen
   in R3.



Functions
---------

.. autoapisummary::

   blackjax.adaptation.metric_estimators.eigenvalue_informativeness
   blackjax.adaptation.metric_estimators.select_top_eigenvalues_by_informativeness
   blackjax.adaptation.metric_estimators.fisher_score_low_rank
   blackjax.adaptation.metric_estimators.draws_singular_value_low_rank
   blackjax.adaptation.metric_estimators.sample_covariance_eigh_low_rank
   blackjax.adaptation.metric_estimators.welford_diagonal
   blackjax.adaptation.metric_estimators.welford_dense
   blackjax.adaptation.metric_estimators.fisher_score_diagonal
   blackjax.adaptation.metric_estimators.sample_variance_diagonal


Module Contents
---------------

.. py:function:: eigenvalue_informativeness(eigenvalues: blackjax.types.Array) -> blackjax.types.Array

   Informativeness of each eigenvalue: how far it deviates from isotropic.

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

   :param eigenvalues: Shape ``(q,)``.  Eigenvalues of a correlation (or preconditioned)
                       matrix.  An isotropic Gaussian has all eigenvalues equal to 1 and
                       informativeness 0.

   :returns: :math:`|\lambda_i - 1|` for each :math:`\lambda_i`.
   :rtype: Array, shape ``(q,)``


.. py:function:: select_top_eigenvalues_by_informativeness(eigenvalues: blackjax.types.Array, eigenvectors: blackjax.types.Array, max_rank: int, *, tail_handling: Literal['mask_pad', 'raw'] = 'mask_pad', cutoff: float = 2.0) -> tuple[blackjax.types.Array, blackjax.types.Array]

   Select the top-``max_rank`` eigenpairs by :func:`eigenvalue_informativeness`.

   Two production consumers use this pattern with **divergent tail handling**
   (D2 in the RFC duplication map):

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

   :param eigenvalues: Shape ``(q,)``.  Must be the eigenvalues associated with the columns
                       of ``eigenvectors``.
   :param eigenvectors: Shape ``(d, q)``.  Columns are the eigenvectors.
   :param max_rank: Number of eigenpairs to return.  Must be a **static Python integer**
                    (determines the output shape).
   :param tail_handling: See above.  Default ``"mask_pad"`` matches the Fisher-score consumer.
   :param cutoff: Only used when ``tail_handling="mask_pad"``.  Eigenvalues in
                  :math:`[1/\text{cutoff},\ \text{cutoff}]` are masked to 1.
                  Default ``2.0`` matches nutpie's ``c=2``.

   :returns: * **U_out** (Array, shape ``(d, max_rank)`` (mask_pad) or ``(d, ≤max_rank)`` (raw)) -- Selected (and possibly masked/padded) eigenvectors.
             * **lam_out** (Array, shape ``(max_rank,)`` or ``(≤max_rank,)``) -- Selected (and possibly masked) eigenvalues.


.. py:function:: fisher_score_low_rank(draws: blackjax.types.Array, grads: blackjax.types.Array, max_rank: int, *, gamma: float = 1e-05, cutoff: float = 2.0) -> blackjax.mcmc.metrics.LowRankInverseMassMatrix

   Fisher-divergence-minimising low-rank inverse mass matrix.

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

   :param draws: Shape ``(n, d)``.  Chain positions (all rows must be valid samples —
                 no zero-padding).
   :param grads: Shape ``(n, d)``.  Log-density gradients at the corresponding draws.
   :param max_rank: Maximum number of eigenvectors in the low-rank correction.  Must be
                    a **static Python integer** (determines output shape).
   :param gamma: Regularisation scale (nutpie convention).  Default ``1e-5`` matches
                 nutpie's ``LowRankSettings::default``.
   :param cutoff: Eigenvalue cutoff for informativeness masking.  Default ``2.0``
                  matches nutpie's ``c=2``.

   :returns: ``(sigma, U, lam)`` with shapes ``(d,)``, ``(d, max_rank)``,
             ``(max_rank,)``.
   :rtype: LowRankInverseMassMatrix

   .. rubric:: Notes

   The optimal translation ``μ* = x̄ + σ² ⊙ ᾱ`` (paper §3.2) is an
   adaptation-layer output, not part of the metric.  Compute it separately
   from the per-draw means if needed by the warmup wiring (R3 concern).


.. py:function:: draws_singular_value_low_rank(draws: blackjax.types.Array, max_rank: int) -> blackjax.mcmc.metrics.LowRankInverseMassMatrix

   Draws-only SVD low-rank inverse mass matrix (MCLMC Scheme A estimator).

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

   This asymmetry is preserved faithfully here; see the RFC duplication map
   D2 for context.

   **G-layer note:** the Geyer-ESS support gate
   (``k_used = min(k, ⌊n_eff/2⌋)``, ``mclmc_lrd_adaptation.py:636``) and
   the ``n ≥ 2·d`` threshold are G-layer concerns.  Ensure ``max_rank ≤
   min(n, d)`` before calling (behaviour is undefined otherwise; JAX will
   return fewer than ``max_rank`` columns when ``max_rank > min(n, d)``).

   :param draws: Shape ``(n, d)``.  All rows must be valid samples (no zero-padding).
   :param max_rank: Number of eigenpairs to return.  Must be a static Python integer and
                    satisfy ``max_rank ≤ min(n, d)``.

   :returns: ``(sigma, U, lam)`` where ``sigma`` is the per-coordinate standard
             deviation and ``lam`` are the raw SVD eigenvalues of the sample
             correlation matrix (not masked to 1 for near-unity values).
   :rtype: LowRankInverseMassMatrix

   .. rubric:: Notes

   The full eigenspectrum needed for :func:`~blackjax.adaptation.mclmc_lrd_adaptation._kappa_eff_pilot`
   diagnostics is NOT returned here — that function operates on the full
   sorted spectrum (``lam_all_sorted``).  If you need the full spectrum,
   call ``_extract_lrd_from_samples`` directly (G-layer concern, not part
   of the pure estimator).


.. py:function:: sample_covariance_eigh_low_rank(m2: blackjax.types.Array, count: blackjax.types.Array | int, max_rank: int) -> blackjax.mcmc.metrics.LowRankInverseMassMatrix

   Chan/Welford-accumulated covariance → low-rank metric via ``eigh``.

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

   :param m2: Shape ``(d, d)``.  Accumulated Chan-Welford sum of squared deviations
              (NOT divided by ``count``; this function applies the Bessel correction
              ``/ max(count - 1, 1)`` internally).
   :param count: Total number of samples accumulated into ``m2``.  May be a traced
                 JAX integer (safe inside ``jax.lax.scan``).
   :param max_rank: Number of eigenpairs to return.  Must be a static Python integer and
                    satisfy ``max_rank ≤ d``.

   :returns: ``(sigma, U, lam)`` with shapes ``(d,)``, ``(d, max_rank)``,
             ``(max_rank,)``.
   :rtype: LowRankInverseMassMatrix


.. py:function:: welford_diagonal(draws: blackjax.types.Array) -> blackjax.types.Array

   Diagonal (per-coordinate) sample variance via Welford's algorithm.

   Thin scan-wrapper over :func:`blackjax.adaptation.mass_matrix.welford_algorithm`
   with ``is_diagonal_matrix=True``.  Computes the Bessel-corrected sample
   variance :math:`s^2_i = \frac{1}{n-1}\sum_{j=1}^n (x_{ji} - \bar{x}_i)^2`
   for each coordinate ``i``.

   **Why a wrapper instead of direct ``jnp.var``:** R2b/R3 must have a
   single estimator import surface; this function ensures future callers can
   swap to streaming Welford (e.g., for online adaptation) without changing
   call sites.  The algorithm itself is NOT moved or duplicated — only the
   call site is unified here.

   **Source:** ``blackjax.adaptation.mass_matrix.welford_algorithm``
   (``is_diagonal_matrix=True``).

   :param draws: Shape ``(n, d)``.

   :returns: Bessel-corrected per-coordinate sample variance (= diagonal of the
             sample covariance matrix).
   :rtype: Array, shape ``(d,)``


.. py:function:: welford_dense(draws: blackjax.types.Array) -> blackjax.types.Array

   Dense sample covariance via Welford's algorithm.

   Thin scan-wrapper over :func:`blackjax.adaptation.mass_matrix.welford_algorithm`
   with ``is_diagonal_matrix=False``.  Computes the Bessel-corrected sample
   covariance matrix.

   **Source:** ``blackjax.adaptation.mass_matrix.welford_algorithm``
   (``is_diagonal_matrix=False``).

   :param draws: Shape ``(n, d)``.

   :returns: Bessel-corrected sample covariance matrix.
   :rtype: Array, shape ``(d, d)``


.. py:function:: fisher_score_diagonal(draws: blackjax.types.Array, grads: blackjax.types.Array) -> blackjax.types.Array

   Fisher-divergence-minimising diagonal inverse mass matrix.

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
   The branch is DORMANT — its ``window_adaptation`` wiring (the diagonal
   estimator opt-in in ``mass_matrix_adaptation``) is NOT reproduced here and
   stays dormant.  Only the pure estimator math is extracted.

   **Near-zero gradient protection** (same as :func:`fisher_score_low_rank`):
   ``Var[∇ log p]`` is floored at ``1e-10`` before division, and the result
   is clipped to nutpie's ``[1e-20, 1e20]`` range before squaring.

   **Note on variance convention:** ``Var[x]`` and ``Var[∇ log p]`` are
   computed with the same normalisation (Bessel-corrected, ``n-1`` divisor,
   via :func:`welford_diagonal`); the ratio is invariant to a shared
   ``n`` vs ``n-1`` factor (branch ``b197f1e2`` commit message, verified).

   :param draws: Shape ``(n, d)``.  Chain positions.
   :param grads: Shape ``(n, d)``.  Log-density gradients at the corresponding draws.

   :returns: Diagonal inverse mass matrix :math:`\sigma^2`.
   :rtype: Array, shape ``(d,)``


.. py:function:: sample_variance_diagonal(draws: blackjax.types.Array) -> blackjax.types.Array

   Coordinate-wise population variance from raw draws.

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

   :param draws: Shape ``(n, d)``.  The draws accumulated over the estimation window.

   :returns: Per-coordinate population variance
             :math:`\mathbb{E}[x^2] - \mathbb{E}[x]^2`.
   :rtype: Array, shape ``(d,)``


