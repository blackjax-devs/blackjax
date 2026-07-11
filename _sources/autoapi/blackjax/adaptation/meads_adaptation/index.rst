blackjax.adaptation.meads_adaptation
====================================

.. py:module:: blackjax.adaptation.meads_adaptation


Classes
-------

.. autoapisummary::

   blackjax.adaptation.meads_adaptation.MEADSAdaptationState


Functions
---------

.. autoapisummary::

   blackjax.adaptation.meads_adaptation.base
   blackjax.adaptation.meads_adaptation.meads_adaptation
   blackjax.adaptation.meads_adaptation.maximum_eigenvalue


Module Contents
---------------

.. py:class:: MEADSAdaptationState



   State of the MEADS adaptation scheme.

   current_iteration
       Current iteration of the adaptation.
   step_size
       Step size for each fold, shape (num_folds,).
   position_sigma
       PyTree with per-fold per-dimension sample standard deviation of the
       position variable, leading axis has size num_folds.
   alpha
       Alpha parameter (momentum persistence) for each fold, shape (num_folds,).
   delta
       Delta parameter (slice translation) for each fold, shape (num_folds,).



   .. py:attribute:: current_iteration
      :type:  int


   .. py:attribute:: step_size
      :type:  blackjax.types.Array


   .. py:attribute:: position_sigma
      :type:  blackjax.types.ArrayTree


   .. py:attribute:: alpha
      :type:  blackjax.types.Array


   .. py:attribute:: delta
      :type:  blackjax.types.Array


.. py:function:: base(num_folds: int = 4, step_size_multiplier: float = 0.5, damping_slowdown: float = 1.0)

   Maximum-Eigenvalue Adaptation of damping and step size for the generalized
   Hamiltonian Monte Carlo kernel :cite:p:`hoffman2022tuning`.

   Full implementation of Algorithm 3 with K-fold cross-chain adaptation and
   chain shuffling. Chains are divided into ``num_folds`` folds; at each step
   statistics from fold ``t mod K`` are used to update the parameters for fold
   ``(t+1) mod K``. Every K steps all chains are reshuffled across folds.

   :param num_folds: Number of folds K to split chains into. Must divide num_chains evenly.
   :param step_size_multiplier: Multiplicative factor applied to the raw step size heuristic (default 0.5
                                as in the paper).
   :param damping_slowdown: Controls the damping floor in early iterations. The floor on γ is
                            ``damping_slowdown / (t·ε)``, so higher values force stronger damping
                            (higher α) in early iterations. Default is 1.0 as in the paper.

   :returns: * *init* -- Function that initializes the warmup state.
             * *update* -- Function that moves the warmup one step forward.


.. py:function:: meads_adaptation(logdensity_fn: Callable, num_chains: int, num_folds: int = 4, step_size_multiplier: float = 0.5, damping_slowdown: float = 1.0, adaptation_info_fn: Callable = return_all_adapt_info, low_rank_rank: int | None = None, low_rank_window_fraction: float = 0.5) -> blackjax.base.AdaptationAlgorithm

   Adapt the parameters of the Generalized HMC algorithm.

   Full implementation of Algorithm 3 from :cite:p:`hoffman2022tuning` with
   K-fold cross-chain adaptation and periodic chain shuffling.

   Chains are divided into ``num_folds`` folds. At adaptation step ``t``,
   fold ``t mod K`` is frozen (its chains do not advance, Algorithm 3 line 4).
   For each active fold k, the step size is computed from fold ``(k-1) mod K``'s
   preconditioned gradients, and the damping is computed from fold k's own
   positions using that step size. Every K steps all chains are reshuffled
   randomly across folds to prevent fold-assignment bias.

   :param logdensity_fn: The log density probability density function from which we wish to sample.
   :param num_chains: Total number of chains. Must be divisible by ``num_folds``.
   :param num_folds: Number of folds K to split chains into. Default is 4 as in the paper.
   :param step_size_multiplier: Multiplicative factor for the step size heuristic. Default is 0.5 as in
                                the paper.
   :param damping_slowdown: Slows the damping decay relative to the iteration count. Default is 1.0
                            as in the paper. Higher values force stronger damping in early iterations.
   :param adaptation_info_fn: Function to select the adaptation info returned. See return_all_adapt_info
                              and get_filter_adapt_info_fn in blackjax.adaptation.base. By default all
                              information is saved - this can result in excessive memory usage if the
                              information is unused.
   :param low_rank_rank: MEADS-LRD extension (opt-in, default ``None``). ``None`` adapts a
                         *diagonal* momentum metric from the fold ensemble -- exactly the
                         original behavior, bit-for-bit. An ``int`` instead adapts a
                         rank-``low_rank_rank`` :class:`~blackjax.mcmc.metrics.LowRankInverseMassMatrix`
                         from the **full population of all ``num_chains`` chains** (requires
                         :func:`blackjax.mcmc.ghmc`'s dense/low-rank momentum-metric support,
                         blackjax#950), generalizing MEADS the way MCLMC-LRD generalized
                         MCLMC. Unlike the diagonal scale (estimated per-fold, from each
                         fold's own ``num_chains // num_folds`` chains), the low-rank
                         eigenbasis is estimated *once per step* from the pooled global
                         population and then shared across all folds: a single fold's
                         ensemble (paper default ``num_folds=4`` gives only 16 chains/fold)
                         is too small for its top-k eigenvectors to be stable step-to-step,
                         and the resulting jitter destabilizes ghmc's persistent momentum
                         (measured regression: low-rank underperformed diagonal at
                         ``num_folds=4`` despite beating it at ``num_folds=1``, where the
                         per-fold estimate happens to already be the global one). The metric
                         is a shared symmetric preconditioner, not a per-fold statistic like
                         step size or damping, so pooling all chains to estimate it needs no
                         special justification -- it is the same practice window adaptation
                         uses for its diagonal/dense metric. The per-fold step-size and
                         damping heuristics (Algorithm 3) are otherwise unchanged, except they
                         now whiten by this shared global metric rather than a per-fold one,
                         so they stay consistent with the metric ghmc actually samples with.
                         The rank is clamped to ``min(low_rank_rank, num_chains - 1, d)`` (raises
                         ``ValueError`` if ``num_chains - 1 < 1``). A rank-``d`` metric equals the
                         full dense metric, so clamping by ``d`` is lossless and prevents shape
                         disagreements in the jax.lax.cond branches. The metric *returned* by
                         ``run()`` is the final state of the same window-accumulated estimator
                         described under ``low_rank_window_fraction`` below.

                         Two further fixes address a validated high-dimension (``d >>
                         num_chains``) failure mode where a single-snapshot low-rank metric
                         made MEADS-LRD *worse* than the diagonal baseline (a p >> n noise-
                         dominated eigenbasis fed into ghmc's step-size heuristic collapsed
                         ``epsilon`` to ~1e-3 and froze the chains):

                         - The step-size heuristic (Algorithm 3, line 8) always whitens its
                           gradients by the plain per-fold diagonal scale (``grad * sigma``),
                           never by the low-rank metric, even when ``low_rank_rank`` is set.
                           Whitening ``epsilon`` by a noisy low-rank eigenbasis couples the
                           step size to whichever direction the estimate currently
                           over-weights, which is what caused the collapse above; the
                           low-rank metric still preconditions the *momentum* (where it
                           helps), just not the step-size proxy.
                         - Selected eigenvalues are floored away from 0 (see
                           ``_floor_lrd_eigenvalues``) so a collinear/rank-deficient initial
                           ensemble can't seed a degenerate metric that self-reinforces into
                           ``rhat = inf``. Collinear / near-collinear initial ensembles
                           (e.g. all chains on a 1-D offset line) do not crash — two redundant
                           guards (the step-size decoupling and the eigenvalue floor) prevent
                           the NaN collapse — but expect severe under-mixing (measured
                           rhat≈5 on a rank-1 init); use a dispersed, full-rank initialization.
   :param low_rank_window_fraction: Only used when ``low_rank_rank`` is not ``None``. Fraction of
                                    warmup steps, counted from the end, over which the low-rank metric's
                                    covariance is accumulated (default ``0.5``: the last half of
                                    warmup). A single ``num_chains``-sized ensemble snapshot is
                                    ``p >> n`` noise-dominated once the dimension ``d`` exceeds
                                    ``num_chains`` -- exactly the regime a single fold's estimate was
                                    already too noisy for (see ``low_rank_rank`` above), just worse,
                                    since now the *whole* population's snapshot is undersized too, and
                                    (measured directly) even routing through it as a one-off fallback
                                    is enough to destabilize the ensemble. Instead, a running
                                    Chan/Welford covariance accumulator (mirroring the pattern
                                    :func:`~blackjax.adaptation.mass_matrix.welford_algorithm` uses for
                                    the mass matrix, generalized to a whole ensemble per step) is
                                    updated with every chain's position at every step *inside* the
                                    window, giving an effective sample size of ``num_chains *
                                    window_steps``. Once that effective size exceeds ``2 * d`` (a bare
                                    minimum for the estimate to not be noise-dominated), the low-rank
                                    momentum metric switches on and keeps improving every further
                                    window step; before that point -- either because the step is
                                    before the window (the initial, still-transient fraction of
                                    warmup, mirroring why Stan's window adaptation excludes its own
                                    initial/final fast windows from mass-matrix estimation), or because
                                    the window hasn't yet pooled ``2 * d`` samples -- the momentum
                                    metric falls back to a purely diagonal one
                                    (:func:`_lrd_diagonal_fallback`, i.e. no low-rank correction at
                                    all, matching ``low_rank_rank=None``'s momentum exactly), never a
                                    low-rank estimate from too little data.
                                    Must be in ``[0.0, 1.0]``; ``0.0`` accumulates from step 0, ``1.0``
                                    disables accumulation entirely (falls back to the purely diagonal
                                    momentum metric throughout the run).

                                    At GPU scale (independent re-validation, num_chains up to 1024 on a
                                    d≈390 hierarchical target): the low-rank metric's de-biasing reproduces
                                    (num_chains ≥ 256 with adequate warmup — cutting warmup at high
                                    num_chains under-accumulates the metric and re-introduces bias via
                                    step-size collapse) and holds as num_chains grows, but residual mean-error
                                    stabilizes slightly above strict certification thresholds; treat the
                                    low-rank metric as a robust improvement over the diagonal rather than a
                                    guarantee of unbiased means on such targets.

   :returns: * *A function that returns the last cross-chain state, a sampling kernel with the*
             * *tuned parameter values (averaged across folds), and all the warm-up states for*
             * *diagnostics.*


.. py:function:: maximum_eigenvalue(matrix: blackjax.types.ArrayLikeTree) -> blackjax.types.Array

   Estimate the largest eigenvalues of a matrix.

   We calculate an unbiased estimate of the ratio between the sum of the
   squared eigenvalues and the sum of the eigenvalues from the input
   matrix. This ratio approximates the largest eigenvalue well except in
   cases when there are a large number of small eigenvalues significantly
   larger than 0 but significantly smaller than the largest eigenvalue.
   This unbiased estimate is used instead of directly computing an unbiased
   estimate of the largest eigenvalue because of the latter's large
   variance.

   :param matrix: A PyTree with equal batch shape as the first dimension of every leaf.
                  The PyTree for each batch is flattened into a one dimensional array and
                  these arrays are stacked vertically, giving a matrix with one row
                  for every batch.


