blackjax.adaptation.low_rank_adaptation
=======================================

.. py:module:: blackjax.adaptation.low_rank_adaptation

.. autoapi-nested-parse::

   Adaptation of the low-rank-modified mass matrix for HMC-family samplers.

   Implements Algorithm 1 of :cite:p:`seyboldt2026preconditioning`, following the
   nutpie reference implementation.  The mass matrix has the form

   .. math::

       M^{-1} = \operatorname{diag}(\sigma)
                \bigl(I + U(\Lambda - I)U^\top\bigr)
                \operatorname{diag}(\sigma)

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

   **Numerical robustness** (round-9 schedule-port audit). ``_compute_low_rank_metric``
   opportunistically promotes its internal computation to ``float64`` when JAX's
   ``jax_enable_x64`` mode is enabled (regardless of the chain's own working
   dtype), and always applies a scale-relative positive-definiteness floor to
   both intermediate eigenspectra and the metric's own final eigenvalues --
   matching nuts-rs's own f64-throughout, PD-by-construction estimator. Enabling
   ``jax_enable_x64`` is strongly recommended for this warmup.

   **Memory**: :func:`window_adaptation_low_rank`'s default ``adaptation_info_fn``
   drops the internal ``draws_buffer``/``grads_buffer`` working buffers from the
   per-step diagnostic trace (an O(``num_steps`` x ``buffer_size`` x ``d``)
   allocation ``jax.lax.scan`` would otherwise stack for no benefit); pass
   ``adaptation_info_fn=blackjax.adaptation.base.return_all_adapt_info``
   explicitly to keep them.



Classes
-------

.. autoapisummary::

   blackjax.adaptation.low_rank_adaptation.LowRankAdaptationState


Functions
---------

.. autoapisummary::

   blackjax.adaptation.low_rank_adaptation.build_growing_window_schedule
   blackjax.adaptation.low_rank_adaptation.base
   blackjax.adaptation.low_rank_adaptation.window_adaptation_low_rank


Module Contents
---------------

.. py:class:: LowRankAdaptationState



   State for the low-rank mass matrix window adaptation.

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


   .. py:attribute:: ss_state
      :type:  blackjax.adaptation.step_size.DualAveragingAdaptationState


   .. py:attribute:: sigma
      :type:  blackjax.types.Array


   .. py:attribute:: mu_star
      :type:  blackjax.types.Array


   .. py:attribute:: U
      :type:  blackjax.types.Array


   .. py:attribute:: lam
      :type:  blackjax.types.Array


   .. py:attribute:: step_size
      :type:  float


   .. py:attribute:: draws_buffer
      :type:  blackjax.types.Array


   .. py:attribute:: grads_buffer
      :type:  blackjax.types.Array


   .. py:attribute:: buffer_idx
      :type:  int


   .. py:attribute:: background_split
      :type:  int


   .. py:attribute:: recompute_counter
      :type:  int


.. py:function:: build_growing_window_schedule(num_steps: int, early_window: float = 0.3, step_size_window: float = 0.15, early_window_size: int = 10, window_size: int = 80, window_growth: float = 1.5) -> blackjax.types.Array

   Proportional-to-tune, geometrically-growing-window warmup schedule.

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
   the window-sizing and gradient-based-init components of nutpie's warmup;
   pair it with ``buffer_policy="accumulating"`` (see :func:`base`) for the
   partial-forget buffer and continuous recompute cadence, matching
   nutpie's other main pieces.

   nutpie's actual schedule is an *online*, per-draw decision
   (``adapt_strategy.rs``'s ``is_late`` look-ahead + a partial-forget
   circular buffer + up-to-every-draw metric recomputation,
   ``mass_matrix_update_freq=1``), whereas blackjax's warmup runs the
   entire schedule as a static array through a single ``jax.lax.scan``
   (fixed ahead of time, like Stan's own :func:`build_schedule`), so this
   function precomputes an equivalent *offline* schedule with the same
   growth/sizing character -- **including the ``is_late`` rule**: the main
   phase does not start a window whose own successor (grown by
   ``window_growth``) would not fit before ``final_buffer_start``; instead
   the in-progress window keeps absorbing draws, unswitched, all the way to
   the step-size-only phase boundary. Without this, the naive
   ``min(current_size, remaining)`` truncation manufactures a tiny final
   window (e.g. 45 draws, under ``d=50``, at ``num_steps=2000``) that
   starves the final low-rank/dense metric recompute -- the ``is_late``
   rule instead gives a large, well-supported final window (e.g. 450 at the
   same budget), matching nuts-rs's own final-recompute support (round-9
   schedule-port audit).

   Unlike Stan's schedule, there is no purely step-size-only *initial*
   buffer: nutpie starts adapting the mass matrix from the very first draw
   (paper §3.2, "More frequent updates"), so the entire region up to the
   final step-size-only window is labelled "slow" (mass-matrix-adapting),
   split into windows of size ``early_window_size`` during the early phase
   and growing windows (starting at ``window_size``, x``window_growth``
   each switch) during the main phase.

   :param num_steps: Total number of warmup steps.
   :param early_window: Fraction of ``num_steps`` devoted to the early phase (fixed small
                        windows of size ``early_window_size``). Default ``0.3`` matches
                        nutpie's ``early_window``.
   :param step_size_window: Fraction of ``num_steps`` devoted to the final step-size-only
                            phase (no mass-matrix updates). Default ``0.15`` matches nutpie's
                            ``step_size_window``.
   :param early_window_size: Fixed window size during the early phase. Default ``10`` matches
                             nutpie's ``early_mass_matrix_switch_freq``.
   :param window_size: Starting window size for the main (post-early) phase, before
                       growth. Default ``80`` matches nutpie's ``mass_matrix_switch_freq``.
   :param window_growth: Multiplicative growth factor applied to the window size after each
                         switch in the main phase. Default ``1.5`` matches nutpie's
                         ``mass_matrix_window_growth``.

   :returns: * A ``(num_steps, 2)`` array of ``(stage, is_window_end)`` pairs, in the
             * same format as :func:`~blackjax.adaptation.window_adaptation.build_schedule`
             * (stage ``0`` = fast/step-size-only, stage ``1`` = slow/mass-matrix-adapting).


.. py:function:: base(max_rank: int = 10, target_acceptance_rate: float = 0.8, gamma: float = 1e-05, cutoff: float = 2.0, gradient_based_init: bool = False, buffer_policy: str = 'reset', recompute_every: int = 1) -> tuple[Callable, Callable, Callable]

   Warmup scheme using the low-rank mass matrix adaptation.

   Mirrors Stan's three-phase schedule but replaces Welford covariance
   estimation with the Fisher-divergence-minimising low-rank metric of
   :cite:p:`seyboldt2026preconditioning`, following nutpie's implementation.

   :param max_rank: Maximum number of eigenvectors retained in the low-rank correction.
   :param target_acceptance_rate: Target acceptance rate for dual-averaging step-size adaptation.
   :param gamma: Regularisation scale.  The projected covariance is divided by ``gamma``
                 (nutpie convention -- no ``n`` scaling).  Default ``1e-5`` matches
                 nutpie's ``LowRankSettings::default``.
   :param cutoff: Eigenvectors with eigenvalue in ``[1/cutoff, cutoff]`` are masked
                  (eigenvalue set to 1).  Default ``2.0`` matches nutpie's ``c=2``.
   :param gradient_based_init: If ``True``, seed the diagonal scale from the initial gradient
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
   :param buffer_policy: ``"reset"`` (default) hard-resets the draw/gradient buffer to empty
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
   :param recompute_every: Only used when ``buffer_policy="accumulating"``. Number of
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

   :returns: The three adaptation primitives expected by the window-adaptation loop.
   :rtype: ``(init, update, final)``


.. py:function:: window_adaptation_low_rank(algorithm, logdensity_fn: Callable, max_rank: int = 10, initial_step_size: float = 1.0, target_acceptance_rate: float = 0.8, gamma: float = 1e-05, cutoff: float = 2.0, adaptation_info_fn: Callable = _default_low_rank_adaptation_info_fn, integrator=mcmc.integrators.velocity_verlet, gradient_based_init: bool = False, schedule_fn: Callable[[int], blackjax.types.Array] = build_schedule, buffer_policy: str = 'reset', recompute_every: int = 1, **extra_parameters) -> blackjax.base.AdaptationAlgorithm

   Adapt step size and a low-rank mass matrix for HMC-family samplers.

   Uses the three-phase Stan-style warmup schedule while replacing Welford
   covariance estimation with the Fisher-divergence-minimising low-rank
   metric of :cite:p:`seyboldt2026preconditioning`.

   The returned ``AdaptationAlgorithm`` has a single ``run`` method::

       (state, params), info = warmup.run(rng_key, position, num_steps=1000)
       nuts = blackjax.nuts(logdensity_fn, **params)

   :param algorithm: An HMC-family algorithm object (e.g. ``blackjax.nuts``).
   :param logdensity_fn: Log-density of the target distribution.
   :param max_rank: Maximum number of eigenvectors in the low-rank correction.
   :param initial_step_size: Starting step size (adapted automatically).
   :param target_acceptance_rate: Target acceptance rate for dual averaging.
   :param gamma: Regularisation scale; projected covariance is divided by ``gamma``
                 before adding identity (nutpie convention -- no ``n`` scaling).
                 Default ``1e-5`` matches nutpie's ``LowRankSettings::default``.
   :param cutoff: Eigenvectors with eigenvalue in ``[1/cutoff, cutoff]`` are masked.
                  Default ``2.0`` matches nutpie's ``c=2``.
   :param adaptation_info_fn: Controls what adaptation info is retained; see
                              ``blackjax.adaptation.base``. Default
                              :func:`_default_low_rank_adaptation_info_fn` drops the raw
                              ``draws_buffer``/``grads_buffer`` internal working buffers from the
                              per-step trace (an O(num_steps * buffer_size * d) allocation
                              otherwise stacked by ``jax.lax.scan`` for no benefit -- the exact
                              root cause of a reported OOM at high d + large
                              ``buffer_policy="accumulating"`` buffers; see that function's
                              docstring). Pass ``blackjax.adaptation.base.return_all_adapt_info``
                              explicitly to keep the raw per-step buffer trace.
   :param integrator: Integrator to pass to ``algorithm.build_kernel``.
   :param gradient_based_init: Seed the diagonal scale from the initial gradient instead of the
                               identity, matching nutpie's own initialisation (see :func:`base`).
                               Default ``False`` reproduces the original behaviour exactly.
   :param schedule_fn: Schedule-generator function ``num_steps -> (num_steps, 2)`` array of
                       ``(stage, is_window_end)`` pairs. Default is Stan's fixed-absolute,
                       2x-doubling :func:`~blackjax.adaptation.window_adaptation.build_schedule`
                       (unchanged default behaviour). Pass
                       :func:`build_growing_window_schedule` for nutpie's proportional-to-tune,
                       1.5x-growing-window schedule -- see that function's docstring for
                       exactly what it does and does not capture relative to nutpie's own
                       (online, per-draw) schedule.
   :param buffer_policy: ``"reset"`` (default, unchanged behaviour) or ``"accumulating"``
                         (nutpie's partial-forget buffer) -- see :func:`base` for the exact
                         semantics. Composes with any ``schedule_fn``.
   :param recompute_every: Only used when ``buffer_policy="accumulating"``; see :func:`base`.
   :param \*\*extra_parameters: Additional keyword arguments forwarded to the kernel at every step
                                (e.g. ``num_integration_steps`` for HMC).

   :returns: * An ``AdaptationAlgorithm`` whose ``run`` method returns
             * ``(AdaptationResults, info)``.  ``AdaptationResults.parameters`` contains
             * ``step_size``, ``inverse_mass_matrix`` (a
             * :class:`~blackjax.mcmc.metrics.LowRankInverseMassMatrix` NamedTuple holding
             * the pure-array payload ``(sigma, U, lam)``), and any ``extra_parameters``.
             * *The kernel layer normalises this into a full*
             * :class:`~blackjax.mcmc.metrics.Metric` via
             * :func:`~blackjax.mcmc.metrics.default_metric` at call time. Returning the
             * pure-array form (rather than the closure-bearing ``Metric``) lets the
             * warmup compose with ``jax.vmap`` over chains; see GH #916.
             * ``AdaptationResults.state`` is re-initialised at the optimal translation
             * *μ* = x̄ + σ²⊙ᾱ, so it can be passed directly as the starting state for*
             * *production sampling.  The last chain state from warmup is available as*
             * ``warmup_info[-1].state``, and μ* as
             * ``warmup_info[-1].adaptation_state.mu_star``.

   .. rubric:: Notes

   Wrap ``warmup.run(...)`` in :func:`blackjax.progress_bar` to display a
   progress bar, e.g. ``with blackjax.progress_bar(): warmup.run(...)``.


