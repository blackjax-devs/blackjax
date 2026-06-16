blackjax.adaptation.mclmc_lrd_adaptation
========================================

.. py:module:: blackjax.adaptation.mclmc_lrd_adaptation

.. autoapi-nested-parse::

   Pilot-free (Scheme A) MCLMC warmup with Low-Rank Diagonal preconditioning.

   Overview
   --------
   This module implements the **Scheme A** warmup for MCLMC via
   :func:`mclmc_lrd_warmup`.  Phases 1–3 are shared between the unadjusted and
   adjusted inner kernels; only Phase 4 switches:

   1. **Pilot phase** — a single unadjusted MCLMC chain with diagonal
      preconditioning (via
      :func:`~blackjax.adaptation.mclmc_adaptation.mclmc_find_L_and_step_size`)
      runs for ``pilot_num_warmup + pilot_num_samples`` steps to reach the
      typical set and collect geometry samples.

   2. **LRD extraction** — the pilot draws are standardised and a thin SVD
      extracts the top-*k* principal directions of the correlation structure.
      The result is a
      :class:`~blackjax.mcmc.metrics.LowRankInverseMassMatrix` ``(sigma, U,
      lam)``.

      **Rank guard (non-negotiable):** before extraction, the effective sample
      size of the pilot chain is estimated via
      :func:`~blackjax.diagnostics.effective_sample_size` (the Geyer
      monotone-sequence estimator, operating on the ravelled flat-parameter
      vector; one chain, ``pilot_num_samples`` draws; basis is *not* the
      az-bulk ESS).  A safe rank bound ``k_safe = floor(n_eff / 2)`` is
      computed, and the requested ``k`` is hard-clamped to ``k_safe`` when it
      would exceed it.  Without this guard, under-mixed pilots produce
      rank-deficient SVDs and the LRD metric degrades sampling quality.  In the
      extreme under-mix regime (n_eff < 5), the Geyer estimator may exceed
      az-bulk ESS on the minimum-ESS dimension; ``k_safe`` is still floored to
      0 → ``k_used=1``, so practical rank clamping is unaffected.

   3. **Multi-chain unadjusted LRD tuning** —
      :func:`~blackjax.adaptation.mclmc_adaptation.mclmc_find_L_and_step_size`
      is run in parallel over ``num_chains`` independent chains (all starting
      from the pilot's final position, with per-chain PRNG keys) with the LRD
      metric kernel (``diagonal_preconditioning=False``).  The per-chain
      ``(L, step_size)`` values are averaged to give stable estimates of the
      trajectory length and step size in LRD geometry.

   4. **Inner-kernel dispatch** (controlled by ``inner_kernel``):

      * ``"mclmc"`` *(default)*: returns the Phase-3 mean
        ``(L, step_size)`` directly.

      * ``"adjusted_mclmc"`` *(experimental)*: warm-starts
        :func:`~blackjax.adaptation.adjusted_mclmc_adaptation.adjusted_mclmc_find_L_and_step_size`
        across ``num_chains`` parallel chains from the Phase-3 mean parameters.
        **Two hard constraints are enforced automatically** to avoid known
        failure modes:

        - ``params`` is set to
          ``MCLMCAdaptationState(L=L_init, step_size=..., inverse_mass_matrix=lrd_imm)``
          so that the ``sqrt(dim)`` default L initialisation (which ignores the
          baked-in LRD metric) is never used.
        - ``frac_tune2=0.0`` disables the variance-based *L* estimator, which
          computes original-space ``trace(Σ)`` and is incompatible with an
          externally-baked LRD IMM.

        ``L_init`` is floored at ``floor_factor * step_unadj`` (default
        ``floor_factor=1.15``) to prevent the Dual-Averaging ceiling
        ``(L / 1.1)`` from binding below the oracle step size.  For geometry
        where the oracle step size exceeds the oracle L (stiff
        high-condition-number targets), the default 1.15 may leave the DA
        ceiling binding; raise to approximately 1.5 and set
        ``adjusted_num_steps`` to at least 5000 for those targets.

   Stan-window analogy
   -------------------
   The staging mirrors Stan's two-phase warmup: a cheap diagonal (fast) phase
   reaches the typical set first; the LRD (slow) phase then builds the metric
   from draws that are already in the right region.  The key difference is that
   Scheme A uses MCLMC (unadjusted, leapfrog-free) for the pilot, eliminating
   the NUTS pilot cost that dominated total gradient expenditure in our
   benchmarks:

   * ill_cond_50 (d=50, κ=1000): k=40, n_pilot=10k → 101.1 % oracle step-size
     recovery, 2.42× ESS/total-grad vs NUTS-pilot, 72.5 % grad savings.
   * german_credit (d=26): k=8, n_pilot=5k.
   * mvn_10 (d=10): k=4, n_pilot=2k.

   Limitations
   -----------
   * **Single-chain pilot.** The pilot is single-chain; overdispersed
     multi-chain initialisation would give stronger geometry coverage and
     tighter n_eff estimates.  This is a known limitation for the rank guard's
     tightness.
   * **NUTS-pilot fallback.** When the target has funnel-class geometry (e.g.
     hierarchical models near unit-root), the MCLMC pilot may under-mix even
     with n_eff/2 clamping.  A NUTS-pilot fallback (replacing Phase 1 with
     :func:`~blackjax.adaptation.window_adaptation.window_adaptation`) is the
     recommended escape hatch but is out of scope here.
   * **Adjusted path experimental.** ``inner_kernel="adjusted_mclmc"`` is
     certified 3/3 on german_credit at the default recipe (4-chain phase 3,
     ``frac_tune2=0``, ``floor_factor=1.15``, ``adjusted_num_steps=3000``).
     ill_cond_50 (stiff, κ=1000) is geometrically compatible (clamp-free at
     ``floor_factor=1.5``, step size 104–114% of oracle, zero DA divergences)
     but was NOT certified at ``adjusted_num_steps=3000`` (marginal R-hat
     1.010–1.011; DA not converged).  For stiff geometry, raise
     ``floor_factor`` to ~1.5 and ``adjusted_num_steps`` to ≥5000.
     The unadjusted default remains the stable, broadly validated path.



Classes
-------

.. autoapisummary::

   blackjax.adaptation.mclmc_lrd_adaptation.MCLMCLRDAdaptationState


Functions
---------

.. autoapisummary::

   blackjax.adaptation.mclmc_lrd_adaptation.mclmc_lrd_warmup


Module Contents
---------------

.. py:class:: MCLMCLRDAdaptationState



   Result of :func:`mclmc_lrd_warmup`.

   L
       Adapted momentum decoherence length from the final tuning phase.
   step_size
       Adapted step size from the final tuning phase.
   inverse_mass_matrix
       The adapted LRD inverse mass matrix as a
       :class:`~blackjax.mcmc.metrics.LowRankInverseMassMatrix` NamedTuple
       ``(sigma, U, lam)``.
   diagnostics
       A plain dict with provenance fields:

       ``inner_kernel``
           Which inner kernel was used (``"mclmc"`` or
           ``"adjusted_mclmc"``).
       ``n_eff``
           Effective sample size of the pilot chain (Geyer monotone-sequence
           estimator, blackjax basis, ravelled flat vector).
       ``k_safe``
           Floor of ``n_eff / 2`` — the maximum rank that is statistically
           supported by the pilot.
       ``k_used``
           The rank actually passed to SVD after clamping (``min(k, k_safe)``).
       ``pilot_num_grad_evals``
           Total gradient evaluations consumed by the pilot phase
           (warmup + samples; unadjusted MCLMC costs 2 grads/step).
       ``pilot_L``
           Trajectory length L adapted during the diagonal pilot warmup.
       ``pilot_step_size``
           Step size adapted during the diagonal pilot warmup.
       ``lrd_L``
           Mean trajectory length L across ``num_chains`` chains after
           Phase-3 unadjusted LRD tuning.
       ``lrd_step_size``
           Mean step size across ``num_chains`` chains after Phase-3
           unadjusted LRD tuning.
       ``L_init``
           *(adjusted path only)* The floor-guarded L initialisation value
           passed to the adjusted warm-start: ``max(lrd_L, floor_factor *
           lrd_step_size)``.  This is the value whose ``/ 1.1`` sets the DA
           ceiling for step-size tuning.
       ``floor_active``
           *(adjusted path only)* ``True`` when the floor guard was
           triggered (``floor_factor * lrd_step_size > lrd_L``), i.e. when
           ``L_init`` was raised above the unadjusted mean.
       ``N_sample``
           *(adjusted path only)* Effective number of leapfrog steps per
           trajectory at the final adapted parameters:
           ``round(L_init / final_step_size)``.  Provided as a bookkeeping
           aid for cert integration.
       ``e1_fired``
           ``True`` when the √d warm-start (E1) was applied to Phase-3 DA
           initialisation; ``False`` when the fallback (pilot step/L) was
           used instead.  Always ``False`` when ``warmup_step_init="default"``.
       ``kappa_eff_pilot``
           Effective condition number κ(M⁻¹ Σ⁻¹) of the pilot-derived LRD
           IMM against the pilot sample covariance.  Computed from the Phase-2
           SVD eigenspectrum.  Values near 1 indicate good whitening; values
           above 5 indicate the IMM is under-preconditioned (E1 falls back to
           pilot step/L).  Present whenever ``warmup_step_init`` is set (both
           ``"law"`` and ``"default"`` paths compute it for observability).


   .. py:attribute:: L
      :type:  float


   .. py:attribute:: step_size
      :type:  float


   .. py:attribute:: inverse_mass_matrix
      :type:  blackjax.mcmc.metrics.LowRankInverseMassMatrix


   .. py:attribute:: diagnostics
      :type:  dict


.. py:function:: mclmc_lrd_warmup(logdensity_fn, position, rng_key, *, k: int = 10, pilot_num_warmup: int = 1000, pilot_num_samples: int = 5000, lrd_num_steps: int = 1000, num_chains: int = 4, inner_kernel: str = 'mclmc', floor_factor: float = 1.15, adjusted_num_steps: int = 3000, adjusted_target: float = 0.9, warmup_step_init: str = 'law')

   Scheme A (pilot-free) MCLMC warmup with Low-Rank Diagonal preconditioning.

   Runs a cheap diagonal unadjusted MCLMC pilot to reach the typical set and
   collect geometry samples, extracts a low-rank diagonal (LRD) inverse mass
   matrix via thin SVD, then calibrates step size and trajectory length L
   across ``num_chains`` parallel chains with the LRD metric kernel.  The
   inner kernel for the final tuning phase is controlled by ``inner_kernel``.

   :param logdensity_fn: Log-density of the target distribution.
   :param position: Initial position (any pytree).
   :param rng_key: JAX PRNG key.
   :param k: Requested LRD rank.  Hard-clamped to ``floor(n_eff / 2)`` if the
             pilot chain is under-mixed; see the rank-guard note in the module
             docstring.
   :param pilot_num_warmup: Number of steps for the diagonal pilot warmup phase (used by
                            :func:`~blackjax.adaptation.mclmc_adaptation.mclmc_find_L_and_step_size`
                            to adapt step size and L with a diagonal mass matrix).
   :param pilot_num_samples: Number of unadjusted MCLMC draws collected *after* the pilot warmup,
                             used for the SVD geometry estimate.
   :param lrd_num_steps: Number of steps passed to the LRD tuning call(s) in Phase 3 (and
                         Phase 4 for the adjusted path).
   :param num_chains: Number of parallel chains for Phase 3 (unadjusted LRD tuning) and
                      Phase 4 (adjusted tuning when ``inner_kernel="adjusted_mclmc"``).
                      All chains start from the pilot's final position; per-chain PRNG keys
                      give independent trajectories.  Per-chain L and step_size are averaged
                      for a stable multi-chain estimate.  Default ``4``.
   :param inner_kernel: Which inner kernel to use for the final tuning phase.  One of:

                        * ``"mclmc"`` *(default, stable)*: unadjusted MCLMC.  Phase-3
                          mean ``(L, step_size)`` is returned directly.
                        * ``"adjusted_mclmc"`` *(experimental)*: after Phase-3 unadjusted
                          LRD tuning, warm-starts
                          :func:`~blackjax.adaptation.adjusted_mclmc_adaptation.adjusted_mclmc_find_L_and_step_size`
                          across ``num_chains`` chains with ``frac_tune2=0.0``
                          (variance-based *L* estimator disabled) and
                          ``diagonal_preconditioning=False``.
   :param floor_factor: For ``inner_kernel="adjusted_mclmc"`` only: the L initialisation
                        floor is ``max(L_unadj, floor_factor * step_unadj)``.  Default
                        ``1.15``; certified on german_credit.  For stiff geometry where the
                        oracle step size exceeds the oracle L, raise to approximately 1.5
                        (the default leaves the DA ceiling ``L / 1.1`` binding for those
                        targets).  Ignored when ``inner_kernel="mclmc"``.
   :param adjusted_num_steps: Number of DA tuning steps for the adjusted Phase-4 path.  Default
                              ``3000`` with ``frac_tune1=0.5`` → 1500 DA steps; the certified
                              recipe for german_credit.  For stiff high-κ geometry, increase to
                              at least 5000.  Ignored when ``inner_kernel="mclmc"``.
   :param adjusted_target: Target acceptance rate for the adjusted MCLMC tuning phase.
                           Default ``0.9``.  Ignored when ``inner_kernel="mclmc"``.
   :param warmup_step_init: Initialisation strategy for the Phase-3 Dual-Averaging (DA) step-size
                            tuner.  One of:

                            * ``"law"`` *(default)*: **√d warm-start (E1)**, gated on κ_eff.

                              When the pilot-derived LRD IMM achieves κ_eff ≤ 5 (the geometry is
                              sufficiently whitened), Phase-3 DA is initialised at the scaling-law
                              values ``step_size = 1.22 × √d``, ``L = 0.85 × √d``.  These
                              constants were derived from the MCLMC scaling-laws study (S3, 2026):
                              at good preconditioning, MCLMC's optimal step and trajectory length
                              are dimension-independent multiples of √d.

                              When κ_eff > 5 (under-preconditioned, e.g. rank-1 LRD on a κ=1000
                              target), E1 is **not applied** and Phase-3 DA falls back to the
                              pilot's own ``(step_size, L)`` — the same behaviour as
                              ``"default"``.  This gate prevents overshoot: at low rank the
                              geometry is not yet whitened and the scaling-law values would place
                              the DA starting point far from the actual optimum.

                              The warm-start only affects DA *convergence speed*, not the final
                              converged value.  At sufficient ``lrd_num_steps`` budget,
                              ``"law"`` and ``"default"`` produce statistically identical
                              ``(step_size, L)`` outputs.  The gain is in sample quality at
                              *low-budget* warmup (measured via ESS/grad):
                              ~20–30% improvement through n_warmup ≈ 1000 at d = 500 in the
                              Q0 sweep (2026-06), scaling with dimension because the default
                              DA init at 0.25 √d falls further below the optimum at larger d.

                            * ``"default"``: Phase-3 DA is initialised by warm-starting from the
                              pilot's own ``(step_size, L)`` (the pre-existing behaviour prior to
                              this parameter).  Use this to reproduce results from the previous
                              code path or to suppress E1 entirely.
   :type warmup_step_init: str

   :returns: A :class:`MCLMCLRDAdaptationState` NamedTuple with fields ``L``,
             ``step_size``, ``inverse_mass_matrix`` (a
             :class:`~blackjax.mcmc.metrics.LowRankInverseMassMatrix`), and
             ``diagnostics`` (see :class:`MCLMCLRDAdaptationState` for keys).
   :rtype: MCLMCLRDAdaptationState

   :raises ValueError: If ``inner_kernel`` is not one of ``"mclmc"`` or
       ``"adjusted_mclmc"``.

   .. rubric:: Examples

   .. code-block:: python

       import jax
       import jax.numpy as jnp
       import blackjax

       logdensity_fn = lambda x: -0.5 * jnp.sum(x**2)
       position = jnp.zeros(10)
       rng_key = jax.random.key(42)

       # Unadjusted (stable default):
       result = blackjax.mclmc_lrd_warmup(
           logdensity_fn, position, rng_key,
           k=4, pilot_num_warmup=500, pilot_num_samples=2000,
           lrd_num_steps=1000, num_chains=4,
       )

       # Adjusted (experimental):
       result_adj = blackjax.mclmc_lrd_warmup(
           logdensity_fn, position, rng_key,
           k=4, pilot_num_warmup=500, pilot_num_samples=2000,
           lrd_num_steps=1000, num_chains=4,
           inner_kernel="adjusted_mclmc",
       )

       # Build the production kernel from the unadjusted result:
       import blackjax.mcmc.mclmc as mclmc_mod
       lrd_imm = result.inverse_mass_matrix
       kernel = mclmc_mod.build_kernel()
       init_state = mclmc_mod.init(position, logdensity_fn, jax.random.key(0))
       next_state, info = kernel(
           jax.random.key(1), init_state, logdensity_fn,
           lrd_imm, result.L, result.step_size,
       )


