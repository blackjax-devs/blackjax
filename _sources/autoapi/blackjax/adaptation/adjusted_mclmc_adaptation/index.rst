blackjax.adaptation.adjusted_mclmc_adaptation
=============================================

.. py:module:: blackjax.adaptation.adjusted_mclmc_adaptation


Attributes
----------

.. autoapisummary::

   blackjax.adaptation.adjusted_mclmc_adaptation.Lratio_lowerbound
   blackjax.adaptation.adjusted_mclmc_adaptation.Lratio_upperbound


Functions
---------

.. autoapisummary::

   blackjax.adaptation.adjusted_mclmc_adaptation.adjusted_mclmc_find_L_and_step_size
   blackjax.adaptation.adjusted_mclmc_adaptation.adjusted_mclmc_make_L_step_size_adaptation
   blackjax.adaptation.adjusted_mclmc_adaptation.adjusted_mclmc_make_adaptation_L
   blackjax.adaptation.adjusted_mclmc_adaptation.handle_nans


Module Contents
---------------

.. py:data:: Lratio_lowerbound
   :value: 0.0


.. py:data:: Lratio_upperbound
   :value: 2.0


.. py:function:: adjusted_mclmc_find_L_and_step_size(mclmc_kernel, logdensity_fn, num_steps, state, rng_key, target, frac_tune1=0.1, frac_tune2=0.1, frac_tune3=0.0, diagonal_preconditioning=True, params=None, max='avg', num_windows=1, tuning_factor=1.3, target_num_integration_steps=2.0)

   Finds the optimal value of the parameters for the MH-MCHMC algorithm.

   :param mclmc_kernel: The kernel function used for the MCMC algorithm.  Must have signature
                        ``(rng_key, state, logdensity_fn, step_size, inverse_mass_matrix,
                        integration_steps_params) -> (state, info)``.
   :param logdensity_fn: The log-density function of the target distribution.  Passed to
                         ``mclmc_kernel`` on every adaptation step.
   :param num_steps: The number of MCMC steps that will subsequently be run, after tuning.
   :param state: The initial state of the MCMC algorithm.
   :param rng_key: The random number generator key.
   :param target: The target acceptance rate for the step size adaptation.
   :param frac_tune1: The fraction of tuning for the first step of the adaptation.
   :param frac_tune2: The fraction of tuning for the second step of the adaptation.
   :param frac_tune3: The fraction of tuning for the third step of the adaptation.
   :param diagonal_preconditioning: Whether to do diagonal preconditioning (i.e. a mass matrix)
   :param params: Initial params to start tuning from (optional)
   :param max: whether to calculate L from maximum or average eigenvalue. Average is advised.
   :param num_windows: how many iterations of the tuning are carried out
   :param tuning_factor: multiplicative factor for L
   :param target_num_integration_steps: The average number of leapfrog integration steps per MH proposal.
                                        The step-size DA is calibrated AT this trajectory length (avg-preserving):
                                        ``L`` is pinned to ``target_num_integration_steps * step_size`` at entry
                                        and tracked throughout adaptation, so the step is calibrated against the
                                        same ``avg`` that the dynamic kernel will use at sampling time.  The final
                                        ``L`` is enforced to ``target_num_integration_steps * step_size`` as an
                                        invariant (a near-NO-OP on the main path; it also fixes any final_da
                                        step/L bookkeeping desync and covers the frac_tune3 > 0 edge path).

                                        **Why this matters at high d:** without avg-preserving calibration, the
                                        step is calibrated against ``avg ≈ 1`` (the √dim reset collapses
                                        ``L/step`` to 1 before pass-2 DA).  Running the dynamic kernel at
                                        ``avg = 2`` with a step sized for ``avg = 1`` doubles the energy error
                                        → acceptance collapses at high dimensionality (d=300: ≈0.22; d=500:
                                        ≈0.21 vs target 0.65).  With avg-preserving calibration, the step is
                                        correctly sized for the operating trajectory length across all d.

                                        **Robustness evidence:** across 7 models × 2 IMM regimes × 3 seeds,
                                        ``avg = 2`` has zero silent failures (inadequate cases fail loudly via
                                        R̂/divergences/acceptance collapse), delivers ≈2× ESS vs ``avg ≈ 1``
                                        (MALA), and ties a per-model ESS/grad search.  Longer trajectories
                                        (``avg = 8``) silently under-sample variance at equal budget.

                                        Default ``2.0`` is the robust sweet spot.  Values below ``1.1`` (the
                                        ``_AVG_FLOOR``) are not reachable with avg-preserving calibration — the
                                        clamp forces ``step ≤ L / 1.1`` and the step converges to zero.  To
                                        recover near-MALA behaviour, use a value like ``1.2`` (just above the
                                        floor); ``1.0`` is not a valid choice with the avg-preserving tuner.

   :rtype: A tuple containing the final state of the MCMC algorithm and the final hyperparameters.


.. py:function:: adjusted_mclmc_make_L_step_size_adaptation(kernel, logdensity_fn, dim, frac_tune1, frac_tune2, target, diagonal_preconditioning, fix_L_first_da=False, max='avg', tuning_factor=1.0, target_num_integration_steps=None)

   Adapts the stepsize and L of the MCLMC kernel. Designed for adjusted MCLMC

   :param target_num_integration_steps: When provided, pass-1 uses ``fix_L=True`` (stable: L anchored at the
                                        entry-pinned value so step cannot diverge) and pass-2 starts with a
                                        re-pin ``L = target_num_integration_steps * step`` to guarantee avg =
                                        target at the start of the avg-preserving DA.  When ``None`` the
                                        pre-2c behaviour is preserved (``fix_L_first_da`` controls pass-1).


.. py:function:: adjusted_mclmc_make_adaptation_L(kernel, logdensity_fn, frac, l_factor, max='avg', eigenvector=None)

   determine L by the autocorrelations (around 10 effective samples are needed for this to be accurate)


.. py:function:: handle_nans(previous_state, next_state, step_size, step_size_max, kinetic_change)

   if there are nans, let's reduce the stepsize, and not update the state. The
   function returns the old state in this case.


