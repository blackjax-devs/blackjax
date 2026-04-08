blackjax.diagnostics
====================

.. py:module:: blackjax.diagnostics

.. autoapi-nested-parse::

   MCMC diagnostics.



Functions
---------

.. autoapisummary::

   blackjax.diagnostics.potential_scale_reduction
   blackjax.diagnostics.effective_sample_size
   blackjax.diagnostics.psis_weights


Module Contents
---------------

.. py:function:: potential_scale_reduction(input_array: blackjax.types.ArrayLike, chain_axis: int = 0, sample_axis: int = 1) -> blackjax.types.Array

   Gelman and Rubin (1992)'s potential scale reduction for computing multiple MCMC chain convergence.

   :param input_array: An array representing multiple chains of MCMC samples. The array must
                       contains a chain dimension and a sample dimension.
   :param chain_axis: The axis indicating the multiple chains. Default to 0.
   :param sample_axis: The axis indicating a single chain of MCMC samples. Default to 1.

   :rtype: NDArray of the resulting statistics (r-hat), with the chain and sample dimensions squeezed.

   .. rubric:: Notes

   The diagnostic is computed by:

   .. math:: \hat{R} = \frac{\hat{V}}{W}

   where :math:`W` is the within-chain variance and :math:`\hat{V}` is the posterior variance
   estimate for the pooled traces. This is the potential scale reduction factor, which
   converges to unity when each of the traces is a sample from the target posterior. Values
   greater than one indicate that one or more chains have not yet converged :cite:p:`stan_rhat,gelman1992inference`.


.. py:function:: effective_sample_size(input_array: blackjax.types.ArrayLike, chain_axis: int = 0, sample_axis: int = 1) -> blackjax.types.Array

   Compute estimate of the effective sample size (ess).

   :param input_array: An array representing multiple chains of MCMC samples. The array must
                       contains a chain dimension and a sample dimension.
   :param chain_axis: The axis indicating the multiple chains. Default to 0.
   :param sample_axis: The axis indicating a single chain of MCMC samples. Default to 1.

   :rtype: NDArray of the resulting statistics (ess), with the chain and sample dimensions squeezed.

   .. rubric:: Notes

   The basic ess (:math:`N_{\mathit{eff}}`) diagnostic is computed by:

   .. math:: \hat{N}_{\mathit{eff}} = \frac{MN}{\hat{\tau}}

   .. math:: \hat{\tau} = -1 + 2 \sum_{t'=0}^K \hat{P}_{t'}

   where :math:`M` is the number of chains, :math:`N` the number of draws,
   :math:`\hat{\rho}_t` is the estimated _autocorrelation at lag :math:`t`, and
   :math:`K` is the last integer for which :math:`\hat{P}_{K} = \hat{\rho}_{2K} +
   \hat{\rho}_{2K+1}` is still positive :cite:p:`stan_ess,gelman1995bayesian`.

   The current implementation is similar to Stan, which uses Geyer's initial monotone sequence
   criterion :cite:p:`geyer1992practical,geyer2011introduction`.


.. py:function:: psis_weights(log_ratios: blackjax.types.Array, r_eff: float = 1.0) -> tuple[blackjax.types.Array, blackjax.types.Array]

   Pareto Smoothed Importance Sampling (PSIS) log weights.

   Implements the PSIS smoothing step from :cite:p:`vehtari2017practical`:
   the ``M`` largest importance ratios (in ratio space) are replaced by sorted
   Generalised Pareto quantiles fitted by the empirical Bayes estimator of
   Zhang & Stephens (2009), then all weights are normalised.

   This is a pure-JAX, JIT-compatible implementation faithful to Algorithm 1
   of Vehtari, Gelman & Gabry (2017).

   :param log_ratios: Log importance ratios ``log p(θ) − log q(θ)``, shape ``(n,)``.
                      Need not be normalised.
   :param r_eff: Relative effective sample size of the proposal, ``S_eff / n``.
                 Use the default of ``1.0`` for i.i.d. draws (e.g. Pathfinder);
                 set to the actual ESS ratio for correlated MCMC chains.  Values
                 below 1 increase the tail size ``M`` to compensate for correlation.

   :returns: * *log_weights* -- Normalised log importance weights, shape ``(n,)``.
               ``jnp.exp(log_weights).sum() == 1`` up to floating-point precision.
             * *pareto_k* -- Pareto shape parameter estimate (scalar ``Array``).  Values below 0.5
               indicate reliable estimates; 0.5–0.7 are moderate; above 0.7 may give
               unreliable estimates.  ``jnp.inf`` means the tail was too small to fit
               (fewer than 5 samples).

   .. rubric:: Notes

   Tail size: ``M = min(floor(3*sqrt(n/r_eff)), n//5)``, matching the paper.
   The GPD is only applied when ``k >= 1/3``; lighter tails are left
   unsmoothed (only normalised).  Fitting uses empirical Bayes in
   importance-ratio space, the same approach as ArviZ.


