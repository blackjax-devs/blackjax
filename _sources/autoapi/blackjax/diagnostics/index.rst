blackjax.diagnostics
====================

.. py:module:: blackjax.diagnostics

.. autoapi-nested-parse::

   MCMC diagnostics.



Functions
---------

.. autoapisummary::

   blackjax.diagnostics.potential_scale_reduction
   blackjax.diagnostics.rhat
   blackjax.diagnostics.effective_sample_size
   blackjax.diagnostics.ess_bulk
   blackjax.diagnostics.ess_tail
   blackjax.diagnostics.pareto_khat
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


.. py:function:: rhat(input_array: blackjax.types.ArrayLike, chain_axis: int = 0, sample_axis: int = 1) -> blackjax.types.Array

   Rank-normalized split-R̂ (Vehtari et al. 2021).

   The modern improved R̂ diagnostic.  Combines two split-chain R̂ values —
   one on rank-normalized draws and one on rank-normalized *folded* draws —
   and returns the maximum.  The folded component catches scale/variance
   non-convergence that the bulk component can miss.

   This matches the default ``az.rhat(method="rank")`` convention in ArviZ.

   :param input_array: An array representing multiple chains of MCMC samples. The array must
                       contain a chain dimension and a sample dimension.  At least 2 chains
                       and at least 4 draws per chain are required.
   :param chain_axis: The axis indicating the multiple chains. Default 0.
   :param sample_axis: The axis indicating a single chain of MCMC samples. Default 1.

   :returns: * *NDArray of the resulting R̂ values, with chain and sample dimensions*
             * *squeezed.  Values close to 1.0 indicate convergence; values above 1.01*
             * *suggest chains have not converged.*

   .. rubric:: Notes

   Algorithm (Vehtari et al. 2021, § 4):

   1. Split each chain in half → 2× chains.
   2. Rank-normalize with the Blom plotting position
      :math:`z_r = \Phi^{-1}((r - 3/8) / (n + 1/4))` over the joint pool.
   3. Compute the standard split-R̂ on the rank-normalized draws (**bulk**).
   4. Compute the folded draws :math:`|x - \mathrm{median}(x)|`, rank-normalize
      them, and compute split-R̂ again (**tail**).
   5. Return :math:`\max(\hat{R}_{\text{bulk}}, \hat{R}_{\text{tail}})`.

   .. rubric:: References

   .. cite:p:`vehtari2021rank`


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


.. py:function:: ess_bulk(input_array: blackjax.types.ArrayLike, chain_axis: int = 0, sample_axis: int = 1) -> blackjax.types.Array

   Bulk effective sample size (rank-normalized split-chain ESS).

   Computes the bulk ESS from Vehtari et al. (2021): rank-normalizes draws
   after splitting each chain in half, then applies the standard
   autocorrelation-based :func:`effective_sample_size` estimator.  This
   diagnostic is robust to non-stationarity and multimodality.

   :param input_array: An array representing multiple chains of MCMC samples. The array must
                       contain a chain dimension and a sample dimension.
   :param chain_axis: The axis indicating the multiple chains. Default 0.
   :param sample_axis: The axis indicating a single chain of MCMC samples. Default 1.

   :rtype: NDArray of the resulting bulk-ESS, with chain and sample dimensions squeezed.

   .. rubric:: Notes

   Algorithm:

   1. Split each chain in half → 2× chains.
   2. Pool all draws and rank-normalize with :math:`z_r = \Phi^{-1}((r-3/8)/(n+1/4))`.
   3. Apply :func:`effective_sample_size` to the rank-normalized draws.

   .. rubric:: References

   .. cite:p:`vehtari2021rank`


.. py:function:: ess_tail(input_array: blackjax.types.ArrayLike, chain_axis: int = 0, sample_axis: int = 1) -> blackjax.types.Array

   Tail effective sample size.

   Computes the tail ESS from Vehtari et al. (2021) as the minimum of the
   ESS of the 5th- and 95th-percentile indicator functions applied to
   split-chain draws.

   :param input_array: An array representing multiple chains of MCMC samples. The array must
                       contain a chain dimension and a sample dimension.
   :param chain_axis: The axis indicating the multiple chains. Default 0.
   :param sample_axis: The axis indicating a single chain of MCMC samples. Default 1.

   :rtype: NDArray of the resulting tail-ESS, with chain and sample dimensions squeezed.

   .. rubric:: Notes

   Algorithm:

   1. Split each chain in half → 2× chains.
   2. Compute pooled 5th and 95th quantiles across all split chains and draws.
   3. Form indicator series :math:`\mathbf{1}(x \le q_{0.05})` and
      :math:`\mathbf{1}(x \ge q_{0.95})`.
   4. Compute :func:`effective_sample_size` for each indicator.
   5. Return :math:`\min(\text{ESS}_\text{lower}, \text{ESS}_\text{upper})`.

   .. rubric:: References

   .. cite:p:`vehtari2021rank`


.. py:function:: pareto_khat(x: blackjax.types.ArrayLike, tail: str = 'both', tail_frac: float = 0.1) -> blackjax.types.Array

   Pareto shape parameter k̂ for tail diagnosis.

   Fits a Generalised Pareto Distribution (GPD) to the upper and/or lower
   tail of a 1-D sample and returns the estimated shape parameter k̂.

   :param x: 1-D array of draws (or any array; it is ravelled before use).
   :param tail: Which tail to fit: ``"upper"``, ``"lower"``, or ``"both"`` (default).
                When ``"both"``, returns the maximum of the two k̂ estimates.
   :param tail_frac: Fraction of samples used as the tail. Default 0.10 (10 %).
                     A minimum of 5 tail samples is always enforced.

   :returns: * **Scalar Array** (*the Pareto shape estimate k̂.  Values below 0.5 indicate*)
             * *reliable tail estimates; 0.5–0.7 are moderate; above 0.7 may be*
             * *unreliable.*

   .. rubric:: Notes

   Uses the Zhang & Stephens (2009) empirical-Bayes estimator implemented
   in the internal :func:`_gpdfit`.  The upper tail is modelled directly;
   the lower tail is reflected and modelled as an upper tail.


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


