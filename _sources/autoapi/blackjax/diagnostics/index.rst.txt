:py:mod:`blackjax.diagnostics`
==============================

.. py:module:: blackjax.diagnostics

.. autoapi-nested-parse::

   MCMC diagnostics.



Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   blackjax.diagnostics.potential_scale_reduction
   blackjax.diagnostics.effective_sample_size



.. py:function:: potential_scale_reduction(input_array: blackjax.types.Array, chain_axis: int = 0, sample_axis: int = 1)

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


.. py:function:: effective_sample_size(input_array: blackjax.types.Array, chain_axis: int = 0, sample_axis: int = 1)

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


