:py:mod:`blackjax.smc.tuning.from_kernel_info`
==============================================

.. py:module:: blackjax.smc.tuning.from_kernel_info

.. autoapi-nested-parse::

   strategies to tune the parameters of mcmc kernels
   used within smc, based on MCMC states



Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   blackjax.smc.tuning.from_kernel_info.update_scale_from_acceptance_rate



.. py:function:: update_scale_from_acceptance_rate(scales: jax.Array, acceptance_rates: jax.Array, target_acceptance_rate: float = 0.234) -> jax.Array

   Given N chains from some MCMC algorithm like Random Walk Metropolis
   and N scale factors, each associated to a different chain.
   Updates the scale factors taking into account acceptance rates and
   the average acceptance rate.

   Under certain assumptions it is known that the optimal acceptance rate
   of Metropolis Hastings is 0.4 for 1 dimension and converges to
   0.234 in infinite dimensions. In practice, 0.234 is a reasonable
   assumption for 5 or more dimensions.

   If certain chain is below optimal acceptance rate, its scale will decrease
   and if its above, its scale will increase,
   -------

   :param scales: (n_chains) array consisting of N scale factors, associated to N markov chains
   :param acceptance_rates: (n_chains) acceptance rate of the N markov chains
   :param target_acceptance_rate: a float with a desirable acceptance rate for the chains.

   :returns: * *(n_chains) new scales, with the aim of getting acceptance rates closer to target*
             * *if the chains were to be run again.*


