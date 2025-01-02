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


.. py:function:: adjusted_mclmc_find_L_and_step_size(mclmc_kernel, num_steps, state, rng_key, target, frac_tune1=0.1, frac_tune2=0.1, frac_tune3=0.0, diagonal_preconditioning=True, params=None, max='avg', num_windows=1, tuning_factor=1.3)

   Finds the optimal value of the parameters for the MH-MCHMC algorithm.

   :param mclmc_kernel: The kernel function used for the MCMC algorithm.
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

   :rtype: A tuple containing the final state of the MCMC algorithm and the final hyperparameters.


.. py:function:: adjusted_mclmc_make_L_step_size_adaptation(kernel, dim, frac_tune1, frac_tune2, target, diagonal_preconditioning, fix_L_first_da=False, max='avg', tuning_factor=1.0)

   Adapts the stepsize and L of the MCLMC kernel. Designed for adjusted MCLMC


.. py:function:: adjusted_mclmc_make_adaptation_L(kernel, frac, Lfactor, max='avg', eigenvector=None)

   determine L by the autocorrelations (around 10 effective samples are needed for this to be accurate)


.. py:function:: handle_nans(previous_state, next_state, step_size, step_size_max, kinetic_change)

   if there are nans, let's reduce the stepsize, and not update the state. The
   function returns the old state in this case.


