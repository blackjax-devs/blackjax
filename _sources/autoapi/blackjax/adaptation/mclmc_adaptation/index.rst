:py:mod:`blackjax.adaptation.mclmc_adaptation`
==============================================

.. py:module:: blackjax.adaptation.mclmc_adaptation

.. autoapi-nested-parse::

   Algorithms to adapt the MCLMC kernel parameters, namely step size and L.



Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   blackjax.adaptation.mclmc_adaptation.MCLMCAdaptationState



Functions
~~~~~~~~~

.. autoapisummary::

   blackjax.adaptation.mclmc_adaptation.mclmc_find_L_and_step_size
   blackjax.adaptation.mclmc_adaptation.make_L_step_size_adaptation
   blackjax.adaptation.mclmc_adaptation.make_adaptation_L
   blackjax.adaptation.mclmc_adaptation.handle_nans



.. py:class:: MCLMCAdaptationState




   Represents the tunable parameters for MCLMC adaptation.

   L
       The momentum decoherent rate for the MCLMC algorithm.
   step_size
       The step size used for the MCLMC algorithm.

   .. py:attribute:: L
      :type: float

      

   .. py:attribute:: step_size
      :type: float

      


.. py:function:: mclmc_find_L_and_step_size(mclmc_kernel, num_steps, state, rng_key, frac_tune1=0.1, frac_tune2=0.1, frac_tune3=0.1, desired_energy_var=0.0005, trust_in_estimate=1.5, num_effective_samples=150)

   Finds the optimal value of the parameters for the MCLMC algorithm.

   :param mclmc_kernel: The kernel function used for the MCMC algorithm.
   :param num_steps: The number of MCMC steps that will subsequently be run, after tuning.
   :param state: The initial state of the MCMC algorithm.
   :param rng_key: The random number generator key.
   :param frac_tune1: The fraction of tuning for the first step of the adaptation.
   :param frac_tune2: The fraction of tuning for the second step of the adaptation.
   :param frac_tune3: The fraction of tuning for the third step of the adaptation.
   :param desired_energy_va: The desired energy variance for the MCMC algorithm.
   :param trust_in_estimate: The trust in the estimate of optimal stepsize.
   :param num_effective_samples: The number of effective samples for the MCMC algorithm.

   :rtype: A tuple containing the final state of the MCMC algorithm and the final hyperparameters.

   .. rubric:: Examples

   .. code::

       # Define the kernel function
       def kernel(x):
           return x ** 2

       # Define the initial state
       initial_state = MCMCState(position=0, momentum=1)

       # Generate a random number generator key
       rng_key = jax.random.key(0)

       # Find the optimal parameters for the MCLMC algorithm
       final_state, final_params = mclmc_find_L_and_step_size(
           mclmc_kernel=kernel,
           num_steps=1000,
           state=initial_state,
           rng_key=rng_key,
           frac_tune1=0.2,
           frac_tune2=0.3,
           frac_tune3=0.1,
           desired_energy_var=1e-4,
           trust_in_estimate=2.0,
           num_effective_samples=200,
       )


.. py:function:: make_L_step_size_adaptation(kernel, dim, frac_tune1, frac_tune2, desired_energy_var=0.001, trust_in_estimate=1.5, num_effective_samples=150)

   Adapts the stepsize and L of the MCLMC kernel. Designed for the unadjusted MCLMC


.. py:function:: make_adaptation_L(kernel, frac, Lfactor)

   determine L by the autocorrelations (around 10 effective samples are needed for this to be accurate)


.. py:function:: handle_nans(previous_state, next_state, step_size, step_size_max, kinetic_change)

   if there are nans, let's reduce the stepsize, and not update the state. The
   function returns the old state in this case.


