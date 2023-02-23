:py:mod:`blackjax.adaptation.meads_adaptation`
==============================================

.. py:module:: blackjax.adaptation.meads_adaptation


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   blackjax.adaptation.meads_adaptation.MEADSAdaptationState



Functions
~~~~~~~~~

.. autoapisummary::

   blackjax.adaptation.meads_adaptation.base
   blackjax.adaptation.meads_adaptation.maximum_eigenvalue



.. py:class:: MEADSAdaptationState



   State of the MEADS adaptation scheme.

   step_size
       Value of the step_size parameter of the generalized HMC algorithm.
   position_sigma
       PyTree containing the per dimension sample standard deviation of the
       position variable. Used to scale the momentum variable on the generalized
       HMC algorithm.
   alpha
       Value of the alpha parameter of the generalized HMC algorithm.
   delta
       Value of the alpha parameter of the generalized HMC algorithm.


   .. py:attribute:: current_iteration
      :type: int

      

   .. py:attribute:: step_size
      :type: float

      

   .. py:attribute:: position_sigma
      :type: blackjax.types.PyTree

      

   .. py:attribute:: alpha
      :type: float

      

   .. py:attribute:: delta
      :type: float

      


.. py:function:: base()

   Maximum-Eigenvalue Adaptation of damping and step size for the generalized
   Hamiltonian Monte Carlo kernel :cite:p:`hoffman2022tuning`.


   This algorithm performs a cross-chain adaptation scheme for the generalized
   HMC algorithm that automatically selects values for the generalized HMC's
   tunable parameters based on statistics collected from a population of many
   chains. It uses heuristics determined by the maximum eigenvalue of the
   covariance and gradient matrices given by the grouped samples of all chains
   with shape.

   This is an implementation of Algorithm 3 of :cite:p:`hoffman2022tuning` using cross-chain
   adaptation instead of parallel ensample chain adaptation.

   :returns: * *init* -- Function that initializes the warmup.
             * *update* -- Function that moves the warmup one step.


.. py:function:: maximum_eigenvalue(matrix: blackjax.types.PyTree)

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


