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
   blackjax.adaptation.meads_adaptation.meads_adaptation
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
      :type: blackjax.types.ArrayTree

      

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


.. py:function:: meads_adaptation(logdensity_fn: Callable, num_chains: int) -> blackjax.base.AdaptationAlgorithm

   Adapt the parameters of the Generalized HMC algorithm.

   The Generalized HMC algorithm depends on three parameters, each controlling
   one element of its behaviour: step size controls the integrator's dynamics,
   alpha controls the persistency of the momentum variable, and delta controls
   the deterministic transformation of the slice variable used to perform the
   non-reversible Metropolis-Hastings accept/reject step.

   The step size parameter is chosen to ensure the stability of the velocity
   verlet integrator, the alpha parameter to make the influence of the current
   state on future states of the momentum variable to decay exponentially, and
   the delta parameter to maximize the acceptance of proposal but with good
   mixing properties for the slice variable. These characteristics are targeted
   by controlling heuristics based on the maximum eigenvalues of the correlation
   and gradient matrices of the cross-chain samples, under simpifyng assumptions.

   Good tuning is fundamental for the non-reversible Generalized HMC sampling
   algorithm to explore the target space efficienty and output uncorrelated, or
   as uncorrelated as possible, samples from the target space. Furthermore, the
   single integrator step of the algorithm lends itself for fast sampling
   on parallel computer architectures.

   :param logdensity_fn: The log density probability density function from which we wish to sample.
   :param num_chains: Number of chains used for cross-chain warm-up training.

   :returns: * *A function that returns the last cross-chain state, a sampling kernel with the*
             * *tuned parameter values, and all the warm-up states for diagnostics.*


.. py:function:: maximum_eigenvalue(matrix: blackjax.types.ArrayLikeTree) -> blackjax.types.Array

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


