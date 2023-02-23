:py:mod:`blackjax.adaptation.mass_matrix`
=========================================

.. py:module:: blackjax.adaptation.mass_matrix

.. autoapi-nested-parse::

   Algorithms to adapt the mass matrix used by algorithms in the Hamiltonian
   Monte Carlo family to the current geometry.

   The Stan Manual :cite:p:`stan_hmc_param` is a very good reference on automatic tuning of
   parameters used in Hamiltonian Monte Carlo.



Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   blackjax.adaptation.mass_matrix.WelfordAlgorithmState
   blackjax.adaptation.mass_matrix.MassMatrixAdaptationState



Functions
~~~~~~~~~

.. autoapisummary::

   blackjax.adaptation.mass_matrix.mass_matrix_adaptation
   blackjax.adaptation.mass_matrix.welford_algorithm



.. py:class:: WelfordAlgorithmState



   State carried through the Welford algorithm.

   mean
       The running sample mean.
   m2
       The running value of the sum of difference of squares. See documentation
       of the `welford_algorithm` function for an explanation.
   sample_size
       The number of successive states the previous values have been computed on;
       also the current number of iterations of the algorithm.


   .. py:attribute:: mean
      :type: blackjax.types.Array

      

   .. py:attribute:: m2
      :type: blackjax.types.Array

      

   .. py:attribute:: sample_size
      :type: int

      


.. py:class:: MassMatrixAdaptationState



   State carried through the mass matrix adaptation.

   inverse_mass_matrix
       The curent value of the inverse mass matrix.
   wc_state
       The current state of the Welford Algorithm.


   .. py:attribute:: inverse_mass_matrix
      :type: blackjax.types.Array

      

   .. py:attribute:: wc_state
      :type: WelfordAlgorithmState

      


.. py:function:: mass_matrix_adaptation(is_diagonal_matrix: bool = True) -> Tuple[Callable, Callable, Callable]

   Adapts the values in the mass matrix by computing the covariance
   between parameters.

   :param is_diagonal_matrix: When True the algorithm adapts and returns a diagonal mass matrix
                              (default), otherwise adaps and returns a dense mass matrix.

   :returns: * *init* -- A function that initializes the step of the mass matrix adaptation.
             * *update* -- A function that updates the state of the mass matrix.
             * *final* -- A function that computes the inverse mass matrix based on the current
               state.


.. py:function:: welford_algorithm(is_diagonal_matrix: bool) -> Tuple[Callable, Callable, Callable]

   Welford's online estimator of covariance.

   It is possible to compute the variance of a population of values in an
   on-line fashion to avoid storing intermediate results. The naive recurrence
   relations between the sample mean and variance at a step and the next are
   however not numerically stable.

   Welford's algorithm uses the sum of square of differences
   :math:`M_{2,n} = \sum_{i=1}^n \left(x_i-\overline{x_n}\right)^2`
   for updating where :math:`x_n` is the current mean and the following
   recurrence relationships

   :param is_diagonal_matrix: When True the algorithm adapts and returns a diagonal mass matrix
                              (default), otherwise adaps and returns a dense mass matrix.

   .. note::

      It might seem pedantic to separate the Welford algorithm from mass adaptation,
      but this covariance estimator is used in other parts of the library.


