:py:mod:`blackjax.optimizers.dual_averaging`
============================================

.. py:module:: blackjax.optimizers.dual_averaging


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   blackjax.optimizers.dual_averaging.DualAveragingState



Functions
~~~~~~~~~

.. autoapisummary::

   blackjax.optimizers.dual_averaging.dual_averaging



.. py:class:: DualAveragingState



   State carried through the dual averaging procedure.

   log_x
       The logarithm of the current state
   log_x_avg
       The time-weighted average of the values that the logarithm of the state
       has taken so far.
   step
       The current iteration step.
   avg_err
       The time average of the value of the quantity :math:`H_t`, the
       difference between the target acceptance rate and the current
       acceptance rate.
   mu
       Arbitrary point the values of log_step_size are shrunk towards. Chose
       to be :math:`\log(10 \epsilon_0)` where :math:`\epsilon_0` is chosen
       in this context to be the step size given by the
       `find_reasonable_step_size` procedure.


   .. py:attribute:: log_x
      :type: float

      

   .. py:attribute:: log_x_avg
      :type: float

      

   .. py:attribute:: step
      :type: int

      

   .. py:attribute:: avg_error
      :type: float

      

   .. py:attribute:: mu
      :type: float

      


.. py:function:: dual_averaging(t0: int = 10, gamma: float = 0.05, kappa: float = 0.75) -> Tuple[Callable, Callable, Callable]

   Find the state that minimizes an objective function using a primal-dual
   subgradient method.

   See :cite:p:`nesterov2009primal` for a detailed explanation of the algorithm and its mathematical
   properties.

   :param t0: Free parameter that stabilizes the initial iterations of the algorithm.
              Large values may slow down convergence. Introduced in :cite:p:`hoffman2014no` with a default
              value of 10.
   :type t0: float >= 0
   :param gamma: Controls the speed of convergence of the scheme. The authors of :cite:p:`hoffman2014no` recommend
                 a value of 0.05.
   :param kappa: Controls the weights of past steps in the current update. The scheme will
                 quickly forget earlier step for a small value of `kappa`. Introduced
                 in :cite:p:`hoffman2014no`, with a recommended value of .75
   :type kappa: float in ]0.5, 1]

   :returns: * *init* -- A function that initializes the state of the dual averaging scheme.
             * *update* -- a function that updates the state of the dual averaging scheme.
             * *final* -- a function that returns the state that minimizes the objective function.


