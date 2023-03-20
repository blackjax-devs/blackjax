:py:mod:`blackjax.mcmc.termination`
===================================

.. py:module:: blackjax.mcmc.termination


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   blackjax.mcmc.termination.IterativeUTurnState



Functions
~~~~~~~~~

.. autoapisummary::

   blackjax.mcmc.termination.iterative_uturn_numpyro



.. py:class:: IterativeUTurnState



   .. py:attribute:: momentum
      :type: blackjax.types.Array

      

   .. py:attribute:: momentum_sum
      :type: blackjax.types.Array

      

   .. py:attribute:: idx_min
      :type: int

      

   .. py:attribute:: idx_max
      :type: int

      


.. py:function:: iterative_uturn_numpyro(is_turning)

   Numpyro style dynamic U-Turn criterion.


