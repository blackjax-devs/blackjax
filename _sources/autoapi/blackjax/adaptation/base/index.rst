:py:mod:`blackjax.adaptation.base`
==================================

.. py:module:: blackjax.adaptation.base


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   blackjax.adaptation.base.AdaptationResults
   blackjax.adaptation.base.AdaptationInfo



Functions
~~~~~~~~~

.. autoapisummary::

   blackjax.adaptation.base.return_all_adapt_info
   blackjax.adaptation.base.get_filter_adapt_info_fn



.. py:class:: AdaptationResults




   .. py:attribute:: state
      :type: blackjax.types.ArrayTree

      

   .. py:attribute:: parameters
      :type: dict

      


.. py:class:: AdaptationInfo




   .. py:attribute:: state
      :type: NamedTuple

      

   .. py:attribute:: info
      :type: NamedTuple

      

   .. py:attribute:: adaptation_state
      :type: NamedTuple

      


.. py:function:: return_all_adapt_info(state, info, adaptation_state)

   Return fully populated AdaptationInfo.  Used for adaptation_info_fn
   parameters of the adaptation algorithms.


.. py:function:: get_filter_adapt_info_fn(state_keys: Set[str] = set(), info_keys: Set[str] = set(), adapt_state_keys: Set[str] = set())

   Generate a function to filter what is saved in AdaptationInfo.  Used
   for adptation_info_fn parameters of the adaptation algorithms.
   adaptation_info_fn=get_filter_adapt_info_fn() saves no auxiliary information


