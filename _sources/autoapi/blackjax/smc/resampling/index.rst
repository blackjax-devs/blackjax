:py:mod:`blackjax.smc.resampling`
=================================

.. py:module:: blackjax.smc.resampling

.. autoapi-nested-parse::

   All things resampling.



Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   blackjax.smc.resampling.systematic
   blackjax.smc.resampling.stratified
   blackjax.smc.resampling.multinomial
   blackjax.smc.resampling.residual



.. py:function:: systematic(rng_key: blackjax.types.PRNGKey, weights: blackjax.types.Array, num_samples: int) -> blackjax.types.Array


.. py:function:: stratified(rng_key: blackjax.types.PRNGKey, weights: blackjax.types.Array, num_samples: int) -> blackjax.types.Array


.. py:function:: multinomial(rng_key: blackjax.types.PRNGKey, weights: blackjax.types.Array, num_samples: int) -> blackjax.types.Array


.. py:function:: residual(rng_key: blackjax.types.PRNGKey, weights: blackjax.types.Array, num_samples: int) -> blackjax.types.Array


