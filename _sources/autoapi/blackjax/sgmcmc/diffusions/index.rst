:py:mod:`blackjax.sgmcmc.diffusions`
====================================

.. py:module:: blackjax.sgmcmc.diffusions

.. autoapi-nested-parse::

   Solvers for Langevin diffusions.



Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   blackjax.sgmcmc.diffusions.overdamped_langevin
   blackjax.sgmcmc.diffusions.sghmc



.. py:function:: overdamped_langevin()

   Euler solver for overdamped Langevin diffusion.

   This algorithm was ported from :cite:p:`coullon2022sgmcmcjax`.



.. py:function:: sghmc(alpha: float = 0.01, beta: float = 0)

   Solver for the diffusion equation of the SGHMC algorithm :cite:p:`chen2014stochastic`.

   This algorithm was ported from :cite:p:`coullon2022sgmcmcjax`.



