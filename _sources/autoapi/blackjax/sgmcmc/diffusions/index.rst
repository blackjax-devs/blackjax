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
   blackjax.sgmcmc.diffusions.sgnht



.. py:function:: overdamped_langevin()

   Euler solver for overdamped Langevin diffusion.

   This algorithm was ported from :cite:p:`coullon2022sgmcmcjax`.



.. py:function:: sghmc(alpha: float = 0.01, beta: float = 0)

   Euler solver for the diffusion equation of the SGHMC algorithm :cite:p:`chen2014stochastic`,
   with parameters alpha and beta scaled according to :cite:p:`ma2015complete`.

   This algorithm was ported from :cite:p:`coullon2022sgmcmcjax`.



.. py:function:: sgnht(alpha: float = 0.01, beta: float = 0)

   Euler solver for the diffusion equation of the SGNHT algorithm :cite:p:`ding2014bayesian`.

   This algorithm was ported from :cite:p:`coullon2022sgmcmcjax`.



