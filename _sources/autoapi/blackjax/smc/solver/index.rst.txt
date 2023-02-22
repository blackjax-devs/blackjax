:py:mod:`blackjax.smc.solver`
=============================

.. py:module:: blackjax.smc.solver

.. autoapi-nested-parse::

   All things solving for adaptive tempering.



Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   blackjax.smc.solver.dichotomy



.. py:function:: dichotomy(fun, _delta0, min_delta, max_delta, eps=0.0001, max_iter=100)

   Solves for delta by dichotomy.

   If max_delta is such that fun(max_delta) > 0, then we assume that max_delta
   can be used as an increment in the tempering.

   :param fun: The decreasing function to solve, we must have fun(min_delta) > 0, fun(max_delta) < 0
   :type fun: Callable
   :param min_delta: Starting point of the interval search
   :type min_delta: float
   :param max_delta: End point of the interval search
   :type max_delta: float
   :param eps: Tolerance for :math:`|f(a) - f(b)|`
   :type eps: float
   :param max_iter: Maximum of iterations in the dichotomy search
   :type max_iter: int

   :returns: **delta** -- The root of `fun`
   :rtype: jnp.ndarray, shape (,)


