blackjax.eca
============

.. py:module:: blackjax.eca

.. autoapi-nested-parse::

   Ensemble Chain Adaptation (ECA) utilities for multi-device parallel sampling.



Functions
---------

.. autoapisummary::

   blackjax.eca.eca_step
   blackjax.eca.add_splitR
   blackjax.eca.add_all_chains_info
   blackjax.eca.while_with_info
   blackjax.eca.run_eca
   blackjax.eca.ensemble_execute_fn


Module Contents
---------------

.. py:function:: eca_step(kernel, summary_statistics_fn, adaptation_update, num_chains, superchain_size=None, all_chains_info=None)

   Construct a single step of ensemble chain adaptation (eca) to be performed in parallel on multiple devices.


.. py:function:: add_splitR(step, num_chains, superchain_size)

.. py:function:: add_all_chains_info(step, all_chains_info)

.. py:function:: while_with_info(step, init, xs, length, while_cond)

   Same syntax and usage as jax.lax.scan, but it is run as a while loop that is terminated if not while_cond(state).
   len(xs) determines the maximum number of iterations.


.. py:function:: run_eca(rng_key, initial_state, kernel, adaptation, num_steps, num_chains, mesh, superchain_size=None, all_chains_info=None, early_stop=False)

   Run ensemble chain adaptation (eca) in parallel on multiple devices.
   -----------------------------------------------------
   :param rng_key: random key
   :param initial_state: initial state of the system
   :param kernel: kernel for the dynamics
   :param adaptation: adaptation object
   :param num_steps: number of steps to run
   :param num_chains: number of chains
   :param mesh: mesh for parallelization
   :param all_chains_info: function that takes the state of the system and returns some summary statistics. Will be applied and stored for all the chains at each step so it can be memory intensive.
   :param early_stop: whether to stop early

   :returns: final state of the system
             final_adaptation_state: final adaptation state
             info_history: history of the information that was stored at each step (if early_stop is False, then this is None)
   :rtype: final_state


.. py:function:: ensemble_execute_fn(func, rng_key, num_chains, mesh, x=None, args=None, summary_statistics_fn=lambda y: 0.0, superchain_size=None)

   Given a sequential function
    func(rng_key, x, args) = y,
   evaluate it with an ensemble and also compute some summary statistics E[theta(y)], where expectation is taken over ensemble.
   :param x: array distributed over all decvices
   :param args: additional arguments for func, not distributed.
   :param summary_statistics_fn: operates on a chain and returns some summary statistics.
   :param rng_key: a single random key, which will then be split, such that chain will get a different random key.

   :returns: array distributed over all decvices. Need not be of the same shape as x.
             Etheta: expected values of the summary statistics
   :rtype: y


